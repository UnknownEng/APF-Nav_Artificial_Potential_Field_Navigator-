#!/usr/bin/env python3
"""
APF 3D Navigation Node - VORTEX
(formerly dwa_inspection_sucess.py)

Attractive force to the current waypoint, repulsive "vortex" force from 2D
lidar returns, wall/floor repulsion, and an orthogonal escape manoeuvre for
local minima.

Fixes over the previous version:
1. Single command publisher: the keepalive timer only publishes while the
   navigator is NOT running (pre-takeoff / after mission), so it no longer
   injects zero-velocity commands into active navigation.
2. Force -> velocity keeps direction and magnitude (no per-axis clipping), so
   opposing attraction/repulsion actually slows the drone down.
3. Dynamic gain tuning is bounded relative to the configured gains, so the
   configured repulsion gain is the one that is used; attraction gain is reset
   for every new waypoint.
4. The drone yaws to face the current waypoint, scan points are remembered for
   a few seconds, and it only creeps (0.1 m/s) toward the lidar's blind sector.
5. The lidar's full mount transform comes from TF. The hector Hokuyo is mounted
   upside down (roll = pi), which mirrors the scan left/right; ignoring that
   placed every obstacle on the wrong side. Points are then projected with the
   full body attitude so floor hits while tilted are dropped.
6. Escape picks the clearer side, near-goal repulsion never drops below a
   floor, stale scans make the drone hover, and a waypoint is skipped (and
   reported) after repeated escapes or a timeout instead of looping forever.
"""

import collections
import threading

import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# Room walls are at +-15 m (see worlds/room_6x6.world).
ROOM_HALF_SIZE = 15.0

# Wall repulsion kicks in when drone gets within this distance of boundary
WALL_INFLUENCE = 5.0
WALL_GAIN      = 16.0
# The floor needs a much shorter reach: with the wall values it pushed up
# hard enough at 2.5 m to cancel the altitude controller, so the drone never
# got within goal_tol_z of 2 m waypoints.
FLOOR_INFLUENCE = 1.0
FLOOR_GAIN      = 4.0


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_rot(q):
    """Rotation matrix from a quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class APFParams:
    def __init__(self):
        self.attraction_gain       = rospy.get_param('~attraction_gain', 0.6)
        # Open-space repulsion gain. Repulsion is summed over every lidar hit
        # within repulsion_distance (~20 hits from a flat face), so small gains
        # are already strong. The previous nominal 15.0 was never actually used
        # (tuning clamped it into 1..3); 1..3 is the range that flew well.
        self.repulsion_gain        = rospy.get_param('~repulsion_gain', 1.0)
        self.base_attraction_gain  = self.attraction_gain
        self.base_repulsion_gain   = self.repulsion_gain
        self.repulsion_distance    = 2.0
        self.dt                    = 0.1
        self.use_adaptive          = True
        self.goal_influence_distance = 3.0
        # Near the goal repulsion is scaled down, but never below this factor,
        # and never for obstacles closer than close_obstacle_dist.
        self.min_adaptive_factor   = 0.3
        self.close_obstacle_dist   = 1.0
        self.enable_escape         = True
        self.escape_force_magnitude = 2.0
        self.escape_duration       = 20
        self.escape_threshold      = 0.05
        self.dynamic_tuning        = True
        # Dynamic tuning range, relative to the configured gains.
        self.min_repulsion_gain    = self.base_repulsion_gain
        self.max_repulsion_gain    = 3.0 * self.base_repulsion_gain
        self.min_attraction_gain   = 0.5 * self.base_attraction_gain
        self.max_rep_force         = 3.0
        self.max_att_force         = 1.5
        self.max_force             = 2.5
        self.robot_radius          = 0.5
        self.v_max  = rospy.get_param('~v_max',  0.3)
        self.v_min  = rospy.get_param('~v_min',  0.0)
        self.w_max  = rospy.get_param('~w_max',  0.8)
        self.vz_max = rospy.get_param('~vz_max', 0.3)
        self.vz_min = rospy.get_param('~vz_min', -0.3)


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Default lidar mount for hector's quadrotor_hokuyo_utm30lx (used only if TF
# is unavailable): 0.097 m below base_link, upside down (roll = pi).
DEFAULT_SCAN_R = np.diag([1.0, -1.0, -1.0])
DEFAULT_SCAN_T = np.array([0.0, 0.0, -0.097])


def scan_fov_in_body(scan_msg, R_bs):
    """(centre, half_width) of the scan's horizontal field of view, expressed
    as body-frame bearings (handles mirrored / rotated mounts)."""
    mid = 0.5 * (scan_msg.angle_min + scan_msg.angle_max)
    d = R_bs @ np.array([np.cos(mid), np.sin(mid), 0.0])
    return float(np.arctan2(d[1], d[0])), 0.5 * (scan_msg.angle_max - scan_msg.angle_min)


def laser_to_obstacles(scan_msg, robot_pos, robot_quat, R_bs=None, t_bs=None,
                       max_rays=80, floor_z=0.15):
    """Project (down-sampled) scan returns into world-frame 3D points.

    R_bs / t_bs: lidar frame -> base_link rotation and translation (from TF).
    Uses the full body orientation, so beams that hit the floor while the
    drone is pitched/rolled are recognised and dropped instead of becoming
    phantom obstacles.
    """
    R_bs = DEFAULT_SCAN_R if R_bs is None else R_bs
    t_bs = DEFAULT_SCAN_T if t_bs is None else t_bs
    r = np.asarray(scan_msg.ranges, dtype=float)
    n = len(r)
    if n == 0:
        return np.empty((0, 3))
    idx = np.arange(0, n, max(1, n // max_rays))
    r = r[idx]
    ang = scan_msg.angle_min + idx * scan_msg.angle_increment
    ok = np.isfinite(r) & (r > scan_msg.range_min) & (r < scan_msg.range_max - 0.1)
    r, ang = r[ok], ang[ok]
    if len(r) == 0:
        return np.empty((0, 3))

    pts_scan = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(len(r))])
    pts_body = pts_scan @ R_bs.T + t_bs
    pts = robot_pos + pts_body @ quat_to_rot(robot_quat).T
    return pts[pts[:, 2] > floor_z]


def wall_repulsion(robot_pos):
    F_wall = np.zeros(3)

    def boundary_force(dist_to_wall, direction, influence=WALL_INFLUENCE, gain=WALL_GAIN):
        if 0 < dist_to_wall < influence:
            mag = gain * (1.0/dist_to_wall - 1.0/influence) ** 2
            return mag * direction
        return np.zeros(3)

    x, y, z = robot_pos
    H = ROOM_HALF_SIZE

    F_wall += boundary_force(x + H, np.array([1, 0, 0]))
    F_wall += boundary_force(H - x, np.array([-1, 0, 0]))
    F_wall += boundary_force(y + H, np.array([0, 1, 0]))
    F_wall += boundary_force(H - y, np.array([0, -1, 0]))
    F_wall += boundary_force(z - 0.0, np.array([0, 0, 1]), FLOOR_INFLUENCE, FLOOR_GAIN)

    wn = np.linalg.norm(F_wall)
    if wn > 4.0:
        F_wall = F_wall / wn * 4.0

    return F_wall


def calculate_apf_forces(robot_pos, goal_pos, obstacle_pts, params):
    dist_to_goal = np.linalg.norm(robot_pos - goal_pos)

    # 1. Attractive force (capped)
    F_att = params.attraction_gain * (goal_pos - robot_pos)
    att_norm = np.linalg.norm(F_att)
    if att_norm > params.max_att_force:
        F_att = F_att / att_norm * params.max_att_force

    F_obs = np.zeros(3)
    closest_dist = np.inf

    # 2. Obstacle repulsion with a vortex (tangential) component
    if len(obstacle_pts) > 0:
        diffs = robot_pos - obstacle_pts
        dists = np.linalg.norm(diffs, axis=1)
        closest_dist = float(np.min(dists))

        rospy.loginfo_throttle(0.5, f"Closest Obstacle Distance: {closest_dist:.2f}m")

        mask = (dists < params.repulsion_distance) & (dists > 0.1)
        near_diffs = diffs[mask]
        near_dists = dists[mask]

        if params.use_adaptive and dist_to_goal < params.goal_influence_distance:
            goal_factor = max((dist_to_goal / params.goal_influence_distance) ** 2,
                              params.min_adaptive_factor)
        else:
            goal_factor = 1.0

        for d, diff in zip(near_dists, near_diffs):
            direction = diff / d
            tangent = np.array([-direction[1], direction[0], 0.0])
            swerve_dir = direction + (1.2 * tangent)
            swerve_dir /= np.linalg.norm(swerve_dir)

            # Never weaken repulsion from an obstacle that is already close.
            factor = 1.0 if d < params.close_obstacle_dist else goal_factor
            magnitude = (params.repulsion_gain * factor *
                         (1.0/d - 1.0/params.repulsion_distance) ** 2)
            F_obs += magnitude * swerve_dir

    obs_norm = np.linalg.norm(F_obs)
    if obs_norm > params.max_rep_force:
        F_obs = F_obs / obs_norm * params.max_rep_force

    F_wall = wall_repulsion(robot_pos)

    # 3. Z height control
    z_error = goal_pos[2] - robot_pos[2]
    F_z = np.array([0, 0, np.clip(0.8 * z_error, params.vz_min * 2, params.vz_max * 2)])
    if robot_pos[2] < 0.8:
        F_z[2] = max(F_z[2], 1.5)

    # 4. Safety override: weaken / remove attraction when close to obstacles.
    if closest_dist < 1.5:
        F_att *= 0.3
        rospy.logwarn_throttle(1.0, "DANGER: Close to obstacle. Slowing down...")

    if closest_dist < 0.8:
        F_att *= 0.0
        rospy.logerr_throttle(1.0, "EMERGENCY: Collision imminent! Repulsion only!")

    total_force = F_att + F_obs + F_wall + F_z

    total_norm = np.linalg.norm(total_force)
    if total_norm > params.max_force:
        total_force = total_force / total_norm * params.max_force

    return total_force, F_att, F_obs + F_wall, closest_dist


def force_to_velocity(force, params):
    """Map a force to a world-frame velocity command.

    Horizontal: proportional (the attraction cap maps to v_max), then the
    vector is scaled -- not clipped per axis -- so its direction is kept and
    opposing forces genuinely slow the drone down. Vertical: clipped.
    """
    v_xy = np.asarray(force[:2], dtype=float) * (params.v_max / params.max_att_force)
    n = np.linalg.norm(v_xy)
    if n > params.v_max:
        v_xy = v_xy / n * params.v_max
    vz = float(np.clip(force[2], params.vz_min, params.vz_max))
    return v_xy, vz


def dynamic_parameter_tuning(params, closest_dist, dist_to_goal):
    if closest_dist < 2.0:
        params.repulsion_gain = min(params.max_repulsion_gain, params.repulsion_gain * 1.05)
    else:
        params.repulsion_gain = max(params.min_repulsion_gain, params.repulsion_gain * 0.95)
    if dist_to_goal < 2.0:
        params.attraction_gain = max(params.min_attraction_gain, params.attraction_gain * 0.95)
    return params


def clearance_along(robot_pos, direction, obstacle_pts, lookahead=(0.5, 1.0, 1.5, 2.0)):
    """Smallest distance from points along `direction` to any obstacle."""
    if len(obstacle_pts) == 0:
        return np.inf
    probes = robot_pos + np.outer(lookahead, direction)
    d = np.linalg.norm(probes[:, None, :] - obstacle_pts[None, :, :], axis=2)
    return float(np.min(d))


def path_clear(start, goal, obstacle_pts, half_width, max_len=6.0):
    """True if no obstacle point lies in the corridor of +-half_width around
    the straight xy segment start->goal (checked up to max_len metres)."""
    seg = (np.asarray(goal) - np.asarray(start))[:2]
    L = float(np.linalg.norm(seg))
    if L < 1e-6 or len(obstacle_pts) == 0:
        return True
    u = seg / L
    rel = obstacle_pts[:, :2] - np.asarray(start)[:2]
    t = rel @ u
    perp = np.abs(rel @ np.array([-u[1], u[0]]))
    return not np.any((t > 0.0) & (t < min(L, max_len)) & (perp < half_width))


def choose_escape_side(robot_pos, goal_pos, obstacle_pts, half_width,
                       offsets=np.arange(0.5, 4.01, 0.25)):
    """Pick the perpendicular side whose detour opens a clear line to the goal
    soonest. Returns (unit direction (3,), detour length or inf)."""
    g = (goal_pos - robot_pos)[:2]
    n = np.linalg.norm(g)
    perp = np.array([-g[1], g[0], 0.0]) / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    best_dir, best_cost = perp, np.inf
    for d in (perp, -perp):
        for s in offsets:
            p = robot_pos + s * d
            if not path_clear(robot_pos, p, obstacle_pts, 0.8 * half_width):
                break                       # this side is walled off
            if path_clear(p, goal_pos, obstacle_pts, half_width):
                if s < best_cost:
                    best_dir, best_cost = d, s
                break
    return best_dir, best_cost


def detect_local_minima(position_history, goal_dir, robot_pos=None, obstacle_pts=None,
                        half_width=0.8):
    """Return (stuck, escape_dir). Stuck = <0.15 m net progress in 2 s.

    The escape direction is perpendicular to the goal direction, on the side
    whose detour reaches a clear line to the goal soonest (falling back to the
    side with more clearance, then to left when there is no scan data).
    """
    if len(position_history) < 20:
        return False, np.array([1.0, 0.0, 0.0])

    net_progress = np.linalg.norm(position_history[-1] - position_history[-20])
    if net_progress >= 0.15:
        return False, np.array([1.0, 0.0, 0.0])

    perp = np.array([-goal_dir[1], goal_dir[0], 0.0])
    if np.linalg.norm(perp) < 1e-9:
        angle = np.random.uniform(0, 2 * np.pi)
        perp = np.array([np.cos(angle), np.sin(angle), 0.0])
    else:
        perp /= np.linalg.norm(perp)

    if robot_pos is not None and obstacle_pts is not None and len(obstacle_pts) > 0:
        side, cost = choose_escape_side(robot_pos, robot_pos + np.asarray(goal_dir),
                                        obstacle_pts, half_width)
        if np.isfinite(cost):
            perp = side
        elif clearance_along(robot_pos, -perp, obstacle_pts) > clearance_along(robot_pos, perp, obstacle_pts):
            perp = -perp

    esc = perp + np.array([0, 0, (np.random.rand() - 0.5) * 0.2])
    esc /= np.linalg.norm(esc)
    return True, esc


def merge_scan_memory(point_sets, voxel=0.15):
    """Union of recent world-frame scan points, thinned to one point per voxel
    so that remembering several scans does not inflate the summed repulsion."""
    sets = [p for p in point_sets if len(p)]
    if not sets:
        return np.empty((0, 3))
    pts = np.vstack(sets)
    if voxel <= 0:
        return pts
    _, idx = np.unique(np.round(pts / voxel).astype(np.int64), axis=0, return_index=True)
    return pts[np.sort(idx)]


def lookup_scan_transform(scan_frame, base_frame, timeout=2.0):
    """(R, t) of the lidar frame in the body frame, from TF. None on failure."""
    try:
        import tf2_ros
        buf = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(buf)  # must stay referenced during lookup
        tr = buf.lookup_transform(base_frame, scan_frame, rospy.Time(0), rospy.Duration(timeout))
        q, p = tr.transform.rotation, tr.transform.translation
        del listener
        return quat_to_rot((q.x, q.y, q.z, q.w)), np.array([p.x, p.y, p.z])
    except Exception as e:  # tf2 missing, frame missing, timeout...
        rospy.logwarn("APF: TF lookup %s -> %s failed (%s)", base_frame, scan_frame, e)
        return None


class APFNavigator:
    def __init__(self):
        rospy.init_node('apf_navigator', anonymous=False)

        self.start_delay  = rospy.get_param('~start_delay',  8.0)
        self.control_rate = rospy.get_param('~control_rate', 10.0)
        self.goal_tol_xy  = rospy.get_param('~goal_tol_xy',  0.6)
        self.goal_tol_z   = rospy.get_param('~goal_tol_z',   0.4)
        self.scan_timeout = rospy.get_param('~scan_timeout', 1.0)
        self.wp_timeout   = rospy.get_param('~wp_timeout',   120.0)
        self.max_escapes_per_wp = rospy.get_param('~max_escapes_per_wp', 6)
        # 'goal': face the current waypoint (default, stable); 'travel': face the
        # commanded velocity (oscillates with a 270 deg lidar, because what the
        # lidar sees changes as the drone turns); 'fixed': never yaw.
        self.heading_mode = rospy.get_param('~heading_mode', 'goal')
        # Scan points are remembered (world frame) for this long so obstacles do
        # not vanish when they rotate into the lidar's blind sector.
        self.obstacle_memory = rospy.get_param('~obstacle_memory', 3.0)
        self.memory_voxel = rospy.get_param('~memory_voxel', 0.15)
        self.scan_buffer = collections.deque()
        self.k_yaw        = rospy.get_param('~k_yaw', 1.5)
        self.fov_margin   = rospy.get_param('~fov_margin', np.radians(15.0))
        self.blind_speed  = rospy.get_param('~blind_speed', 0.1)
        # Below this speed the travel direction is too noisy to steer by:
        # no yaw command and no blind-sector limit.
        self.yaw_min_speed = rospy.get_param('~yaw_min_speed', 0.1)
        self.base_frame   = rospy.get_param('~base_frame', 'base_link')
        # None => auto-detect from TF on first scan.
        self.scan_yaw_offset = rospy.get_param('~scan_yaw_offset', None)

        self.waypoints = [
            np.array([-10.0,  10.0,  1.5]),
            np.array([ -8.0,   8.0,  2.0]),
            np.array([  6.0,   8.0,  2.0]),
            np.array([  7.0,   2.0,  2.0]),
            np.array([  7.0,  -6.0,  2.0]),
            np.array([  3.0,  -8.0,  2.0]),
            np.array([ -4.0,  -8.0,  2.0]),
            np.array([ -7.0,  -2.0,  2.0]),
            np.array([ -7.0,   6.0,  2.0]),
            np.array([-10.0,  10.0,  1.5]),
        ]
        self.current_wp = 0
        self.goal_pos   = self.waypoints[0].copy()
        self.skipped_wps = []

        self.params    = APFParams()
        self.lock      = threading.Lock()
        self.robot_pos     = np.zeros(3)
        self.robot_vel     = np.zeros(3)
        self.robot_yaw     = 0.0
        self.robot_quat    = np.array([0.0, 0.0, 0.0, 1.0])
        self.pose_received = False
        self.obstacle_points = np.empty((0, 3))
        self.scan_received   = False
        self.scan_time       = None
        self.scan_fov        = (0.0, np.pi)        # (centre, half-width), body frame
        self.scan_R = None                         # lidar -> base_link rotation
        self.scan_t = None
        self.vel_history = []
        self.pos_history = []
        self.MAX_HIST    = 30

        self.escape_mode      = False
        self.escape_counter   = 0
        self.escape_direction = np.zeros(3)
        self.escape_elapsed   = 0
        self.stuck_counter    = 0
        self.iteration        = 0
        self.wp_start_time    = None

        # Only one publisher of /command/twist may be active at a time:
        # the keepalive while nav_active is False, the control loop otherwise.
        self.nav_active = False

        self.cmd_pub    = rospy.Publisher('/command/twist', TwistStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/apf/markers',  MarkerArray,  queue_size=1)

        rospy.Subscriber('/ground_truth/state', Odometry,  self.odom_cb, queue_size=1)
        rospy.Subscriber('/scan',               LaserScan, self.scan_cb, queue_size=1)

        self.keepalive_timer = rospy.Timer(rospy.Duration(0.4), self._keepalive)

        rospy.loginfo("APF: waiting %.1f s for Gazebo to settle...", self.start_delay)
        rospy.sleep(self.start_delay)

        if not self.scan_received:
            rospy.logwarn("APF: /scan not received!")

        rospy.loginfo("APF: waiting for drone to be airborne (z > 0.5m)...")
        rospy.loginfo("APF: --> Enable motors now: rosservice call /enable_motors \"{enable: true}\"")
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            with self.lock:
                z = self.robot_pos[2]
                got_pose = self.pose_received
            if got_pose and z > 0.5:
                rospy.loginfo("APF: drone airborne at z=%.2f - starting navigation!", z)
                break
            rate.sleep()

        rospy.loginfo("APF: STARTED. %d waypoints.", len(self.waypoints))

        self.wp_start_time = rospy.Time.now()
        self.nav_active = True
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)

    def _keepalive(self, event):
        # Never compete with the control loop while it is navigating.
        if self.nav_active:
            return
        cmd = TwistStamped()
        cmd.header.stamp    = rospy.Time.now()
        cmd.header.frame_id = 'world'
        with self.lock:
            # Take-off assist only before navigation has started.
            if self.pose_received and self.robot_pos[2] < 0.5 and self.wp_start_time is None:
                cmd.twist.linear.z = 0.5

        self.cmd_pub.publish(cmd)

    def odom_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            self.robot_pos = np.array([p.x, p.y, p.z])
            v = msg.twist.twist.linear
            self.robot_vel = np.array([v.x, v.y, v.z])
            q = msg.pose.pose.orientation
            self.robot_quat = np.array([q.x, q.y, q.z, q.w])
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = np.arctan2(siny, cosy)
            self.pose_received = True

    def scan_cb(self, msg):
        if self.scan_R is None:
            if self.scan_yaw_offset is not None:
                # Manual override: upright lidar rotated by scan_yaw_offset.
                self.scan_R, self.scan_t = rot_z(self.scan_yaw_offset), DEFAULT_SCAN_T.copy()
            else:
                tf = lookup_scan_transform(msg.header.frame_id, self.base_frame)
                if tf is None:
                    rospy.logwarn("APF: using default upside-down hector lidar mount")
                    tf = (DEFAULT_SCAN_R, DEFAULT_SCAN_T)
                self.scan_R, self.scan_t = tf
            upside_down = self.scan_R[2, 2] < 0
            rospy.loginfo("APF: lidar frame %s mount: %s, z offset %.3f m",
                          msg.header.frame_id, "UPSIDE DOWN (scan mirrored)" if upside_down
                          else "upright", self.scan_t[2])
        with self.lock:
            if not self.pose_received:
                return
            pts = laser_to_obstacles(msg, self.robot_pos, self.robot_quat, self.scan_R, self.scan_t)
            now = rospy.Time.now()
            self.scan_buffer.append((now, pts))
            while self.scan_buffer and (now - self.scan_buffer[0][0]).to_sec() > self.obstacle_memory:
                self.scan_buffer.popleft()
            self.obstacle_points = merge_scan_memory([p for _, p in self.scan_buffer],
                                                     self.memory_voxel)
            self.scan_fov = scan_fov_in_body(msg, self.scan_R)
            self.scan_received = True
            self.scan_time = rospy.Time.now()

    def _advance_waypoint(self, skipped_reason=None):
        if skipped_reason:
            rospy.logwarn("APF: SKIPPING waypoint %d/%d (%s)",
                          self.current_wp + 1, len(self.waypoints), skipped_reason)
            self.skipped_wps.append(self.current_wp + 1)
        else:
            rospy.loginfo("APF: Waypoint %d/%d reached!", self.current_wp + 1, len(self.waypoints))
        self.current_wp += 1
        self.stuck_counter = 0
        self.escape_mode = False
        self.pos_history.clear()
        self.vel_history.clear()
        self.params.attraction_gain = self.params.base_attraction_gain
        self.wp_start_time = rospy.Time.now()

        if self.current_wp >= len(self.waypoints):
            if self.skipped_wps:
                rospy.logwarn("APF: Mission finished with SKIPPED waypoints: %s", self.skipped_wps)
            else:
                rospy.loginfo("APF: Mission complete! All waypoints reached.")
            self.timer.shutdown()
            self.nav_active = False   # keepalive takes over (hover)
            return False
        with self.lock:
            self.goal_pos = self.waypoints[self.current_wp].copy()
        rospy.loginfo("APF: Next waypoint %d/%d: %s", self.current_wp + 1,
                      len(self.waypoints), self.goal_pos)
        return True

    def control_loop(self, event):
        if not self.nav_active:
            return
        with self.lock:
            if not self.pose_received:
                return
            robot_pos    = self.robot_pos.copy()
            robot_vel    = self.robot_vel.copy()
            robot_yaw    = self.robot_yaw
            goal_pos     = self.goal_pos.copy()
            obstacle_pts = self.obstacle_points.copy()
            scan_time    = self.scan_time
            scan_fov     = self.scan_fov

        self.iteration += 1

        # Never fly on stale (or missing) obstacle data: hover instead.
        if scan_time is None or (rospy.Time.now() - scan_time).to_sec() > self.scan_timeout:
            rospy.logwarn_throttle(2.0, "APF: /scan stale - hovering")
            hover = np.array([0.0, 0.0, 0.8 * (goal_pos[2] - robot_pos[2])])
            self._send_cmd(hover, 0.0)
            return

        err_xy = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
        err_z  = abs(robot_pos[2] - goal_pos[2])
        if err_xy < self.goal_tol_xy and err_z < self.goal_tol_z:
            if not self._advance_waypoint():
                return
            goal_pos = self.goal_pos.copy()
        elif (rospy.Time.now() - self.wp_start_time).to_sec() > self.wp_timeout:
            if not self._advance_waypoint("timeout %.0f s" % self.wp_timeout):
                return
            goal_pos = self.goal_pos.copy()

        self.pos_history.append(robot_pos.copy())
        self.vel_history.append(robot_vel.copy())
        if len(self.pos_history) > self.MAX_HIST:
            self.pos_history.pop(0)
        if len(self.vel_history) > self.MAX_HIST:
            self.vel_history.pop(0)

        total_force, F_att, F_rep, closest_dist = calculate_apf_forces(
            robot_pos, goal_pos, obstacle_pts, self.params)

        if self.params.dynamic_tuning:
            dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
            self.params = dynamic_parameter_tuning(self.params, closest_dist, dist_to_goal)

        if self.params.enable_escape and not self.escape_mode and closest_dist < 2.5:
            stuck, esc_dir = detect_local_minima(
                self.pos_history, goal_pos - robot_pos, robot_pos, obstacle_pts)
            if stuck:
                self.stuck_counter += 1
                if self.stuck_counter > self.max_escapes_per_wp:
                    self._advance_waypoint("stuck after %d escapes" % self.max_escapes_per_wp)
                    return
                rospy.logwarn("APF: Local minimum (iter %d, escape %d/%d). Orthogonal escape.",
                              self.iteration, self.stuck_counter, self.max_escapes_per_wp)
                self.escape_mode      = True
                # Longer budget on every attempt (2x, 4x, ... escape_duration).
                self.escape_counter   = 2 * self.params.escape_duration * self.stuck_counter
                self.escape_elapsed   = 0
                self.escape_direction = esc_dir

        if self.escape_mode:
            # Weaken attraction so the goal does not pull the drone straight
            # back into the trap; keep repulsion and altitude hold.
            total_force = (0.3 * F_att + F_rep +
                           np.array([0.0, 0.0, total_force[2]]) +
                           self.params.escape_force_magnitude * self.escape_direction)
            tn = np.linalg.norm(total_force)
            if tn > self.params.max_force * 1.5:
                total_force = total_force / tn * self.params.max_force * 1.5
            self.escape_counter -= 1
            self.escape_elapsed += 1
            # Stop once a straight line to the goal is clear (after a short
            # minimum), or when the budget runs out.
            # The corridor is wider than the robot plus margin so the drone is
            # clear of the obstacle's repulsion zone, not just its surface.
            line_clear = (self.escape_elapsed >= 15 and
                          path_clear(robot_pos, goal_pos, obstacle_pts,
                                     self.params.robot_radius + 0.7))
            if self.escape_counter <= 0 or line_clear:
                self.escape_mode = False
                self.pos_history.clear()
                self.vel_history.clear()
                rospy.loginfo("APF: Escape done iter %d", self.iteration)

        if closest_dist < self.params.robot_radius + 0.1:
            rospy.logerr_throttle(2.0, "APF: COLLISION RISK! dist=%.2f m", closest_dist)

        # ---- velocity, heading and blind-sector handling ----
        v_xy, vz = force_to_velocity(total_force, self.params)
        yaw_rate = 0.0
        speed = np.linalg.norm(v_xy)
        travel = np.arctan2(v_xy[1], v_xy[0])
        target_heading = None
        if self.heading_mode == 'goal' and np.linalg.norm((goal_pos - robot_pos)[:2]) > self.goal_tol_xy:
            target_heading = np.arctan2(goal_pos[1] - robot_pos[1], goal_pos[0] - robot_pos[0])
        elif self.heading_mode == 'travel' and speed > self.yaw_min_speed:
            target_heading = travel
        if target_heading is not None:
            yaw_rate = float(np.clip(self.k_yaw * wrap_to_pi(target_heading - robot_yaw),
                                     -self.params.w_max, self.params.w_max))

        if speed > self.yaw_min_speed:
            # Direction of travel relative to the body; is it inside the lidar FOV?
            rel = wrap_to_pi(travel - robot_yaw)
            centre, half = scan_fov
            full_circle = half >= np.pi - 0.03
            if not full_circle and abs(wrap_to_pi(rel - centre)) > half - self.fov_margin:
                # Only creep where the lidar cannot currently see; remembered
                # points still provide repulsion there.
                v_xy = v_xy / speed * min(speed, self.blind_speed)
                rospy.logwarn_throttle(2.0, "APF: travel direction in lidar blind sector "
                                       "- limiting speed to %.2f m/s", self.blind_speed)

        self._send_cmd(np.array([v_xy[0], v_xy[1], vz]), yaw_rate)
        self._publish_markers(robot_pos, goal_pos, total_force, F_att, F_rep, obstacle_pts)

    def _send_cmd(self, vel, yaw_rate):
        cmd = TwistStamped()
        cmd.header.stamp    = rospy.Time.now()
        cmd.header.frame_id = 'world'
        cmd.twist.linear.x = float(vel[0])
        cmd.twist.linear.y = float(vel[1])
        cmd.twist.linear.z = float(np.clip(vel[2], self.params.vz_min, self.params.vz_max))
        cmd.twist.angular.z = float(yaw_rate)
        self.cmd_pub.publish(cmd)

    def _publish_markers(self, robot_pos, goal_pos, total_force, F_att, F_rep, obstacle_pts):
        ma = MarkerArray()

        g = Marker()
        g.header.frame_id = 'world'
        g.header.stamp    = rospy.Time.now()
        g.ns, g.id = 'goal', 0
        g.type   = Marker.SPHERE
        g.action = Marker.ADD
        g.pose.position.x = float(goal_pos[0])
        g.pose.position.y = float(goal_pos[1])
        g.pose.position.z = float(goal_pos[2])
        g.pose.orientation.w = 1.0
        g.scale.x = g.scale.y = g.scale.z = 0.6
        g.color   = ColorRGBA(0, 1, 0, 0.9)
        g.lifetime = rospy.Duration(0.5)
        ma.markers.append(g)

        for i, wp in enumerate(self.waypoints):
            w = Marker()
            w.header.frame_id = 'world'
            w.header.stamp    = rospy.Time.now()
            w.ns, w.id = 'waypoints', i
            w.type   = Marker.SPHERE
            w.action = Marker.ADD
            w.pose.position.x = float(wp[0])
            w.pose.position.y = float(wp[1])
            w.pose.position.z = float(wp[2])
            w.pose.orientation.w = 1.0
            w.scale.x = w.scale.y = w.scale.z = 0.3
            w.color   = ColorRGBA(1, 1, 0, 0.9 if i == self.current_wp else 0.25)
            w.lifetime = rospy.Duration(0.5)
            ma.markers.append(w)

        if len(obstacle_pts) > 0:
            pm = Marker()
            pm.header.frame_id = 'world'
            pm.header.stamp    = rospy.Time.now()
            pm.ns, pm.id = 'scan', 0
            pm.type   = Marker.POINTS
            pm.action = Marker.ADD
            pm.scale.x = pm.scale.y = 0.1
            pm.color   = ColorRGBA(1, 0.3, 0.3, 0.7)
            pm.lifetime = rospy.Duration(0.15)
            for pt in obstacle_pts:
                pm.points.append(Point(float(pt[0]), float(pt[1]), float(pt[2])))
            ma.markers.append(pm)

        def arrow(ns, fid, vec, color):
            a = Marker()
            a.header.frame_id = 'world'
            a.header.stamp    = rospy.Time.now()
            a.ns, a.id = ns, fid
            a.type   = Marker.ARROW
            a.action = Marker.ADD
            p0 = Point(float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2]))
            p1 = Point(float(robot_pos[0]+vec[0]*0.5),
                       float(robot_pos[1]+vec[1]*0.5),
                       float(robot_pos[2]+vec[2]*0.5))
            a.points  = [p0, p1]
            a.scale.x = 0.05
            a.scale.y = 0.12
            a.scale.z = 0.12
            a.color   = color
            a.lifetime = rospy.Duration(0.5)
            return a

        ma.markers.append(arrow('force_total', 0, total_force, ColorRGBA(1, 1, 0, 1.0)))
        ma.markers.append(arrow('force_att',   1, F_att,       ColorRGBA(0, 1, 0, 0.8)))
        ma.markers.append(arrow('force_rep',   2, F_rep,       ColorRGBA(1, 0, 0, 0.8)))
        self.marker_pub.publish(ma)


if __name__ == '__main__':
    try:
        APFNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
