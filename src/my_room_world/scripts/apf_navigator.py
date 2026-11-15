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
4. The drone yaws to face its direction of travel, and horizontal speed is cut
   whenever it would move into the lidar's blind sector.
5. Lidar mounting yaw comes from TF (or ~scan_yaw_offset), and scan points are
   projected with full roll/pitch/yaw so floor hits while tilted are dropped.
6. Escape picks the clearer side, near-goal repulsion never drops below a
   floor, stale scans make the drone hover, and a waypoint is skipped (and
   reported) after repeated escapes or a timeout instead of looping forever.
"""

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
        self.repulsion_gain        = rospy.get_param('~repulsion_gain', 15.0)
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
        self.max_repulsion_gain    = 2.0 * self.base_repulsion_gain
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


def laser_to_obstacles(scan_msg, robot_pos, robot_quat, scan_yaw_offset=0.0,
                       max_rays=80, floor_z=0.15, lidar_height=-0.097):
    """Project (down-sampled) scan returns into world-frame 3D points.

    Uses the full body orientation, so beams that hit the floor while the
    drone is pitched/rolled are recognised and dropped instead of becoming
    phantom obstacles.
    """
    r = np.asarray(scan_msg.ranges, dtype=float)
    n = len(r)
    if n == 0:
        return np.empty((0, 3))
    idx = np.arange(0, n, max(1, n // max_rays))
    r = r[idx]
    ang = scan_msg.angle_min + idx * scan_msg.angle_increment + scan_yaw_offset
    ok = np.isfinite(r) & (r > scan_msg.range_min) & (r < scan_msg.range_max - 0.1)
    r, ang = r[ok], ang[ok]
    if len(r) == 0:
        return np.empty((0, 3))

    pts_body = np.column_stack([r * np.cos(ang), r * np.sin(ang),
                                np.full(len(r), lidar_height)])
    pts = robot_pos + pts_body @ quat_to_rot(robot_quat).T
    return pts[pts[:, 2] > floor_z]


def wall_repulsion(robot_pos):
    F_wall = np.zeros(3)

    def boundary_force(dist_to_wall, direction):
        if 0 < dist_to_wall < WALL_INFLUENCE:
            mag = WALL_GAIN * (1.0/dist_to_wall - 1.0/WALL_INFLUENCE) ** 2
            return mag * direction
        return np.zeros(3)

    x, y, z = robot_pos
    H = ROOM_HALF_SIZE

    F_wall += boundary_force(x + H, np.array([1, 0, 0]))
    F_wall += boundary_force(H - x, np.array([-1, 0, 0]))
    F_wall += boundary_force(y + H, np.array([0, 1, 0]))
    F_wall += boundary_force(H - y, np.array([0, -1, 0]))
    F_wall += boundary_force(z - 0.0, np.array([0, 0, 1]))

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


def detect_local_minima(position_history, goal_dir, robot_pos=None, obstacle_pts=None):
    """Return (stuck, escape_dir). Stuck = <0.15 m net progress in 2 s.

    The escape direction is perpendicular to the goal direction, on whichever
    side has more clearance (falls back to left when there is no scan data).
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
        if clearance_along(robot_pos, -perp, obstacle_pts) > clearance_along(robot_pos, perp, obstacle_pts):
            perp = -perp

    esc = perp + np.array([0, 0, (np.random.rand() - 0.5) * 0.2])
    esc /= np.linalg.norm(esc)
    return True, esc


def lookup_scan_yaw_offset(scan_frame, base_frame, timeout=2.0):
    """Yaw of the lidar frame relative to the body frame, from TF. None on failure."""
    try:
        import tf2_ros
        buf = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(buf)  # must stay referenced during lookup
        tr = buf.lookup_transform(base_frame, scan_frame, rospy.Time(0), rospy.Duration(timeout))
        q = tr.transform.rotation
        yaw = float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        del listener
        return yaw
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
        self.max_escapes_per_wp = rospy.get_param('~max_escapes_per_wp', 4)
        self.face_travel_direction = rospy.get_param('~face_travel_direction', True)
        self.k_yaw        = rospy.get_param('~k_yaw', 1.5)
        self.fov_margin   = rospy.get_param('~fov_margin', np.radians(15.0))
        self.blind_speed  = rospy.get_param('~blind_speed', 0.1)
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
        self.scan_fov        = (-np.pi, np.pi)
        self.vel_history = []
        self.pos_history = []
        self.MAX_HIST    = 30

        self.escape_mode      = False
        self.escape_counter   = 0
        self.escape_direction = np.zeros(3)
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
        if self.scan_yaw_offset is None:
            off = lookup_scan_yaw_offset(msg.header.frame_id, self.base_frame)
            self.scan_yaw_offset = off if off is not None else 0.0
            rospy.loginfo("APF: lidar yaw offset = %.1f deg (frame %s)",
                          np.degrees(self.scan_yaw_offset), msg.header.frame_id)
        with self.lock:
            if not self.pose_received:
                return
            self.obstacle_points = laser_to_obstacles(
                msg, self.robot_pos, self.robot_quat, self.scan_yaw_offset)
            self.scan_fov = (msg.angle_min + self.scan_yaw_offset,
                             msg.angle_max + self.scan_yaw_offset)
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
                self.escape_counter   = self.params.escape_duration
                self.escape_direction = esc_dir

        if self.escape_mode:
            total_force += self.params.escape_force_magnitude * self.escape_direction
            tn = np.linalg.norm(total_force)
            if tn > self.params.max_force * 1.5:
                total_force = total_force / tn * self.params.max_force * 1.5
            self.escape_counter -= 1
            if self.escape_counter <= 0:
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
        if speed > 0.05:
            travel = np.arctan2(v_xy[1], v_xy[0])
            if self.face_travel_direction:
                yaw_rate = float(np.clip(self.k_yaw * wrap_to_pi(travel - robot_yaw),
                                         -self.params.w_max, self.params.w_max))
            # Direction of travel relative to the body; is it inside the lidar FOV?
            rel = wrap_to_pi(travel - robot_yaw)
            lo, hi = scan_fov[0] + self.fov_margin, scan_fov[1] - self.fov_margin
            full_circle = (scan_fov[1] - scan_fov[0]) >= 2 * np.pi - 0.05
            if not full_circle and not (lo <= rel <= hi):
                limit = 0.0 if self.face_travel_direction else self.blind_speed
                if speed > limit:
                    v_xy = v_xy / speed * limit
                rospy.logwarn_throttle(2.0, "APF: travel direction in lidar blind sector "
                                       "- limiting speed to %.2f m/s", limit)

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
