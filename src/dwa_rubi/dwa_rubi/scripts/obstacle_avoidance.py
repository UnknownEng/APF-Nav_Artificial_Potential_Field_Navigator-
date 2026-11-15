#!/usr/bin/env python3
"""3D Dynamic Window Approach (DWA) waypoint follower for the hector quadrotor.

Obstacles come from a PointCloud2 (by default the RTAB-Map cloud map), are
voxel-clustered into spheres, and every candidate (v, vz, w) command is
forward-simulated against them. All rollouts are evaluated in one vectorised
NumPy pass so the planner keeps up with the control rate.
"""
import math
from threading import Lock

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool


# --------------------------------------------------------------------------
# Pure helpers (no ROS state) -- kept at module level so they can be tested.
# --------------------------------------------------------------------------
def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def path_progress(points, path, path_dists):
    """Arc-length of the closest projection of each point onto the polyline.

    points: (2,) or (K, 2). Returns a float or a (K,) array.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    best_d2 = np.full(len(pts), np.inf)
    best_s = np.zeros(len(pts))
    for i in range(len(path) - 1):
        a = path[i]
        ab = path[i + 1] - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-12:
            continue
        t = np.clip((pts - a) @ ab / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d2 = np.sum((pts - proj) ** 2, axis=1)
        better = d2 < best_d2
        best_d2[better] = d2[better]
        best_s[better] = path_dists[i] + t[better] * math.sqrt(L2)
    return best_s if np.ndim(points) > 1 else float(best_s[0])


def point_at_arc(path, path_dists, s_query):
    s_query = max(0.0, min(s_query, path_dists[-1]))
    for i in range(len(path_dists) - 1):
        if path_dists[i] <= s_query <= path_dists[i + 1]:
            seg = (s_query - path_dists[i]) / (path_dists[i + 1] - path_dists[i] + 1e-12)
            return path[i] + seg * (path[i + 1] - path[i])
    return path[-1]


def min_clearance(points, obstacles, inflate):
    """Clearance (surface distance minus inflation) from each point to the
    nearest obstacle sphere. points: (3,) or (K, 3)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if len(obstacles) == 0:
        out = np.full(len(pts), np.inf)
    else:
        d = np.linalg.norm(pts[:, None, :] - obstacles[None, :, :3], axis=2)
        out = np.min(d - (obstacles[None, :, 3] + inflate), axis=1)
    return out if np.ndim(points) > 1 else float(out[0])


def rollout_batch(p0_xy, z0, psi0, v, vz, w, T, dt, obstacles, inflate):
    """Forward-simulate K unicycle+vertical commands at once.

    v, vz, w: (K,) arrays. Returns (minclr (K,), p_end_xy (K, 2), z_end (K,),
    psi_end (K,)). Matches the original per-candidate loop: heading is updated
    before position on every step; an obstacle-free scene reports 5.0 clearance.
    """
    N = max(1, int(round(T / dt)))
    steps = np.arange(1, N + 1, dtype=float)                  # (N,)
    psi = psi0 + np.outer(w, steps) * dt                      # (K, N)
    dx = np.cumsum(v[:, None] * np.cos(psi) * dt, axis=1)     # (K, N)
    dy = np.cumsum(v[:, None] * np.sin(psi) * dt, axis=1)
    xs = p0_xy[0] + dx
    ys = p0_xy[1] + dy
    zs = z0 + np.outer(vz, steps) * dt

    K = len(v)
    minclr = np.full(K, np.inf)
    if len(obstacles) > 0:
        # Only obstacles that any trajectory could possibly touch.
        reach = (np.max(np.abs(v)) * T + np.max(np.abs(vz)) * T
                 + np.max(obstacles[:, 3]) + inflate + 0.1)
        rel = obstacles[:, :3] - np.array([p0_xy[0], p0_xy[1], z0])
        obs = obstacles[np.linalg.norm(rel, axis=1) <= reach]
        if len(obs) > 0:
            c = obs[:, :3]
            r = obs[:, 3] + inflate
            c2 = np.sum(c ** 2, axis=1)
            for n in range(N):
                P = np.column_stack([xs[:, n], ys[:, n], zs[:, n]])   # (K, 3)
                d2 = np.sum(P ** 2, axis=1)[:, None] + c2[None, :] - 2.0 * P @ c.T
                d = np.sqrt(np.maximum(d2, 0.0)) - r[None, :]
                minclr = np.minimum(minclr, np.min(d, axis=1))
    minclr[np.isinf(minclr)] = 5.0
    return minclr, np.column_stack([xs[:, -1], ys[:, -1]]), zs[:, -1], wrap_to_pi(psi[:, -1])


class DWALawnmowerFollower3D(object):
    def __init__(self):
        rospy.init_node('dwa_lawnmower_follower')

        self.cfg = {
            # Goal tolerance
            'goal_tol_xy': rospy.get_param("~goal_tol_xy", 0.5),
            'goal_tol_z': rospy.get_param("~goal_tol_z", 0.25),

            # Control timing
            'control_rate': rospy.get_param("~control_rate", 10.0),
            'dt': 1.0 / rospy.get_param("~control_rate", 10.0),

            # Motion limits
            'inflate': rospy.get_param("~inflate", 0.8),
            'v_min': rospy.get_param("~v_min", 0.0),
            'v_max': rospy.get_param("~v_max", 0.6),
            'w_max': rospy.get_param("~w_max", 1.8),
            'vz_min': rospy.get_param("~vz_min", -0.35),
            'vz_max': rospy.get_param("~vz_max", 0.35),

            'a_max': rospy.get_param("~a_max", 1.0),
            'alpha_max': rospy.get_param("~alpha_max", 3.0),
            'az_max': rospy.get_param("~az_max", 0.8),

            # Vertical hold gain
            'kz': rospy.get_param("~kz", 0.8),

            # DWA horizon
            'sim_dt': rospy.get_param("~sim_dt", 0.08),
            'pred_time_open': rospy.get_param("~pred_time_open", 1.8),
            'pred_time_tight': rospy.get_param("~pred_time_tight", 1.2),

            # Sampling
            'Nv_base': rospy.get_param("~Nv_base", 9),
            'Nw_base': rospy.get_param("~Nw_base", 25),
            'Nz_base': rospy.get_param("~Nz_base", 7),

            # Carrot lookahead
            'L_open': rospy.get_param("~L_open", 1.2),
            'L_tight': rospy.get_param("~L_tight", 0.6),

            # Obstacle processing
            'obs_z_window': rospy.get_param("~obs_z_window", 3.0),
            'obs_range_max': rospy.get_param("~obs_range_max", 10.0),
            'cluster_cell_xy': rospy.get_param("~cluster_cell_xy", 0.35),
            'cluster_cell_z': rospy.get_param("~cluster_cell_z", 0.35),
            'default_obs_r': rospy.get_param("~default_obs_r", 0.45),
            'max_points': rospy.get_param("~max_points", 3000),

            # Timeouts. pc_timeout is measured from when a cloud was last
            # *received* (cloud_map is republished on every map update, ~1 Hz).
            'data_timeout': rospy.get_param("~data_timeout", 2.0),
            'pc_timeout': rospy.get_param("~pc_timeout", 3.0),
            'require_pointcloud': rospy.get_param("~require_pointcloud", True),
            'waypoint_handoff_time': rospy.get_param("~waypoint_handoff_time", 0.2),
            # Only pause for a hand-off when the goal jumps by more than this.
            # Smoothly interpolated carrots (e.g. lawnmower_pose) never trigger it.
            'waypoint_jump_dist': rospy.get_param("~waypoint_jump_dist", 1.0),

            # Safety
            'critical_clearance': rospy.get_param("~critical_clearance", 0.8),
            'stop_clearance': rospy.get_param("~stop_clearance", 0.4),

            # Vertical scoring
            'z_score_weight': rospy.get_param("~z_score_weight", 2.0),
            'z_progress_weight': rospy.get_param("~z_progress_weight", 1.5),
            'z_speed_weight': rospy.get_param("~z_speed_weight", 0.2),
            'max_goal_z_error_for_level_flight': rospy.get_param(
                "~max_goal_z_error_for_level_flight", 0.15
            ),
        }

        self.state = {
            'p': np.array([0.0, 0.0, 0.0], dtype=float),
            'yaw': 0.0,
            'v_xy': 0.0,
            'vz': 0.0,
            'wz': 0.0,
            'stamp': rospy.Time(0),
        }

        self.goal = np.array([0.0, 0.0, 0.0], dtype=float)
        self.goal_stamp = rospy.Time(0)

        # obstacles: [x, y, z, r]
        self.obstacles = np.zeros((0, 4), dtype=float)
        self.pc_rx_time = None   # wall/sim time of the last cloud received

        self.survey_done = False
        self.have_odom = False
        self.have_goal = False

        self.state_lock = Lock()
        self.goal_lock = Lock()
        self.obs_lock = Lock()

        self.waypoint_pause_until = rospy.Time(0)
        self.last_waypoint = None

        odom_topic = rospy.get_param("~odom_topic", "/ground_truth/state")
        waypoint_topic = rospy.get_param("~waypoint_topic", "/dwa/goal")
        survey_topic = rospy.get_param("~survey_topic", "/survey_done")
        pc_topic = rospy.get_param("~pointcloud_topic", "/rtabmap/cloud_map")
        cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(waypoint_topic, PoseStamped, self.waypoint_callback, queue_size=1)
        rospy.Subscriber(survey_topic, Bool, self.survey_callback, queue_size=1)
        # Large buff_size so a big cloud_map is not truncated/queued.
        rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback, queue_size=1,
                         buff_size=2 ** 24)

        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        rospy.on_shutdown(self.safe_stop)

        # The run loop already waits for odom/goal, so a timeout here is only
        # informational instead of crashing the node.
        for topic, mtype, what in ((odom_topic, Odometry, "odometry"),
                                   (waypoint_topic, PoseStamped, "waypoint")):
            rospy.loginfo("Waiting for first %s on %s...", what, topic)
            try:
                rospy.wait_for_message(topic, mtype, timeout=10.0)
            except rospy.ROSException:
                rospy.logwarn("No %s on %s yet; will keep waiting in the control loop.",
                              what, topic)

        rospy.loginfo("DWA lawnmower follower ready. Publishing velocity to %s", cmd_topic)
        self.rate = rospy.Rate(self.cfg['control_rate'])

    # ------------------------------------------------------------------ ROS
    def odom_callback(self, msg):
        with self.state_lock:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            t = msg.twist.twist
            self.state['p'] = np.array([p.x, p.y, p.z], dtype=float)
            self.state['yaw'] = yaw
            self.state['v_xy'] = math.hypot(t.linear.x, t.linear.y)
            self.state['vz'] = t.linear.z
            self.state['wz'] = t.angular.z
            self.state['stamp'] = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
            self.have_odom = True

    def waypoint_callback(self, msg):
        with self.goal_lock:
            new_goal = np.array([msg.pose.position.x,
                                 msg.pose.position.y,
                                 msg.pose.position.z], dtype=float)

            # Pause briefly only on a real hand-off (a jump), not on every
            # slightly-moved carrot -- otherwise a 10 Hz interpolating
            # publisher keeps the follower permanently paused.
            if (self.last_waypoint is None or
                    np.linalg.norm(new_goal - self.last_waypoint) > self.cfg['waypoint_jump_dist']):
                self.waypoint_pause_until = rospy.Time.now() + rospy.Duration(self.cfg['waypoint_handoff_time'])
            self.last_waypoint = new_goal.copy()

            self.goal = new_goal
            self.goal_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
            self.have_goal = True

    def survey_callback(self, msg):
        self.survey_done = msg.data

    def _set_obstacles(self, obstacles):
        with self.obs_lock:
            self.obstacles = obstacles
            self.pc_rx_time = rospy.Time.now()

    def pc_callback(self, msg):
        try:
            xyz = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
                           dtype=float).reshape(-1, 3)
        except Exception as e:
            # Do not refresh pc_rx_time: an unreadable cloud is not valid data.
            rospy.logwarn_throttle(5.0, "Failed to read point cloud: %s", str(e))
            return

        with self.state_lock:
            current_pos = self.state['p'].copy()
            current_yaw = self.state['yaw']

        if len(xyz):
            # Height filter
            xyz = xyz[np.abs(xyz[:, 2] - current_pos[2]) <= self.cfg['obs_z_window'] / 2.0]
        if len(xyz):
            # Range filter
            dist = np.hypot(xyz[:, 0] - current_pos[0], xyz[:, 1] - current_pos[1])
            xyz = xyz[dist <= self.cfg['obs_range_max']]
        if len(xyz):
            # Prefer front points, but keep nearby side points
            dx = xyz[:, 0] - current_pos[0]
            dy = xyz[:, 1] - current_pos[1]
            forward = dx * math.cos(current_yaw) + dy * math.sin(current_yaw)
            xyz = xyz[(forward > -0.5) | (np.hypot(dx, dy) < 2.0)]
        if len(xyz) > self.cfg['max_points']:
            # Keep the closest points rather than a random subset, so a
            # nearby obstacle is never dropped by down-sampling.
            d = np.linalg.norm(xyz - current_pos, axis=1)
            xyz = xyz[np.argsort(d)[:self.cfg['max_points']]]

        if len(xyz) == 0:
            self._set_obstacles(np.zeros((0, 4), dtype=float))
            return

        # 3D voxel clustering
        cxy, cz = self.cfg['cluster_cell_xy'], self.cfg['cluster_cell_z']
        cells = np.unique(np.column_stack([np.round(xyz[:, 0] / cxy) * cxy,
                                           np.round(xyz[:, 1] / cxy) * cxy,
                                           np.round(xyz[:, 2] / cz) * cz]), axis=0)
        self._set_obstacles(np.column_stack([cells, self.cfg['default_obs_r'] * np.ones(len(cells))]))

    def is_data_fresh(self):
        now = rospy.Time.now()
        timeout = rospy.Duration(self.cfg['data_timeout'])

        with self.state_lock:
            odom_fresh = (now - self.state['stamp']) < timeout
        with self.goal_lock:
            goal_fresh = (now - self.goal_stamp) < timeout
        with self.obs_lock:
            if self.pc_rx_time is None:
                pc_fresh = not self.cfg['require_pointcloud']
            else:
                # An empty cloud is valid, but only while clouds keep arriving.
                pc_fresh = (now - self.pc_rx_time) < rospy.Duration(self.cfg['pc_timeout'])

        if not pc_fresh:
            rospy.logwarn_throttle(5.0, "Point cloud stale or never received - not flying blind")
        return odom_fresh and goal_fresh and pc_fresh

    # ------------------------------------------------------------- planning
    def min_clearance_3d(self, p3, obstacles=None):
        if obstacles is None:
            with self.obs_lock:
                obstacles = self.obstacles.copy()
        return min_clearance(p3, obstacles, self.cfg['inflate'])

    def hold_altitude(self, z, z_goal):
        err = z_goal - z
        return max(min(self.cfg['kz'] * err, self.cfg['vz_max']), self.cfg['vz_min'])

    def compute_cmd_dwa_3d(self, pose, global_path, path_s, goal3):
        p_xy = pose['p']
        z = pose['z']
        psi = pose['psi']
        v = pose['v']
        vz = pose['vz']
        w = pose['w']
        cfg = self.cfg

        with self.obs_lock:
            obstacles = self.obstacles.copy()

        clear_now = self.min_clearance_3d(np.array([p_xy[0], p_xy[1], z]), obstacles)

        if clear_now < cfg['stop_clearance']:
            rospy.logwarn_throttle(1.0, "Emergency stop. 3D clearance: %.2f", clear_now)
            return 0.0, 0.0, 0.0

        if clear_now < cfg['critical_clearance']:
            L, predT = cfg['L_tight'], cfg['pred_time_tight']
            Nv, Nw, Nz = max(cfg['Nv_base'], 9), max(cfg['Nw_base'], 31), max(cfg['Nz_base'], 9)
            speed_limit = 0.25
            w_clear, w_speed = 7.0, 0.05
        elif clear_now < 1.0:
            L, predT = cfg['L_tight'], cfg['pred_time_tight']
            Nv, Nw, Nz = max(cfg['Nv_base'], 11), max(cfg['Nw_base'], 25), max(cfg['Nz_base'], 7)
            speed_limit = cfg['v_max']
            w_clear, w_speed = 5.0, 0.15
        else:
            L, predT = cfg['L_open'], cfg['pred_time_open']
            Nv, Nw, Nz = cfg['Nv_base'], cfg['Nw_base'], cfg['Nz_base']
            speed_limit = cfg['v_max']
            w_clear, w_speed = 2.5, 0.8

        s_now = path_progress(p_xy, global_path, path_s)
        carrot = point_at_arc(global_path, path_s, s_now + L)

        # Dynamic window
        v_lo = max(cfg['v_min'], v - cfg['a_max'] * cfg['dt'])
        v_hi = min(cfg['v_max'], v + cfg['a_max'] * cfg['dt'], speed_limit)
        vz_lo = max(cfg['vz_min'], vz - cfg['az_max'] * cfg['dt'])
        vz_hi = min(cfg['vz_max'], vz + cfg['az_max'] * cfg['dt'])
        w_lo = max(-cfg['w_max'], w - cfg['alpha_max'] * cfg['dt'])
        w_hi = min(cfg['w_max'], w + cfg['alpha_max'] * cfg['dt'])

        z_goal_err = goal3[2] - z
        if abs(z_goal_err) < cfg['max_goal_z_error_for_level_flight']:
            vz_lo = min(vz_lo, 0.0)
            vz_hi = max(vz_hi, 0.0)

        v_hi = max(v_hi, 0.20)
        v_lo = min(v_lo, 0.05)

        V, VZ, W = np.meshgrid(np.linspace(v_lo, v_hi, Nv),
                               np.linspace(vz_lo, vz_hi, Nz),
                               np.linspace(w_lo, w_hi, Nw), indexing='ij')
        V, VZ, W = V.ravel(), VZ.ravel(), W.ravel()

        minclr, p_end, z_end, psi_end = rollout_batch(
            p_xy, z, psi, V, VZ, W, predT, cfg['sim_dt'], obstacles, cfg['inflate'])

        valid = minclr > 0
        if not np.any(valid):
            return self.fallback_turn_creep_3d(p_xy, z, psi, goal3, obstacles)

        s_clear = np.minimum(minclr / 2.0, 1.0)
        angC = np.arctan2(carrot[1] - p_end[:, 1], carrot[0] - p_end[:, 0])
        s_head = 1.0 - np.abs(wrap_to_pi(angC - psi_end)) / math.pi
        s_dist = 1.0 / (1.0 + np.linalg.norm(carrot - p_end, axis=1))
        s_speed = (V - cfg['v_min']) / (cfg['v_max'] - cfg['v_min'] + 1e-6)
        s_goal_xy = 1.0 / (1.0 + np.linalg.norm(goal3[:2] - p_end, axis=1))
        z_err_end = np.abs(goal3[2] - z_end)
        s_goal_z = 1.0 / (1.0 + z_err_end)
        s_zspeed = 1.0 - np.minimum(np.abs(VZ) / max(abs(cfg['vz_max']), 1e-6), 1.0)
        s_prog = np.clip((path_progress(p_end, global_path, path_s) - s_now) / 0.30, -1.0, 1.0)
        s_zprog = np.clip((abs(z_goal_err) - z_err_end) / 0.30, -1.0, 1.0)
        rot_pen = np.abs(W) / (cfg['w_max'] + 1e-6)

        score = (2.0 * s_head + 1.6 * s_dist + w_clear * s_clear + w_speed * s_speed +
                 1.2 * s_goal_xy + 4.0 * s_prog +
                 cfg['z_score_weight'] * s_goal_z +
                 cfg['z_progress_weight'] * s_zprog +
                 cfg['z_speed_weight'] * s_zspeed - 0.25 * rot_pen)
        score = score - 1.5 * (V < 0.08)
        score[~valid] = -np.inf

        k = int(np.argmax(score))   # first maximum, same tie-break as the loop
        return float(V[k]), float(VZ[k]), float(W[k])

    def fallback_turn_creep_3d(self, p_xy, z, psi, goal3, obstacles=None):
        angC = math.atan2(goal3[1] - p_xy[1], goal3[0] - p_xy[0])
        e = wrap_to_pi(angC - psi)

        w_cmd = max(min(2.0 * e, self.cfg['w_max']), -self.cfg['w_max'])
        v_cmd = 0.08
        vz_cmd = self.hold_altitude(z, goal3[2])

        if self.min_clearance_3d(np.array([p_xy[0], p_xy[1], z]), obstacles) < self.cfg['stop_clearance']:
            v_cmd = 0.0
            w_cmd = 0.8 * (1.0 if e >= 0.0 else -1.0)

            z_err = goal3[2] - z
            vz_cmd = max(min(0.25 * np.sign(z_err + 1e-6), self.cfg['vz_max']), self.cfg['vz_min'])

        return v_cmd, vz_cmd, w_cmd

    def safe_stop(self):
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()

            if not self.have_odom or not self.have_goal:
                self.rate.sleep()
                continue

            if not self.is_data_fresh():
                rospy.logwarn_throttle(5.0, "Data stale - stopping")
                self.safe_stop()
                self.rate.sleep()
                continue

            if self.survey_done:
                rospy.loginfo("Survey complete")
                self.safe_stop()
                break

            with self.state_lock:
                current_state = {
                    'p': self.state['p'].copy(),
                    'yaw': self.state['yaw'],
                    'v_xy': self.state['v_xy'],
                    'vz': self.state['vz'],
                    'wz': self.state['wz'],
                }
            with self.goal_lock:
                current_goal = self.goal.copy()
                pause_until = self.waypoint_pause_until

            dist_xy = np.linalg.norm(current_state['p'][:2] - current_goal[:2])
            dist_z = abs(current_state['p'][2] - current_goal[2])

            hold = now < pause_until or (dist_xy < self.cfg['goal_tol_xy'] and
                                         dist_z < self.cfg['goal_tol_z'])
            if hold:
                cmd = Twist()
                cmd.linear.z = self.hold_altitude(current_state['p'][2], current_goal[2])
                self.cmd_pub.publish(cmd)
                self.rate.sleep()
                continue

            global_path = np.array([current_state['p'][:2], current_goal[:2]], dtype=float)
            path_s = np.array([0.0, max(dist_xy, 1e-6)], dtype=float)

            pose = {
                'p': current_state['p'][:2],
                'z': current_state['p'][2],
                'psi': current_state['yaw'],
                'v': current_state['v_xy'],
                'vz': current_state['vz'],
                'w': current_state['wz'],
            }

            v_cmd, vz_cmd, w_cmd = self.compute_cmd_dwa_3d(pose, global_path, path_s, current_goal)

            cmd = Twist()
            cmd.linear.x = v_cmd
            cmd.linear.z = vz_cmd
            cmd.angular.z = w_cmd
            self.cmd_pub.publish(cmd)

            rospy.loginfo_throttle(1.0, "cmd: vx=%.2f vz=%.2f wz=%.2f | dxy=%.2f dz=%.2f",
                                   v_cmd, vz_cmd, w_cmd, dist_xy, dist_z)
            self.rate.sleep()

        self.safe_stop()


if __name__ == '__main__':
    try:
        node = DWALawnmowerFollower3D()
        node.run()
    except rospy.ROSInterruptException:
        pass
