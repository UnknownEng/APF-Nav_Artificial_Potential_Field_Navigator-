#!/usr/bin/env python3
"""
APF 3D Navigation Node - FIXED VERSION
Key fixes implemented:
1. Removed the "near wall" blindspot filter so the drone sees objects near walls.
2. Fixed the Repulsion Vector math (it now pushes AWAY from obstacles).
3. Summed the obstacle forces instead of averaging them (stronger avoidance).
4. Improved local minima escape to push orthogonally away from the trap.
"""

import rospy
import numpy as np
import threading

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# Room boundary (walls at +-15m but keep drone 2.5m away)
ROOM_MIN_X = -12.5
ROOM_MAX_X =  12.5
ROOM_MIN_Y = -12.5
ROOM_MAX_Y =  12.5
ROOM_MIN_Z =  0.5
ROOM_MAX_Z =  5.0

# Wall repulsion kicks in when drone gets within this distance of boundary
WALL_INFLUENCE = 4.0
WALL_GAIN      = 8.0


class APFParams:
    def __init__(self):
        self.attraction_gain       = 0.6  # Slightly increased to maintain forward progress
        self.repulsion_gain        = 1.5  # Increased for stronger pushback
        self.repulsion_distance    = 2.5  # Look further ahead for obstacles
        self.dt                    = 0.1
        self.use_adaptive          = True
        self.goal_influence_distance = 3.0
        self.enable_escape         = True
        self.escape_force_magnitude = 2.0 # Stronger escape push
        self.escape_duration       = 30
        self.escape_threshold      = 0.15
        self.dynamic_tuning        = True
        self.min_repulsion_gain    = 1.0
        self.max_repulsion_gain    = 3.0
        self.max_rep_force         = 3.0  # Increased limit for summed forces
        self.max_force             = 2.5
        self.robot_radius          = 0.5  # Increased safety buffer
        self.v_max  = rospy.get_param('~v_max',  0.4)
        self.v_min  = rospy.get_param('~v_min',  0.0)
        self.w_max  = rospy.get_param('~w_max',  0.8)
        self.vz_max = rospy.get_param('~vz_max', 0.3)
        self.vz_min = rospy.get_param('~vz_min', -0.3)


def laser_to_obstacles(scan_msg, robot_pos, robot_yaw):
    """
    Convert scan to world-frame obstacle points.
    FIXED: Removed the wall filter. All points are now evaluated to ensure 
    obstacles near walls are not ignored.
    """
    obstacles = []
    angle = scan_msg.angle_min
    scan_z = robot_pos[2] - 0.097
    step = max(1, len(scan_msg.ranges) // 80)

    for i, r in enumerate(scan_msg.ranges):
        if i % step != 0:
            angle += scan_msg.angle_increment
            continue

        if scan_msg.range_min < r < scan_msg.range_max - 0.1:
            world_angle = robot_yaw + angle
            ox = robot_pos[0] + r * np.cos(world_angle)
            oy = robot_pos[1] + r * np.sin(world_angle)
            
            # Keep all valid laser points
            obstacles.append([ox, oy, scan_z])

        angle += scan_msg.angle_increment

    return np.array(obstacles) if obstacles else np.empty((0, 3))


def wall_repulsion(robot_pos):
    """Explicit repulsion from all 4 room walls + floor."""
    F_wall = np.zeros(3)

    def boundary_force(dist_to_wall, direction):
        if 0 < dist_to_wall < WALL_INFLUENCE:
            mag = WALL_GAIN * (1.0/dist_to_wall - 1.0/WALL_INFLUENCE) ** 2
            return mag * direction
        return np.zeros(3)

    x, y, z = robot_pos

    F_wall += boundary_force(x - (-15.0), np.array([1, 0, 0]))  # West
    F_wall += boundary_force(15.0 - x,    np.array([-1, 0, 0])) # East
    F_wall += boundary_force(y - (-15.0), np.array([0, 1, 0]))  # South
    F_wall += boundary_force(15.0 - y,    np.array([0, -1, 0])) # North
    F_wall += boundary_force(z - 0.0,     np.array([0, 0, 1]))  # Floor

    wn = np.linalg.norm(F_wall)
    if wn > 4.0:
        F_wall = F_wall / wn * 4.0

    return F_wall


def calculate_apf_forces(robot_pos, goal_pos, obstacle_pts, params):
    dist_to_goal = np.linalg.norm(robot_pos - goal_pos)

    # Attractive force
    F_att = params.attraction_gain * (goal_pos - robot_pos)

    # Obstacle repulsion from laser
    F_obs = np.zeros(3)
    closest_dist = np.inf

    if len(obstacle_pts) > 0:
        # FIXED: Vector from obstacle TO robot (points away from danger)
        diffs = robot_pos - obstacle_pts 
        dists = np.linalg.norm(diffs, axis=1)
        closest_dist = float(np.min(dists))
        
        mask = (dists < params.repulsion_distance) & (dists > 0.1)
        near_diffs = diffs[mask]
        near_dists = dists[mask]
        n_near = len(near_dists)
        
        if n_near > 0:
            for i in range(n_near):
                d = near_dists[i]
                direction = near_diffs[i] / d # Normalized vector pointing AWAY
                
                if params.use_adaptive and dist_to_goal < params.goal_influence_distance:
                    adaptive_factor = (dist_to_goal / params.goal_influence_distance) ** 2
                else:
                    adaptive_factor = 1.0
                    
                magnitude = (params.repulsion_gain * adaptive_factor *
                             (1.0/d - 1.0/params.repulsion_distance) ** 2)
                
                # FIXED: Accumulate forces instead of averaging
                F_obs += magnitude * direction 

    # Clamp the summed obstacle force to prevent chaotic jumps
    obs_norm = np.linalg.norm(F_obs)
    if obs_norm > params.max_rep_force:
        F_obs = F_obs / obs_norm * params.max_rep_force

    # Wall repulsion
    F_wall = wall_repulsion(robot_pos)

    # Z height control
    z_error = goal_pos[2] - robot_pos[2]
    F_z = np.array([0, 0, np.clip(0.8 * z_error,
                                   params.vz_min * 2,
                                   params.vz_max * 2)])
    if robot_pos[2] < 0.8:
        F_z[2] = max(F_z[2], 1.5)

    total_force = F_att + F_obs + F_wall + F_z

    total_norm = np.linalg.norm(total_force)
    if total_norm > params.max_force:
        total_force = total_force / total_norm * params.max_force

    return total_force, F_att, F_obs + F_wall, closest_dist


def dynamic_parameter_tuning(params, closest_dist, dist_to_goal):
    if closest_dist < 1.0:
        params.repulsion_gain = min(params.max_repulsion_gain,
                                    params.repulsion_gain * 1.05)
    elif closest_dist > 2.0:
        params.repulsion_gain = max(params.min_repulsion_gain,
                                    params.repulsion_gain * 0.95)
    if dist_to_goal < 2.0:
        params.attraction_gain = max(0.3, params.attraction_gain * 0.95)
    return params


def detect_local_minima(velocity_history, position_history, threshold, goal_dir):
    if len(velocity_history) < 15 or len(position_history) < 15:
        return False, np.array([1.0, 0.0, 0.0])
        
    avg_vel = np.mean([np.linalg.norm(v) for v in velocity_history[-15:]])
    recent_p = np.array(position_history[-10:])
    span = recent_p.max(axis=0) - recent_p.min(axis=0)
    oscillation = np.linalg.norm(span)
    
    if avg_vel < threshold and oscillation < 0.8:
        # FIXED: Push orthogonally to the goal to slide along the obstacle
        perp_dir = np.array([-goal_dir[1], goal_dir[0], 0.0])
        if np.linalg.norm(perp_dir) == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            perp_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
        else:
            perp_dir /= np.linalg.norm(perp_dir)
            
        # Add slight random z variation
        esc = perp_dir + np.array([0, 0, (np.random.rand() - 0.5) * 0.2])
        esc /= np.linalg.norm(esc)
        return True, esc
        
    return False, np.array([1.0, 0.0, 0.0])


class APFNavigator:
    def __init__(self):
        rospy.init_node('dwa_inspection', anonymous=False)

        self.start_delay  = rospy.get_param('~start_delay',  8.0)
        self.control_rate = rospy.get_param('~control_rate', 10.0)
        self.goal_tol_xy  = rospy.get_param('~goal_tol_xy',  0.6)
        self.goal_tol_z   = rospy.get_param('~goal_tol_z',   0.4)

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

        self.params    = APFParams()
        self.lock      = threading.Lock()
        self.robot_pos     = np.zeros(3)
        self.robot_vel     = np.zeros(3)
        self.robot_yaw     = 0.0
        self.pose_received = False
        self.obstacle_points = np.empty((0, 3))
        self.scan_received   = False
        self.vel_history = []
        self.pos_history = []
        self.MAX_HIST    = 20
        self.escape_mode      = False
        self.escape_counter   = 0
        self.escape_direction = np.zeros(3)
        self.stuck_counter    = 0
        self.iteration        = 0

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
                rospy.loginfo("APF: drone airborne at z=%.2f — starting navigation!", z)
                break
            rate.sleep()

        rospy.loginfo("APF: STARTED. %d waypoints.", len(self.waypoints))

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)

    def _keepalive(self, event):
        cmd = TwistStamped()
        cmd.header.stamp    = rospy.Time.now()
        cmd.header.frame_id = 'world'
        with self.lock:
            if self.pose_received and self.robot_pos[2] < 0.5:
                cmd.twist.linear.z = 0.5  # 0.5 m/s upwards velocity
                
        self.cmd_pub.publish(cmd)

    def odom_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            self.robot_pos = np.array([p.x, p.y, p.z])
            v = msg.twist.twist.linear
            self.robot_vel = np.array([v.x, v.y, v.z])
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = np.arctan2(siny, cosy)
            self.pose_received = True

    def scan_cb(self, msg):
        with self.lock:
            if not self.pose_received:
                return
            self.obstacle_points = laser_to_obstacles(msg, self.robot_pos, self.robot_yaw)
            self.scan_received = True

    def control_loop(self, event):
        with self.lock:
            if not self.pose_received:
                return
            robot_pos    = self.robot_pos.copy()
            robot_vel    = self.robot_vel.copy()
            goal_pos     = self.goal_pos.copy()
            obstacle_pts = self.obstacle_points.copy()

        self.iteration += 1

        err_xy = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
        err_z  = abs(robot_pos[2] - goal_pos[2])
        if err_xy < self.goal_tol_xy and err_z < self.goal_tol_z:
            rospy.loginfo("APF: Waypoint %d/%d reached!", self.current_wp + 1, len(self.waypoints))
            self.current_wp += 1
            if self.current_wp >= len(self.waypoints):
                rospy.loginfo("APF: Mission complete!")
                self.timer.shutdown()
                return
            with self.lock:
                self.goal_pos = self.waypoints[self.current_wp].copy()
                goal_pos      = self.goal_pos.copy()
            rospy.loginfo("APF: Next waypoint %d/%d: %s", self.current_wp + 1, len(self.waypoints), goal_pos)

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

        if self.params.enable_escape and not self.escape_mode:
            goal_dir = goal_pos - robot_pos
            stuck, esc_dir = detect_local_minima(
                self.vel_history, self.pos_history, self.params.escape_threshold, goal_dir)
            if stuck:
                rospy.logwarn("APF: Local minima iter %d. Executing orthogonal escape.", self.iteration)
                self.escape_mode      = True
                self.escape_counter   = self.params.escape_duration
                self.escape_direction = esc_dir
                self.stuck_counter   += 1

        if self.escape_mode:
            total_force += self.params.escape_force_magnitude * self.escape_direction
            tn = np.linalg.norm(total_force)
            if tn > self.params.max_force * 1.5:
                total_force = total_force / tn * self.params.max_force * 1.5
            self.escape_counter -= 1
            if self.escape_counter <= 0:
                self.escape_mode = False
                rospy.loginfo("APF: Escape done iter %d", self.iteration)

        if closest_dist < self.params.robot_radius + 0.1:
            rospy.logerr_throttle(2.0, "APF: COLLISION RISK! dist=%.2f m", closest_dist)

        self._send_cmd(total_force, robot_pos, goal_pos)
        self._publish_markers(robot_pos, goal_pos, total_force, F_att, F_rep, obstacle_pts)

    def _send_cmd(self, force, robot_pos, goal_pos):
        cmd = TwistStamped()
        cmd.header.stamp    = rospy.Time.now()
        cmd.header.frame_id = 'world'
        
        cmd.twist.linear.x = float(np.clip(force[0], -self.params.v_max, self.params.v_max))
        cmd.twist.linear.y = float(np.clip(force[1], -self.params.v_max, self.params.v_max))
        cmd.twist.linear.z = float(np.clip(force[2], self.params.vz_min, self.params.vz_max))
        cmd.twist.angular.z = 0.0
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
