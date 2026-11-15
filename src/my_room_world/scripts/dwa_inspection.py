#!/usr/bin/env python3
"""Rule-based reactive navigator ("BulletproofNav").

Despite the file name this is neither DWA nor APF: it is a state machine that
stops / slides / backs off based on four lidar sectors.

Fixes over the previous version:
- The drone yaws toward the goal and only translates once roughly facing it,
  so the body-frame FRONT sector really is the direction of travel.
- Lidar mounting comes from TF (or ~scan_yaw_offset) instead of a hard-coded
  +90 deg rotation. The hector Hokuyo is mounted upside down (roll = pi),
  which mirrors the scan: LEFT and RIGHT were swapped before.
- SLIDE_BACK checks the rear sector, and rotates in place instead of reversing
  when the lidar cannot see behind the drone (Hokuyo UTM-30LX = 270 deg).
- Shared state is read under the lock; stale scans make the drone hover.
- Skipped (timed-out) waypoints are reported instead of claiming success.
"""
import threading

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


def wrap_to_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


# hector quadrotor_hokuyo_utm30lx mount (fallback if TF is unavailable):
# upside down, i.e. roll = pi -> scan y axis points to the body's right.
DEFAULT_SCAN_R = np.diag([1.0, -1.0, -1.0])


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_bearings(scan, R_bs):
    """Body-frame bearing of every beam, honouring the lidar mount rotation."""
    a = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
    d = np.column_stack([np.cos(a), np.sin(a), np.zeros(len(a))]) @ R_bs.T
    return np.arctan2(d[:, 1], d[:, 0])


def sector_minima(scan, R_bs):
    """Min range in FRONT/LEFT/RIGHT/BACK body sectors (99.0 = nothing seen)."""
    r = np.asarray(scan.ranges, dtype=float)
    ang = body_bearings(scan, R_bs)
    ok = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
    r, ang = r[ok], ang[ok]

    def m(mask):
        return float(np.min(r[mask])) if np.any(mask) else 99.0

    q = np.pi / 4
    return (m((ang >= -q) & (ang <= q)),            # front
            m((ang > q) & (ang <= 3 * q)),          # left
            m((ang >= -3 * q) & (ang < -q)),        # right
            m((ang > 3 * q) | (ang < -3 * q)))      # back


def rear_is_visible(scan, R_bs):
    """True if some beam points (nearly) straight behind the body."""
    if scan.angle_max - scan.angle_min >= 2 * np.pi - 0.05:
        return True
    ang = body_bearings(scan, R_bs)
    tol = max(abs(scan.angle_increment) * 2, np.radians(2.0))
    return bool(np.any(np.abs(wrap_to_pi(ang - np.pi)) < tol))


class BulletproofNav:
    def __init__(self):
        rospy.init_node('bulletproof_nav')

        # --- Strict Rules ---
        self.SPEED = 0.5
        self.FRONT_THRESH = 1.5
        self.SIDE_THRESH = 0.9
        self.CRASH_THRESH = 0.6
        self.CLEAR_BUFFER = 0.2

        self.TAKEOFF_H = 2.0
        self.WP_TOL = 1.0
        self.STOP_TIME = 1.0
        self.WP_TIMEOUT = 45.0
        self.SCAN_TIMEOUT = 1.0

        # Heading control: turn toward the goal, translate only when facing it.
        self.K_YAW = 1.5
        self.W_MAX = 0.8
        self.FACE_TOL = np.radians(30.0)

        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.scan_yaw_offset = rospy.get_param('~scan_yaw_offset', None)  # None => TF
        self.scan_R = None

        # --- State ---
        self.pos = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        # Resting height on the landing gear (~0.18 m for hector), taken from
        # the first odometry message; "landed" = back within 10 cm of it.
        self.ground_z = None
        self.last_scan = None
        self.last_scan_time = None
        self.mission_state = "TAKEOFF"

        self.nav_state = "GOAL_TRACKING"
        self.stop_timer = rospy.Time.now()
        self.wp_start_time = rospy.Time.now()
        self.rotate_dir = 1.0
        self.skipped = []

        # --- Waypoints ---
        self.waypoints = [
            [-10.0, 10.0, 2.0], [-8.0, 8.0, 2.0], [6.0, 8.0, 2.0],
            [7.0, 2.0, 2.0],    [7.0, -6.0, 2.0], [3.0, -8.0, 2.0],
            [-4.0, -8.0, 2.0],  [-7.0, -2.0, 2.0], [-7.0, 6.0, 2.0], [0.0, 0.0, 2.0]
        ]
        self.current_wp = 0
        self.lock = threading.Lock()

        self.pub = rospy.Publisher('/command/twist', TwistStamped, queue_size=1)
        rospy.Subscriber('/ground_truth/state', Odometry, self.odom_cb)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        rospy.loginfo("BULLETPROOF NAV READY")
        self.control_loop()

    def odom_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            self.pos = np.array([p.x, p.y, p.z])
            if self.ground_z is None:
                self.ground_z = p.z
            q = msg.pose.pose.orientation
            self.yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))

    def scan_cb(self, msg):
        if self.scan_R is None:
            if self.scan_yaw_offset is not None:
                self.scan_R = rot_z(self.scan_yaw_offset)
            else:
                self.scan_R = self._lookup_mount(msg.header.frame_id)
            rospy.loginfo("Lidar frame %s mount: %s", msg.header.frame_id,
                          "UPSIDE DOWN (scan mirrored)" if self.scan_R[2, 2] < 0 else "upright")
        with self.lock:
            self.last_scan = msg
            self.last_scan_time = rospy.Time.now()

    def _lookup_mount(self, scan_frame):
        """Rotation lidar frame -> base_link from TF (default: hector mount)."""
        try:
            import tf2_ros
            buf = tf2_ros.Buffer()
            listener = tf2_ros.TransformListener(buf)  # keep referenced during lookup
            tr = buf.lookup_transform(self.base_frame, scan_frame, rospy.Time(0), rospy.Duration(2.0))
            x, y, z, w = (tr.transform.rotation.x, tr.transform.rotation.y,
                          tr.transform.rotation.z, tr.transform.rotation.w)
            del listener
            return np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
        except Exception as e:
            rospy.logwarn("TF lookup %s -> %s failed (%s); assuming hector upside-down mount",
                          self.base_frame, scan_frame, e)
            return DEFAULT_SCAN_R.copy()

    def trigger_stop(self, reason):
        self.nav_state = "STOPPED"
        self.stop_timer = rospy.Time.now()
        rospy.logwarn(f"[{reason}] BRAKES APPLIED! Thinking...")

    def next_waypoint(self, skipped=False):
        if skipped:
            rospy.logwarn(f"\n!!! WP {self.current_wp + 1} SKIPPED (TIMEOUT) !!!\n")
            self.skipped.append(self.current_wp + 1)
        else:
            rospy.loginfo(f"\n>>> WP {self.current_wp + 1} REACHED! <<<\n")
        self.current_wp += 1
        self.nav_state = "GOAL_TRACKING"
        self.wp_start_time = rospy.Time.now()
        if self.current_wp >= len(self.waypoints):
            self.mission_state = "LAND"

    def control_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.header.frame_id = "world"

            with self.lock:
                pos = self.pos.copy()
                yaw = self.yaw
                scan = self.last_scan
                scan_time = self.last_scan_time

            if self.mission_state == "TAKEOFF":
                cmd.twist.linear.z = 0.6
                if pos[2] >= self.TAKEOFF_H - 0.2:
                    self.mission_state = "MISSION"
                    self.wp_start_time = rospy.Time.now()
                    rospy.loginfo("\n=== TAKEOFF DONE: STARTING MISSION ===\n")

            elif self.mission_state == "MISSION":
                target = np.array(self.waypoints[self.current_wp])
                diff = target - pos
                dist_xy = np.linalg.norm(diff[:2])

                if dist_xy < self.WP_TOL:
                    self.next_waypoint()
                    continue

                if (rospy.Time.now() - self.wp_start_time).to_sec() > self.WP_TIMEOUT:
                    self.next_waypoint(skipped=True)
                    continue

                cmd.twist.linear.z = 0.5 * (target[2] - pos[2])

                if scan is None or (rospy.Time.now() - scan_time).to_sec() > self.SCAN_TIMEOUT:
                    rospy.logwarn_throttle(2.0, "/scan stale - hovering")
                    self.pub.publish(cmd)
                    rate.sleep()
                    continue

                min_f, min_l, min_r, min_b = sector_minima(scan, self.scan_R)
                rear_ok = rear_is_visible(scan, self.scan_R)

                v_world = np.array([0.0, 0.0])
                v_body_x = 0.0
                v_body_y = 0.0
                yaw_rate = 0.0

                # --- STATE MACHINE LOGIC ---
                if self.nav_state == "GOAL_TRACKING":
                    heading_err = wrap_to_pi(np.arctan2(diff[1], diff[0]) - yaw)
                    yaw_rate = float(np.clip(self.K_YAW * heading_err, -self.W_MAX, self.W_MAX))
                    if min_f < self.FRONT_THRESH and abs(heading_err) < self.FACE_TOL:
                        self.trigger_stop("FRONT_BLOCKED")
                        yaw_rate = 0.0
                    elif min_l < self.CRASH_THRESH:
                        self.nav_state = "PUSH_RIGHT"
                        rospy.logwarn("Scraping Left Wall! Pushing Right.")
                    elif min_r < self.CRASH_THRESH:
                        self.nav_state = "PUSH_LEFT"
                        rospy.logwarn("Scraping Right Wall! Pushing Left.")
                    elif abs(heading_err) < self.FACE_TOL:
                        # Facing the goal: the FRONT sector covers the travel direction.
                        v_world = (diff[:2] / dist_xy) * self.SPEED
                    # else: turn in place until facing the goal

                elif self.nav_state == "PUSH_LEFT":
                    if min_r > self.SIDE_THRESH:
                        self.nav_state = "GOAL_TRACKING"
                    elif min_f < self.FRONT_THRESH:
                        self.trigger_stop("FRONT_BLOCKED")
                    elif min_l < self.CRASH_THRESH:
                        self.trigger_stop("SQUEEZED")
                    else:
                        v_body_y = self.SPEED

                elif self.nav_state == "PUSH_RIGHT":
                    if min_l > self.SIDE_THRESH:
                        self.nav_state = "GOAL_TRACKING"
                    elif min_f < self.FRONT_THRESH:
                        self.trigger_stop("FRONT_BLOCKED")
                    elif min_r < self.CRASH_THRESH:
                        self.trigger_stop("SQUEEZED")
                    else:
                        v_body_y = -self.SPEED

                elif self.nav_state == "STOPPED":
                    if (rospy.Time.now() - self.stop_timer).to_sec() > self.STOP_TIME:
                        if min_f >= self.FRONT_THRESH:
                            self.nav_state = "GOAL_TRACKING"
                        elif min_l > self.SIDE_THRESH and min_l >= min_r:
                            self.nav_state = "SLIDE_LEFT"
                        elif min_r > self.SIDE_THRESH and min_r > min_l:
                            self.nav_state = "SLIDE_RIGHT"
                        elif rear_ok and min_b > self.FRONT_THRESH:
                            self.nav_state = "SLIDE_BACK"
                        else:
                            # Cannot see (or no room) behind: rotate to look for a gap.
                            self.rotate_dir = 1.0 if min_l >= min_r else -1.0
                            self.nav_state = "ROTATE"

                elif self.nav_state == "SLIDE_LEFT":
                    if min_f > self.FRONT_THRESH + self.CLEAR_BUFFER:
                        self.nav_state = "GOAL_TRACKING"
                    elif min_l < self.CRASH_THRESH:
                        self.trigger_stop("LEFT_TRAP")
                    else:
                        v_body_y = self.SPEED

                elif self.nav_state == "SLIDE_RIGHT":
                    if min_f > self.FRONT_THRESH + self.CLEAR_BUFFER:
                        self.nav_state = "GOAL_TRACKING"
                    elif min_r < self.CRASH_THRESH:
                        self.trigger_stop("RIGHT_TRAP")
                    else:
                        v_body_y = -self.SPEED

                elif self.nav_state == "SLIDE_BACK":
                    if min_f > self.FRONT_THRESH + self.CLEAR_BUFFER:
                        self.nav_state = "GOAL_TRACKING"
                    elif min_l > self.SIDE_THRESH or min_r > self.SIDE_THRESH:
                        self.trigger_stop("FOUND_ESCAPE_ROUTE")
                    elif not rear_ok or min_b < self.CRASH_THRESH:
                        self.trigger_stop("REAR_BLOCKED")
                    else:
                        v_body_x = -self.SPEED

                elif self.nav_state == "ROTATE":
                    if min_f > self.FRONT_THRESH + self.CLEAR_BUFFER:
                        self.nav_state = "SLIDE_FORWARD"
                    else:
                        yaw_rate = self.rotate_dir * self.W_MAX

                elif self.nav_state == "SLIDE_FORWARD":
                    # Move into the gap found by ROTATE before turning back to the goal.
                    if min_f < self.FRONT_THRESH:
                        self.trigger_stop("FRONT_BLOCKED")
                    elif min_l > self.SIDE_THRESH and min_r > self.SIDE_THRESH and \
                            (rospy.Time.now() - self.stop_timer).to_sec() > 3.0:
                        self.nav_state = "GOAL_TRACKING"
                    else:
                        v_body_x = self.SPEED

                # --- TRANSLATE BODY-FRAME MOTION TO WORLD ---
                if self.nav_state not in ["GOAL_TRACKING", "STOPPED"]:
                    v_world[0] = v_body_x * np.cos(yaw) - v_body_y * np.sin(yaw)
                    v_world[1] = v_body_x * np.sin(yaw) + v_body_y * np.cos(yaw)

                # --- DASHBOARD ---
                dashboard = (f"[WP {self.current_wp+1}] Dist: {dist_xy:.1f}m | F:{min_f:.1f} "
                             f"L:{min_l:.1f} R:{min_r:.1f} B:{min_b:.1f} | {self.nav_state}")
                rospy.loginfo_throttle(0.5, dashboard)

                cmd.twist.linear.x = v_world[0]
                cmd.twist.linear.y = v_world[1]
                cmd.twist.angular.z = yaw_rate

            elif self.mission_state == "LAND":
                cmd.twist.linear.z = -0.5
                ground = self.ground_z if self.ground_z is not None else 0.0
                if pos[2] < ground + 0.1:
                    if self.skipped:
                        rospy.logwarn(f"\n=== LANDED. Waypoints SKIPPED: {self.skipped} ===\n")
                    else:
                        rospy.loginfo("\n=== MISSION COMPLETE. ALL WAYPOINTS REACHED. LANDED. ===\n")
                    break

            self.pub.publish(cmd)
            rate.sleep()


if __name__ == '__main__':
    try:
        BulletproofNav()
    except rospy.ROSInterruptException:
        pass
