#!/usr/bin/env python3
import rospy
import numpy as np
import threading
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped

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
        
        # --- State ---
        self.pos = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.last_scan = None
        self.mission_state = "TAKEOFF"
        
        self.nav_state = "GOAL_TRACKING"   
        self.stop_timer = rospy.Time.now()
        self.wp_start_time = rospy.Time.now()
        
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

        rospy.loginfo("BULLETPROOF NAV READY: 90-Degree Lidar Fix Applied!")
        self.control_loop()

    def odom_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            self.pos = np.array([p.x, p.y, p.z])
            q = msg.pose.pose.orientation
            self.yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))

    def scan_cb(self, msg):
        with self.lock:
            self.last_scan = msg

    def trigger_stop(self, reason):
        self.nav_state = "STOPPED"
        self.stop_timer = rospy.Time.now()
        rospy.logwarn(f"[{reason}] BRAKES APPLIED! Thinking...")

    def control_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.header.frame_id = "world"

            if self.mission_state == "TAKEOFF":
                cmd.twist.linear.z = 0.6
                if self.pos[2] >= self.TAKEOFF_H - 0.2:
                    self.mission_state = "MISSION"
                    self.wp_start_time = rospy.Time.now()
                    rospy.loginfo("\n=== TAKEOFF DONE: STARTING MISSION ===\n")

            elif self.mission_state == "MISSION":
                target = np.array(self.waypoints[self.current_wp])
                diff = target - self.pos
                dist_xy = np.linalg.norm(diff[:2])

                if dist_xy < self.WP_TOL:
                    rospy.loginfo(f"\n>>> WP {self.current_wp + 1} REACHED! <<<\n")
                    self.current_wp += 1
                    self.nav_state = "GOAL_TRACKING" 
                    self.wp_start_time = rospy.Time.now()
                    if self.current_wp >= len(self.waypoints):
                        self.mission_state = "LAND"
                    continue

                if (rospy.Time.now() - self.wp_start_time).to_sec() > self.WP_TIMEOUT:
                    rospy.logwarn(f"\n!!! WP {self.current_wp + 1} SKIPPED (TIMEOUT) !!!\n")
                    self.current_wp += 1
                    self.nav_state = "GOAL_TRACKING"
                    self.wp_start_time = rospy.Time.now()
                    if self.current_wp >= len(self.waypoints):
                        self.mission_state = "LAND"
                    continue

                # --- 4-ZONE RADAR (WITH 90-DEGREE FIX) ---
                min_f = 99.0
                min_b = 99.0
                min_l = 99.0
                min_r = 99.0
                
                with self.lock:
                    if self.last_scan:
                        angle = self.last_scan.angle_min
                        for r in self.last_scan.ranges:
                            if self.last_scan.range_min < r < self.last_scan.range_max:
                                
                                # THE FIX: Rotate the sensor map by exactly 90 degrees (+ pi/2)
                                corrected_angle = angle + (np.pi / 2.0) 
                                a = np.arctan2(np.sin(corrected_angle), np.cos(corrected_angle))
                                
                                if -np.pi/4 <= a <= np.pi/4:           
                                    if r < min_f: min_f = r
                                elif np.pi/4 < a <= 3*np.pi/4:         
                                    if r < min_l: min_l = r
                                elif -3*np.pi/4 <= a < -np.pi/4:       
                                    if r < min_r: min_r = r
                                else: 
                                    if r < min_b: min_b = r
                            angle += self.last_scan.angle_increment

                v_world = np.array([0.0, 0.0])
                v_body_x = 0.0
                v_body_y = 0.0

                # --- STATE MACHINE LOGIC ---
                if self.nav_state == "GOAL_TRACKING":
                    if min_f < self.FRONT_THRESH: 
                        self.trigger_stop("FRONT_BLOCKED")
                    elif min_l < self.CRASH_THRESH:
                        self.nav_state = "PUSH_RIGHT"
                        rospy.logwarn("Scraping Left Wall! Pushing Right.")
                    elif min_r < self.CRASH_THRESH:
                        self.nav_state = "PUSH_LEFT"
                        rospy.logwarn("Scraping Right Wall! Pushing Left.")
                    else:
                        v_world = (diff[:2] / dist_xy) * self.SPEED

                elif self.nav_state == "PUSH_LEFT":
                    if min_r > self.SIDE_THRESH: 
                        self.nav_state = "GOAL_TRACKING" 
                    elif min_f < self.FRONT_THRESH:
                        self.trigger_stop("FRONT_BLOCKED")
                    else:
                        v_body_y = self.SPEED 

                elif self.nav_state == "PUSH_RIGHT":
                    if min_l > self.SIDE_THRESH: 
                        self.nav_state = "GOAL_TRACKING" 
                    elif min_f < self.FRONT_THRESH:
                        self.trigger_stop("FRONT_BLOCKED")
                    else:
                        v_body_y = -self.SPEED 

                elif self.nav_state == "STOPPED":
                    v_world = np.array([0.0, 0.0]) 
                    
                    if (rospy.Time.now() - self.stop_timer).to_sec() > self.STOP_TIME:
                        if min_f >= self.FRONT_THRESH:
                            self.nav_state = "GOAL_TRACKING" 
                        elif min_l > self.SIDE_THRESH and min_l >= min_r:
                            self.nav_state = "SLIDE_LEFT"
                        elif min_r > self.SIDE_THRESH and min_r > min_l:
                            self.nav_state = "SLIDE_RIGHT"
                        else:
                            self.nav_state = "SLIDE_BACK" 

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
                    else:
                        v_body_x = -self.SPEED

                # --- TRANSLATE SLIDES TO WORLD MAP ---
                if self.nav_state not in ["GOAL_TRACKING", "STOPPED"]:
                    v_world[0] = v_body_x * np.cos(self.yaw) - v_body_y * np.sin(self.yaw)
                    v_world[1] = v_body_x * np.sin(self.yaw) + v_body_y * np.cos(self.yaw)

                # --- DASHBOARD ---
                dashboard = f"[WP {self.current_wp+1}] Dist: {dist_xy:.1f}m | F:{min_f:.1f} L:{min_l:.1f} R:{min_r:.1f} | {self.nav_state}"
                rospy.loginfo_throttle(0.5, dashboard)

                cmd.twist.linear.x = v_world[0]
                cmd.twist.linear.y = v_world[1]
                cmd.twist.linear.z = 0.5 * (target[2] - self.pos[2])

            elif self.mission_state == "LAND":
                cmd.twist.linear.z = -0.5
                if self.pos[2] < 0.1:
                    rospy.loginfo("\n=== MISSION COMPLETE. LANDED SUCCESSFULLY. ===\n")
                    break

            self.pub.publish(cmd)
            rate.sleep()

if __name__ == '__main__':
    try:
        BulletproofNav()
    except rospy.ROSInterruptException: pass
