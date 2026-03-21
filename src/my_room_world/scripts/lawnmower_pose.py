#!/usr/bin/env python2
import math
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from hector_uav_msgs.srv import EnableMotors
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates # Useful for checking if models exist

def enable_motors(service_name="/enable_motors", enable=True, timeout=10.0):
    rospy.loginfo("Waiting for %s ...", service_name)
    rospy.wait_for_service(service_name, timeout=timeout)
    try:
        srv = rospy.ServiceProxy(service_name, EnableMotors)
        resp = srv(enable)
        rospy.loginfo("enable_motors(%s) -> success=%s", enable, resp.success)
        return resp.success
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call %s: %s", service_name, str(e))
        return False

def yaw_to_quat(yaw):
    return Quaternion(0.0, 0.0, math.sin(yaw/2.0), math.cos(yaw/2.0))

# --- NEW FUNCTION TO SPAWN RED MARKERS IN GAZEBO ---
def spawn_gazebo_marker(marker_name, x, y, z):
    # Increase timeout to 15.0 to give the hangar time to load
    try:
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=15.0)
        spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        # Improved SDF with a unique name for the link to avoid conflicts
        sdf_xml = """<?xml version='1.0'?>
        <sdf version='1.6'>
          <model name='{name}'>
            <static>true</static>
            <link name='link_{name}'>
              <visual name='visual'>
                <geometry><sphere><radius>0.1</radius></sphere></geometry>
                <material>
                  <script>
                    <name>Gazebo/Red</name>
                    <uri>file://media/materials/scripts/gazebo.material</uri>
                  </script>
                </material>
              </visual>
            </link>
          </model>
        </sdf>""".format(name=marker_name)
        
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        
        resp = spawn_srv(marker_name, sdf_xml, "", pose, "world")
        if resp.success:
            rospy.loginfo("Successfully spawned marker: %s", marker_name)
        else:
            rospy.logwarn("Gazebo refused to spawn %s: %s", marker_name, resp.status_message)
            
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed for %s: %s", marker_name, e)
# ---------------------------------------------------

def main():
    rospy.init_node("foi_inspection_pose")

    # Optional startup delay
    delay = float(rospy.get_param("~start_delay", 0.0))
    if delay > 0:
        rospy.loginfo("Delaying start by %.1f seconds...", delay)
        rospy.sleep(delay)

    # Wait for controller manager
    rospy.loginfo("Waiting for controller manager services...")
    rospy.wait_for_service("/controller_manager/switch_controller")
    rospy.wait_for_service("/controller_manager/unload_controller")

    pose_topic = rospy.get_param("~pose_topic", "/command/pose")
    enable_srv = rospy.get_param("~enable_motors_srv", "/enable_motors")

    # Enable motors
    ok = enable_motors(enable_srv, True, timeout=10.0)
    if not ok:
        rospy.logwarn("Motors not enabled; continuing anyway.")


    # FLIGHT PARAMETERS
    rate_hz = float(rospy.get_param("~rate", 10.0))
    transition_time = float(rospy.get_param("~transition_time", 8.0)) # Time to move between points
    hover_time = 5.0 # Hover time at each FOI in seconds

    pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
    r = rospy.Rate(rate_hz)

    # 1. DEFINE RTL AND FOIs
    # Spawn location used as Return To Launch (RTL)
    rtl_pos = (-14.0, -9.0, 1.0)
   
    # Coordinates in the shaded region (bottom-left quadrant relative to the plane at 0,0)
    # Keeping z=1.0 as requested and staying clear of the center to avoid collision.
    fois = [
        (-8.5,  -1.75, 3.0),
       (-8.5, 1.75, 3.0),
        (-5.0,  0.0, 3.0),
        (0.0,  0.0, 2.75),
    ]

# --- UPDATED: SPAWN THE MARKERS AT GROUND LEVEL (z=1.0) ---
    rospy.loginfo("Spawning FOI visual markers in Gazebo...")
    for index, (fx, fy, fz) in enumerate(fois):
        marker_name = "foi_marker_{}".format(index)
        spawn_gazebo_marker(marker_name, fx, fy, fz)

# --- UPDATED: FLY 1m ABOVE THE FOIs (z + 1.5) ---
    # We use a list comprehension to add 1.5 to the altitude of every waypoint
    mission_plan = [(x, y, z +1.5) for (x, y, z) in [rtl_pos] + fois + [rtl_pos]]
   
    msg = PoseStamped()
    msg.header.frame_id = "world"

    # Start at the RTL position
    current_x, current_y, current_z = mission_plan[0]
    current_yaw = 0.0

    rospy.loginfo("Starting FOI inspection mission. Total waypoints: %d", len(mission_plan)-1)

    # Execute the mission plan
    for i in range(1, len(mission_plan)):
        if rospy.is_shutdown():
            break
           
        target_x, target_y, target_z = mission_plan[i]
       
        # Calculate yaw to face the next waypoint
        target_yaw = math.atan2(target_y - current_y, target_x - current_x)
       
        rospy.loginfo("Moving to Waypoint %d: (x:%.1f, y:%.1f, z:%.1f)", i, target_x, target_y, target_z)

        # 2. TRANSITION TO WAYPOINT (Interpolation for smooth flight)
        pos_steps = max(1, int(transition_time * rate_hz))
        for s in range(pos_steps):
            if rospy.is_shutdown():
                break
           
            t = float(s + 1) / float(pos_steps)
           
            # Linear interpolation for position
            xi = current_x + (target_x - current_x) * t
            yi = current_y + (target_y - current_y) * t
            zi = current_z + (target_z - current_z) * t
           
            # Simple yaw interpolation
            dy = (target_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
            yawi = current_yaw + dy * t

            msg.header.stamp = rospy.Time.now()
            msg.pose.position.x = xi
            msg.pose.position.y = yi
            msg.pose.position.z = zi
            msg.pose.orientation = yaw_to_quat(yawi)
            pub.publish(msg)
            r.sleep()

        # Update current position
        current_x, current_y, current_z = target_x, target_y, target_z
        current_yaw = target_yaw

        # 3. HOVER AT WAYPOINT
        if i < len(mission_plan) - 1:
            rospy.loginfo("Hovering at FOI %d for %.1f seconds...", i, hover_time)
            hover_ticks = int(hover_time * rate_hz)
            for _ in range(hover_ticks):
                if rospy.is_shutdown():
                    break
                msg.header.stamp = rospy.Time.now()
                pub.publish(msg)
                r.sleep()
        else:
            rospy.loginfo("Returned to RTL position.")

    # 4. HOLD RTL POSITION INDEFINITELY
    rospy.loginfo("Mission complete. Holding position at RTL.")
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        r.sleep()

if __name__ == "__main__":
    main()
