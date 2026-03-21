#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np

def scan_cb(msg):
    # Convert ranges to a numpy array and replace "inf" with 10.0
    ranges = np.array(msg.ranges)
    ranges[np.isinf(ranges)] = 10.0
    
    num_rays = len(ranges)
    
    # Grab the 4 exact compass points of the raw sensor
    index_0   = 0                     # The physical "Zero" of the sensor
    index_90  = int(num_rays * 0.25)  # 1/4 of the way through the array (+90 deg)
    index_180 = int(num_rays * 0.50)  # 1/2 of the way through the array (180 deg)
    index_270 = int(num_rays * 0.75)  # 3/4 of the way through the array (-90 deg)

    val_0   = ranges[index_0]
    val_90  = ranges[index_90]
    val_180 = ranges[index_180]
    val_270 = ranges[index_270]

    # Print the raw hardware data
    print(f"RAW SENSOR MOUNT ->  0°: {val_0:.1f}m  |  +90°: {val_90:.1f}m  |  180°: {val_180:.1f}m  |  -90°: {val_270:.1f}m")

if __name__ == '__main__':
    rospy.init_node('lidar_tester')
    rospy.Subscriber('/scan', LaserScan, scan_cb)
    print("Testing Raw Lidar Orientation. Press Ctrl+C to stop.")
    rospy.spin()
