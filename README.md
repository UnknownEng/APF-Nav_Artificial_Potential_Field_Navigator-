# APF-Nav: Artificial Potential Field navigator (ROS Noetic + Gazebo)

Hector quadrotor flying waypoints through a 30 x 30 m hangar-style room
(`my_room_world/worlds/room_6x6.world`), using a 270 deg Hokuyo UTM-30LX lidar
and Gazebo ground-truth pose.

## Packages

| Package | Contents |
|---|---|
| `my_room_world` | World, launch files and nodes: `apf_navigator.py` (APF), `dwa_inspection.py` (rule-based state machine), `lawnmower_pose.py` (POI survey), `mapping_supervisor.py` (RTAB-Map start/stop/save), `test.py` (lidar orientation check) |
| `dwa_rubi` (`src/dwa_rubi/dwa_rubi`) | `obstacle_avoidance.py`: 3D DWA follower on an RTAB-Map point cloud |
| `my_hector_addons` | Extra sensor xacros (360 lidar, RGB-D, stereo) |
| `adaptive_dwa_ros` | Placeholder (test node + RViz config) |

## Setup

```bash
sudo apt install python3-vcstool ros-noetic-hector-gazebo-plugins ros-noetic-rtabmap-ros
cd <workspace>
vcs import src < dependencies.repos
rosdep install --from-paths src --ignore-src -y
catkin build   # or catkin_make
source devel/setup.bash
```

## Run

APF navigator:

```bash
roslaunch my_room_world apf_navigation.launch
rosservice call /enable_motors "{enable: true}"
```

Rule-based navigator: `roslaunch my_room_world room.launch`, then
`rosrun my_room_world dwa_inspection.py`.

DWA follower: `roslaunch dwa_rubi dwa_follower.launch` with waypoints on
`/dwa/goal` (not `/command/pose`, which the position controller uses) and a
point cloud on `/rtabmap/cloud_map`.

## Lidar orientation

Both lidar nodes read the lidar's yaw relative to `base_link` from TF. If
`rosrun my_room_world test.py` shows the scan is rotated relative to TF, set
`~scan_yaw_offset` (radians) on the node instead.
