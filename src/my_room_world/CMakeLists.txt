cmake_minimum_required(VERSION 3.0.2)
project(my_room_world)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/apf_navigator.py
  scripts/dwa_inspection.py
  scripts/lawnmower_pose.py
  scripts/mapping_supervisor.py
  scripts/test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch worlds DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
