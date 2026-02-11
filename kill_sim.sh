#!/bin/bash
# kill all ROS 2 and Gazebo processes, clean up stale shared memory locks

killall -9 gz ruby rviz2 parameter_bridge 2>/dev/null
pkill -9 -f "ros2 launch" 2>/dev/null
pkill -9 -f "turtlebot4" 2>/dev/null
pkill -9 -f "nav2" 2>/dev/null
pkill -9 -f "slam_toolbox" 2>/dev/null

rm -rf /dev/shm/fastrtps_*

echo "Simulation killed and shared memory cleaned."
