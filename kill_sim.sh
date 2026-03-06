#!/bin/bash
# kill all ROS 2 and Gazebo processes, clean up stale shared memory locks

# try graceful SIGINT first
killall -INT gz ruby rviz2 parameter_bridge 2>/dev/null
pkill -INT -f "ros2 launch" 2>/dev/null
pkill -INT -f "cuttlebot_nodes" 2>/dev/null
pkill -INT -f "turtlebot4" 2>/dev/null
pkill -INT -f "nav2" 2>/dev/null
pkill -INT -f "slam_toolbox" 2>/dev/null

sleep 3

# force-kill anything still running
killall -9 gz ruby rviz2 parameter_bridge 2>/dev/null
pkill -9 -f "ros2 launch" 2>/dev/null
pkill -9 -f "cuttlebot_nodes" 2>/dev/null
pkill -9 -f "turtlebot4" 2>/dev/null
pkill -9 -f "nav2" 2>/dev/null
pkill -9 -f "slam_toolbox" 2>/dev/null

rm -rf /dev/shm/fastrtps_*

echo "Simulation killed and shared memory cleaned."
