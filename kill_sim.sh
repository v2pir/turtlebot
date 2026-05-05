#!/bin/bash
# kill all ROS 2 and Gazebo processes, clean up stale shared memory locks

echo "Stopping simulation processes..."

# try graceful SIGINT first
killall -INT gz ruby rviz2 parameter_bridge 2>/dev/null
pkill -INT -f "ros2 launch" 2>/dev/null
pkill -INT -f "ros2 run" 2>/dev/null
pkill -INT -f "ros2 daemon" 2>/dev/null
pkill -INT -f "cuttlebot_nodes" 2>/dev/null
pkill -INT -f "turtlebot4" 2>/dev/null
pkill -INT -f "irobot_create" 2>/dev/null
pkill -INT -f "nav2" 2>/dev/null
pkill -INT -f "amcl" 2>/dev/null
pkill -INT -f "map_server" 2>/dev/null
pkill -INT -f "lifecycle_manager" 2>/dev/null
pkill -INT -f "robot_state_publisher" 2>/dev/null
pkill -INT -f "ros_gz" 2>/dev/null
pkill -INT -f "slam_toolbox" 2>/dev/null
pkill -INT -f "controller_manager" 2>/dev/null
pkill -INT -f "spawner" 2>/dev/null

sleep 3

# force-kill anything still running
killall -9 gz ruby rviz2 parameter_bridge 2>/dev/null
pkill -9 -f "ros2 launch" 2>/dev/null
pkill -9 -f "ros2 run" 2>/dev/null
pkill -9 -f "ros2 daemon" 2>/dev/null
pkill -9 -f "cuttlebot_nodes" 2>/dev/null
pkill -9 -f "turtlebot4" 2>/dev/null
pkill -9 -f "irobot_create" 2>/dev/null
pkill -9 -f "nav2" 2>/dev/null
pkill -9 -f "amcl" 2>/dev/null
pkill -9 -f "map_server" 2>/dev/null
pkill -9 -f "lifecycle_manager" 2>/dev/null
pkill -9 -f "robot_state_publisher" 2>/dev/null
pkill -9 -f "ros_gz" 2>/dev/null
pkill -9 -f "slam_toolbox" 2>/dev/null
pkill -9 -f "controller_manager" 2>/dev/null
pkill -9 -f "spawner" 2>/dev/null

# clean shared memory
rm -f /dev/shm/fastrtps*

echo "Simulation killed and shared memory cleaned."
