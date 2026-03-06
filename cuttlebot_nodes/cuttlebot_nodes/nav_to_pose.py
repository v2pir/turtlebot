#!/usr/bin/env python3

# Copyright 2022 Clearpath Robotics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Roni Kreinin (rkreinin@clearpathrobotics.com)

import rclpy

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator


def main():
    rclpy.init()

    navigator = TurtleBot4Navigator()

    # # Start on dock (we are not starting from dock in our simulations yet)
    if not navigator.getDockedStatus():
         navigator.info('Docking before intialising pose')
         navigator.dock()

    # Set initial pose (cuttlebot_world: spawn at center-south, facing +y = WEST)
    initial_pose = navigator.getPoseStamped([0.0, -1.0], TurtleBot4Directions.WEST)
    navigator.setInitialPose(initial_pose)
    print("[DEBUG] Initial pose set")

    # Wait for Nav2
    navigator.waitUntilNav2Active()

    # Set goal pose (left chamber in cuttlebot_world)
    goal_pose = navigator.getPoseStamped([-2.5, 1.5], TurtleBot4Directions.NORTH)
    print("[DEBUG] Goal Pose set")

    # # Undock (no need to undock in our case)
    navigator.undock()

    # Go to each goal pose
    navigator.startToPose(goal_pose)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
