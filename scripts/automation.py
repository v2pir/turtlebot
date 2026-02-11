import subprocess
import time
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_PATH = os.path.join(REPO_ROOT, "maps", "map_2.yaml")

interval = 10

physical_commands = [f"ros2 launch turtlebot4_navigation localization.launch.py map:={MAP_PATH}",
            "ros2 launch turtlebot4_navigation nav2.launch.py",
            "ros2 launch turtlebot4_viz view_navigation.launch.py",
            "ros2 run cuttlebot_nodes mail_delivery"]
            
simulation_commands = ["ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py nav2:=true slam:=false localization:=true rviz:=true",
                        "ros2 launch turtlebot4_viz view_robot.launch.py"]
                        # "ros2 run cuttlebot_nodes nav_through_poses"]

for cmd in physical_commands:
    print(f"Launching: {cmd}")
    subprocess.Popen([
        "x-terminal-emulator",
        "-e",
        f"bash -c '{cmd}; exec bash'"
    ])
    time.sleep(interval)
