import subprocess
import time

interval = 10

physical_commands = ["ros2 launch turtlebot4_navigation localization.launch.py map:=map_2.yaml",
            "ros2 launch turtlebot4_navigation nav2.launch.py",
            "ros2 launch turtlebot4_viz view_navigation.launch.py",
            "ros2 run turtlebot4_python_tutorials mail_delivery"]
            
simulation_commands = ["ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py nav2:=true slam:=false localization:=true rviz:=true",
                        "ros2 launch turtlebot4_viz view_robot.launch.py"]
                        # "ros2 run turtlebot4_python_tutorials nav_through_poses"]

for cmd in physical_commands:
    print(f"Launching: {cmd}")
    subprocess.Popen([
        "x-terminal-emulator",
        "-e",
        f"bash -c '{cmd}; exec bash'"
    ])
    time.sleep(interval)
