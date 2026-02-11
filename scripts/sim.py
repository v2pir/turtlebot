import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDS_DIR = os.path.join(REPO_ROOT, "cuttlefish_sim", "sim_gazebo", "worlds")

WORLD_NAME = "cuttlebot_world"

def main():
    world_sdf = os.path.join(WORLDS_DIR, WORLD_NAME + ".sdf")

    system_worlds = "/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds"
    link_path = os.path.join(system_worlds, WORLD_NAME + ".sdf")
    if not os.path.exists(link_path):
        print(f"Symlinking world into {system_worlds}...")
        subprocess.run(["sudo", "ln", "-sf", world_sdf, link_path], check=True)

    cmd = [
        "ros2", "launch", "turtlebot4_gz_bringup", "turtlebot4_gz.launch.py",
        f"world:={WORLD_NAME}",
        "slam:=true",
        "nav2:=true",
        "rviz:=true",
    ]

    print(f"world sdf:  {world_sdf}")
    print(f"launching:  {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nsimulation stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nlaunch failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
