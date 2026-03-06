import os
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDS_DIR = os.path.join(REPO_ROOT, "cuttlefish_sim", "sim_gazebo", "worlds")
LAUNCH_DIR = os.path.join(REPO_ROOT, "cuttlefish_sim", "sim_gazebo", "launch")
MAP_FILE = os.path.join(REPO_ROOT, "maps", "cuttlebot_world.yaml")

WORLD_NAME = "cuttlebot_world"

# spawn position: center-south of the room, facing +y (toward divider gap)
SPAWN_Y = "-1.0"
SPAWN_YAW = "1.5708"

# colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# expirement nodes to launch after gazebo is up
NODES = [
    ["ros2", "run", "cuttlebot_nodes", "location_awareness",
     "--ros-args", "-p", "use_sim_time:=true"],
    ["ros2", "run", "cuttlebot_nodes", "brain_node",
     "--ros-args", "-p", "use_sim_time:=true"],
    ["ros2", "run", "cuttlebot_nodes", "state_manager",
     "--ros-args", "-p", "use_sim_time:=true"],
]

# takes like 15 seconds for gazebo and nav2 to run lets assume
GAZEBO_STARTUP_DELAY = 15


def main():
    world_sdf = os.path.join(WORLDS_DIR, WORLD_NAME + ".sdf")
    launch_file = os.path.join(LAUNCH_DIR, "cuttlebot_gz.launch.py")

    if not os.path.exists(MAP_FILE):
        print(f"{RED}[error]{RESET} map file not found: {MAP_FILE}", file=sys.stderr)
        sys.exit(1)

    # symlink world SDF so Gazebo can find it by name
    system_worlds = "/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds"
    link_path = os.path.join(system_worlds, WORLD_NAME + ".sdf")
    if not os.path.exists(link_path):
        print(f"{YELLOW}symlinking world into {system_worlds}...{RESET}")
        subprocess.run(["sudo", "ln", "-sf", world_sdf, link_path], check=True)

    # launch gazebo with our custom launch file, localization, and nav2
    gazebo_cmd = [
        "ros2", "launch", launch_file,
        f"world:={WORLD_NAME}",
        "localization:=true",
        f"map:={MAP_FILE}",
        "nav2:=true",
        "rviz:=true",
        f"y:={SPAWN_Y}",
        f"yaw:={SPAWN_YAW}",
    ]

    print(f"{DIM}world sdf:  {world_sdf}{RESET}")
    print(f"{CYAN}{BOLD}launching:  {' '.join(gazebo_cmd)}{RESET}")
    print()

    processes = []

    try:
        # launch gazebo + nav2 + rviz
        print(f"{CYAN}[gazebo]{RESET} starting gazebo + nav2 + rviz...")
        gz_proc = subprocess.Popen(gazebo_cmd)
        processes.append(("gazebo", gz_proc))

        print(f"{YELLOW}[wait]{RESET} waiting {GAZEBO_STARTUP_DELAY}s for gazebo and nav2 to start...")
        time.sleep(GAZEBO_STARTUP_DELAY)

        # check that Gazebo is still running
        if gz_proc.poll() is not None:
            print(f"{RED}[error]{RESET} gazebo exited early!", file=sys.stderr)
            sys.exit(gz_proc.returncode or 1)

        print(f"{GREEN}[ready]{RESET} gazebo is running\n")

        # launch experiment nodes
        node_colors = {
            "location_awareness": BLUE,
            "brain_node": MAGENTA,
            "state_manager": GREEN,
        }
        for cmd in NODES:
            name = cmd[3]  # executable name (e.g. "brain_node")
            color = node_colors.get(name, CYAN)
            print(f"{color}[{name}]{RESET} launching: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd)
            processes.append((name, proc))
            time.sleep(1)  # wait after 1 node is started before launching the next one

        print(f"\n{GREEN}{BOLD}all nodes launched. cntrl+c to stop everything.{RESET}\n")

        # wait for gazebo to exit (basically just cntrl c)
        gz_proc.wait()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[shutdown]{RESET} shutting down all processes...")

    finally:
        # kill everything in reverse order
        for name, proc in reversed(processes):
            if proc.poll() is None:
                print(f"{YELLOW}  stopping {name} (pid {proc.pid}){RESET}")
                proc.send_signal(signal.SIGINT)

        # wait 5 seconds for evertyhign to exit then force kill
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"{RED}  force-killing {name}{RESET}")
                proc.kill()

        print(f"{DIM}simulation stopped.{RESET}")


if __name__ == "__main__":
    main()
