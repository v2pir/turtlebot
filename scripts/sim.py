import os
import signal
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDS_DIR = os.path.join(REPO_ROOT, "cuttlefish_sim", "sim_gazebo", "worlds")
LAUNCH_DIR = os.path.join(REPO_ROOT, "cuttlefish_sim", "sim_gazebo", "launch")
MAP_FILE = os.path.join(REPO_ROOT, "maps", "cuttlebot_world.yaml")

TELEOP = "--teleop" in sys.argv
TRAIN = "--train" in sys.argv

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

# experiment nodes to launch after gazebo is up
MODE = "train" if TRAIN else "test"

NODES = [
    ["ros2", "run", "cuttlebot_nodes", "location_awareness",
     "--ros-args", "-p", "use_sim_time:=true"],
    ["ros2", "run", "cuttlebot_nodes", "vision_node",
     "--ros-args", "-p", "use_sim_time:=true"],
    ["ros2", "run", "cuttlebot_nodes", "brain_node",
     "--ros-args", "-p", "use_sim_time:=true"],
    ["ros2", "run", "cuttlebot_nodes", "state_manager",
     "--ros-args", "-p", "use_sim_time:=true",
     "-p", f"mode:={MODE}"],
]

GAZEBO_STARTUP_DELAY = 15


def wait_for_topic(topic, timeout=120):
    """Poll until a ROS topic exists, meaning the robot is ready."""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True, text=True, timeout=10)
        if topic in result.stdout.splitlines():
            return True
        time.sleep(3)
    return False


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

    # patch diffdrive cmd_vel_timeout (baked into URDF, only read at startup)
    control_yaml = "/opt/ros/jazzy/share/irobot_create_control/config/control.yaml"
    try:
        with open(control_yaml) as f:
            content = f.read()
        if "cmd_vel_timeout: 0.5" in content:
            print(f"{YELLOW}patching cmd_vel_timeout in {control_yaml}...{RESET}")
            patched = content.replace("cmd_vel_timeout: 0.5", "cmd_vel_timeout: 5.0")
            subprocess.run(
                ["sudo", "tee", control_yaml],
                input=patched.encode(), stdout=subprocess.DEVNULL, check=True)
            print(f"{GREEN}cmd_vel_timeout set to 5.0s{RESET}")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"{RED}[error]{RESET} failed to patch cmd_vel_timeout: {e}",
              file=sys.stderr)

    # teleop mode: keep localization (so the map loads in RViz) but skip nav2
    # (its controller_server fights teleop for cmd_vel)
    gazebo_cmd = [
        "ros2", "launch", launch_file,
        f"world:={WORLD_NAME}",
        "localization:=true",
        f"map:={MAP_FILE}",
        f"nav2:={'false' if TELEOP else 'true'}",
        "rviz:=true",
        f"y:={SPAWN_Y}",
        f"yaw:={SPAWN_YAW}",
    ]

    print(f"{DIM}world sdf:  {world_sdf}{RESET}")
    print(f"{CYAN}{BOLD}launching:  {' '.join(gazebo_cmd)}{RESET}")
    print()

    processes = []
    gz_log = None

    try:
        mode = "teleop" if TELEOP else ("train" if TRAIN else "test")
        print(f"{CYAN}[gazebo]{RESET} starting gazebo + rviz ({mode} mode)...")
        if TELEOP:
            gz_log = open("/tmp/cuttlebot_gazebo.log", "w")
            print(f"{DIM}  gazebo output → /tmp/cuttlebot_gazebo.log{RESET}")
            gz_proc = subprocess.Popen(gazebo_cmd, stdout=gz_log, stderr=gz_log)
        else:
            gz_proc = subprocess.Popen(gazebo_cmd)
        processes.append(("gazebo", gz_proc))

        if TELEOP:
            print(f"{YELLOW}[wait]{RESET} waiting for robot to load (checking for /cmd_vel topic)...")
            if not wait_for_topic("/cmd_vel"):
                print(f"{RED}[error]{RESET} timed out waiting for robot", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"{YELLOW}[wait]{RESET} waiting {GAZEBO_STARTUP_DELAY}s for gazebo to start...")
            time.sleep(GAZEBO_STARTUP_DELAY)

        # check that Gazebo is still running
        if gz_proc.poll() is not None:
            print(f"{RED}[error]{RESET} gazebo exited early!", file=sys.stderr)
            sys.exit(gz_proc.returncode or 1)

        print(f"{GREEN}[ready]{RESET} gazebo is running\n")

        if TELEOP:
            print(f"{MAGENTA}{BOLD}teleop mode{RESET}\n")

            # undock the robot — Create3 ignores cmd_vel while docked
            print(f"{YELLOW}[undock]{RESET} undocking robot...")
            try:
                subprocess.run(
                    ["ros2", "action", "send_goal", "/undock",
                     "irobot_create_msgs/action/Undock", "{}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=60)
                time.sleep(2)
                print(f"{GREEN}[undock]{RESET} done\n")
            except subprocess.TimeoutExpired:
                print(f"{YELLOW}[undock]{RESET} timed out — continuing anyway\n")

            print(f"{BOLD}controls:{RESET}")
            print()
            print(f"        {BOLD}u{RESET}    {BOLD}i{RESET}    {BOLD}o{RESET}")
            print(f"        {BOLD}j{RESET}    {BOLD}k{RESET}    {BOLD}l{RESET}")
            print(f"        {BOLD}m{RESET}    {BOLD},{RESET}    {BOLD}.{RESET}")
            print()
            print(f"  {BOLD}i{RESET} = forward    {BOLD},{RESET} = backward")
            print(f"  {BOLD}j{RESET} = turn left  {BOLD}l{RESET} = turn right")
            print(f"  {BOLD}u{RESET} = fwd+left   {BOLD}o{RESET} = fwd+right")
            print(f"  {BOLD}m{RESET} = bwd+left   {BOLD}.{RESET} = bwd+right")
            print(f"  {BOLD}k{RESET} = stop")
            print(f"  {BOLD}q{RESET}/{BOLD}z{RESET} = increase/decrease max speed")
            print(f"  {BOLD}ctrl+c{RESET} = quit\n")

            # run teleop directly in this terminal (reads keypresses from stdin)
            # stamped:=true because Create3's motion_control expects TwistStamped
            teleop_proc = subprocess.Popen(
                ["ros2", "run", "teleop_twist_keyboard",
                 "teleop_twist_keyboard",
                 "--ros-args", "-p", "use_sim_time:=true",
                 "-p", "stamped:=true"])
            processes.append(("teleop", teleop_proc))
            teleop_proc.wait()
        else:
            # launch experiment nodes
            node_colors = {
                "location_awareness": BLUE,
                "brain_node": MAGENTA,
                "vision_node": YELLOW,
                "state_manager": GREEN,
            }
            for cmd in NODES:
                name = cmd[3]  # executable name (e.g. "brain_node")
                color = node_colors.get(name, CYAN)
                print(f"{color}[{name}]{RESET} launching: {' '.join(cmd)}")
                proc = subprocess.Popen(cmd)
                processes.append((name, proc))
                time.sleep(1)

            print(f"\n{GREEN}{BOLD}all nodes launched. ctrl+c to stop everything.{RESET}\n")

        if not TELEOP:
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

        if gz_log:
            gz_log.close()
        print(f"{DIM}simulation stopped.{RESET}")


if __name__ == "__main__":
    main()
