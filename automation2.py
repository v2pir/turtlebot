import subprocess
import time
import os
import re

# This program launches the necessary commands and waits until the previous command sends a message allowing for the next command to go underway.  It creates a new terminal logging the previous terminal to read for messages

interval = 0.2  # just a tiny delay for spawn stability

# ------------------------------------------MESSAGE FLAGS -------------------------------------------------------

# [amcl-2] [WARN] [NUMBER] [amcl]: AMCL cannot publish a pose or update the transform. Please set the initial pose...

# [controller_server-1] [ERROR] [NUMBER] [local_costmap.local_costmap]: StaticLayer: "map" passed to lookupTransform argument target_frame does not exist.

# [rviz2-1] [INFO] [NUMBER] [rviz2]: Message Filter dropping message: frame 'rplidar_link' at time NUMBER for reason 'discarding message because the queue is full'

# ---------------------------------------------------------------------------------------------------------------

commands = [
    ("localization",
     "ros2 launch turtlebot4_navigation localization.launch.py map:=map_2.yaml",
     r"Please set the initial pose"),
    ("nav2",
     "ros2 launch turtlebot4_navigation nav2.launch.py",
     r"lookupTransform argument"),
    ("viz",
     "ros2 launch turtlebot4_viz view_navigation.launch.py",
     r"Trying to create a map"),
    ("mail",
     "ros2 run turtlebot4_python_tutorials mail_delivery",
     r"dont exit"),
]

def wait_for_message(log_path: str, pattern: str, timeout: float = 300.0):
    """Block until regex pattern appears in log file"""
    regex = re.compile(pattern)
    start = time.time()

    # Wait until file exists
    while not os.path.exists(log_path):
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for log file: {log_path}")
        time.sleep(0.05)

    # Read through file for pattern
    with open(log_path, "r", errors="ignore") as f:
        while True:
            line = f.readline()
            if line:
                print(f"[log:{os.path.basename(log_path)}] {line}", end="")
                if regex.search(line):
                    print(f"\n✅ Matched: {pattern}\n")
                    return
            else:
                # no new line yet
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timed out waiting for message /{pattern}/ in {log_path}")
                time.sleep(0.05)

for name, cmd, ready_pattern in commands:
    log = f"/tmp/{name}.log"
    # clear old log so you don't match stale messages
    try:
        os.remove(log)
    except FileNotFoundError:
        pass

    print(f"Launching: {cmd}")

    # Run command in a new terminal, but pipe output to a log we can read.
    # stdbuf makes output line-buffered so the log updates immediately.
    wrapped = f"stdbuf -oL -eL {cmd} 2>&1 | tee {log}; exec bash"

    subprocess.Popen([
        "x-terminal-emulator",
        "-e",
        f"bash -lc \"{wrapped}\""
    ])

    time.sleep(interval)         # tiny spawn delay
    wait_for_message(log, ready_pattern, timeout=300)

print("All commands launched after readiness messages.")

