#!/usr/bin/env bash
# ============================================================================
# Cuttlebot - Source ROS 2 Environment
#
# This sets up your shell so ros2 commands work. Must be sourced, not executed:
#
#   source source_setup.bash
#
# You usually don't need this — install.sh adds these lines to ~/.bashrc
# so new terminals source automatically. Use this if you need to manually
# re-source in an existing terminal.
# ============================================================================

ROS_DISTRO="jazzy"
WS_DIR="$HOME/turtlebot4_ws"

# Guard: must be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed."
    echo "  Usage: source source_setup.bash"
    exit 1
fi

# Source ROS 2 base
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    echo "[OK] Sourced ROS 2 ${ROS_DISTRO}"
else
    echo "[FAIL] ROS 2 ${ROS_DISTRO} not found at /opt/ros/${ROS_DISTRO}/setup.bash"
    return 1
fi

# Source workspace overlay
if [ -f "${WS_DIR}/install/setup.bash" ]; then
    source "${WS_DIR}/install/setup.bash"
    echo "[OK] Sourced workspace at ${WS_DIR}"
else
    echo "[WARN] Workspace not built yet. Run ./build.sh first."
    return 1
fi

echo ""
echo "Ready. You can now run:"
echo "  ros2 run cuttlebot_nodes <node_name>"
echo "  python3 scripts/automation2.py"
echo ""
