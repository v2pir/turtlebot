#!/usr/bin/env bash
# ============================================================================
# Cuttlebot - Rebuild Workspace
#
# Run this after making changes to any code in cuttlebot_nodes/.
# Uses --symlink-install so Python changes are picked up immediately,
# but you still need to rebuild if you add new nodes/entry points.
#
# Usage:
#   ./build.sh            # full rebuild
#   ./build.sh --this     # rebuild only cuttlebot_nodes (faster)
# ============================================================================
set -euo pipefail

ROS_DISTRO="jazzy"
WS_DIR="$HOME/turtlebot4_ws"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# Check workspace exists
if [ ! -d "${WS_DIR}/src" ]; then
    fail "Workspace not found at ${WS_DIR}. Run install.sh first."
fi

# Source ROS 2
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
else
    fail "ROS 2 ${ROS_DISTRO} not found. Run install.sh first."
fi

cd "${WS_DIR}"

if [[ "${1:-}" == "--this" ]]; then
    info "Rebuilding cuttlebot_nodes only..."
    colcon build --symlink-install --packages-select cuttlebot_nodes
else
    info "Rebuilding entire workspace..."
    colcon build --symlink-install
fi

ok "Build complete"
echo ""
echo "  If this is a new terminal, run:"
echo "    source setup.bash"
echo "  or open a new terminal (bashrc sources automatically)."
echo ""
