#!/usr/bin/env bash
# ============================================================================
# Cuttlebot - Full System Setup
# Target: Ubuntu 24.04 LTS + ROS 2 Jazzy
#
# Safe to re-run — skips anything already installed.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# After install, run the automation launcher:
#   python3 scripts/automation2.py
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="jazzy"
WS_DIR="$HOME/turtlebot4_ws"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
skip()  { echo -e "${YELLOW}[SKIP]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# Helper: check if ALL listed packages are installed
all_installed() {
    for pkg in "$@"; do
        if ! dpkg -s "$pkg" > /dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

# Helper: check if ALL listed pip packages are importable
all_pip_installed() {
    for mod in "$@"; do
        if ! python3 -c "import $mod" > /dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

# ---- Pre-checks -----------------------------------------------------------
info "Checking system requirements..."

if [[ "$(lsb_release -cs 2>/dev/null)" != "noble" ]]; then
    fail "This script requires Ubuntu 24.04 (Noble Numbat). Detected: $(lsb_release -ds 2>/dev/null || echo 'unknown')"
fi
ok "Ubuntu 24.04 detected"

# ---- 1. ROS 2 Jazzy -------------------------------------------------------
ROS_PKGS=(ros-${ROS_DISTRO}-desktop)
if all_installed "${ROS_PKGS[@]}"; then
    skip "ROS 2 ${ROS_DISTRO} already installed"
else
    info "Installing ROS 2 ${ROS_DISTRO}..."

    # Enable Ubuntu Universe repo
    sudo apt-get update && sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y universe

    # Add ROS 2 GPG key
    sudo apt-get install -y curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg

    # Add ROS 2 apt repo
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y ros-${ROS_DISTRO}-desktop
    ok "ROS 2 ${ROS_DISTRO} installed"
fi

# Source ROS 2 for the rest of this script
source /opt/ros/${ROS_DISTRO}/setup.bash

# ---- 2. Dev tools ----------------------------------------------------------
DEV_PKGS=(python3-colcon-common-extensions python3-rosdep python3-pip build-essential git)
if all_installed "${DEV_PKGS[@]}"; then
    skip "Dev tools already installed"
else
    info "Installing development tools..."
    sudo apt-get install -y "${DEV_PKGS[@]}"
    ok "Dev tools installed"
fi

# ---- 3. TurtleBot4 packages -----------------------------------------------
TB4_PKGS=(
    ros-${ROS_DISTRO}-turtlebot4-desktop
    ros-${ROS_DISTRO}-turtlebot4-navigation
    ros-${ROS_DISTRO}-turtlebot4-viz
    ros-${ROS_DISTRO}-turtlebot4-msgs
    ros-${ROS_DISTRO}-irobot-create-msgs
)
if all_installed "${TB4_PKGS[@]}"; then
    skip "TurtleBot4 packages already installed"
else
    info "Installing TurtleBot4 packages..."
    sudo apt-get install -y "${TB4_PKGS[@]}"
    ok "TurtleBot4 packages installed"
fi

# ---- 4. Nav2 (full navigation stack) --------------------------------------
NAV_PKGS=(
    ros-${ROS_DISTRO}-navigation2
    ros-${ROS_DISTRO}-nav2-bringup
    ros-${ROS_DISTRO}-slam-toolbox
)
if all_installed "${NAV_PKGS[@]}"; then
    skip "Nav2 already installed"
else
    info "Installing Nav2 packages..."
    sudo apt-get install -y "${NAV_PKGS[@]}"
    ok "Nav2 installed"
fi

# ---- 5. Simulation packages (optional, for Gazebo sim) --------------------
SIM_PKGS=(
    ros-${ROS_DISTRO}-turtlebot4-simulator
    ros-${ROS_DISTRO}-turtlebot4-gz-bringup
)
if all_installed "${SIM_PKGS[@]}"; then
    skip "Simulation packages already installed"
else
    info "Installing simulation packages..."
    sudo apt-get install -y "${SIM_PKGS[@]}" \
        || skip "Simulation packages not available — physical robot only"
fi

# ---- 6. Python dependencies ------------------------------------------------
if all_pip_installed numpy scipy matplotlib; then
    skip "Python dependencies already installed (numpy, scipy, matplotlib)"
else
    info "Installing Python dependencies..."
    pip3 install --user numpy scipy matplotlib
    ok "Python deps installed"
fi

# ---- 7. Initialize rosdep --------------------------------------------------
if [ -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    skip "rosdep already initialized"
else
    info "Initializing rosdep..."
    sudo rosdep init || true
fi
rosdep update --rosdistro=${ROS_DISTRO}
ok "rosdep ready"

# ---- 8. Set up colcon workspace -------------------------------------------
info "Setting up colcon workspace at ${WS_DIR}..."
mkdir -p "${WS_DIR}/src"

# Symlink this repo's ROS 2 package into the workspace
if [ -e "${WS_DIR}/src/cuttlebot_nodes" ]; then
    skip "cuttlebot_nodes already linked in workspace"
else
    ln -s "${REPO_DIR}/cuttlebot_nodes" "${WS_DIR}/src/cuttlebot_nodes"
    ok "Symlinked cuttlebot_nodes into workspace"
fi

# ---- 9. Install package dependencies with rosdep --------------------------
info "Resolving package dependencies via rosdep..."
cd "${WS_DIR}"
rosdep install --from-paths src --ignore-src -r -y
ok "Package dependencies resolved"

# ---- 10. Build the workspace -----------------------------------------------
info "Building workspace with colcon..."
cd "${WS_DIR}"
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --symlink-install
ok "Workspace built"

# ---- 11. Shell setup -------------------------------------------------------
SHELL_RC="$HOME/.bashrc"
ROS_SOURCE_LINE="source /opt/ros/${ROS_DISTRO}/setup.bash"
WS_SOURCE_LINE="source ${WS_DIR}/install/setup.bash"

add_to_rc() {
    local line="$1"
    if grep -qF "$line" "$SHELL_RC" 2>/dev/null; then
        skip "Already in .bashrc: $line"
    else
        echo "$line" >> "$SHELL_RC"
        ok "Added to .bashrc: $line"
    fi
}

add_to_rc "$ROS_SOURCE_LINE"
add_to_rc "$WS_SOURCE_LINE"

# ---- Done ------------------------------------------------------------------
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Workspace:  ${WS_DIR}"
echo "  Package:    cuttlebot_nodes"
echo ""
echo "  To use in a new terminal:"
echo "    source ~/.bashrc"
echo ""
echo "  To run nodes:"
echo "    ros2 run cuttlebot_nodes nav_to_pose"
echo "    ros2 run cuttlebot_nodes turtlebot4_first_python_node"
echo ""
echo "  To launch full automation:"
echo "    cd ${REPO_DIR}"
echo "    python3 scripts/automation2.py"
echo ""
echo "  To rebuild after code changes:"
echo "    cd ${WS_DIR} && colcon build --symlink-install"
echo ""
