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

# ---- Section colors (each phase gets its own color) -----------------------
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_DIM='\033[2m'

# Status colors
C_OK='\033[1;32m'       # bright green
C_SKIP='\033[0;33m'     # yellow
C_FAIL='\033[1;31m'     # bright red
C_WARN='\033[0;33m'     # yellow

# Section colors — each install phase has a unique color
C_PRECHECK='\033[1;37m'  # bright white
C_ROS='\033[1;36m'       # bright cyan
C_DEV='\033[1;34m'       # bright blue
C_TB4='\033[1;35m'       # bright magenta
C_NAV='\033[1;33m'       # bright yellow
C_SIM='\033[0;36m'       # cyan
C_PY='\033[0;32m'        # green
C_ROSDEP='\033[0;35m'    # magenta
C_WS='\033[1;34m'        # bright blue
C_DEPS='\033[0;34m'      # blue
C_BUILD='\033[1;33m'     # bright yellow
C_SHELL='\033[1;32m'     # bright green

# Current section color (updated per phase)
SEC=""

# ---- Logging functions ----------------------------------------------------
section() {
    SEC="$1"
    local label="$2"
    echo ""
    echo -e "${SEC}${C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
    echo -e "${SEC}${C_BOLD}  $label${C_RESET}"
    echo -e "${SEC}${C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
}
info()  { echo -e "${SEC}  ▸${C_RESET} $*"; }
ok()    { echo -e "${C_OK}  ✔${C_RESET} $*"; }
skip()  { echo -e "${C_SKIP}  ⏭${C_RESET} ${C_DIM}$*${C_RESET}"; }
fail()  { echo -e "${C_FAIL}  ✘ $*${C_RESET}"; exit 1; }

# ---- Helpers --------------------------------------------------------------
all_installed() {
    for pkg in "$@"; do
        if ! dpkg -s "$pkg" > /dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

all_pip_installed() {
    for mod in "$@"; do
        if ! python3 -c "import $mod" > /dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

# ---- Pre-checks -----------------------------------------------------------
section "$C_PRECHECK" "PRE-FLIGHT CHECKS"

info "Checking system requirements..."
if [[ "$(lsb_release -cs 2>/dev/null)" != "noble" ]]; then
    fail "This script requires Ubuntu 24.04 (Noble Numbat). Detected: $(lsb_release -ds 2>/dev/null || echo 'unknown')"
fi
ok "Ubuntu 24.04 detected"

# ---- 1. ROS 2 Jazzy -------------------------------------------------------
section "$C_ROS" "1/11  ROS 2 JAZZY"

ROS_PKGS=(ros-${ROS_DISTRO}-desktop)
if all_installed "${ROS_PKGS[@]}"; then
    skip "ROS 2 ${ROS_DISTRO} already installed"
else
    info "Installing ROS 2 ${ROS_DISTRO}..."

    sudo apt-get update && sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y universe

    sudo apt-get install -y curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y ros-${ROS_DISTRO}-desktop
    ok "ROS 2 ${ROS_DISTRO} installed"
fi

source /opt/ros/${ROS_DISTRO}/setup.bash

# ---- 2. Dev tools ----------------------------------------------------------
section "$C_DEV" "2/11  DEVELOPMENT TOOLS"

DEV_PKGS=(python3-colcon-common-extensions python3-rosdep python3-pip build-essential git)
if all_installed "${DEV_PKGS[@]}"; then
    skip "Dev tools already installed"
else
    info "Installing colcon, rosdep, pip, build-essential, git..."
    sudo apt-get install -y "${DEV_PKGS[@]}"
    ok "Dev tools installed"
fi

# ---- 3. TurtleBot4 packages -----------------------------------------------
section "$C_TB4" "3/11  TURTLEBOT4 PACKAGES"

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
    info "Installing TurtleBot4 desktop, navigation, viz, msgs..."
    sudo apt-get install -y "${TB4_PKGS[@]}"
    ok "TurtleBot4 packages installed"
fi

# ---- 4. Nav2 (full navigation stack) --------------------------------------
section "$C_NAV" "4/11  NAV2 NAVIGATION STACK"

NAV_PKGS=(
    ros-${ROS_DISTRO}-navigation2
    ros-${ROS_DISTRO}-nav2-bringup
    ros-${ROS_DISTRO}-slam-toolbox
)
if all_installed "${NAV_PKGS[@]}"; then
    skip "Nav2 already installed"
else
    info "Installing navigation2, nav2-bringup, slam-toolbox..."
    sudo apt-get install -y "${NAV_PKGS[@]}"
    ok "Nav2 installed"
fi

# ---- 5. Simulation packages (optional) ------------------------------------
section "$C_SIM" "5/11  SIMULATION (GAZEBO)"

SIM_PKGS=(
    ros-${ROS_DISTRO}-turtlebot4-simulator
    ros-${ROS_DISTRO}-turtlebot4-gz-bringup
)
if all_installed "${SIM_PKGS[@]}"; then
    skip "Simulation packages already installed"
else
    info "Installing Gazebo simulation packages..."
    sudo apt-get install -y "${SIM_PKGS[@]}" \
        || skip "Simulation packages not available — physical robot only"
fi

# ---- 6. Python dependencies ------------------------------------------------
section "$C_PY" "6/11  PYTHON DEPENDENCIES"

if all_pip_installed numpy scipy matplotlib; then
    skip "numpy, scipy, matplotlib already installed"
else
    info "Installing numpy, scipy, matplotlib..."
    pip3 install --user numpy scipy matplotlib
    ok "Python deps installed"
fi

# ---- 7. Initialize rosdep --------------------------------------------------
section "$C_ROSDEP" "7/11  ROSDEP INIT"

if [ -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    skip "rosdep already initialized"
else
    info "Initializing rosdep..."
    sudo rosdep init || true
fi
rosdep update --rosdistro=${ROS_DISTRO}
ok "rosdep ready"

# ---- 8. Set up colcon workspace -------------------------------------------
section "$C_WS" "8/11  COLCON WORKSPACE"

info "Setting up workspace at ${WS_DIR}..."
mkdir -p "${WS_DIR}/src"

if [ -e "${WS_DIR}/src/cuttlebot_nodes" ]; then
    skip "cuttlebot_nodes already linked in workspace"
else
    ln -s "${REPO_DIR}/cuttlebot_nodes" "${WS_DIR}/src/cuttlebot_nodes"
    ok "Symlinked cuttlebot_nodes into workspace"
fi

# ---- 9. Install package dependencies with rosdep --------------------------
section "$C_DEPS" "9/11  PACKAGE DEPENDENCIES"

info "Resolving package dependencies via rosdep..."
cd "${WS_DIR}"
rosdep install --from-paths src --ignore-src -r -y
ok "Package dependencies resolved"

# ---- 10. Build the workspace -----------------------------------------------
section "$C_BUILD" "10/11  BUILD WORKSPACE"

info "Building with colcon (symlink-install)..."
cd "${WS_DIR}"
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --symlink-install
ok "Workspace built"

# ---- 11. Shell setup -------------------------------------------------------
section "$C_SHELL" "11/11  SHELL CONFIGURATION"

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
echo -e "${C_OK}${C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo -e "${C_OK}${C_BOLD}  SETUP COMPLETE${C_RESET}"
echo -e "${C_OK}${C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
echo ""
echo -e "  ${C_BOLD}Workspace${C_RESET}   ${WS_DIR}"
echo -e "  ${C_BOLD}Package${C_RESET}     cuttlebot_nodes"
echo -e "  ${C_BOLD}Repo${C_RESET}        ${REPO_DIR}"
echo ""
echo -e "  ${C_DIM}Source environment (new terminal or manual refresh):${C_RESET}"
echo -e "    source ~/.bashrc        ${C_DIM}# auto-added by this script${C_RESET}"
echo -e "    source setup.bash       ${C_DIM}# manual alternative${C_RESET}"
echo ""
echo -e "  ${C_DIM}Run nodes:${C_RESET}"
echo "    ros2 run cuttlebot_nodes nav_to_pose"
echo "    ros2 run cuttlebot_nodes turtlebot4_first_python_node"
echo ""
echo -e "  ${C_DIM}Launch full automation (opens multiple terminals):${C_RESET}"
echo "    cd ${REPO_DIR}"
echo "    python3 scripts/automation2.py"
echo ""
echo -e "  ${C_DIM}Rebuild after code changes:${C_RESET}"
echo -e "    ./build.sh              ${C_DIM}# full rebuild${C_RESET}"
echo -e "    ./build.sh --this       ${C_DIM}# rebuild cuttlebot_nodes only${C_RESET}"
echo ""
