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
# After install, run the simulation:
#   python3 scripts/sim.py
# ============================================================================
set -eo pipefail

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
C_GZ='\033[0;36m'        # cyan       (Gazebo)
C_WB='\033[1;37m'        # bright white (Webots)
C_TBSIM='\033[0;36m'     # cyan       (TB4 sim packages)
C_PY='\033[0;32m'        # green
C_ROSDEP='\033[0;35m'    # magenta
C_WS='\033[1;34m'        # bright blue
C_DEPS='\033[0;34m'      # blue
C_BUILD='\033[1;33m'     # bright yellow
C_SHELL='\033[1;32m'     # bright green

# Current section color (updated per phase)
SEC=""

# Track warnings/failures that didn't halt the script
WARNINGS=()

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
warn()  { echo -e "${C_WARN}  ⚠${C_RESET} $*"; WARNINGS+=("$*"); }
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
section "$C_ROS" "1/13  ROS 2 JAZZY"

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
section "$C_DEV" "2/13  DEVELOPMENT TOOLS"

DEV_PKGS=(python3-colcon-common-extensions python3-rosdep python3-pip build-essential git)
if all_installed "${DEV_PKGS[@]}"; then
    skip "Dev tools already installed"
else
    info "Installing colcon, rosdep, pip, build-essential, git..."
    sudo apt-get install -y "${DEV_PKGS[@]}"
    ok "Dev tools installed"
fi

# ---- 3. TurtleBot4 packages -----------------------------------------------
section "$C_TB4" "3/13  TURTLEBOT4 PACKAGES"

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
section "$C_NAV" "4/13  NAV2 NAVIGATION STACK"

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

# ---- 5. Gazebo Harmonic (simulation engine) --------------------------------
section "$C_GZ" "5/13  GAZEBO HARMONIC"

GZ_PKGS=(ros-${ROS_DISTRO}-ros-gz)
if all_installed "${GZ_PKGS[@]}"; then
    skip "Gazebo Harmonic (ros-gz) already installed"
else
    info "Installing Gazebo Harmonic via ros-gz..."
    sudo apt-get install -y "${GZ_PKGS[@]}" \
        || warn "Gazebo Harmonic — packages not available for this system"
fi

# ---- 6. TurtleBot4 Gazebo simulation packages -----------------------------
section "$C_TBSIM" "6/13  TURTLEBOT4 GAZEBO SIM"

TBSIM_PKGS=(
    ros-${ROS_DISTRO}-turtlebot4-simulator
    ros-${ROS_DISTRO}-turtlebot4-gz-bringup
)
if all_installed "${TBSIM_PKGS[@]}"; then
    skip "TurtleBot4 Gazebo sim packages already installed"
else
    info "Installing TurtleBot4 simulator, gz-bringup..."
    sudo apt-get install -y "${TBSIM_PKGS[@]}" \
        || warn "TurtleBot4 Gazebo sim — packages not available"
fi

# ---- 7. Webots simulator (for cuttlefish sim) -----------------------------
section "$C_WB" "7/13  WEBOTS"

if command -v webots > /dev/null 2>&1; then
    skip "Webots already installed ($(webots --version 2>/dev/null || echo 'unknown version'))"
else
    info "Installing Webots..."
    if snap list webots > /dev/null 2>&1; then
        skip "Webots snap already installed"
    else
        sudo snap install webots \
            || warn "Webots — snap install failed, install manually from https://cyberbotics.com"
    fi
fi

# ---- 8. Python dependencies ------------------------------------------------
section "$C_PY" "8/13  PYTHON DEPENDENCIES"

if all_pip_installed numpy scipy matplotlib; then
    skip "numpy, scipy, matplotlib already installed"
else
    info "Installing numpy, scipy, matplotlib..."
    pip3 install --user numpy scipy matplotlib
    ok "Python deps installed"
fi

# ---- 9. Initialize rosdep --------------------------------------------------
section "$C_ROSDEP" "9/13  ROSDEP INIT"

if [ -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    skip "rosdep already initialized"
else
    info "Initializing rosdep..."
    sudo rosdep init || true
fi
rosdep update --rosdistro=${ROS_DISTRO}
ok "rosdep ready"

# ---- 10. Set up colcon workspace ------------------------------------------
section "$C_WS" "10/13  COLCON WORKSPACE"

info "Setting up workspace at ${WS_DIR}..."
mkdir -p "${WS_DIR}/src"

if [ -e "${WS_DIR}/src/cuttlebot_nodes" ]; then
    skip "cuttlebot_nodes already linked in workspace"
else
    ln -s "${REPO_DIR}/cuttlebot_nodes" "${WS_DIR}/src/cuttlebot_nodes"
    ok "Symlinked cuttlebot_nodes into workspace"
fi

# ---- 11. Install package dependencies with rosdep -------------------------
section "$C_DEPS" "11/13  PACKAGE DEPENDENCIES"

info "Resolving package dependencies via rosdep..."
cd "${WS_DIR}"
rosdep install --from-paths src --ignore-src -r -y
ok "Package dependencies resolved"

# ---- 12. Build the workspace -----------------------------------------------
section "$C_BUILD" "12/13  BUILD WORKSPACE"

info "Building with colcon (symlink-install)..."
cd "${WS_DIR}"
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --symlink-install
ok "Workspace built"

# ---- 13. Shell setup -------------------------------------------------------
section "$C_SHELL" "13/13  SHELL CONFIGURATION"

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

# Show failure summary if there were any warnings
if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo ""
    echo -e "${C_FAIL}${C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
    echo -e "${C_FAIL}${C_BOLD}  WARNINGS (${#WARNINGS[@]})${C_RESET}"
    echo -e "${C_FAIL}${C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
    echo ""
    for w in "${WARNINGS[@]}"; do
        echo -e "  ${C_FAIL}✘${C_RESET} ${w}"
    done
    echo ""
    echo -e "  ${C_DIM}These are non-critical — the rest of the install succeeded.${C_RESET}"
    echo -e "  ${C_DIM}Re-run this script after fixing the issues above.${C_RESET}"
fi

echo ""
echo -e "${C_OK}${C_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo -e "${C_OK}${C_BOLD}  SETUP COMPLETE (with ${#WARNINGS[@]} warning(s))${C_RESET}"
else
    echo -e "${C_OK}${C_BOLD}  SETUP COMPLETE${C_RESET}"
fi
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
echo ""
echo -e "  ${C_DIM}Launch simulation (Gazebo + Nav2 + experiment nodes):${C_RESET}"
echo "    cd ${REPO_DIR}"
echo "    python3 scripts/sim.py"
echo ""
echo -e "  ${C_DIM}Rebuild after code changes:${C_RESET}"
echo -e "    ./build.sh              ${C_DIM}# full rebuild${C_RESET}"
echo -e "    ./build.sh --this       ${C_DIM}# rebuild cuttlebot_nodes only${C_RESET}"
echo ""
echo -e "  ${C_DIM}Launch Gazebo simulation:${C_RESET}"
echo "    ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py"
echo ""
echo -e "  ${C_DIM}Launch Webots simulation:${C_RESET}"
echo "    webots cuttlefish_sim/sim_webots/worlds/my_first_simulation_test.wbt"
echo ""
