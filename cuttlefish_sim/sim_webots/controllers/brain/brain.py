from controller import Robot
# ^ meant for webots simulation environment
# logic should preserve in Gazebo though(?) I hope lol
import numpy as np
import random
import time
from math import exp, erf, sqrt

ACTIONS = 2
LEFT = 0
RIGHT = 1

STATES = 4
EXPM_LR = 0
EXPM_RL = 1
CTRL_LR = 2
CTRL_RL = 3

LIVE_RWD = 5.0
DEAD_RWD = 1.0
UNOBTAINABLE_RWD = 0.0

ALPHA = 0.10
BETA = 1.0

PRETRAIN_TRIALS = 100

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
camera = robot.getDevice("camera")
gps = robot.getDevice("gps")

camera.enable(timestep)
gps.enable(timestep)

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

random.seed(time.time())

qTbl = [[0.0 for _ in range(ACTIONS)] for _ in range(STATES)]

def _norm_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0 if x < mean else 1.0
    z = (x - mean) / (std * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))

def prob_wait(tim: float) -> float:
    mean = 70.0
    std_dev = 20.0
    beta_weight = 2.0
    return 1.0 / exp(beta_weight * _norm_cdf(tim, mean, std_dev))

def action_select(q_values, beta):
    denom = 0.0
    for v in q_values:
        denom += exp(beta * v)
    r = random.random()
    cum = 0.0
    for i, v in enumerate(q_values):
        p = exp(beta * v) / denom
        cum += p
        if cum >= r:
            return i
    return len(q_values) - 1

def reward_from_state_action(state, act):
    if state == EXPM_LR:
        return LIVE_RWD if act == LEFT else DEAD_RWD
    elif state == EXPM_RL:
        return DEAD_RWD if act == LEFT else LIVE_RWD
    elif state == CTRL_LR:
        return UNOBTAINABLE_RWD if act == LEFT else DEAD_RWD
    else:
        return DEAD_RWD if act == LEFT else UNOBTAINABLE_RWD

def parse_custom_data(raw: str):
    if raw is None:
        return None, None
    raw = raw.strip()
    if not raw:
        return None, None
    s = None
    d = None
    for part in raw.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "state":
            try:
                s = int(v)
            except:
                pass
        elif k == "delay":
            try:
                d = float(v)
            except:
                pass
    return s, d

def run_trial_logic(delay_s, forced_state=None):
    if forced_state is None:
        r = random.random()
        if r < 0.25:
            state = EXPM_LR
        elif r < 0.50:
            state = EXPM_RL
        elif r < 0.75:
            state = CTRL_LR
        else:
            state = CTRL_RL
    else:
        state = int(forced_state)

    p_wait = prob_wait(delay_s)

    if state == EXPM_LR:
        qTbl[state][LEFT] *= p_wait
    elif state == EXPM_RL:
        qTbl[state][RIGHT] *= p_wait

    act = action_select(qTbl[state], BETA)
    rwd = reward_from_state_action(state, act)

    qTbl[state][act] = qTbl[state][act] + ALPHA * (rwd - qTbl[state][act])

    return state, act, rwd, p_wait

def stop():
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

def move_forward(speed=5.0):
    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)

def turn_left(speed=6.0):
    left_motor.setVelocity(-speed)
    right_motor.setVelocity(speed)

def turn_right(speed=6.0):
    left_motor.setVelocity(speed)
    right_motor.setVelocity(-speed)

def spin_in_place(speed=6.0, direction=1):
    if direction >= 0:
        left_motor.setVelocity(speed)
        right_motor.setVelocity(-speed)
    else:
        left_motor.setVelocity(-speed)
        right_motor.setVelocity(speed)

def detect_means(image, width, height):
    reds, blues, blacks = [], [], []
    y0 = max(0, height // 3)
    y1 = min(height - 1, 2 * height // 3)
    for x in range(0, width, 2):
        for y in range(y0, y1, 2):
            r = camera.imageGetRed(image, width, x, y)
            g = camera.imageGetGreen(image, width, x, y)
            b = camera.imageGetBlue(image, width, x, y)
            if r > g * 1.5 and r > b * 1.5:
                reds.append(x)
            elif b > r * 1.5 and b > g * 1.5:
                blues.append(x)
            elif r < 50 and g < 50 and b < 50:
                blacks.append(x)
    mean_red = np.mean(reds) if reds else None
    mean_blue = np.mean(blues) if blues else None
    mean_black = np.mean(blacks) if blacks else None
    return mean_red, mean_blue, mean_black

def door_color_visible(image, width, height, color):
    cnt = 0
    total = 0
    x0 = max(0, width // 2 - 30)
    x1 = min(width - 1, width // 2 + 30)
    y0 = max(0, height // 3)
    y1 = min(height - 1, 2 * height // 3)
    for x in range(x0, x1, 2):
        for y in range(y0, y1, 2):
            r = camera.imageGetRed(image, width, x, y)
            g = camera.imageGetGreen(image, width, x, y)
            b = camera.imageGetBlue(image, width, x, y)
            total += 1
            if color == "blue":
                if b > r * 1.5 and b > g * 1.5:
                    cnt += 1
            else:
                if r > g * 1.5 and r > b * 1.5:
                    cnt += 1
    if total <= 0:
        return False
    return (cnt / total) >= 0.10

def action_to_color(act):
    return "blue" if act == LEFT else "red"

MOVE_SPEED = 5.0
TURN_SPEED = 6.0

DOOR_Y = 0.083
DOOR_APPROACH_MARGIN = 0.010
WAIT_START_Y = DOOR_Y - DOOR_APPROACH_MARGIN

CONFUSE_TURN_SPEED = 2.0
CONFUSE_TURN_PERIOD = 0.60

PANIC_START = 150.0
PANIC_SPIN_SPEED = 14.0

CHECK_OPEN_PERIOD = 0.7

for _ in range(PRETRAIN_TRIALS):
    state, act, rwd, pw = run_trial_logic(0.0, None)
    if pw != 0:
        if state == EXPM_LR:
            qTbl[state][LEFT] /= pw
        elif state == EXPM_RL:
            qTbl[state][RIGHT] /= pw

# print("[brain] Pretraining done. Starting embodied trials.")

last_custom = None
decision_state = "idle"
pre_trial_timer = 0.0
confuse_timer = 0.0
trial_elapsed = 0.0
panic_dir = 1
chosen_color = None
check_timer = 0.0

while robot.step(timestep) != -1:
    raw_custom = robot.getCustomData()
    if raw_custom != last_custom:
        last_custom = raw_custom
        decision_state = "idle"
        pre_trial_timer = 0.0
        confuse_timer = 0.0
        trial_elapsed = 0.0
        panic_dir = 1 if random.random() < 0.5 else -1
        chosen_color = None
        check_timer = 0.0

    dt = timestep / 1000.0
    trial_elapsed += dt

    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    mean_red, mean_blue, mean_black = detect_means(image, width, height)

    y = gps.getValues()[1]

    if trial_elapsed >= PANIC_START and decision_state not in ("idle", "turn_blue", "turn_red"):
        decision_state = "panic"

    if decision_state == "panic":
        spin_in_place(PANIC_SPIN_SPEED, panic_dir)
        continue

    if decision_state == "idle":
        stop()
        pre_trial_timer += dt
        if pre_trial_timer >= 1.0:
            pre_trial_timer = 0.0
            forced_state, forced_delay = parse_custom_data(raw_custom)
            trial_delay = float(forced_delay) if forced_delay is not None else 10.0
            trial_state, trial_act, trial_reward, trial_pw = run_trial_logic(trial_delay, forced_state)
            chosen_color = action_to_color(trial_act)
            print(
                f"[Trial] delay={trial_delay:.1f}s state={trial_state} "
                f"act={trial_act} ({chosen_color}) "
                f"p_wait={trial_pw:.3f} predicted_reward={trial_reward:.2f}"
            )
            decision_state = f"turn_{chosen_color}"

    elif decision_state == "turn_blue":
        if mean_blue is None:
            turn_left(TURN_SPEED)
        elif mean_blue < width / 2 - 10:
            turn_left(TURN_SPEED)
        elif mean_blue > width / 2 + 10:
            turn_right(TURN_SPEED)
        else:
            stop()
            decision_state = "approach"

    elif decision_state == "turn_red":
        if mean_red is None:
            turn_left(TURN_SPEED)
        elif mean_red < width / 2 - 10:
            turn_left(TURN_SPEED)
        elif mean_red > width / 2 + 10:
            turn_right(TURN_SPEED)
        else:
            stop()
            decision_state = "approach"

    elif decision_state == "approach":
        if y >= WAIT_START_Y:
            if door_color_visible(image, width, height, chosen_color):
                stop()
                confuse_timer = 0.0
                check_timer = 0.0
                decision_state = "wait"
            else:
                decision_state = "through"
        else:
            move_forward(MOVE_SPEED)

    elif decision_state == "wait":
        confuse_timer += dt
        phase = int(confuse_timer / CONFUSE_TURN_PERIOD) % 2
        if phase == 0:
            turn_left(CONFUSE_TURN_SPEED)
        else:
            turn_right(CONFUSE_TURN_SPEED)

        check_timer += dt
        if check_timer >= CHECK_OPEN_PERIOD:
            check_timer = 0.0
            if not door_color_visible(image, width, height, chosen_color):
                decision_state = "search_black"

    elif decision_state == "search_black":
        if mean_black is None:
            spin_in_place(6.0, 1)
        else:
            decision_state = "center_black"

    elif decision_state == "center_black":
        if mean_black is None:
            decision_state = "search_black"
        elif mean_black < width / 2 - 10:
            turn_left(TURN_SPEED)
        elif mean_black > width / 2 + 10:
            turn_right(TURN_SPEED)
        else:
            stop()
            decision_state = "through"

    elif decision_state == "through":
        if mean_black is None:
            move_forward(MOVE_SPEED)
        elif mean_black < width / 2 - 10:
            left_motor.setVelocity(MOVE_SPEED * 0.6)
            right_motor.setVelocity(MOVE_SPEED)
        elif mean_black > width / 2 + 10:
            left_motor.setVelocity(MOVE_SPEED)
            right_motor.setVelocity(MOVE_SPEED * 0.6)
        else:
            move_forward(MOVE_SPEED)
