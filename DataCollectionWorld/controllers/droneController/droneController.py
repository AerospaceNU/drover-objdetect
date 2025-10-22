"""droneController controller."""

from controller import Robot, Keyboard
from math import pow
import time
import math
from utils import ControlMode, Status, clamp, load_waypoints, print_manual_controls, print_waypoints_status, wrap_to_pi, dist

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize sensors
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# Initialize keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

# Initialize camera motors
camera_roll_motor = robot.getDevice("camera roll")
camera_pitch_motor = robot.getDevice("camera pitch")
try:
    camera_yaw_motor = robot.getDevice("camera yaw")
except Exception:
    camera_yaw_motor = None

# Initialize LEDs
front_left_led = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

# Get initial orientation
initial_roll, initial_pitch, _ = imu.getRollPitchYaw()

# Initialize propeller motors
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

# Set motors to velocity control mode
for m in (front_left_motor, front_right_motor, rear_left_motor, rear_right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

print("Start the drone...")

# Wait for sensors to initialize
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

# ============================================================================
# Control Parameters
# ============================================================================

k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

control_mode = ControlMode.MANUAL
target_altitude = 1.0

# Waypoint navigation setup
waypoints = load_waypoints()
curr_waypoint_idx = 0
waypoints_travel = [Status.UNVISITED for i in range(len(waypoints))]
if waypoints:
    waypoints_travel[curr_waypoint_idx] = Status.APPROACHING

# Camera control parameters
cam_pitch_offset = 0.0
cam_yaw_offset = 0.0
cam_pitch_step = 0.005
cam_yaw_step = 0.005
cam_pitch_limit = 0.5
cam_yaw_limit = 1.6

# Key debouncing
last_p_key_time = 0.0
p_key_debounce_delay = 0.5

print_manual_controls()

# ============================================================================
# Main Control Loop
# ============================================================================

while robot.step(timestep) != -1:
    now = robot.getTime()

    # Read sensor data
    roll, pitch, _yaw = imu.getRollPitchYaw()
    gx, gy, _gz = gyro.getValues()
    _x, _y, altitude = gps.getValues()

    # LED blinking
    led_state = int(now) % 2
    front_left_led.set(led_state)
    front_right_led.set(1 - led_state)

    # Camera stabilization and control
    camera_roll_motor.setPosition(-0.115 * gx)
    camera_pitch_motor.setPosition(-0.1 * gy + cam_pitch_offset)
    if camera_yaw_motor is not None:
        camera_yaw_motor.setPosition(cam_yaw_offset)

    # Initialize control disturbances
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0

    # Keyboard input handling
    key = keyboard.getKey()
    while key != -1:
        # Mode switching (P key)
        if key in (ord("P"), ord("p")):
            if now - last_p_key_time > p_key_debounce_delay:
                last_p_key_time = now
                if control_mode == ControlMode.MANUAL:
                    control_mode = ControlMode.AUTO
                    target_altitude = 20.0
                    if waypoints:
                        curr_waypoint_idx = 0
                        waypoints_travel = [
                            Status.UNVISITED for i in range(len(waypoints))
                        ]
                        waypoints_travel[curr_waypoint_idx] = Status.APPROACHING
                    clear_terminal()
                    print(f"\n*** SWITCHED TO {control_mode.value.upper()} MODE ***")
                    time.sleep(0.1)
                else:
                    control_mode = ControlMode.MANUAL
                    target_altitude = 1.0
                    print_manual_controls()
        
        # Manual control inputs
        elif control_mode == ControlMode.MANUAL:
            if key in (ord("W"), ord("w")):
                pitch_disturbance = -2.0
            elif key in (ord("S"), ord("s")):
                pitch_disturbance = 2.0
            elif key in (ord("A"), ord("a")):
                roll_disturbance = 1.0
            elif key in (ord("D"), ord("d")):
                roll_disturbance = -1.0
            elif key in (ord("Q"), ord("q")):
                yaw_disturbance = 2.0
            elif key in (ord("E"), ord("e")):
                yaw_disturbance = -2.0
            elif key in (ord("R"), ord("r")):
                target_altitude += 0.05
                print(f"target altitude: {target_altitude:.2f} m")
            elif key in (ord("F"), ord("f")):
                target_altitude -= 0.05
                print(f"target altitude: {target_altitude:.2f} m")

        # Camera control (arrow keys)
        if key == Keyboard.UP:
            cam_pitch_offset = clamp(
                cam_pitch_offset - cam_pitch_step, -cam_pitch_limit, cam_pitch_limit
            )
        elif key == Keyboard.DOWN:
            cam_pitch_offset = clamp(
                cam_pitch_offset + cam_pitch_step, -cam_pitch_limit, cam_pitch_limit
            )
        elif key == Keyboard.LEFT:
            cam_yaw_offset = clamp(
                cam_yaw_offset + cam_yaw_step, -cam_yaw_limit, cam_yaw_limit
            )
        elif key == Keyboard.RIGHT:
            cam_yaw_offset = clamp(
                cam_yaw_offset - cam_yaw_step, -cam_yaw_limit, cam_yaw_limit
            )

        key = keyboard.getKey()

    # Autopilot waypoint navigation
    if control_mode == ControlMode.AUTO and waypoints:
        target_altitude = 20.0
        pitch_disturbance = -2.0

        wx, wy = waypoints[curr_waypoint_idx]
        theta_desired = math.atan2(wy - _y, wx - _x)
        yaw_disturbance = wrap_to_pi(theta_desired - _yaw)

        if dist(_x, _y, wx, wy) < 8.5:
            waypoints_travel[curr_waypoint_idx] = Status.VISITED
            curr_waypoint_idx = (curr_waypoint_idx + 1) % len(waypoints)
            waypoints_travel[curr_waypoint_idx] = Status.APPROACHING
            wx, wy = waypoints[curr_waypoint_idx]

        frame_count = int(now * 1000 / timestep)
        print_waypoints_status(
            waypoints, waypoints_travel, _x, _y, frame_count, control_mode
        )

    # Calculate motor inputs
    pitch_input = (
        k_pitch_p * clamp(pitch - initial_pitch, -1.0, 1.0) + gy + pitch_disturbance
    )
    roll_input = (
        k_roll_p * clamp(roll - initial_roll, -1.0, 1.0) + gx + roll_disturbance
    )

    if control_mode == ControlMode.AUTO:
        yaw_input = yaw_disturbance * 0.35
    else:
        yaw_input = yaw_disturbance

    clamped_diff_alt = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * pow(clamped_diff_alt, 3.0)

    front_left_motor_input = (
        k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    )
    front_right_motor_input = (
        k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    )
    rear_left_motor_input = (
        k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    )
    rear_right_motor_input = (
        k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
    )

    # Apply motor velocities
    front_left_motor.setVelocity(front_left_motor_input)
    front_right_motor.setVelocity(-front_right_motor_input)
    rear_left_motor.setVelocity(-rear_left_motor_input)
    rear_right_motor.setVelocity(rear_right_motor_input)
