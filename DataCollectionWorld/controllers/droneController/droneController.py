"""droneController controller with object detection and logging."""

from controller import Supervisor, Keyboard
from math import pow
import time
import math
import numpy as np
import cv2
from datetime import datetime
from utils import (ControlMode, Status, clamp, load_waypoints, 
                   print_manual_controls, print_waypoints_status, wrap_to_pi, dist, clear_terminal)
from object_detector import ObjectDetector
from detection_db import DetectionDatabase

# ============================================================================
# Robot Initialization
# ============================================================================

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# Initialize camera
camera = supervisor.getDevice("camera")
camera.enable(timestep)

cam_width = camera.getWidth()
cam_height = camera.getHeight()
print(f"Camera device: {camera.getName()} | {cam_width}x{cam_height}")

# Initialize sensors
imu = supervisor.getDevice("inertial unit")
imu.enable(timestep)

gps = supervisor.getDevice("gps")
gps.enable(timestep)

compass = supervisor.getDevice("compass")
compass.enable(timestep)

gyro = supervisor.getDevice("gyro")
gyro.enable(timestep)

# Initialize keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

# Initialize camera motors
camera_roll_motor = supervisor.getDevice("camera roll")
camera_pitch_motor = supervisor.getDevice("camera pitch")
try:
    camera_yaw_motor = supervisor.getDevice("camera yaw")
except Exception:
    camera_yaw_motor = None

# Initialize LEDs
front_left_led = supervisor.getDevice("front left led")
front_right_led = supervisor.getDevice("front right led")

# Get initial orientation
initial_roll, initial_pitch, _ = imu.getRollPitchYaw()

# Initialize propeller motors
front_left_motor = supervisor.getDevice("front left propeller")
front_right_motor = supervisor.getDevice("front right propeller")
rear_left_motor = supervisor.getDevice("rear left propeller")
rear_right_motor = supervisor.getDevice("rear right propeller")

# Set motors to velocity control mode
for m in (front_left_motor, front_right_motor, rear_left_motor, rear_right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

print("Start the drone...")

# Wait for sensors to initialize
while supervisor.step(timestep) != -1:
    if supervisor.getTime() > 1.0:
        break

# ============================================================================
# Detection and Recording Setup
# ============================================================================

# Initialize object detector
detector = ObjectDetector(camera, supervisor)

# Initialize database
session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
db = DetectionDatabase("detections.db")
session_id = db.create_session(session_name)

# Video recording setup
raw_video_path = f"videos/raw_{session_name}.mp4"
labeled_video_path = f"videos/labeled_{session_name}.mp4"

# Create videos directory if it doesn't exist
import os
os.makedirs("videos", exist_ok=True)

# Video writers (will be initialized on first frame)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
fps = 30.0  # Approximate FPS
raw_video_writer = None
labeled_video_writer = None

frame_idx = 0
total_detections = 0

def camera_frame_bgr():
    """Get camera frame as BGR image."""
    buf = camera.getImage()
    if buf is None:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size != cam_width * cam_height * 4:
        return None
    arr = arr.reshape((cam_height, cam_width, 4))
    bgr = arr[:, :, :3].copy()
    return bgr

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    frame_copy = frame.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        color_type = det['type']
        
        # Color for bounding box (green for visibility)
        bbox_color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), bbox_color, 2)
        
        # Draw label with background
        label = f"{color_type}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_copy, (x, y - label_h - 4), (x + label_w, y), bbox_color, -1)
        cv2.putText(frame_copy, label, (x, y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw 3D position info
        pos_text = f"3D: ({det['position_3d'][0]:.1f}, {det['position_3d'][1]:.1f}, {det['position_3d'][2]:.1f})"
        cv2.putText(frame_copy, pos_text, (x, y + h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Add frame info
    info_text = f"Frame: {frame_idx} | Detections: {len(detections)}"
    cv2.putText(frame_copy, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_copy

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

try:
    while supervisor.step(timestep) != -1:
        now = supervisor.getTime()

        # Read sensor data
        roll, pitch, _yaw = imu.getRollPitchYaw()
        gx, gy, _gz = gyro.getValues()
        _x, _y, altitude = gps.getValues()

        # Get camera frame
        frame_bgr = camera_frame_bgr()
        
        if frame_bgr is not None:
            # Initialize video writers on first frame
            if raw_video_writer is None:
                raw_video_writer = cv2.VideoWriter(
                    raw_video_path, fourcc, fps, (cam_width, cam_height))
                labeled_video_writer = cv2.VideoWriter(
                    labeled_video_path, fourcc, fps, (cam_width, cam_height))
                print(f"Started recording videos:")
                print(f"  Raw: {raw_video_path}")
                print(f"  Labeled: {labeled_video_path}")
            
            # Write raw frame
            raw_video_writer.write(frame_bgr)
            
            # Detect red spheres
            detections = detector.detect_visible_spheres(
                imu, gps, cam_pitch_offset, cam_yaw_offset, filter_color="red"
            )
            
            # Log detections to database
            if detections:
                for det in detections:
                    db.log_detection(
                        frame_idx=frame_idx,
                        timestamp=now,
                        object_name=det['name'],
                        object_type=det['type'],
                        bbox=det['bbox'],
                        position_3d=det['position_3d']
                    )
                total_detections += len(detections)
            
            # Draw detections and write labeled frame
            labeled_frame = draw_detections(frame_bgr, detections)
            labeled_video_writer.write(labeled_frame)
            
            frame_idx += 1

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

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    # Cleanup and finalize
    print("\n" + "=" * 60)
    print("FINALIZING SESSION")
    print("=" * 60)
    
    # Release video writers
    if raw_video_writer is not None:
        raw_video_writer.release()
        print(f"Saved raw video: {raw_video_path}")
    
    if labeled_video_writer is not None:
        labeled_video_writer.release()
        print(f"Saved labeled video: {labeled_video_path}")
    
    # Update session in database
    db.update_session(session_id, frame_idx, total_detections, 
                     raw_video_path, labeled_video_path)
    
    # Print statistics
    stats = db.get_statistics()
    print(f"\nSession Statistics:")
    print(f"  Total frames: {frame_idx}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Frames with detections: {stats['frames_with_detections']}")
    print(f"  Unique objects: {stats['unique_objects']}")
    
    # Close database
    db.close()
    print("\nDatabase closed.")
    print("=" * 60)
