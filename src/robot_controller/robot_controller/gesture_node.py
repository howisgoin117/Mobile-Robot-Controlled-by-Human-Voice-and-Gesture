import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import cv2
import json
import os
import numpy as np
import mediapipe as mp
import csv
import time
import math
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


# Camera index
DEFAULT_CAMERA_INDEX = '/dev/amr_camera'
MAX_CAMERA_SCAN      = 5   


# 1.HEURISTIC FUNCTIONS
def dist(a, b):
    return np.linalg.norm(np.array([a.x - b.x, a.y - b.y, a.z - b.z]))

def angle(a, b, c):
    """Angle at joint B, with A and C as neighbours. Returns degrees."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def is_extended(lm, tip, pip):
    """True if tip is above PIP on screen (y-axis method)."""
    return lm[tip].y < lm[pip].y

def is_extended_inversed(lm, tip, pip):
    return lm[tip].y > lm[pip].y

def hand_scale(lm):
    return dist(lm[0], lm[9])   # wrist -> middle MCP

def calculate_pixel_distance(p1, p2):
    """Euclidean distance between two (x, y) pixel coordinate tuples."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Gesture definitions
def gesture_backward(lm):
    """Index finger points up — only index extended."""
    return (
        is_extended(lm, 8, 6)   and      # index up
        not is_extended(lm, 12, 10) and  # middle curled
        not is_extended(lm, 16, 14) and  # ring curled
        not is_extended(lm, 20, 18)      # pinky curled
    )

def gesture_stop(lm):
    """Open palm — all four fingers extended."""
    fingers_straight = (
        angle(lm[5], lm[6], lm[7]) > 150 and
        angle(lm[9], lm[10], lm[11]) > 150 and
        angle(lm[13], lm[14], lm[15]) > 150 and
        angle(lm[17], lm[18], lm[19]) > 150
    )
    #all fingers pointing up
    fingers_up = (
        lm[8].y < lm[5].y and
        lm[12].y < lm[9].y and
        lm[16].y < lm[13].y and
        lm[20].y < lm[17].y
    )

    rest_is_open = (
        is_extended(lm, 8, 6)  and
        is_extended(lm, 12, 10) and
        is_extended(lm, 16, 14) and
        is_extended(lm, 20, 18)
    )
    return fingers_straight and rest_is_open and fingers_up

def gesture_fist(lm):
    """curled into a fist"""
    return(
        not is_extended(lm, 8, 5) and 
        not is_extended(lm, 12, 10) and
        not is_extended(lm, 16, 13) and
        not is_extended(lm, 20, 17) 
    )    

def gesture_ok(lm):
    return(is_extended(lm, 20, 18) and is_extended(lm, 16, 15) and
           is_extended(lm, 12, 11) and angle(lm[5], lm[6], lm[7]) < 120 and
           angle(lm[17], lm[18], lm[19]) > 150 and angle(lm[13], lm[14], lm[15]) > 150)


def gesture_forward(lm):
    """Index finger points down - tip is below the pip joint"""
    #the index finger must be straight
    index_straight = angle(lm[5], lm[6], lm[7]) > 150
    tip_below_mcp = lm[8].y > lm[5].y
    rest_is_curled = (
        not is_extended_inversed(lm, 12, 10) and
        not is_extended_inversed(lm, 16, 14) and
        not is_extended_inversed(lm, 20, 18)      
    )
    return index_straight and tip_below_mcp and rest_is_curled

def gesture_move_left(lm):
    return(
        is_extended(lm, 8, 6)   and  
        is_extended(lm, 12, 10) and  
        not is_extended(lm, 16, 14) and  
        not is_extended(lm, 20, 18)     
    )
    
def gesture_move_right(lm):
    return(
        is_extended(lm, 8, 6)   and     
        is_extended(lm, 12, 10) and  
        is_extended(lm, 16, 14) and  
        not is_extended(lm, 20, 18)      
    )

#command data logger
previous_command = None
def log_command(command_name, fps, inference_time, filepath="robot_commands.csv"):
    """
    Appends a timestamp, command name, FPS, and inference time to a CSV file.
    Creates the file if it doesn't exist.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, command_name, f"{fps:.1f}", f"{inference_time:.1f}"])
        
    print(f"Logged: {command_name} at {timestamp} | FPS: {fps:.1f} | Inference: {inference_time:.1f}ms")

# Create folder to save captures
CAPTURE_DIR = "captures"
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

#DISPATCHER
#gesture for two hands
GESTURE_TWO = [
    ("forward",  (gesture_forward, gesture_ok)),
    ("backward", (gesture_backward, gesture_ok)),
    ("left",     (gesture_move_left, gesture_ok)),
    ("right",    (gesture_move_right, gesture_ok)),
    ("standby",  (gesture_fist, gesture_ok))
]
#gesture for one hand
GESTURE_ONE = [
    ("stop",  gesture_stop)
]

def classify_one_hand(lm) -> str | None:
    for name, fn in GESTURE_ONE:
            if fn(lm):
                return name
    return None

def classify_two_hands(lm1, lm2) -> str | None:
    for name, (fn_action, fn_confirm) in GESTURE_TWO:
        scenario_a = fn_action(lm1) and fn_confirm(lm2)
        scenario_b = fn_action(lm2) and fn_confirm(lm1)
        if scenario_a or scenario_b:
            return name
    return None

class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        
        # 1. ROS 2 PUBLISHER
        # This creates a topic that other nodes can listen to
        self.command_publisher = self.create_publisher(String, '/gesture/command', 10)
        
        # 2. MEDIAPIPE SETUP
        # Derive model path relative to this script so it works regardless of
        # where the workspace is mounted (e.g. /ros2_ws inside Docker).
        _script_dir = os.path.dirname(os.path.abspath(__file__))

        # --- HAND MODEL ---
        _default_hand_model = os.path.join(_script_dir, 'model', 'hand_landmarker.task')
        hand_model_path = self.declare_parameter('model_path', _default_hand_model).value

        if not os.path.isfile(hand_model_path):
            self.get_logger().fatal(
                f'Hand-landmarker model not found at: {hand_model_path}\n'
                'Set the "model_path" parameter or verify the file exists.')
            raise FileNotFoundError(f'Model not found: {hand_model_path}')

        self.get_logger().info(f'Loading hand model from: {hand_model_path}')
        hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=6,  # High limit to catch all background hands for filtering
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # --- POSE MODEL ---
        _default_pose_model = os.path.join(_script_dir, 'model', 'pose_landmarker_lite.task')
        pose_model_path = self.declare_parameter('pose_model_path', _default_pose_model).value

        if not os.path.isfile(pose_model_path):
            self.get_logger().fatal(
                f'Pose-landmarker model not found at: {pose_model_path}\n'
                'Set the "pose_model_path" parameter or verify the file exists.')
            raise FileNotFoundError(f'Model not found: {pose_model_path}')

        self.get_logger().info(f'Loading pose model from: {pose_model_path}')
        pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        # Distance threshold (pixels) between Hand Wrist and Body Wrist
        self.MATCH_THRESHOLD = 100
        
        # 3. CAMERA SETUP — auto-detect a working camera
        camera_index = self.declare_parameter(
            'camera_index', DEFAULT_CAMERA_INDEX
        ).value
        self.cap = self._open_camera(camera_index)
        
        # 4. STATE TRACKER
        self.previous_command = None
        self.CAPTURE_DIR = "captures"
        if not os.path.exists(self.CAPTURE_DIR):
            os.makedirs(self.CAPTURE_DIR)

        # 5. OUTPUT VIDEO (fallback when cv2.imshow is unavailable, e.g. Docker)
        self.video_writer = None
        self.gui_available = self._check_gui_available()
        if not self.gui_available:
            self.get_logger().warn(
                'No display detected — will save annotated output to '
                'gesture_output.avi instead of showing a GUI window.')

        # 6. FPS & INFERENCE TRACKING
        self.prev_frame_time = 0
        self.fps = 0.0
        self.inference_time_ms = 0.0

        # 7. POSE STATE — persisted between frames for asymmetric frame skipping
        self.frame_counter = 0
        self.saved_pose_landmarks = None
        self.user_left_wrist = None
        self.user_right_wrist = None
            
        # 8. THE TIMER (Replaces the while loop)
        # 0.05 seconds = 20 Frames Per Second
        self.timer = self.create_timer(0.05, self.timer_callback)

        # 9. WAKE STATE
        self.standalone = self.declare_parameter('standalone', False).value
        if self.standalone:
            self.is_awake = True
            self.get_logger().info('Running in STANDALONE mode — always awake')
        else:
            self.is_awake = False
            self.create_subscription(Bool, '/is_awake', self._on_awake, 10)

        # 10. CAMERA SLEEP TOGGLE
        # When True, camera processing pauses during sleep to save CPU.
        # When False, camera & detection keep running but commands are not published.
        self.camera_sleep_when_idle = self.declare_parameter(
            'camera_sleep_when_idle', False
        ).value

    # ── Camera helpers ───────────────────────────────────────────────────
    def _open_camera(self, preferred_index: int) -> cv2.VideoCapture:
        """Try preferred index first, then scan 0..MAX_CAMERA_SCAN-1."""
        indices_to_try = [preferred_index] + [
            i for i in range(MAX_CAMERA_SCAN) if i != preferred_index
        ]
        for idx in indices_to_try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.get_logger().info(
                        f'Camera opened on /dev/video{idx} '
                        f'({frame.shape[1]}x{frame.shape[0]}). Node is running.')
                    return cap
            cap.release()

        self.get_logger().fatal(
            f'Could not open ANY camera (tried indices 0–{MAX_CAMERA_SCAN - 1}).\n'
            'If running inside Docker, make sure you pass the camera device, e.g.:\n'
            '  docker run --device=/dev/video0:/dev/video0 ...')
        raise RuntimeError('No working camera found')

    @staticmethod
    def _check_gui_available() -> bool:
        """Return True if a GUI window can be created (X11 / Wayland)."""
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            return False
        try:
            # Attempt to create a tiny test window
            cv2.namedWindow('__test__', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('__test__')
            cv2.waitKey(1)
            return True
        except cv2.error:
            return False

    # ── Awake state callback ────────────────────────────────────────────
    def _on_awake(self, msg: Bool):
        prev = self.is_awake
        self.is_awake = msg.data
        if self.is_awake and not prev:
            self.get_logger().info('Gesture node ACTIVE — accepting commands')
        elif not self.is_awake and prev:
            self.get_logger().info('Gesture node SLEEPING — ignoring gestures')

    def timer_callback(self):
        """This function runs automatically 20 times a second."""
        # If camera_sleep_when_idle is enabled, skip everything while sleeping
        if self.camera_sleep_when_idle and not self.is_awake:
            return

        success, frame = self.cap.read()
        if not success:
            self.get_logger().warning("Ignoring empty camera frame.")
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ── FPS Calculation ──────────────────────────────────────────────
        new_frame_time = time.time()
        self.fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = new_frame_time

        start_inference = time.time()

        # ── 1. ALWAYS Run Hand Model (Fast) ──────────────────────────────
        hand_results = self.hand_landmarker.detect(mp_image)

        # ── 2. ASYMMETRIC FRAME SKIPPING: Run Pose Model every 5 frames ─
        self.frame_counter += 1
        if self.frame_counter % 5 == 0:
            pose_results = self.pose_landmarker.detect(mp_image)

            # Update saved pose data if a body is detected
            if pose_results.pose_landmarks:
                self.saved_pose_landmarks = pose_results.pose_landmarks[0]
                lm15 = self.saved_pose_landmarks[15]  # Left Wrist
                lm16 = self.saved_pose_landmarks[16]  # Right Wrist

                self.user_left_wrist = (int(lm15.x * w), int(lm15.y * h)) if lm15.visibility > 0.5 else None
                self.user_right_wrist = (int(lm16.x * w), int(lm16.y * h)) if lm16.visibility > 0.5 else None
            else:
                self.saved_pose_landmarks = None
                self.user_left_wrist = None
                self.user_right_wrist = None

        self.inference_time_ms = (time.time() - start_inference) * 1000

        current_command = None
        user_hand_landmarks = []

        # ── 3. DRAW THE POSE SKELETON ────────────────────────────────────
        if self.saved_pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=pt.x, y=pt.y, z=pt.z)
                for pt in self.saved_pose_landmarks
            ])
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks_proto,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # ── 4. FILTERING: ISOLATE THE USER'S HANDS ──────────────────────
        if hand_results.hand_landmarks:
            for hand_lms in hand_results.hand_landmarks:
                hand_wrist_px = (int(hand_lms[0].x * w), int(hand_lms[0].y * h))
                is_users_hand = False

                if self.user_left_wrist and calculate_pixel_distance(hand_wrist_px, self.user_left_wrist) < self.MATCH_THRESHOLD:
                    is_users_hand = True
                elif self.user_right_wrist and calculate_pixel_distance(hand_wrist_px, self.user_right_wrist) < self.MATCH_THRESHOLD:
                    is_users_hand = True

                if is_users_hand:
                    user_hand_landmarks.append(hand_lms)

                    # Draw the valid hands
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=pt.x, y=pt.y, z=pt.z)
                        for pt in hand_lms
                    ])
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks_proto, self.mp_hands.HAND_CONNECTIONS)
                else:
                    cv2.putText(frame, "IGNORED", hand_wrist_px,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # ── 5. APPLYING HEURISTICS ───────────────────────────────────────
        if user_hand_landmarks:
            stop_detected = False

            for lm in user_hand_landmarks:
                if classify_one_hand(lm) == "stop":
                    stop_detected = True
                    break

            # GLOBAL SAFETY OVERRIDE
            if stop_detected:
                current_command = "stop"
                cv2.putText(frame, "Gesture: EMERGENCY STOP", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # STANDARD COMMAND LOGIC
            else:
                num_valid_hands = len(user_hand_landmarks)

                if num_valid_hands >= 2:
                    lm1 = user_hand_landmarks[0]
                    lm2 = user_hand_landmarks[1]

                    current_command = classify_two_hands(lm1, lm2)
                    is_ok_present = gesture_ok(lm1) or gesture_ok(lm2)

                    if is_ok_present:
                        if current_command:
                            cv2.putText(frame, f"Gesture: {current_command.upper()}", (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        else:
                            cv2.putText(frame, "Waiting for valid combo...", (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    else:
                        cv2.putText(frame, "Requires OK confirmation", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

                elif num_valid_hands == 1:
                    cv2.putText(frame, "Bring up second hand", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

        # ── 6. LOGGING, CAPTURING, AND PUBLISHING ────────────────────────
        if current_command is not None:
            # Only publish if robot is awake
            if not self.is_awake:
                self.get_logger().debug(
                    f'Ignored gesture "{current_command}" — robot is sleeping')
                self.previous_command = current_command
                return

            # PUBLISH TO ROS 2 — send EVERY frame so the arbiter watchdog
            # knows the gesture is still active (continuous movement).
            msg = String()
            msg.data = json.dumps({
                "source": "gesture",
                "command": current_command,
                "confidence": 1.0,
            })
            self.command_publisher.publish(msg)

            # Log, capture, and info-log only on CHANGE to avoid spam
            if current_command != self.previous_command:
                self.get_logger().info(f"Published command: {current_command}")
                log_command(current_command, self.fps, self.inference_time_ms)

                safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.CAPTURE_DIR}/{current_command}_{safe_time}.jpg"
                cv2.imwrite(filename, frame)

            self.previous_command = current_command

        elif current_command is None:
            self.previous_command = None

        # ── 7. DISPLAY METRICS ───────────────────────────────────────────
        cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Inference: {int(self.inference_time_ms)} ms", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # ── 8. OUTPUT — GUI window or video file ─────────────────────────
        if self.gui_available:
            cv2.imshow('Hand Gesture Recognition', frame)
            cv2.waitKey(1)
        else:
            # Lazy-init the video writer on the first frame
            if self.video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    'gesture_output.avi', fourcc, 20.0, (w, h))
                self.get_logger().info(
                    f'Video writer initialised: gesture_output.avi ({w}x{h})')
            self.video_writer.write(frame)

    def destroy_node(self):
        """Cleanup when the node shuts down."""
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info('Output video saved: gesture_output.avi')
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    
    try:
        # spin() keeps the node running and listening for timer events
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
