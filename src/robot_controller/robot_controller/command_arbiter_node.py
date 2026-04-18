import rclpy
import json
import time
from rclpy.node import Node
from std_msgs.msg import String, Bool

WAKE_TIMEOUT    = 10.0   # seconds of active listening after wake word
COMMAND_TIMEOUT = 1.5    # seconds before auto-stop within awake window
GESTURE_ACTIVE_WINDOW = 2.0  # if gesture received within this window, robot stays awake


class CommandArbiterNode(Node):
    def __init__(self):
        super().__init__('command_arbiter_node')

        # ── State ──────────────────────────────────────────────────────────
        self.is_awake       = False
        self.awake_deadline = 0.0       # epoch time when awake window expires
        self.last_command   = "stop"
        self.last_received  = time.time()
        self.last_gesture_time = 0.0    # tracks when last gesture command arrived
        self.dispatch_seq   = 0         # sequence number for dispatched commands

        # ── Subscriptions ──────────────────────────────────────────────────
        self.create_subscription(String, '/gesture/command', self._on_gesture, 10)
        self.create_subscription(String, '/voice/command',   self._on_voice,   10)
        self.create_subscription(String, '/serial/cmd_ack',  self._on_cmd_ack, 10)

        # ── Publishers ─────────────────────────────────────────────────────
        self.cmd_pub   = self.create_publisher(String, '/robot/command', 10)
        self.awake_pub = self.create_publisher(Bool,   '/is_awake',      10)

        # ── Watchdog timer (100 ms) ────────────────────────────────────────
        self.create_timer(0.1, self._watchdog)

        # Publish initial sleeping state
        self.awake_pub.publish(Bool(data=False))
        self.get_logger().info('Command arbiter node ready  [SLEEPING]')

    # ── Wake / Sleep helpers ───────────────────────────────────────────────
    def _wake_up(self):
        """Transition to AWAKE state and start the timeout window."""
        self.is_awake       = True
        self.awake_deadline = time.time() + WAKE_TIMEOUT
        self.awake_pub.publish(Bool(data=True))
        self.get_logger().info(
            f'Robot AWAKENED — listening for {WAKE_TIMEOUT}s')

    def _go_to_sleep(self, reason: str):
        """Transition to SLEEPING state and stop the robot."""
        self.is_awake = False
        self.awake_pub.publish(Bool(data=False))
        self.cmd_pub.publish(String(data="stop"))
        self.last_command = "stop"
        self.get_logger().info(f'Robot SLEEPING — {reason}')

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _on_voice(self, msg: String):
        self.get_logger().info(f'[DEBUG] Raw voice msg received: {msg.data}')
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'[DEBUG] Failed to parse voice JSON: {e}')
            return
        command = data.get('command', '')
        self.get_logger().info(
            f'[DEBUG] Voice parsed — command="{command}"  awake={self.is_awake}')

        if command == 'wake_word':
            self._wake_up()
            return                      # don't dispatch wake_word as a motor cmd

        # Voice always overrides gesture (higher priority)
        if self.is_awake:
            self._dispatch(command, source='voice')
        else:
            self.get_logger().info(
                f'Ignored voice "{command}" — robot is sleeping')

    def _on_gesture(self, msg: String):
        self.get_logger().info(f'[DEBUG] Raw gesture msg received: {msg.data}')
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'[DEBUG] Failed to parse gesture JSON: {e}')
            return
        command = data.get('command', '')
        confidence = data.get('confidence', 0)
        self.get_logger().info(
            f'[DEBUG] Gesture parsed — command="{command}"  '
            f'confidence={confidence}  awake={self.is_awake}')

        if not self.is_awake:
            self.get_logger().info(
                f'Ignored gesture "{command}" — robot is sleeping')
            return

        if confidence >= 0.75:
            # Track gesture activity to keep robot awake during continuous gestures
            self.last_gesture_time = time.time()
            self._dispatch(command, source='gesture')
        else:
            self.get_logger().info(
                f'Gesture "{command}" below threshold ({confidence} < 0.75)')

    # ── Command dispatch ───────────────────────────────────────────────────
    def _dispatch(self, command: str, source: str):
        self.last_command  = command
        self.last_received = time.time()
        self.dispatch_seq += 1

        # Reset the wake timeout on every accepted command
        self.awake_deadline = time.time() + WAKE_TIMEOUT

        self.cmd_pub.publish(String(data=command))
        remaining = self.awake_deadline - time.time()
        self.get_logger().info(
            f'[DISPATCH #{self.dispatch_seq}] [{source}] → "{command}"  '
            f'(wake window: {remaining:.1f}s remaining)')
        self.get_logger().info(
            f'[TX→AVR #{self.dispatch_seq}] Sent "{command}" to '
            f'/robot/command for avr_serial_node')

    # ── ACK from avr_serial_node ───────────────────────────────────────────
    def _on_cmd_ack(self, msg: String):
        self.get_logger().info(
            f'[ACK←AVR] avr_serial_node confirmed: {msg.data}')

    # ── Watchdog (runs every 100 ms) ───────────────────────────────────────
    def _watchdog(self):
        if not self.is_awake:
            return

        now = time.time()
        gesture_active = (now - self.last_gesture_time) < GESTURE_ACTIVE_WINDOW

        # 1. Wake timeout — but keep alive if gesture commands are flowing
        if now > self.awake_deadline:
            if gesture_active:
                # Gesture commands still coming in — extend the wake window
                self.awake_deadline = now + WAKE_TIMEOUT
                self.get_logger().debug(
                    '[WATCHDOG] Gesture still active — extending wake window')
            else:
                self._go_to_sleep('wake timeout expired (no active gesture)')
                return

        # 2. Command timeout — no new command within COMMAND_TIMEOUT → auto-stop
        #    But NOT if gestures are continuously streaming a movement command
        if (now - self.last_received > COMMAND_TIMEOUT
                and self.last_command != "stop"
                and not gesture_active):
            self._dispatch("stop", source="watchdog")


def main():
    rclpy.init()
    node = CommandArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()