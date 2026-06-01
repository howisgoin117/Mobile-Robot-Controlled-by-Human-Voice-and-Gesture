import rclpy
import json
import time
from rclpy.node import Node
from std_msgs.msg import String, Bool

WAKE_TIMEOUT    = 50.0   # seconds of active listening after wake word
COMMAND_TIMEOUT = 1.5    # seconds before auto-stop within awake window
GESTURE_ACTIVE_WINDOW = 2.0  # if gesture received within this window, robot stays awake


class CommandArbiterNode(Node):
    """Subsumption-based command arbiter.

    Three priority levels control which input source drives the robot:

      Level 0 (highest) — EMERGENCY STOP
        "stop" from ANY source (voice or gesture) is always accepted
        immediately. It also clears the voice-active lock so gesture
        can take over afterwards.

      Level 1 — VOICE COMMANDS
        When voice sends a movement command, a ``voice_active`` flag
        is raised.  While the flag is set, all gesture commands except
        "stop" are suppressed.  The flag is cleared when:
          • voice sends "stop"  (explicit end of voice session), or
          • gesture sends "stop" (Level 0 override), or
          • the watchdog fires   (COMMAND_TIMEOUT with no new input).

      Level 2 (lowest) — GESTURE COMMANDS
        Gesture commands are only accepted when ``voice_active`` is
        False (i.e. no active voice session).
    """

    def __init__(self):
        super().__init__('command_arbiter_node')

        # ── State ──────────────────────────────────────────────────────────
        self.is_awake       = False
        self.awake_deadline = 0.0       # epoch time when awake window expires
        self.last_command   = "stop"
        self.last_received  = time.time()
        self.last_gesture_time = 0.0    # tracks when last gesture command arrived
        self.dispatch_seq   = 0         # sequence number for dispatched commands

        # ── Subsumption flags ──────────────────────────────────────────────
        self.voice_active = False       # True while voice owns the robot
        self.gesture_stop_active = False # True while gesture 'stop' is active
        self.last_gesture_stop_time = 0.0
        self.gesture_blocked_until = 0.0 # Time until gestures are ignored after voice 'stop'

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
        self.voice_active   = False
        self.awake_pub.publish(Bool(data=True))
        self.get_logger().info(
            f'Robot AWAKENED — listening for {WAKE_TIMEOUT}s')

    def _go_to_sleep(self, reason: str):
        """Transition to SLEEPING state and stop the robot."""
        self.is_awake     = False
        self.voice_active = False
        self.awake_pub.publish(Bool(data=False))
        self.cmd_pub.publish(String(data="stop"))
        self.last_command = "stop"
        self.get_logger().info(f'Robot SLEEPING — {reason}')

    # ══════════════════════════════════════════════════════════════════════
    #  VOICE CALLBACK  (Level 0 + Level 1)
    # ══════════════════════════════════════════════════════════════════════
    def _on_voice(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        command = data.get('command', '')

        # ── Meta commands (always processed) ──────────────────────────────
        if command == 'wake_word':
            self._wake_up()
            return

        if command == 'sleep':
            if self.is_awake:
                self._go_to_sleep('sleep command received via voice')
            return

        if not self.is_awake:
            return

        # ── Check if gesture 'stop' is actively blocking ──────────────────
        # Gesture 'stop' has ultimate priority when held
        if self.gesture_stop_active and (time.time() - self.last_gesture_stop_time < 0.5):
            self.get_logger().debug(
                f'Voice "{command}" suppressed — gesture "stop" is active')
            return

        # ── Level 0: voice "stop" — execute + end voice session ──────────
        if command == 'stop':
            self.voice_active = False
            
            # Only start the 10s block if we aren't already blocked, to prevent looping
            if time.time() >= self.gesture_blocked_until:
                self.gesture_blocked_until = time.time() + 10.0
                self.get_logger().info(
                    '[SUBSUMPTION] Voice session ENDED (voice said "stop"). '
                    'Gestures BLOCKED for 10 seconds.')
                
            self._dispatch(command, source='voice')
            return

        # ── Level 1: voice movement — raise flag, suppress gesture ───────
        if not self.voice_active:
            self.get_logger().info(
                f'[SUBSUMPTION] Voice session STARTED — '
                f'gesture suppressed until voice "stop"')
        self.voice_active = True
        self._dispatch(command, source='voice')

    # ══════════════════════════════════════════════════════════════════════
    #  GESTURE CALLBACK  (Level 0 + Level 2)
    # ══════════════════════════════════════════════════════════════════════
    def _on_gesture(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        command    = data.get('command', '')
        confidence = data.get('confidence', 0)

        if not self.is_awake:
            return

        # ── 10-Second Block after Voice "stop" ───────────────────────────
        if time.time() < self.gesture_blocked_until:
            remaining = self.gesture_blocked_until - time.time()
            self.get_logger().debug(
                f'Gesture "{command}" ignored — blocked for {remaining:.1f}s after voice "stop"')
            return

        if confidence < 0.75:
            return

        # ── Level 0: gesture "stop" — always passes, clears voice lock ───
        if command == 'stop':
            self.voice_active = False
            self.gesture_stop_active = True
            self.last_gesture_stop_time = time.time()
            self.last_gesture_time = time.time()
            self._dispatch(command, source='gesture')
            self.get_logger().info(
                '[SUBSUMPTION] Voice session OVERRIDDEN by gesture "stop"')
            return

        # ── Level 2: gesture movement — only when voice is NOT active ────
        if self.voice_active:
            self.get_logger().debug(
                f'Gesture "{command}" suppressed — voice has priority')
            return

        self.last_gesture_time = time.time()
        self._dispatch(command, source='gesture')

    # ── Command dispatch ───────────────────────────────────────────────────
    def _dispatch(self, command: str, source: str):
        command_changed = (command != self.last_command)

        self.last_command  = command
        self.last_received = time.time()

        # Reset the wake timeout on every accepted command
        self.awake_deadline = time.time() + WAKE_TIMEOUT

        self.cmd_pub.publish(String(data=command))

        if command_changed:
            self.dispatch_seq += 1
            remaining = self.awake_deadline - time.time()
            self.get_logger().info(
                f'[DISPATCH #{self.dispatch_seq}] [{source}] → "{command}"  '
                f'(wake window: {remaining:.1f}s remaining)')

    # ── ACK from avr_serial_node ───────────────────────────────────────────
    def _on_cmd_ack(self, msg: String):
        self.get_logger().debug(
            f'[ACK←AVR] avr_serial_node confirmed: {msg.data}')

    # ── Watchdog (runs every 100 ms) ───────────────────────────────────────
    def _watchdog(self):
        if not self.is_awake:
            return

        now = time.time()
        gesture_active = (now - self.last_gesture_time) < GESTURE_ACTIVE_WINDOW

        # 0. Check if gesture block just expired
        if self.gesture_blocked_until > 0 and now >= self.gesture_blocked_until:
            self.gesture_blocked_until = 0.0  # Reset so it logs only once
            self.get_logger().info('[SUBSUMPTION] 10-second block expired. Gesture command is READY.')

        # 1. Wake timeout — but keep alive if gesture commands are flowing
        if now > self.awake_deadline:
            if gesture_active:
                self.awake_deadline = now + WAKE_TIMEOUT
                self.get_logger().debug(
                    '[WATCHDOG] Gesture still active — extending wake window')
            else:
                self._go_to_sleep('wake timeout expired (no active gesture)')
                return

        # 2. Command timeout — no new command within COMMAND_TIMEOUT
        #    Also clears voice_active so gesture can take over after silence.
        if (now - self.last_received > COMMAND_TIMEOUT
                and self.last_command != "stop"
                and not gesture_active):
            if self.voice_active:
                self.get_logger().info(
                    '[SUBSUMPTION] Voice session EXPIRED (command timeout)')
            self.voice_active = False
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