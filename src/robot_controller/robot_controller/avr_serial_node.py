# avr_serial_node.py
#
# Compatible with Arduino diffbot_agv_preset v2.1
# Arduino protocol:
#   e         → toggle enable/ESTOP
#   d         → emergency stop (motorOn=false)
#   stop      → soft brake (tgtL=tgtR=0, motorOn stays true)
#   v <f> [s] → set velocity (requires MODE_SER + motorOn)
#
import rclpy
import serial
import serial.tools.list_ports
import threading
import time
import re
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import SetParametersResult


# ── Tuning (must match Arduino FIXED_VEL / STEER_RATIO) ───────────────────────
FIXED_VEL   = 0.5           # matches Arduino FIXED_VEL
STEER_RATIO = 0.30          # fraction of FIXED_VEL used for steering
CONNECTION_CHECK_INTERVAL = 2.0


class AVRSerialNode(Node):
    def __init__(self):
        super().__init__('avr_serial_node')

        self.port     = self.declare_parameter('port',     '/dev/ttyACM1').value
        self.baudrate = self.declare_parameter('baudrate', 115200).value

        self.serial = None
        self.serial_lock = threading.Lock()
        self.last_cmd = "stop"

        # ── Arduino state (tracked from serial responses) ──────────────────
        self.motor_on    = False      # tracks Arduino motorOn
        self.serial_mode = False      # tracks if Arduino is in MODE_SER
        self.enable_sent = False      # whether we've sent 'e' after connect

        # ── Debug counters ─────────────────────────────────────────────────
        self.cmds_received  = 0
        self.cmds_sent_ok   = 0
        self.cmds_sent_fail = 0

        # Publisher: serial connection status for other nodes / debugging
        self.status_pub = self.create_publisher(String, '/serial/status', 10)

        # Publisher: command acknowledgement back to arbiter
        self.ack_pub = self.create_publisher(String, '/serial/cmd_ack', 10)

        # Connect AFTER publishers exist (so _publish_status works)
        self._connect(self.port, self.baudrate)

        # Subscribe to /robot/command from command_arbiter_node
        self.create_subscription(String, '/robot/command', self._on_command, 10)

        # Subscriber: allow changing port at runtime via topic
        self.create_subscription(String, '/serial/set_port', self._on_set_port, 10)

        # Heartbeat every 500 ms — resend last MOVEMENT command so Arduino
        # keeps the target velocity; do NOT resend stop/d.
        self.create_timer(0.5, self._heartbeat)

        # Periodic connection health check
        self.create_timer(CONNECTION_CHECK_INTERVAL, self._check_connection)

        # Allow parameter changes at runtime (port, baudrate)
        self.add_on_set_parameters_callback(self._on_param_change)

        # Start background serial reader thread
        reader_thread = threading.Thread(target=self._serial_reader, daemon=True)
        reader_thread.start()

        self._log_available_ports()
        self.get_logger().info(
            'AVR serial node ready  '
            '(protocol: diffbot_agv_preset v2.1)')

    # ── List available serial ports ────────────────────────────────────────
    def _log_available_ports(self):
        ports = serial.tools.list_ports.comports()
        if ports:
            self.get_logger().info('── Available serial ports ──')
            for p in ports:
                self.get_logger().info(f'  {p.device}  [{p.description}]')
        else:
            self.get_logger().warn('No serial ports detected on this machine')

    # ── Serial connection ──────────────────────────────────────────────────
    def _connect(self, port, baudrate):
        with self.serial_lock:
            if self.serial and self.serial.is_open:
                try:
                    self.serial.close()
                    self.get_logger().info('Previous serial connection closed')
                except Exception:
                    pass

            try:
                self.serial = serial.Serial(port, baudrate, timeout=1)
                time.sleep(2)   # wait for AVR to come out of reset
                self.get_logger().info(
                    f'✔ Serial CONNECTED: {port} @ {baudrate}')
                self._publish_status('connected', port)
                self.enable_sent = False  # will try to enable on next cycle
            except serial.SerialException as e:
                self.serial = None
                self.get_logger().error(
                    f'✘ Serial FAILED to open {port}: {e}')
                self._publish_status('disconnected', port, str(e))

    # ── Publish status to /serial/status ───────────────────────────────────
    def _publish_status(self, state: str, port: str, error: str = ''):
        msg = String()
        msg.data = (
            f'{{"state":"{state}","port":"{port}",'
            f'"motor_on":{str(self.motor_on).lower()},'
            f'"serial_mode":{str(self.serial_mode).lower()},'
            f'"cmds_received":{self.cmds_received},'
            f'"cmds_sent_ok":{self.cmds_sent_ok},'
            f'"cmds_sent_fail":{self.cmds_sent_fail}'
            + (f',"error":"{error}"' if error else '')
            + '}'
        )
        self.status_pub.publish(msg)

    # ── Periodic connection health check ───────────────────────────────────
    def _check_connection(self):
        with self.serial_lock:
            connected = self.serial is not None and self.serial.is_open

        if connected:
            self.get_logger().debug(
                f'[HEALTH] Serial OK on {self.port}  '
                f'(motor={self.motor_on}  ser_mode={self.serial_mode}  '
                f'rx={self.cmds_received}  tx_ok={self.cmds_sent_ok})')
            self._publish_status('connected', self.port)

            # Auto-enable motors once after connection
            if not self.enable_sent:
                self.enable_sent = True
                self.get_logger().info(
                    '[INIT] Sending "e" to enable motors…')
                self._write_serial('e')
        else:
            self.get_logger().warn(
                f'[HEALTH] Serial DISCONNECTED on {self.port} — '
                f'attempting reconnect…')
            self._publish_status('disconnected', self.port)
            self._connect(self.port, self.baudrate)

    # ── Runtime port change via /serial/set_port topic ─────────────────────
    def _on_set_port(self, msg: String):
        new_port = msg.data.strip()
        if not new_port:
            self.get_logger().warn('[SET_PORT] Empty port string, ignoring')
            return
        self.get_logger().info(
            f'[SET_PORT] Changing port: {self.port} → {new_port}')
        self.port = new_port
        self._connect(self.port, self.baudrate)

    # ── Runtime parameter change (ros2 param set) ──────────────────────────
    def _on_param_change(self, params):
        for param in params:
            if param.name == 'port':
                self.get_logger().info(
                    f'[PARAM] Port changed: {self.port} → {param.value}')
                self.port = param.value
                self._connect(self.port, self.baudrate)
            elif param.name == 'baudrate':
                self.get_logger().info(
                    f'[PARAM] Baudrate changed: {self.baudrate} → {param.value}')
                self.baudrate = param.value
                self._connect(self.port, self.baudrate)
        return SetParametersResult(successful=True)

    # ── Background serial reader ───────────────────────────────────────────
    def _serial_reader(self):
        """Read lines from the AVR and track motor/mode state."""
        while True:
            with self.serial_lock:
                ser = self.serial
            if ser is None or not ser.is_open:
                time.sleep(0.5)
                continue
            try:
                line = ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue
                self._parse_arduino_response(line)
            except serial.SerialException as e:
                self.get_logger().warn(f'[ARD] Read error (port lost?): {e}')
                time.sleep(1)
            except Exception:
                pass

    def _parse_arduino_response(self, line: str):
        """Parse Arduino output to track state and log important events."""

        # ── Motor ON ───────────────────────────────────────────────────────
        if '[OK] Motor ON' in line:
            self.motor_on = True
            self.get_logger().info(f'[ARD] ✔ {line}')
            return

        # ── ESTOP (motor OFF) ──────────────────────────────────────────────
        if '[ESTOP]' in line:
            self.motor_on = False
            self.get_logger().warn(f'[ARD] ⚠ {line}')
            return

        # ── Enable FAIL ────────────────────────────────────────────────────
        if '[ERR] Enable FAIL' in line:
            self.motor_on = False
            self.get_logger().error(f'[ARD] ✘ {line}')
            return

        # ── Mode changes ──────────────────────────────────────────────────
        if '->SER' in line:
            self.serial_mode = True
            self.get_logger().info(f'[ARD] Mode → SERIAL')
            return
        if '->PS2' in line:
            self.serial_mode = False
            self.get_logger().info(f'[ARD] Mode → PS2')
            return

        # ── Velocity command accepted ─────────────────────────────────────
        if line.startswith('[OK] L='):
            self.get_logger().info(f'[ARD] ✔ {line}')
            return

        # ── Errors that indicate commands are rejected ────────────────────
        if '[ERR] Motor off' in line:
            self.motor_on = False
            self.get_logger().warn(
                '[ARD] ⚠ Motor is OFF — send "e" to enable or press R2')
            return
        if '[ERR] Mode PS2' in line:
            self.serial_mode = False
            self.get_logger().warn(
                '[ARD] ⚠ Arduino is in PS2 mode — press △ on PS2 to switch')
            return

        # ── ODrive errors ─────────────────────────────────────────────────
        if '[ODRV ERR]' in line:
            self.motor_on = False
            self.get_logger().error(f'[ARD] ✘ {line}')
            return

        # ── Everything else at debug level ────────────────────────────────
        self.get_logger().debug(f'[ARD] {line}')

    # ── ROS callback ───────────────────────────────────────────────────────
    def _on_command(self, msg: String):
        cmd = msg.data.strip()
        self.cmds_received += 1
        prev_cmd = self.last_cmd
        self.last_cmd = cmd

        cmd_changed = (cmd != prev_cmd)
        if cmd_changed:
            self.get_logger().info(
                f'[RX #{self.cmds_received}] Command changed: '
                f'"{prev_cmd}" → "{cmd}"')

        success = self._send_command(cmd, log=cmd_changed)

        # Publish ACK/NACK back to arbiter
        ack_msg = String()
        if success:
            ack_msg.data = (
                f'ACK #{self.cmds_received} cmd="{cmd}" '
                f'tx_ok={self.cmds_sent_ok}')
            if cmd_changed:
                self.get_logger().info(
                    f'[ACK #{self.cmds_received}] Confirmed "{cmd}" '
                    f'sent to Arduino')
        else:
            ack_msg.data = (
                f'NACK #{self.cmds_received} cmd="{cmd}" '
                f'tx_fail={self.cmds_sent_fail}')
            self.get_logger().warn(
                f'[NACK #{self.cmds_received}] Failed to send "{cmd}" '
                f'to Arduino')
        self.ack_pub.publish(ack_msg)

    # ── Heartbeat ──────────────────────────────────────────────────────────
    def _heartbeat(self):
        # Only resend movement commands; stop/standby don't need repeating
        # because Arduino keeps tgtL=tgtR=0 until a new velocity is set.
        if self.last_cmd in ("stop", "standby"):
            return
        self._send_command(self.last_cmd, log=False)

    # ── Translate command → Arduino serial protocol ────────────────────────
    def _send_command(self, cmd: str, log: bool = True) -> bool:
        """Translate high-level command to Arduino serial string.

        Arduino protocol (diffbot_agv_preset v2.1):
          v <fwd> [steer]  — set velocity (needs MODE_SER + motorOn)
          stop / brake     — soft brake (tgtL=tgtR=0, motorOn stays true)
          e                — toggle enable/ESTOP
          d                — emergency stop (motorOn=false)
        """
        v = FIXED_VEL
        s = v * STEER_RATIO

        serial_cmd = {
            # Movement commands → velocity protocol
            "forward":  f"v  {v:.2f}  0.00",
            "backward": f"v -{v:.2f}  0.00",
            "left":     f"v  {v:.2f} -{s:.2f}",
            "right":    f"v  {v:.2f}  {s:.2f}",
            # Stop commands → soft brake (keeps motors enabled!)
            "stop":     "stop",
            "standby":  "stop",
        }.get(cmd.lower())

        if serial_cmd is None:
            self.get_logger().warn(f'Unknown command: {cmd}')
            return False

        return self._write_serial(serial_cmd, label=cmd, log=log)

    # ── Low-level serial write ─────────────────────────────────────────────
    def _write_serial(self, data: str, label: str = '', log: bool = True) -> bool:
        """Write a string to the serial port. Returns True on success."""
        with self.serial_lock:
            ser = self.serial
            is_open = ser is not None and ser.is_open

        if is_open:
            try:
                ser.write((data + '\n').encode())
                self.cmds_sent_ok += 1
                if log:
                    tag = f' ({label})' if label else ''
                    self.get_logger().info(
                        f'[TX OK #{self.cmds_sent_ok}] → "{data}"{tag}')
                return True
            except serial.SerialException as e:
                self.cmds_sent_fail += 1
                self.get_logger().error(
                    f'[TX FAIL #{self.cmds_sent_fail}] Serial write failed: '
                    f'{e}')
                return False
        else:
            self.cmds_sent_fail += 1
            self.get_logger().error(
                f'[TX FAIL #{self.cmds_sent_fail}] Cannot send "{data}" — '
                f'serial port {self.port} is not connected')
            return False


def main():
    rclpy.init()
    node = AVRSerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()