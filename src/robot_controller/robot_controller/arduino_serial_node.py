import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time

class ArduinoSerialNode(Node):
    def __init__(self):
        super().__init__('arduino_serial_node')
        
        # Configure Serial Port
        self.declare_parameter('port', '/dev/ttyACM1')
        self.declare_parameter('baudrate', 115200)
        
        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        
        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2) # Wait for Arduino reset
            self.get_logger().info(f"Successfully connected to Arduino on {port}")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect to Arduino: {e}")
            raise SystemExit

        # Publisher: Data coming FROM Arduino
        self.publisher_ = self.create_publisher(String, 'arduino_rx', 10)
        
        # Subscriber: Data going TO Arduino
        self.subscription = self.create_subscription(
            String,
            'arduino_tx',
            self.send_to_arduino_callback,
            10)

        # Timer to check for incoming serial data
        self.timer = self.create_timer(0.1, self.read_from_arduino_callback)

    def send_to_arduino_callback(self, msg):
        """Triggered when a message is published to /arduino_tx"""
        serial_msg = msg.data + '\n'
        self.serial_conn.write(serial_msg.encode('utf-8'))
        self.get_logger().debug(f"Sent to Arduino: {msg.data}")

    def read_from_arduino_callback(self):
        """Constantly checks the serial buffer for incoming data"""
        if self.serial_conn.in_waiting > 0:
            try:
                # Read line and decode
                incoming_data = self.serial_conn.readline().decode('utf-8').rstrip()
                if incoming_data:
                    # Publish to ROS2 topic
                    ros_msg = String()
                    ros_msg.data = incoming_data
                    self.publisher_.publish(ros_msg)
                    self.get_logger().info(f"Received: {incoming_data}")
            except Exception as e:
                self.get_logger().warning(f"Serial read error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArduinoSerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.serial_conn.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()