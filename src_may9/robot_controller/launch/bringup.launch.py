from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_controller',
            executable='gesture_node',
            name='gesture_controller',
            output='screen' # This makes print/log statements show in the terminal
        ),
        Node(
            package='robot_controller',
            executable='avr_serial_node',
            name='arduino_bridge',
            output='screen'
        ),
	Node(
            package='robot_controller',
            executable='voice_node',
            name='voice_controller',
            output='screen'
        ),
        Node(
            package='robot_controller',
            executable='command_arbiter_node',
            name='command_arbiter',
            output='screen'
	)
    ])
