from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    package_data={
        package_name: ['model/*.task'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='manh',
    maintainer_email='pham.tienmanh.school@gmail.com',
    description='Robot controlled by Human Voice and Gesture',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gesture_node    = robot_controller.gesture_node:main',
            'voice_node      = robot_controller.voice_node:main',
            'command_arbiter_node   = robot_controller.command_arbiter_node:main',
            'avr_serial_node  = robot_controller.avr_serial_node:main',
            'arduino_serial_node = robot_controller.arduino_serial_node:main'
        ],
    },
    

)
