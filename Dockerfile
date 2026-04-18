# Start from the official ROS 2 Humble Desktop image
FROM osrf/ros:humble-desktop

# Install pip, required graphics for MediaPipe, and audio headers for PyAudio
RUN apt-get update && apt-get install -y \
    python3-pip \
    nano \
    libgl1-mesa-glx \
    libglib2.0-0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

#dependencies for voice_node and gesture_node
RUN pip3 install --no-cache-dir numpy==1.26.4 protobuf==4.25.3 mediapipe==0.10.14 pyaudio vosk pyserial

# Automatically source ROS 2 when the container starts
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc


# Set the default working directory
WORKDIR /ros2_ws


