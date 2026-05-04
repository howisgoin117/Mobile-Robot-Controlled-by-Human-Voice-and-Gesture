#!/bin/bash

echo "Granting GUI (X11) display access for Docker..."
xhost +local:root

echo "Starting AMR..."
# Sử dụng --gpus all để cấp quyền truy cập phần cứng AI
# Sử dụng -w /ros2_ws để thiết lập thư mục làm việc mặc định trong container
# Sử dụng cờ bash -c để chạy gộp nhiều lệnh bên trong container

docker run -it --rm \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/video1:/dev/video1 \
  --device=/dev/snd:/dev/snd \
  --device=/dev/ttyACM0:/dev/ttyACM0 \
  --device=/dev/ttyACM1:/dev/ttyACM1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/manh/ros2_workspace:/ros2_ws \
  -e DISPLAY=$DISPLAY \
  -w /ros2_ws \
  my_robot_env \
  bash -c "source install/setup.bash && ros2 launch robot_controller bringup.launch.py"
