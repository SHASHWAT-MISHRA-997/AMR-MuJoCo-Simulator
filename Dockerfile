FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WS=/opt/ros2_ws
ENV DISPLAY=:1
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MUJOCO_GL=glx
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-tk \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    openbox \
    wmctrl \
    xterm \
    libglfw3 \
    libglew2.2 \
    libgl1-mesa-dri \
    libglu1-mesa \
    libosmesa6 \
    mesa-utils \
    ros-jazzy-ament-index-python \
    ros-jazzy-geometry-msgs \
    ros-jazzy-launch \
    ros-jazzy-launch-ros \
    ros-jazzy-nav-msgs \
    ros-jazzy-nav2-msgs \
    ros-jazzy-rclpy \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-rviz2 \
    ros-jazzy-sensor-msgs \
    ros-jazzy-std-msgs \
    ros-jazzy-std-srvs \
    ros-jazzy-tf2-ros \
    ros-jazzy-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${ROS_WS}

COPY src/mujoco_amr_sim/package.xml src/mujoco_amr_sim/package.xml
COPY src/mujoco_amr_sim/setup.py src/mujoco_amr_sim/setup.py
COPY src/mujoco_amr_sim/setup.cfg src/mujoco_amr_sim/setup.cfg
COPY src/mujoco_amr_sim/resource src/mujoco_amr_sim/resource
COPY src/mujoco_amr_sim/mujoco_amr_sim src/mujoco_amr_sim/mujoco_amr_sim
COPY src/mujoco_amr_sim/launch src/mujoco_amr_sim/launch
COPY src/mujoco_amr_sim/config src/mujoco_amr_sim/config
COPY src/mujoco_amr_sim/docs src/mujoco_amr_sim/docs
COPY src/mujoco_amr_sim/models src/mujoco_amr_sim/models
COPY src/mujoco_amr_sim/rviz src/mujoco_amr_sim/rviz
COPY src/mujoco_amr_sim/urdf src/mujoco_amr_sim/urdf
COPY src/mujoco_amr_sim/README.md src/mujoco_amr_sim/README.md

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    . /opt/ros/jazzy/setup.sh && \
    "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir \
      numpy mujoco glfw PyOpenGL Pillow absl-py etils && \
    colcon build --symlink-install --packages-select mujoco_amr_sim

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/healthcheck.py /healthcheck.py
COPY docker/amr-favicon.svg /opt/amr-favicon.svg
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 CMD python3 /healthcheck.py

EXPOSE 6080

ENTRYPOINT ["/entrypoint.sh"]
