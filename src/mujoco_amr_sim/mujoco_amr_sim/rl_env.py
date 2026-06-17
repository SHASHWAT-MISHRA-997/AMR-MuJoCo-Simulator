import math
from pathlib import Path

import mujoco
import numpy as np

from .config_utils import load_dock_config
from .controllers import diff_drive_inverse, wrap_to_pi
from .mjcf_builder import (
    LIDAR_MAX_RANGE,
    LIDAR_MOUNTS,
    PICK_STATION_X,
    PICK_STATION_Y,
    PLACE_STATION_X,
    PLACE_STATION_Y,
    ROBOT2_START_X,
    ROBOT2_START_Y,
    SERVICE_DOCK_X,
    SERVICE_DOCK_Y,
    WHEEL_RADIUS,
    WHEEL_TRACK,
    build_model_xml,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class MujocoDockNavEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(self, dock_config_file: str | None = None, lidar_beams: int = 31) -> None:
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required for RL training. Install it in WSL first.")
        self.lidar_beams = lidar_beams
        self.model = mujoco.MjModel.from_xml_string(build_model_xml(lidar_beams=lidar_beams, lidar_fov_deg=220.0))
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)

        default_dock = Path(dock_config_file) if dock_config_file else None
        if default_dock is None:
            default_dock = Path("/mnt/c/Users/shash/OneDrive/Desktop/New folder/ros2_ws/src/mujoco_amr_sim/config/dock_station.json")
        self.dock = load_dock_config(str(default_dock))

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base_quat_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "base_quat")
        self.base_x_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_x_motor")
        self.base_y_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_y_motor")
        self.base_yaw_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_yaw_motor")

        primary_mount = LIDAR_MOUNTS[0].name
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{primary_mount}_lidar_{index:03d}")
            for index in range(lidar_beams)
        ]
        self.lidar_angles = np.linspace(-math.radians(110.0), math.radians(110.0), lidar_beams, dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lidar_beams + 7,), dtype=np.float32)

        self.goal = np.array([4.5, 3.2], dtype=np.float32)
        self.battery_pct = 1.0
        self.steps = 0
        self.max_steps = 1200
        self.last_goal_distance = None

    def _read_pose(self) -> tuple[float, float, float]:
        x = float(self.data.xpos[self.base_body_id][0])
        y = float(self.data.xpos[self.base_body_id][1])
        start = self.model.sensor_adr[self.base_quat_sensor]
        w = float(self.data.sensordata[start + 0])
        xq = float(self.data.sensordata[start + 1])
        yq = float(self.data.sensordata[start + 2])
        zq = float(self.data.sensordata[start + 3])
        yaw = math.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))
        return x, y, yaw

    def _read_lidar(self) -> np.ndarray:
        ranges = np.full((self.lidar_beams,), LIDAR_MAX_RANGE, dtype=np.float32)
        for index, sensor_id in enumerate(self.lidar_sensor_ids):
            value = float(self.data.sensordata[self.model.sensor_adr[sensor_id]])
            ranges[index] = LIDAR_MAX_RANGE if value < 0.0 else min(value, LIDAR_MAX_RANGE)
        return ranges

    def _get_obs(self) -> np.ndarray:
        px, py, yaw = self._read_pose()
        lidar = self._read_lidar() / LIDAR_MAX_RANGE
        dx = float(self.goal[0] - px)
        dy = float(self.goal[1] - py)
        dock_dx = self.dock.dock_pose[0] - px
        dock_dy = self.dock.dock_pose[1] - py
        return np.concatenate(
            [
                lidar,
                np.array(
                    [
                        dx,
                        dy,
                        wrap_to_pi(math.atan2(dy, dx) - yaw),
                        dock_dx,
                        dock_dy,
                        self.battery_pct,
                        yaw,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def _apply_action(self, action: np.ndarray) -> None:
        linear = float(np.clip(action[0], -1.0, 1.0)) * 0.7
        angular = float(np.clip(action[1], -1.0, 1.0)) * 1.5
        left_speed, right_speed = diff_drive_inverse(linear, angular, WHEEL_RADIUS, WHEEL_TRACK)
        del left_speed, right_speed

        px, py, yaw = self._read_pose()
        del px, py
        self.data.ctrl[self.base_x_actuator_id] = linear * math.cos(yaw)
        self.data.ctrl[self.base_y_actuator_id] = linear * math.sin(yaw)
        self.data.ctrl[self.base_yaw_actuator_id] = angular

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0] = np.random.uniform(-5.5, 3.0)
        self.data.qpos[1] = np.random.uniform(-4.0, 2.0)
        self.data.qpos[2] = np.random.uniform(-math.pi, math.pi)
        mujoco.mj_forward(self.model, self.data)
        self.goal = np.array(
            [np.random.uniform(-4.8, 4.8), np.random.uniform(-3.8, 3.8)],
            dtype=np.float32,
        )
        self.battery_pct = np.random.uniform(0.25, 1.0)
        self.steps = 0
        px, py, _ = self._read_pose()
        self.last_goal_distance = math.hypot(self.goal[0] - px, self.goal[1] - py)
        return self._get_obs(), {}

    def step(self, action):
        self._apply_action(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        px, py, _ = self._read_pose()
        lidar = self._read_lidar()
        goal_distance = math.hypot(self.goal[0] - px, self.goal[1] - py)
        dock_distance = math.hypot(self.dock.dock_pose[0] - px, self.dock.dock_pose[1] - py)

        motion_draw = 0.0002 + 0.0007 * (abs(float(action[0])) + abs(float(action[1])))
        self.battery_pct = max(0.0, self.battery_pct - motion_draw)

        reward = 1.8 * (self.last_goal_distance - goal_distance)
        reward -= 0.003
        if np.min(lidar) < 0.20:
            reward -= 1.5
        if self.battery_pct < 0.20:
            reward += max(0.0, 1.2 - dock_distance)
        if goal_distance < 0.30:
            reward += 15.0
            self.goal = np.array([self.dock.dock_pose[0], self.dock.dock_pose[1]], dtype=np.float32)
        if self.battery_pct < 0.20 and dock_distance < self.dock.charge_distance:
            reward += 25.0

        terminated = bool(np.min(lidar) < 0.11 or dock_distance < self.dock.contact_distance or goal_distance > 20.0)
        truncated = self.steps >= self.max_steps
        self.last_goal_distance = goal_distance
        return self._get_obs(), reward, terminated, truncated, {}


class MultiRobotWarehouseEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(self, dock_config_file: str | None = None, lidar_beams: int = 31) -> None:
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required for RL training. Install it in WSL first.")
        self.lidar_beams = lidar_beams
        self.model = mujoco.MjModel.from_xml_string(build_model_xml(lidar_beams=lidar_beams, lidar_fov_deg=220.0))
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)

        default_dock = Path(dock_config_file) if dock_config_file else None
        if default_dock is None:
            default_dock = Path("/mnt/c/Users/shash/OneDrive/Desktop/New folder/ros2_ws/src/mujoco_amr_sim/config/dock_station.json")
        self.dock = load_dock_config(str(default_dock))

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base_quat_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "base_quat")
        self.base_x_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_x_motor")
        self.base_y_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_y_motor")
        self.base_yaw_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_yaw_motor")
        self.service_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_bot")
        self.service_mocap_id = int(self.model.body_mocapid[self.service_body_id]) if self.service_body_id >= 0 else -1

        primary_mount = LIDAR_MOUNTS[0].name
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{primary_mount}_lidar_{index:03d}")
            for index in range(lidar_beams)
        ]
        self.lidar_angles = np.linspace(-math.radians(110.0), math.radians(110.0), lidar_beams, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lidar_beams + 16,), dtype=np.float32)

        self.service_pose = np.array([ROBOT2_START_X, ROBOT2_START_Y], dtype=np.float32)
        self.service_yaw = math.pi
        self.primary_goal = np.array([PICK_STATION_X, PICK_STATION_Y], dtype=np.float32)
        self.service_goal = np.array([SERVICE_DOCK_X, SERVICE_DOCK_Y], dtype=np.float32)
        self.steps = 0
        self.max_steps = 1800
        self.phase = 0

    def _read_pose(self) -> tuple[float, float, float]:
        x = float(self.data.xpos[self.base_body_id][0])
        y = float(self.data.xpos[self.base_body_id][1])
        start = self.model.sensor_adr[self.base_quat_sensor]
        w = float(self.data.sensordata[start + 0])
        xq = float(self.data.sensordata[start + 1])
        yq = float(self.data.sensordata[start + 2])
        zq = float(self.data.sensordata[start + 3])
        yaw = math.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))
        return x, y, yaw

    def _read_lidar(self) -> np.ndarray:
        ranges = np.full((self.lidar_beams,), LIDAR_MAX_RANGE, dtype=np.float32)
        for index, sensor_id in enumerate(self.lidar_sensor_ids):
            value = float(self.data.sensordata[self.model.sensor_adr[sensor_id]])
            ranges[index] = LIDAR_MAX_RANGE if value < 0.0 else min(value, LIDAR_MAX_RANGE)
        return ranges

    def _get_obs(self) -> np.ndarray:
        px, py, yaw = self._read_pose()
        lidar = self._read_lidar() / LIDAR_MAX_RANGE
        dx = float(self.primary_goal[0] - px)
        dy = float(self.primary_goal[1] - py)
        service_dx = float(self.service_goal[0] - self.service_pose[0])
        service_dy = float(self.service_goal[1] - self.service_pose[1])
        robot_gap = self.service_pose - np.array([px, py], dtype=np.float32)
        return np.concatenate(
            [
                lidar,
                np.array(
                    [
                        dx,
                        dy,
                        wrap_to_pi(math.atan2(dy, dx) - yaw),
                        self.service_pose[0],
                        self.service_pose[1],
                        self.service_yaw,
                        service_dx,
                        service_dy,
                        robot_gap[0],
                        robot_gap[1],
                        self.primary_goal[0],
                        self.primary_goal[1],
                        self.service_goal[0],
                        self.service_goal[1],
                        float(self.phase),
                        float(self.steps / max(self.max_steps, 1)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def _apply_action(self, action: np.ndarray) -> None:
        linear = float(np.clip(action[0], -1.0, 1.0)) * 0.7
        angular = float(np.clip(action[1], -1.0, 1.0)) * 1.5
        service_linear = float(np.clip(action[2], -1.0, 1.0)) * 0.55
        service_angular = float(np.clip(action[3], -1.0, 1.0)) * 1.3

        px, py, yaw = self._read_pose()
        del px, py
        self.data.ctrl[self.base_x_actuator_id] = linear * math.cos(yaw)
        self.data.ctrl[self.base_y_actuator_id] = linear * math.sin(yaw)
        self.data.ctrl[self.base_yaw_actuator_id] = angular

        self.service_yaw = wrap_to_pi(self.service_yaw + service_angular * self.sim_dt)
        self.service_pose = self.service_pose + np.array(
            [math.cos(self.service_yaw), math.sin(self.service_yaw)],
            dtype=np.float32,
        ) * service_linear * self.sim_dt
        if self.service_mocap_id >= 0:
            self.data.mocap_pos[self.service_mocap_id] = np.array([self.service_pose[0], self.service_pose[1], 0.11], dtype=float)
            self.data.mocap_quat[self.service_mocap_id] = np.array(
                [math.cos(0.5 * self.service_yaw), 0.0, 0.0, math.sin(0.5 * self.service_yaw)],
                dtype=float,
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0] = np.random.uniform(-7.2, -4.8)
        self.data.qpos[1] = np.random.uniform(-5.0, -3.5)
        self.data.qpos[2] = np.random.uniform(-math.pi, math.pi)
        self.service_pose = np.array([np.random.uniform(5.8, 7.8), np.random.uniform(3.8, 5.2)], dtype=np.float32)
        self.service_yaw = np.random.uniform(-math.pi, math.pi)
        self.primary_goal = np.array([PICK_STATION_X, PICK_STATION_Y], dtype=np.float32)
        self.service_goal = np.array([7.0, -4.2], dtype=np.float32)
        self.steps = 0
        self.phase = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self._apply_action(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        px, py, _ = self._read_pose()
        lidar = self._read_lidar()
        base_goal_distance = math.hypot(self.primary_goal[0] - px, self.primary_goal[1] - py)
        service_goal_distance = math.hypot(self.service_goal[0] - self.service_pose[0], self.service_goal[1] - self.service_pose[1])
        gap = math.hypot(self.service_pose[0] - px, self.service_pose[1] - py)

        reward = -0.004
        reward += max(0.0, 1.1 - base_goal_distance) * 0.7
        reward += max(0.0, 1.1 - service_goal_distance) * 0.6
        reward -= max(0.0, 0.95 - gap) * 3.5
        if np.min(lidar) < 0.18:
            reward -= 1.2

        if self.phase == 0 and base_goal_distance < 0.45:
            self.phase = 1
            self.primary_goal = np.array([PLACE_STATION_X, PLACE_STATION_Y], dtype=np.float32)
            reward += 10.0
        elif self.phase == 1 and base_goal_distance < 0.50 and service_goal_distance < 0.55:
            self.phase = 2
            reward += 18.0

        terminated = bool(np.min(lidar) < 0.10 or gap < 0.42 or self.phase >= 2)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}
