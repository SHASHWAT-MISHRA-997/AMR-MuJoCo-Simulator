import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, LaserScan
from std_msgs.msg import Bool

from .config_utils import load_dock_config, load_waypoints
from .controllers import ObstacleAwareWaypointNavigator, wrap_to_pi


class RlPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_policy_node")
        package_share = Path(get_package_share_directory("mujoco_amr_sim"))
        package_root = Path(__file__).resolve().parents[1]
        default_model_path = package_root / "models" / "mujoco_amr_ppo.zip"
        self.declare_parameter("model_path", str(default_model_path))
        self.declare_parameter("dock_config_file", str(package_share / "config" / "dock_station.json"))
        self.declare_parameter("waypoints_file", str(package_share / "config" / "waypoints.json"))
        self.declare_parameter("odom_topic", "/odometry/filtered")

        self.model = None
        self.policy_backend = "heuristic_fallback"
        model_path = Path(str(self.get_parameter("model_path").value))
        if model_path.is_file():
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(str(model_path))
                self.policy_backend = "ppo"
            except Exception as exc:
                self.get_logger().warning(f"RL model unavailable, using heuristic fallback: {exc}")
        else:
            self.get_logger().warning(
                f"RL model not found at {model_path}. Using heuristic fallback policy instead."
            )

        self.dock = load_dock_config(str(self.get_parameter("dock_config_file").value))
        self.waypoints = load_waypoints(str(self.get_parameter("waypoints_file").value))
        self.navigator = ObstacleAwareWaypointNavigator(self.waypoints)
        self.goal_index = 0

        self.pose_x = None
        self.pose_y = None
        self.pose_yaw = None
        self.battery_pct = 1.0
        self.scan = []
        self.scan_angles = []
        self.dock_contact = False

        self.pub = self.create_publisher(Twist, "/cmd_vel_rl", 10)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self._odom_callback, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_callback, 10)
        self.create_subscription(BatteryState, "/battery_state", self._battery_callback, 10)
        self.create_subscription(Bool, "/dock/in_contact", self._dock_contact_callback, 10)
        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(f"RL policy node ready with backend '{self.policy_backend}'")

    def _odom_callback(self, msg: Odometry) -> None:
        self.pose_x = float(msg.pose.pose.position.x)
        self.pose_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.pose_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _scan_callback(self, msg: LaserScan) -> None:
        self.scan = list(msg.ranges)
        self.scan_angles = [msg.angle_min + index * msg.angle_increment for index in range(len(msg.ranges))]

    def _battery_callback(self, msg: BatteryState) -> None:
        if msg.percentage is not None and math.isfinite(msg.percentage):
            self.battery_pct = max(0.0, min(1.0, float(msg.percentage)))

    def _dock_contact_callback(self, msg: Bool) -> None:
        self.dock_contact = bool(msg.data)

    def _build_observation(self) -> np.ndarray:
        lidar = np.array(self.scan, dtype=np.float32)
        if lidar.size == 0:
            lidar = np.full((31,), 8.0, dtype=np.float32)
        lidar = np.interp(np.linspace(0, len(lidar) - 1, 31), np.arange(len(lidar)), lidar)
        lidar = np.clip(lidar / 8.0, 0.0, 1.0)

        if self.battery_pct < 0.20:
            goal_x, goal_y = self.dock.dock_pose[:2]
        else:
            goal_x, goal_y = self.waypoints[self.goal_index]
            if math.hypot(goal_x - self.pose_x, goal_y - self.pose_y) < 0.35:
                self.goal_index = (self.goal_index + 1) % len(self.waypoints)
                goal_x, goal_y = self.waypoints[self.goal_index]

        dx = goal_x - self.pose_x
        dy = goal_y - self.pose_y
        dock_dx = self.dock.dock_pose[0] - self.pose_x
        dock_dy = self.dock.dock_pose[1] - self.pose_y
        return np.concatenate(
            [
                lidar,
                np.array(
                    [
                        dx,
                        dy,
                        wrap_to_pi(math.atan2(dy, dx) - self.pose_yaw),
                        dock_dx,
                        dock_dy,
                        self.battery_pct,
                        self.pose_yaw,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def _select_goal(self) -> tuple[float, float]:
        if self.battery_pct < 0.20:
            return float(self.dock.dock_pose[0]), float(self.dock.dock_pose[1])

        goal_x, goal_y = self.waypoints[self.goal_index]
        if math.hypot(goal_x - self.pose_x, goal_y - self.pose_y) < 0.35:
            self.goal_index = (self.goal_index + 1) % len(self.waypoints)
            goal_x, goal_y = self.waypoints[self.goal_index]
        return float(goal_x), float(goal_y)

    def _heuristic_command(self) -> Twist:
        goal_x, goal_y = self._select_goal()
        dx = goal_x - self.pose_x
        dy = goal_y - self.pose_y
        distance = math.hypot(dx, dy)
        heading_error = wrap_to_pi(math.atan2(dy, dx) - self.pose_yaw)

        linear = float(np.clip(0.55 * distance, 0.0, 0.6))
        linear *= max(0.0, 1.0 - abs(heading_error) / 1.25)
        angular = float(np.clip(1.6 * heading_error, -1.3, 1.3))

        if self.scan:
            center = len(self.scan) // 2
            front = self.scan[max(0, center - 12) : min(len(self.scan), center + 12)]
            left = self.scan[min(len(self.scan) - 1, center + 8) : min(len(self.scan), center + 28)]
            right = self.scan[max(0, center - 28) : max(1, center - 8)]
            front_min = min(front) if front else 8.0
            left_mean = float(np.mean(left)) if left else 8.0
            right_mean = float(np.mean(right)) if right else 8.0

            if front_min < 0.70:
                linear *= max(0.0, (front_min - 0.16) / 0.54)
                angular += 0.95 if left_mean > right_mean else -0.95

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        return msg

    def _tick(self) -> None:
        if None in (self.pose_x, self.pose_y, self.pose_yaw) or not self.scan or self.dock_contact:
            return
        if self.model is not None:
            action, _ = self.model.predict(self._build_observation(), deterministic=True)
            msg = Twist()
            msg.linear.x = float(np.clip(action[0], -1.0, 1.0) * 0.7)
            msg.angular.z = float(np.clip(action[1], -1.0, 1.0) * 1.5)
        else:
            msg = self._heuristic_command()
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RlPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
