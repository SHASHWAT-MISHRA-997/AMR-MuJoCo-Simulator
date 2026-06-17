import json
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger

from .config_utils import load_dock_config, normalize_service_target_key, table_service_goal
from .mjcf_builder import SERVICE_TARGETS


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class Nav2MissionBridge(Node):
    def __init__(self) -> None:
        super().__init__("nav2_mission_bridge")

        package_share = Path(get_package_share_directory("mujoco_amr_sim"))
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("dock_config_file", str(package_share / "config" / "dock_station.json"))
        self.declare_parameter("goal_frame", "map")
        self.declare_parameter("default_mission_mode", "idle")
        self.declare_parameter("table_service_dwell_seconds", 5.0)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.dock_config_file = str(self.get_parameter("dock_config_file").value)
        self.goal_frame = str(self.get_parameter("goal_frame").value)
        self.default_mission_mode = str(self.get_parameter("default_mission_mode").value).strip().lower() or "idle"
        self.table_service_dwell_seconds = float(self.get_parameter("table_service_dwell_seconds").value)
        self.dock = load_dock_config(self.dock_config_file)

        self.state_pub = self.create_publisher(String, "/autonomy/state", 10)
        self.mission_pub = self.create_publisher(String, "/autonomy/mission_status", 10)
        self.event_pub = self.create_publisher(String, "/autonomy/event_log", 10)
        self.patrol_enabled_pub = self.create_publisher(Bool, "/autonomy/patrol_enabled", 10)
        self.force_dock_pub = self.create_publisher(Bool, "/autonomy/force_dock_active", 10)

        self.create_subscription(String, "/autonomy/mission_command", self._mission_command_callback, 10)
        self.create_subscription(Float32, "/autonomy/speed_limit", self._speed_limit_callback, 10)
        self.create_subscription(BatteryState, "/battery_state", self._battery_callback, 10)
        self.create_subscription(Bool, "/dock/in_contact", self._dock_contact_callback, 10)
        self.create_subscription(Bool, "/dock/is_charging", self._charging_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)

        self.create_service(SetBool, "/autonomy/set_patrol_enabled", self._set_patrol_enabled)
        self.create_service(Trigger, "/autonomy/force_dock", self._force_dock)
        self.create_service(Trigger, "/autonomy/resume_patrol", self._resume_patrol)
        self.create_service(Trigger, "/autonomy/skip_waypoint", self._skip_waypoint)
        self.create_service(Trigger, "/autonomy/reload_mission", self._reload_mission)

        self.navigate_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.timer = self.create_timer(0.10, self._tick)

        self.pose_x: float | None = None
        self.pose_y: float | None = None
        self.pose_yaw: float | None = None
        self.battery_pct = 1.0
        self.dock_contact = False
        self.is_charging = False
        self.speed_limit_scale = 1.0
        self.patrol_enabled = False
        self.force_dock_requested = False
        self.pending_table_dock_cycle = False
        self.table_hold_started_at: float | None = None
        self.state = "WAITING_TASK"
        self.mission_mode = "idle"
        self.active_goal_handle = None
        self.active_goal_xy: tuple[float, float] | None = None
        self._queued_mission_mode: str | None = None

        self._emit_event("Nav2 mission bridge initialized")
        if self.default_mission_mode != "idle":
            self._queue_or_send_mission(self.default_mission_mode)

    def _emit_event(self, message: str) -> None:
        self.get_logger().info(message)
        self.event_pub.publish(String(data=message))

    def _publish_state(self, state: str) -> None:
        self.state = state
        self.state_pub.publish(String(data=state))

    def _odom_callback(self, msg: Odometry) -> None:
        self.pose_x = float(msg.pose.pose.position.x)
        self.pose_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.pose_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _battery_callback(self, msg: BatteryState) -> None:
        if msg.percentage is not None and math.isfinite(msg.percentage):
            self.battery_pct = max(0.0, min(1.0, float(msg.percentage)))

    def _dock_contact_callback(self, msg: Bool) -> None:
        self.dock_contact = bool(msg.data)

    def _charging_callback(self, msg: Bool) -> None:
        self.is_charging = bool(msg.data)

    def _speed_limit_callback(self, msg: Float32) -> None:
        self.speed_limit_scale = max(0.20, min(1.25, float(msg.data)))

    def _set_patrol_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.patrol_enabled = bool(request.data)
        response.success = True
        response.message = "Nav2 patrol flag updated"
        return response

    def _force_dock(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self._queue_or_send_mission("dock_return")
        response.success = True
        response.message = "Nav2 dock return requested"
        return response

    def _resume_patrol(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self._queue_or_send_mission(self.default_mission_mode)
        response.success = True
        response.message = "Nav2 default mission resumed"
        return response

    def _skip_waypoint(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        response.success = False
        response.message = "Skip waypoint is not supported in Nav2 mission mode"
        return response

    def _reload_mission(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.dock = load_dock_config(self.dock_config_file)
        response.success = True
        response.message = "Nav2 mission configuration reloaded"
        return response

    def _mission_command_callback(self, msg: String) -> None:
        mission_mode = normalize_service_target_key(str(msg.data).strip().lower())
        if mission_mode:
            self._queue_or_send_mission(mission_mode)

    def _mission_goal_pose(self, mission_mode: str) -> tuple[float, float, float] | None:
        if mission_mode in SERVICE_TARGETS:
            return table_service_goal(mission_mode, "a")
        if mission_mode == "dock_return":
            return self.dock.dock_pose
        return None

    def _cancel_active_goal(self) -> None:
        if self.active_goal_handle is None:
            return
        cancel_future = self.active_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(lambda _future: self._send_queued_if_ready())

    def _queue_or_send_mission(self, mission_mode: str) -> None:
        mission_mode = (mission_mode or "").strip().lower()
        if not mission_mode:
            return
        if mission_mode == "idle":
            self.mission_mode = "idle"
            self.force_dock_requested = False
            self.pending_table_dock_cycle = False
            self.table_hold_started_at = None
            self._queued_mission_mode = None
            self._cancel_active_goal()
            self.active_goal_handle = None
            self.active_goal_xy = None
            self._publish_state("CHARGING" if self.is_charging else "WAITING_TASK")
            self._emit_event("Nav2 mission set to idle")
            return
        self._queued_mission_mode = mission_mode
        if self.active_goal_handle is not None:
            self._cancel_active_goal()
            return
        self._send_queued_if_ready()

    def _send_queued_if_ready(self) -> None:
        if self._queued_mission_mode is None:
            return
        mission_mode = self._queued_mission_mode
        goal_pose = self._mission_goal_pose(mission_mode)
        if goal_pose is None:
            self._emit_event(f"Unknown Nav2 mission command: {mission_mode}")
            self._queued_mission_mode = None
            return
        if not self.navigate_client.wait_for_server(timeout_sec=0.10):
            self._publish_state("NAV2_WAIT_SERVER")
            return

        goal_x, goal_y, goal_yaw = goal_pose
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.goal_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(goal_x)
        goal.pose.pose.position.y = float(goal_y)
        qx, qy, qz, qw = yaw_to_quaternion(float(goal_yaw))
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        self.mission_mode = mission_mode
        self.force_dock_requested = mission_mode == "dock_return"
        self.pending_table_dock_cycle = False
        self.table_hold_started_at = None
        self.active_goal_xy = (float(goal_x), float(goal_y))
        self._publish_state("NAV2_PLANNING")
        self._emit_event(f"Nav2 mission request: {mission_mode}")
        self._queued_mission_mode = None

        send_future = self.navigate_client.send_goal_async(goal, feedback_callback=self._feedback_callback)
        send_future.add_done_callback(self._goal_response_callback)

    def _feedback_callback(self, _feedback_msg) -> None:
        if self.mission_mode == "dock_return":
            self._publish_state("SEEK_DOCK")
        else:
            self._publish_state("NAV2_ACTIVE")

    def _goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.active_goal_handle = None
            self.active_goal_xy = None
            self._publish_state("NAV2_REJECTED")
            self._emit_event(f"Nav2 goal rejected for mission {self.mission_mode}")
            return
        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future) -> None:
        self.active_goal_handle = None
        self.active_goal_xy = None
        result = future.result()
        status = None if result is None else int(result.status)
        if status == 5:
            self._emit_event(f"Nav2 goal cancelled for mission {self.mission_mode}")
            self._publish_state("WAITING_TASK")
            self._send_queued_if_ready()
            return
        if status == 4:
            if self.mission_mode in SERVICE_TARGETS:
                self.table_hold_started_at = self.get_clock().now().nanoseconds * 1e-9
                self._publish_state("TABLE_SERVICE")
                self._emit_event(f"{SERVICE_TARGETS[self.mission_mode]['label']} reached with Nav2")
            elif self.mission_mode == "dock_return":
                self.force_dock_requested = False
                self._publish_state("CHARGING" if self.is_charging else "WAITING_TASK")
                self._emit_event("Nav2 dock return completed")
            else:
                self._publish_state("WAITING_TASK")
        else:
            self._publish_state("NAV2_FAILED")
            self._emit_event(f"Nav2 mission failed: {self.mission_mode}")
        self._send_queued_if_ready()

    def _publish_mission_status(self) -> None:
        payload = {
            "state": self.state,
            "mission_mode": self.mission_mode,
            "speed_limit_scale": round(self.speed_limit_scale, 3),
            "battery_pct": round(self.battery_pct, 4),
            "patrol_enabled": self.patrol_enabled,
            "force_dock_requested": self.force_dock_requested,
            "dock_contact": self.dock_contact,
            "is_charging": self.is_charging,
            "waypoint_index": 0,
            "waypoint_count": 1 if self.active_goal_xy is not None else 0,
            "current_goal": list(self.active_goal_xy) if self.active_goal_xy is not None else None,
            "pose": {
                "x": None if self.pose_x is None else round(self.pose_x, 3),
                "y": None if self.pose_y is None else round(self.pose_y, 3),
                "yaw": None if self.pose_yaw is None else round(self.pose_yaw, 3),
            },
        }
        self.mission_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self.patrol_enabled_pub.publish(Bool(data=self.patrol_enabled))
        self.force_dock_pub.publish(Bool(data=self.force_dock_requested))

    def _tick(self) -> None:
        if self._queued_mission_mode is not None and self.active_goal_handle is None:
            self._send_queued_if_ready()
        if self.table_hold_started_at is not None and self.mission_mode in SERVICE_TARGETS:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if now_s - self.table_hold_started_at >= self.table_service_dwell_seconds:
                completed_table = self.mission_mode
                self.table_hold_started_at = None
                self.pending_table_dock_cycle = True
                self._emit_event(f"{SERVICE_TARGETS[completed_table]['label']} served by Nav2; returning to auto dock")
                self._queue_or_send_mission("dock_return")
        if self.mission_mode == "idle" and self.is_charging:
            self._publish_state("CHARGING")
        elif self.mission_mode == "idle" and self.active_goal_handle is None and self.table_hold_started_at is None:
            self._publish_state("WAITING_TASK")
        self._publish_mission_status()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Nav2MissionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
