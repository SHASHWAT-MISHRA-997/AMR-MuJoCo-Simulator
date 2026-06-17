import json
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, LaserScan
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger

from .config_utils import (
    build_main_dock_return_route,
    build_service_route,
    build_visibility_route,
    load_dock_config,
    load_waypoints,
    normalize_service_target_key,
    service_approach_goal,
    service_target_goal,
)
from .controllers import ObstacleAwareWaypointNavigator, PoseGoal, PoseTrackingController, RobotCommand
from .mjcf_builder import (
    DOCK_X,
    DOCK_Y,
    HOTEL_TABLES,
    SERVICE_TARGETS,
    PICK_STATION_X,
    PICK_STATION_Y,
    PLACE_STATION_X,
    PLACE_STATION_Y,
    ROBOT_START_X,
    ROBOT_START_Y,
)


class AutonomyManager(Node):
    def __init__(self) -> None:
        super().__init__("autonomy_manager")

        package_share = Path(get_package_share_directory("mujoco_amr_sim"))

        self.declare_parameter("patrol_enabled", False)
        self.declare_parameter("battery_dock_threshold", 0.30)
        self.declare_parameter("battery_resume_threshold", 0.98)
        self.declare_parameter("battery_latch_samples", 4)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("waypoints_file", str(package_share / "config" / "waypoints.json"))
        self.declare_parameter("dock_config_file", str(package_share / "config" / "dock_station.json"))
        self.declare_parameter("default_mission_mode", "idle")
        self.declare_parameter("speed_limit_scale", 1.00)

        self.patrol_enabled = bool(self.get_parameter("patrol_enabled").value)
        self.battery_dock_threshold = float(self.get_parameter("battery_dock_threshold").value)
        self.battery_resume_threshold = float(self.get_parameter("battery_resume_threshold").value)
        self.battery_latch_samples = max(1, int(self.get_parameter("battery_latch_samples").value))
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.dock_config_file = str(self.get_parameter("dock_config_file").value)
        self.waypoints_file = str(self.get_parameter("waypoints_file").value)
        self.default_mission_mode = str(self.get_parameter("default_mission_mode").value)
        self.speed_limit_scale = max(0.20, min(1.25, float(self.get_parameter("speed_limit_scale").value)))
        self.navigator_speed_boost = 1.10
        self.table_service_dwell_seconds = 5.0

        self.base_waypoints = [list(point[:2]) for point in load_waypoints(self.waypoints_file)]
        self.navigator = ObstacleAwareWaypointNavigator(
            self.base_waypoints,
            max_linear=0.92,
            max_angular=2.05,
            goal_tolerance=0.22,
            stop_distance=0.54,
            slow_distance=0.96,
        )
        self._apply_navigation_profile()
        self.approach_controller = PoseTrackingController(
            max_linear=0.64,
            max_angular=1.55,
            position_tolerance=0.16,
            yaw_tolerance=0.20,
        )
        self.final_docking_controller = PoseTrackingController(
            max_linear=0.34,
            max_angular=0.9,
            position_tolerance=0.06,
            yaw_tolerance=0.10,
        )
        self.table_service_controller = PoseTrackingController(
            max_linear=0.38,
            max_angular=1.18,
            position_tolerance=0.06,
            yaw_tolerance=0.16,
        )
        self.dock = load_dock_config(self.dock_config_file)
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.command_callback_group = ReentrantCallbackGroup()

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_auto", 10)
        self.state_pub = self.create_publisher(String, "/autonomy/state", 10)
        self.mission_pub = self.create_publisher(String, "/autonomy/mission_status", 10)
        self.event_pub = self.create_publisher(String, "/autonomy/event_log", 10)
        self.patrol_enabled_pub = self.create_publisher(Bool, "/autonomy/patrol_enabled", 10)
        self.force_dock_pub = self.create_publisher(Bool, "/autonomy/force_dock_active", 10)

        self.create_subscription(BatteryState, "/battery_state", self._battery_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_callback, 10)
        self.create_subscription(Bool, "/dock/in_contact", self._dock_contact_callback, 10)
        self.create_subscription(Bool, "/dock/is_charging", self._charging_callback, 10)
        self.create_subscription(
            String,
            "/autonomy/mission_command",
            self._mission_command_callback,
            10,
            callback_group=self.command_callback_group,
        )
        self.create_subscription(
            Float32,
            "/autonomy/speed_limit",
            self._speed_limit_callback,
            10,
            callback_group=self.command_callback_group,
        )

        self.create_service(SetBool, "/autonomy/set_patrol_enabled", self._set_patrol_enabled)
        self.create_service(Trigger, "/autonomy/force_dock", self._force_dock)
        self.create_service(Trigger, "/autonomy/resume_patrol", self._resume_patrol)
        self.create_service(Trigger, "/autonomy/skip_waypoint", self._skip_waypoint)
        self.create_service(Trigger, "/autonomy/reload_mission", self._reload_mission)

        self.timer = self.create_timer(0.05, self._control_loop, callback_group=self.control_callback_group)

        self.pose_x = None
        self.pose_y = None
        self.pose_yaw = None
        self.scan_ranges: list[float] = []
        self.scan_angles: list[float] = []
        self.battery_pct = 1.0
        self.battery_samples_received = 0
        self.low_battery_sample_count = 0
        self.high_battery_sample_count = 0
        self.dock_contact = False
        self.is_charging = False
        self.low_battery_latched = False
        self.force_dock_requested = False
        self.dock_request_reason: str | None = None
        self.table_hold_started_at: float | None = None
        self.pending_table_dock_cycle = False
        self.post_service_escape_until: float | None = None
        self.rebuild_route_after_undock = False
        self.undock_required = False
        self.route_start_override: tuple[float, float] | None = None
        self.last_route_progress_time = 0.0
        self.last_route_progress_pose: tuple[float, float] | None = None
        self.last_route_progress_index = 0
        self.last_service_replan_time = -10.0
        self.state = "IDLE"
        self.mission_mode = "idle"
        self._apply_mission_mode(self.default_mission_mode, emit_event=False)
        self._emit_event("Autonomy manager initialized")

    def _table_route(self, table_key: str) -> list[list[float]]:
        route_start = self.route_start_override if self.route_start_override is not None else self._route_start_xy()
        return build_service_route(route_start, table_key, "a")

    def _route_start_xy(self) -> tuple[float, float]:
        if self._current_pose_available():
            return float(self.pose_x), float(self.pose_y)
        if self.dock_contact or self.is_charging:
            return DOCK_X, DOCK_Y
        return ROBOT_START_X, ROBOT_START_Y

    def _table_service_pose_goal(self, table_key: str) -> PoseGoal:
        goal_x, goal_y, goal_yaw = service_target_goal(table_key, "a")
        return PoseGoal(goal_x, goal_y, goal_yaw)

    def _table_service_approach_goal(self, table_key: str) -> PoseGoal:
        goal_x, goal_y, goal_yaw = service_approach_goal(table_key, "a")
        return PoseGoal(goal_x, goal_y, goal_yaw)

    def _navigator_speed_scale(self) -> float:
        return max(0.20, min(1.25, self.speed_limit_scale * self.navigator_speed_boost))

    def _apply_navigation_profile(self) -> None:
        service_route_active = getattr(self, "mission_mode", "idle") in SERVICE_TARGETS
        dock_return_active = getattr(self, "mission_mode", "idle") == "dock_return"
        if service_route_active:
            self.navigator.set_goal_tolerance(0.20)
            self.navigator.set_max_linear(0.64)
        elif dock_return_active:
            self.navigator.set_goal_tolerance(0.22)
            self.navigator.set_max_linear(0.72)
        else:
            self.navigator.set_goal_tolerance(0.22)
            self.navigator.set_max_linear(0.92)
        speed_scale = self._navigator_speed_scale()
        if service_route_active:
            speed_scale = min(speed_scale, 0.82)
        elif dock_return_active:
            speed_scale = min(0.78, max(0.58, speed_scale))
        self.navigator.set_speed_scale(speed_scale)

    def _build_mission_library(self) -> dict[str, list[list[float]]]:
        route_start = self._route_start_xy()
        pre_dock_x, pre_dock_y, _ = self.dock.pre_dock_pose
        missions = {
            "idle": [],
            "lobby_patrol": [list(point) for point in self.base_waypoints],
            "room_delivery": [
                [ROBOT_START_X, ROBOT_START_Y],
                [PICK_STATION_X - 1.10, PICK_STATION_Y],
                [PICK_STATION_X, PICK_STATION_Y],
                [1.20, -1.00],
                [4.80, 1.40],
                [PLACE_STATION_X, PLACE_STATION_Y],
                [4.20, 5.10],
                [-1.80, 5.10],
                [-6.80, 4.80],
            ],
            "table_service": [
                [ROBOT_START_X, ROBOT_START_Y],
                [-4.80, -1.50],
                [-1.20, 0.60],
                [2.60, 2.40],
                [4.80, 0.40],
                [1.10, -1.90],
                [-3.60, -0.20],
            ],
            "dock_return": build_main_dock_return_route(route_start, (pre_dock_x, pre_dock_y)),
        }
        for table_key in SERVICE_TARGETS:
            missions[table_key] = self._table_route(table_key)
        return missions

    def _reset_route_progress_tracking(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self.last_route_progress_time = now_s
        self.last_route_progress_index = self.navigator.current_index
        if self._current_pose_available():
            self.last_route_progress_pose = (float(self.pose_x), float(self.pose_y))
        else:
            self.last_route_progress_pose = None

    def _front_obstacle_distance(self) -> float:
        readings = [
            distance
            for angle, distance in zip(self.scan_angles, self.scan_ranges, strict=False)
            if math.isfinite(distance) and abs(angle) <= math.radians(26.0)
        ]
        if not readings:
            return float("inf")
        return min(readings)

    def _service_route_stalled(self) -> bool:
        if self.mission_mode not in SERVICE_TARGETS or not self._current_pose_available():
            return False
        if self.navigator.waypoint_count == 0 or self.navigator.route_complete(self.pose_x, self.pose_y):
            return False

        now_s = self.get_clock().now().nanoseconds * 1e-9
        current_pose = (float(self.pose_x), float(self.pose_y))
        moved_distance = 0.0
        if self.last_route_progress_pose is not None:
            moved_distance = math.hypot(
                current_pose[0] - self.last_route_progress_pose[0],
                current_pose[1] - self.last_route_progress_pose[1],
            )
        front_distance = self._front_obstacle_distance()

        if self.navigator.current_index != self.last_route_progress_index or moved_distance >= 0.16:
            self.last_route_progress_index = self.navigator.current_index
            self.last_route_progress_pose = current_pose
            self.last_route_progress_time = now_s
            return False

        stalled_long_enough = now_s - self.last_route_progress_time >= 3.0
        obstacle_blocking = front_distance < 0.42
        replan_cooldown_over = now_s - self.last_service_replan_time >= 3.0
        return stalled_long_enough and obstacle_blocking and replan_cooldown_over

    def _ready_for_final_service_approach(self) -> bool:
        if self.mission_mode not in SERVICE_TARGETS or not self._current_pose_available():
            return False
        if self.navigator.waypoint_count == 0:
            return False
        if self.mission_mode == "table_1" and self.navigator.current_index >= 6 and self.pose_y >= 0.20:
            return True
        if self.mission_mode == "sofa_1" and self.navigator.current_index >= 5 and self.pose_y >= 1.80:
            return True
        if self.mission_mode == "table_6" and self.navigator.current_index >= 7 and self.pose_y >= 2.10:
            return True
        if self.mission_mode == "sofa_3" and self.navigator.current_index >= 13 and self.pose_x >= 3.80:
            return True
        goal_x, goal_y, _goal_yaw = service_target_goal(self.mission_mode, "a")
        approach_x, approach_y, _approach_yaw = service_approach_goal(self.mission_mode, "a")
        distance_to_goal = math.hypot(goal_x - self.pose_x, goal_y - self.pose_y)
        distance_to_approach = math.hypot(approach_x - self.pose_x, approach_y - self.pose_y)
        remaining_waypoints = max(0, self.navigator.waypoint_count - self.navigator.current_index - 1)
        approach_stop_gap = math.hypot(goal_x - approach_x, goal_y - approach_y)
        if approach_stop_gap >= 0.22:
            return remaining_waypoints <= 2 and (
                distance_to_approach <= 0.95 or distance_to_goal <= 1.05
            )
        return distance_to_goal <= 1.55 and remaining_waypoints <= 5

    def _service_waypoint_ready_to_advance(self) -> bool:
        if self.mission_mode not in SERVICE_TARGETS or not self._current_pose_available():
            return False
        if self.navigator.waypoint_count == 0 or self.navigator.current_index >= self.navigator.waypoint_count - 1:
            return False
        remaining_waypoints = max(0, self.navigator.waypoint_count - self.navigator.current_index - 1)
        if remaining_waypoints <= 4:
            return False
        current_goal = self.navigator.current_goal
        if current_goal is None:
            return False
        distance_to_waypoint = math.hypot(float(current_goal[0]) - self.pose_x, float(current_goal[1]) - self.pose_y)
        if remaining_waypoints <= 3:
            return distance_to_waypoint <= 0.24
        if remaining_waypoints >= 8:
            return distance_to_waypoint <= 0.46
        return distance_to_waypoint <= 0.30

    def _rebuild_service_route_from_current_pose(self) -> None:
        if self.mission_mode not in SERVICE_TARGETS:
            return
        self.last_service_replan_time = self.get_clock().now().nanoseconds * 1e-9
        self._apply_mission_mode(self.mission_mode, emit_event=False)
        self._reset_route_progress_tracking()
        self._emit_event(f"Replanned path to {SERVICE_TARGETS[self.mission_mode]['label']} from current pose")

    def _apply_mission_mode(self, mission_mode: str, emit_event: bool = True) -> bool:
        normalized = (mission_mode or "").strip().lower()
        mission_library = self._build_mission_library()
        if normalized not in mission_library:
            return False
        self.mission_mode = normalized
        self.navigator.set_waypoints(
            mission_library[normalized],
            loop=normalized not in SERVICE_TARGETS and normalized not in {"dock_return", "idle"},
        )
        self._apply_navigation_profile()
        self._reset_route_progress_tracking()
        self.table_hold_started_at = None
        self.post_service_escape_until = None
        if normalized != "dock_return":
            self.pending_table_dock_cycle = False
        if emit_event:
            self._emit_event(f"Mission set to {normalized}")
        return True

    def _battery_callback(self, msg: BatteryState) -> None:
        if msg.percentage is not None and math.isfinite(msg.percentage):
            self.battery_pct = max(0.0, min(1.0, float(msg.percentage)))
            self.battery_samples_received += 1

    def _odom_callback(self, msg: Odometry) -> None:
        self.pose_x = float(msg.pose.pose.position.x)
        self.pose_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.pose_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _scan_callback(self, msg: LaserScan) -> None:
        self.scan_ranges = list(msg.ranges)
        self.scan_angles = [msg.angle_min + index * msg.angle_increment for index in range(len(msg.ranges))]

    def _dock_contact_callback(self, msg: Bool) -> None:
        self.dock_contact = bool(msg.data)

    def _charging_callback(self, msg: Bool) -> None:
        self.is_charging = bool(msg.data)
        if self.is_charging and self.force_dock_requested:
            self.force_dock_requested = False
            self.dock_request_reason = None
            self.undock_required = False
            self._apply_mission_mode("idle", emit_event=False)
            self._emit_event("Force dock request completed; charger engaged")

    def _mission_command_callback(self, msg: String) -> None:
        self.get_logger().info(f"Mission command callback received: {msg.data}")
        mission_mode = normalize_service_target_key(str(msg.data).strip().lower())
        if not mission_mode:
            return
        if (
            mission_mode == self.mission_mode
            and mission_mode in SERVICE_TARGETS
            and self.state not in {"WAITING_FOR_ODOM", "CHARGING"}
            and not self.force_dock_requested
        ):
            return
        if mission_mode == "dock_return":
            self.force_dock_requested = True
            self.dock_request_reason = "manual"
            self.pending_table_dock_cycle = False
            self.table_hold_started_at = None
            self.undock_required = False
            self._apply_mission_mode(mission_mode, emit_event=False)
            self._emit_event("Mission set to dock_return; returning to charging bay")
            return
        if self._apply_mission_mode(mission_mode):
            self.force_dock_requested = False
            self.low_battery_latched = False
            self.dock_request_reason = None
            self.pending_table_dock_cycle = False
            self.table_hold_started_at = None
            self.undock_required = mission_mode in SERVICE_TARGETS and (
                not self._current_pose_available()
                or self.dock_contact
                or self.is_charging
                or self._main_dock_distance() <= 0.95
            )
            self.rebuild_route_after_undock = self.undock_required
        else:
            self._emit_event(f"Unknown mission command: {mission_mode}")

    def _speed_limit_callback(self, msg: Float32) -> None:
        requested = max(0.20, min(1.25, float(msg.data)))
        self.speed_limit_scale = requested
        self._apply_navigation_profile()
        self._emit_event(f"Speed limit set to {requested:.2f}x")

    def _publish_state(self, state: str) -> None:
        if state != self.state:
            self._emit_event(f"State changed: {self.state} -> {state}")
            self.state = state
        self.state_pub.publish(String(data=self.state))

    def _publish_cmd(self, command: RobotCommand) -> None:
        msg = Twist()
        msg.linear.x = float(command.linear)
        msg.angular.z = float(command.angular)
        self.cmd_pub.publish(msg)

    def _current_pose_available(self) -> bool:
        return self.pose_x is not None and self.pose_y is not None and self.pose_yaw is not None

    def _main_dock_distance(self) -> float:
        if not self._current_pose_available():
            return float("inf")
        dock_x, dock_y, _dock_yaw = self.dock.dock_pose
        return math.hypot(float(self.pose_x) - dock_x, float(self.pose_y) - dock_y)

    def _in_final_docking_corridor(self) -> bool:
        if not self._current_pose_available():
            return False
        dock_x, dock_y, dock_yaw = self.dock.dock_pose
        dx = dock_x - float(self.pose_x)
        dy = dock_y - float(self.pose_y)
        local_x = math.cos(dock_yaw) * dx + math.sin(dock_yaw) * dy
        local_y = -math.sin(dock_yaw) * dx + math.cos(dock_yaw) * dy
        yaw_error = abs(math.atan2(math.sin(dock_yaw - self.pose_yaw), math.cos(dock_yaw - self.pose_yaw)))
        return -0.20 <= local_x <= 0.60 and abs(local_y) <= 0.30 and yaw_error <= 0.42

    def _needs_undock_departure(self) -> bool:
        if self.mission_mode in {"idle", "dock_return"} or not self._current_pose_available():
            return False
        if not self.undock_required:
            return False
        pre_dock_x, pre_dock_y, _pre_dock_yaw = self.dock.pre_dock_pose
        if math.hypot(pre_dock_x - self.pose_x, pre_dock_y - self.pose_y) <= max(
            0.18,
            self.approach_controller.position_tolerance,
        ):
            self.undock_required = False
            return False
        return True

    def _compute_undock_command(self) -> RobotCommand:
        dock_x, dock_y, dock_yaw = self.dock.dock_pose
        pre_dock_x, pre_dock_y, pre_dock_yaw = self.dock.pre_dock_pose
        exit_heading = math.atan2(pre_dock_y - dock_y, pre_dock_x - dock_x)
        heading_error = math.atan2(math.sin(exit_heading - self.pose_yaw), math.cos(exit_heading - self.pose_yaw))
        yaw_error = math.atan2(math.sin(pre_dock_yaw - self.pose_yaw), math.cos(pre_dock_yaw - self.pose_yaw))
        distance_to_pre_dock = math.hypot(pre_dock_x - self.pose_x, pre_dock_y - self.pose_y)
        dock_yaw_error = math.atan2(math.sin(dock_yaw - self.pose_yaw), math.cos(dock_yaw - self.pose_yaw))
        dock_distance = self._main_dock_distance()

        if distance_to_pre_dock <= self.approach_controller.position_tolerance:
            return RobotCommand()

        speed_scale = max(0.72, min(1.0, self.speed_limit_scale))
        if dock_distance < 1.05:
            if abs(heading_error) > 0.14:
                launch_angular = max(-1.15, min(1.15, 2.4 * heading_error))
                return RobotCommand(linear=0.0, angular=launch_angular)
            launch_angular = max(-0.14, min(0.14, 0.45 * heading_error + 0.10 * dock_yaw_error))
            launch_linear = 0.40 if abs(heading_error) < 0.05 else 0.30
            return RobotCommand(linear=launch_linear * speed_scale, angular=launch_angular)

        angular = max(-1.45, min(1.45, 2.6 * heading_error + 0.55 * yaw_error))
        if abs(heading_error) > 0.62:
            return RobotCommand(linear=0.0, angular=angular)

        linear = 0.34
        if abs(heading_error) > 0.34:
            linear = 0.18
        elif distance_to_pre_dock < 0.55:
            linear = 0.16
        if dock_distance < 0.40:
            linear = max(linear, 0.26)

        linear *= speed_scale
        return RobotCommand(linear=linear, angular=angular)

    def _emit_event(self, message: str) -> None:
        self.get_logger().info(message)
        self.event_pub.publish(String(data=message))

    def _publish_mission_status(self) -> None:
        current_goal = self.navigator.current_goal
        payload = {
            "state": self.state,
            "mission_mode": self.mission_mode,
            "speed_limit_scale": round(self.speed_limit_scale, 3),
            "battery_pct": round(self.battery_pct, 4),
            "battery_samples_received": self.battery_samples_received,
            "patrol_enabled": self.patrol_enabled,
            "force_dock_requested": self.force_dock_requested,
            "dock_request_reason": self.dock_request_reason,
            "low_battery_latched": self.low_battery_latched,
            "dock_contact": self.dock_contact,
            "is_charging": self.is_charging,
            "waypoint_index": self.navigator.current_index,
            "waypoint_count": self.navigator.waypoint_count,
            "current_goal": list(current_goal) if current_goal is not None else None,
            "pose": {
                "x": None if self.pose_x is None else round(self.pose_x, 3),
                "y": None if self.pose_y is None else round(self.pose_y, 3),
                "yaw": None if self.pose_yaw is None else round(self.pose_yaw, 3),
            },
        }
        self.mission_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self.patrol_enabled_pub.publish(Bool(data=self.patrol_enabled))
        self.force_dock_pub.publish(Bool(data=self.force_dock_requested))

    def _set_patrol_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.patrol_enabled = bool(request.data)
        state_text = "enabled" if self.patrol_enabled else "disabled"
        self._emit_event(f"Hotel route execution {state_text} by operator")
        response.success = True
        response.message = f"Hotel route execution {state_text}"
        return response

    def _force_dock(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.force_dock_requested = True
        self.dock_request_reason = "manual"
        self._apply_mission_mode("dock_return", emit_event=False)
        self._emit_event("Operator requested return to charging bay")
        response.success = True
        response.message = "Return-to-charge request accepted"
        return response

    def _resume_patrol(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.battery_pct < self.battery_dock_threshold:
            response.success = False
            response.message = "Battery is still below dock threshold"
            return response
        self.force_dock_requested = False
        self.low_battery_latched = False
        self.dock_request_reason = None
        self._apply_mission_mode(self.default_mission_mode, emit_event=False)
        self.navigator.reset()
        self._emit_event("Operator requested hotel route resume")
        response.success = True
        response.message = "Hotel route resumed"
        return response

    def _skip_waypoint(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        next_goal = self.navigator.skip_to_next()
        if next_goal is None:
            response.success = False
            response.message = "No waypoints available"
            return response
        self._emit_event(f"Operator skipped to waypoint {self.navigator.current_index}")
        response.success = True
        response.message = f"Next waypoint: {next_goal[0]:.2f}, {next_goal[1]:.2f}"
        return response

    def _reload_mission(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        try:
            self.dock = load_dock_config(self.dock_config_file)
            self.base_waypoints = [list(point[:2]) for point in load_waypoints(self.waypoints_file)]
            mission_mode = self.mission_mode if self.mission_mode != "dock_return" else self.default_mission_mode
            self._apply_mission_mode(mission_mode, emit_event=False)
        except Exception as exc:
            response.success = False
            response.message = f"Reload failed: {exc}"
            return response
        self.force_dock_requested = False
        self.low_battery_latched = False
        self.dock_request_reason = None
        self._apply_navigation_profile()
        self._emit_event("Hotel mission configuration reloaded from disk")
        response.success = True
        response.message = "Mission configuration reloaded"
        return response

    def _compute_docking_command(self) -> RobotCommand:
        pre_dock = PoseGoal(*self.dock.pre_dock_pose)
        dock_pose = PoseGoal(*self.dock.dock_pose)
        final_docking_latched = self.state == "FINAL_DOCKING"
        pre_dock_distance = math.hypot(pre_dock.x - self.pose_x, pre_dock.y - self.pose_y)
        pre_dock_yaw_error = abs(math.atan2(math.sin(pre_dock.yaw - self.pose_yaw), math.cos(pre_dock.yaw - self.pose_yaw)))
        relaxed_pre_dock_ready = pre_dock_distance <= 0.45 and pre_dock_yaw_error <= 1.05

        if self.dock_contact or self.is_charging:
            self._publish_state("CHARGING" if self.is_charging else "FINAL_DOCKING")
            return RobotCommand()

        if (
            not final_docking_latched
            and not relaxed_pre_dock_ready
            and self.navigator.waypoint_count > 0
            and not self.navigator.route_complete(self.pose_x, self.pose_y)
        ):
            self._publish_state("SEEK_DOCK")
            current_goal = self.navigator.current_goal
            if current_goal is not None and math.hypot(float(current_goal[0]) - self.pose_x, float(current_goal[1]) - self.pose_y) <= 0.26:
                self.navigator.skip_to_next()
                current_goal = self.navigator.current_goal
            if current_goal is None:
                return RobotCommand()
            target_heading = math.atan2(float(current_goal[1]) - self.pose_y, float(current_goal[0]) - self.pose_x)
            command = self.approach_controller.compute_command(
                self.pose_x,
                self.pose_y,
                self.pose_yaw,
                PoseGoal(float(current_goal[0]), float(current_goal[1]), target_heading),
            )
            return RobotCommand(
                linear=min(0.78, command.linear) * self.speed_limit_scale,
                angular=command.angular,
            )

        if not final_docking_latched and not relaxed_pre_dock_ready and not self.approach_controller.goal_reached(
            self.pose_x,
            self.pose_y,
            self.pose_yaw,
            pre_dock,
        ):
            self._publish_state("SEEK_DOCK")
            command = self.approach_controller.compute_command(self.pose_x, self.pose_y, self.pose_yaw, pre_dock)
            command.linear *= self.speed_limit_scale
            return command

        self._publish_state("FINAL_DOCKING")
        dx = dock_pose.x - self.pose_x
        dy = dock_pose.y - self.pose_y
        local_x = math.cos(self.pose_yaw) * dx + math.sin(self.pose_yaw) * dy
        local_y = -math.sin(self.pose_yaw) * dx + math.cos(self.pose_yaw) * dy
        yaw_error = math.atan2(math.sin(dock_pose.yaw - self.pose_yaw), math.cos(dock_pose.yaw - self.pose_yaw))

        if abs(yaw_error) > 0.16:
            return RobotCommand(linear=0.0, angular=max(-0.85, min(0.85, 2.2 * yaw_error)))

        reverse_speed = max(0.08, min(0.34, abs(local_x) * 0.85))
        command = RobotCommand(
            linear=-reverse_speed * self.speed_limit_scale,
            angular=max(-0.45, min(0.45, -1.8 * local_y + 0.35 * yaw_error)),
        )

        rear_distances = [
            distance
            for angle, distance in zip(self.scan_angles, self.scan_ranges)
            if math.isfinite(distance) and abs(abs(angle) - math.pi) < math.radians(24.0)
        ]
        if rear_distances and min(rear_distances) < 0.10:
            command.linear = max(command.linear, -0.04)
        if self.is_charging or self.final_docking_controller.goal_reached(
            self.pose_x,
            self.pose_y,
            self.pose_yaw,
            dock_pose,
        ):
            return RobotCommand()
        command.linear *= self.speed_limit_scale
        return command

    def _control_loop(self) -> None:
        if not self._current_pose_available():
            self._publish_state("WAITING_FOR_ODOM")
            self._publish_cmd(RobotCommand())
            self._publish_mission_status()
            return

        if self.is_charging and self.mission_mode == "idle" and not self.force_dock_requested:
            self._publish_state("CHARGING")
            self._publish_cmd(RobotCommand())
            self._publish_mission_status()
            return

        if self.force_dock_requested:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if self.post_service_escape_until is not None:
                if now_s < self.post_service_escape_until:
                    self._publish_state("SEEK_DOCK")
                    self._publish_cmd(RobotCommand(linear=-0.16 * self.speed_limit_scale, angular=0.0))
                    self._publish_mission_status()
                    return
                self.post_service_escape_until = None
                if self.mission_mode == "dock_return":
                    self._apply_mission_mode("dock_return", emit_event=False)
                    self._emit_event("Rebuilt dock return route after service escape")
            self._publish_cmd(self._compute_docking_command())
            self._publish_mission_status()
            return

        if self.mission_mode == "idle":
            self._publish_state("WAITING_TASK")
            self._publish_cmd(RobotCommand())
            self._publish_mission_status()
            return

        if self._needs_undock_departure():
            self._publish_state("UNDOCKING")
            command = self._compute_undock_command()
            self._publish_cmd(command)
            self._publish_mission_status()
            return

        if self.rebuild_route_after_undock and self.mission_mode in SERVICE_TARGETS:
            self.route_start_override = (
                float(self.dock.pre_dock_pose[0]),
                float(self.dock.pre_dock_pose[1]),
            )
            self._apply_mission_mode(self.mission_mode, emit_event=False)
            self.route_start_override = None
            self.rebuild_route_after_undock = False
            self._emit_event(f"Rebuilt {SERVICE_TARGETS[self.mission_mode]['label']} route after dock exit")

        if self._service_route_stalled():
            self._publish_state("REPLAN")
            self._rebuild_service_route_from_current_pose()

        if self._service_waypoint_ready_to_advance():
            self.navigator.skip_to_next()
            self._reset_route_progress_tracking()

        service_approach_active = self.state == "TABLE_APPROACH" and self.mission_mode in SERVICE_TARGETS
        if self.mission_mode in SERVICE_TARGETS and (
            service_approach_active
            or self.navigator.route_complete(self.pose_x, self.pose_y)
            or self._ready_for_final_service_approach()
        ):
            table_goal = self._table_service_pose_goal(self.mission_mode)
            approach_goal = self._table_service_approach_goal(self.mission_mode)
            final_goal_reached = self.table_service_controller.goal_reached(
                self.pose_x,
                self.pose_y,
                self.pose_yaw,
                table_goal,
            )
            distance_to_approach = math.hypot(approach_goal.x - self.pose_x, approach_goal.y - self.pose_y)
            distance_from_approach_to_stop = math.hypot(table_goal.x - approach_goal.x, table_goal.y - approach_goal.y)
            using_approach_pose = distance_from_approach_to_stop > 0.10 and distance_to_approach > 0.24
            tracking_goal = approach_goal if using_approach_pose else table_goal
            if not final_goal_reached:
                self._publish_state("TABLE_APPROACH")
                distance_to_table = math.hypot(tracking_goal.x - self.pose_x, tracking_goal.y - self.pose_y)
                heading_to_goal = math.atan2(tracking_goal.y - self.pose_y, tracking_goal.x - self.pose_x)
                heading_error = math.atan2(
                    math.sin(heading_to_goal - self.pose_yaw),
                    math.cos(heading_to_goal - self.pose_yaw),
                )
                yaw_error = math.atan2(
                    math.sin(tracking_goal.yaw - self.pose_yaw),
                    math.cos(tracking_goal.yaw - self.pose_yaw),
                )
                if distance_to_table > 0.38:
                    linear_cap = 0.60 if using_approach_pose else 0.44
                    linear = min(linear_cap, 0.90 * distance_to_table)
                    if abs(heading_error) > 0.75:
                        linear *= 0.30 if using_approach_pose else 0.15
                    elif abs(heading_error) > 0.40:
                        linear *= 0.60 if using_approach_pose else 0.45
                    angular = max(-1.10, min(1.10, 1.85 * heading_error + 0.20 * yaw_error))
                else:
                    if abs(heading_error) > 0.32:
                        linear = 0.0
                    else:
                        linear = min(0.18, 0.85 * distance_to_table)
                        if abs(yaw_error) > 0.60:
                            linear *= 0.45
                    angular = max(-1.10, min(1.10, 1.35 * heading_error + 2.10 * yaw_error))
                command = RobotCommand(linear=linear, angular=angular)
                command.linear *= min(1.0, self.speed_limit_scale)
                self._publish_cmd(command)
                self._publish_mission_status()
                return
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if self.table_hold_started_at is None:
                self.table_hold_started_at = now_s
                self._emit_event(f"{SERVICE_TARGETS[self.mission_mode]['label']} reached; holding service position for 5 seconds")
            if now_s - self.table_hold_started_at < self.table_service_dwell_seconds:
                self._publish_state("TABLE_SERVICE")
                self._publish_cmd(RobotCommand())
                self._publish_mission_status()
                return
            if not self.pending_table_dock_cycle:
                finished_table = self.mission_mode
                self.pending_table_dock_cycle = True
                self.table_hold_started_at = None
                self.force_dock_requested = True
                self.dock_request_reason = "service_complete"
                self.post_service_escape_until = now_s + 1.8
                self._apply_mission_mode("dock_return", emit_event=False)
                self._emit_event(f"{SERVICE_TARGETS[finished_table]['label']} served; returning to auto dock")
            self._publish_cmd(self._compute_docking_command())
            self._publish_mission_status()
            return

        if self.scan_ranges:
            self._publish_state("PATROL")
            command = self.navigator.compute_command(
                self.pose_x,
                self.pose_y,
                self.pose_yaw,
                self.scan_ranges,
                self.scan_angles,
            )
            self._publish_cmd(command)
            self._publish_mission_status()
            return

        self._publish_state("IDLE")
        self._publish_cmd(RobotCommand())
        self._publish_mission_status()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutonomyManager()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
