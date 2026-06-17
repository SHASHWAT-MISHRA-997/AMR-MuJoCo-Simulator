#!/usr/bin/env python3
import argparse
import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from mujoco_amr_sim.config_utils import normalize_service_target_key, service_target_goal


class ServiceTargetChecker(Node):
    def __init__(self, robot: str, target: str) -> None:
        super().__init__("service_target_checker")
        self.robot = robot
        self.target = normalize_service_target_key(target)
        self.goal_x, self.goal_y, self.goal_yaw = service_target_goal(self.target, robot)
        self.command_topic = "/service_amr/mission_command" if robot == "b" else "/operator/amr_a_mission_command"

        self.pub = self.create_publisher(
            String,
            self.command_topic,
            10,
        )
        self.create_subscription(String, "/autonomy/state", self._on_autonomy_state, 10)
        self.create_subscription(String, "/autonomy/mission_status", self._on_mission_status, 10)
        self.create_subscription(String, "/simulation/status", self._on_sim_status, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_ground_truth, 10)
        self.create_subscription(
            String,
            self.command_topic,
            self._on_command_echo,
            10,
        )

        self.autonomy_state = ""
        self.autonomy_status: dict[str, object] = {}
        self.sim_status: dict[str, object] = {}
        self.main_pose: dict[str, float] = {}
        self.last_publish_time = 0.0
        self.publish_count = 0
        self.command_echo_count = 0
        self.last_command_echo = ""
        self.ready_since = 0.0
        self.dispatch_started_at = 0.0
        self.trace: list[dict[str, object]] = []
        self.last_trace_time = 0.0

    def ready_for_dispatch(self) -> bool:
        if not isinstance(self.sim_status, dict) or not self.sim_status:
            self.ready_since = 0.0
            return False
        if self.pub.get_subscription_count() <= 0:
            self.ready_since = 0.0
            return False
        if self.robot == "a":
            ready_now = self.autonomy_state not in {"", "WAITING_FOR_ODOM"}
        else:
            ready_now = True
        if not ready_now:
            self.ready_since = 0.0
            return False
        now_s = time.time()
        if self.ready_since <= 0.0:
            self.ready_since = now_s
            return False
        return now_s - self.ready_since >= 2.0

    def _on_autonomy_state(self, msg: String) -> None:
        self.autonomy_state = str(msg.data)

    def _on_mission_status(self, msg: String) -> None:
        try:
            self.autonomy_status = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            self.autonomy_status = {"raw": msg.data}

    def _on_sim_status(self, msg: String) -> None:
        try:
            self.sim_status = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            self.sim_status = {"raw": msg.data}

    def _on_ground_truth(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.main_pose = {
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "yaw": math.atan2(siny_cosp, cosy_cosp),
        }

    def _on_command_echo(self, msg: String) -> None:
        self.command_echo_count += 1
        self.last_command_echo = str(msg.data)

    def publish_command_if_needed(self, now_s: float) -> None:
        if not self.ready_for_dispatch():
            return
        if self.dispatch_started_at <= 0.0:
            self.dispatch_started_at = now_s
        if self.last_publish_time > 0.0 and now_s - self.last_publish_time < 2.0:
            return
        self.pub.publish(String(data=self.target))
        self.last_publish_time = now_s
        self.publish_count += 1

    def mission_acknowledged(self) -> bool:
        if self.robot == "a":
            sim_mode = str(self.sim_status.get("mission_mode", "")).lower() if isinstance(self.sim_status, dict) else ""
            auto_mode = str(self.autonomy_status.get("mission_mode", "")).lower()
            return sim_mode == self.target or auto_mode == self.target
        service_status = self.sim_status.get("service_amr")
        return isinstance(service_status, dict) and str(service_status.get("mission_mode", "")).lower() == self.target

    def current_pose(self) -> tuple[float, float, float] | None:
        if self.robot == "a":
            if not self.main_pose:
                return None
            return self.main_pose["x"], self.main_pose["y"], self.main_pose["yaw"]
        service_status = self.sim_status.get("service_amr")
        if not isinstance(service_status, dict):
            return None
        pose = service_status.get("pose")
        if not isinstance(pose, dict):
            return None
        try:
            return float(pose["x"]), float(pose["y"]), float(pose["yaw"])
        except (KeyError, TypeError, ValueError):
            return None

    def pose_metrics(self) -> dict[str, float] | None:
        pose = self.current_pose()
        if pose is None:
            return None
        x, y, yaw = pose
        return {
            "x": round(x, 3),
            "y": round(y, 3),
            "yaw": round(yaw, 3),
            "distance_to_goal": round(math.hypot(self.goal_x - x, self.goal_y - y), 3),
            "yaw_error": round(abs(math.atan2(math.sin(self.goal_yaw - yaw), math.cos(self.goal_yaw - yaw))), 3),
        }

    def success(self) -> bool:
        metrics = self.pose_metrics()
        if metrics is None:
            return False
        if self.robot == "a":
            return (
                self.mission_acknowledged()
                and self.autonomy_state == "TABLE_SERVICE"
                and metrics["distance_to_goal"] <= 0.12
                and metrics["yaw_error"] <= 0.20
            )
        service_status = self.sim_status.get("service_amr")
        service_state = str(service_status.get("state", "")) if isinstance(service_status, dict) else ""
        return (
            self.mission_acknowledged()
            and service_state == "SERVING"
            and metrics["distance_to_goal"] <= 0.12
            and metrics["yaw_error"] <= 0.20
        )

    def maybe_record_trace(self) -> None:
        now_s = time.time()
        if now_s - self.last_trace_time < 2.0:
            return
        self.last_trace_time = now_s
        trace_item = {
            "t": round(now_s, 2),
            "autonomy_state": self.autonomy_state,
            "ack": self.mission_acknowledged(),
            "pose_metrics": self.pose_metrics(),
            "publish_count": self.publish_count,
            "mission_command_subscribers": self.pub.get_subscription_count(),
            "command_echo_count": self.command_echo_count,
            "last_command_echo": self.last_command_echo,
        }
        if isinstance(self.sim_status, dict):
            trace_item["sim_main"] = {
                "mission_mode": self.sim_status.get("mission_mode"),
                "autonomy_state": self.sim_status.get("autonomy_state"),
                "cmd_source": self.sim_status.get("cmd_source"),
                "command": self.sim_status.get("command"),
                "pose": self.sim_status.get("pose"),
                "joint_state_main": self.sim_status.get("joint_state_main"),
            }
        if isinstance(self.autonomy_status, dict):
            trace_item["autonomy_status"] = {
                "mission_mode": self.autonomy_status.get("mission_mode"),
                "state": self.autonomy_status.get("state"),
                "waypoint_index": self.autonomy_status.get("waypoint_index"),
                "waypoint_count": self.autonomy_status.get("waypoint_count"),
                "current_goal": self.autonomy_status.get("current_goal"),
                "pose": self.autonomy_status.get("pose"),
            }
        self.trace.append(trace_item)
        self.trace = self.trace[-25:]

    def snapshot(self) -> dict[str, object]:
        service_status = self.sim_status.get("service_amr") if isinstance(self.sim_status, dict) else None
        sim_main = None
        if isinstance(self.sim_status, dict):
            sim_main = {
                "mission_mode": self.sim_status.get("mission_mode"),
                "autonomy_state": self.sim_status.get("autonomy_state"),
                "cmd_source": self.sim_status.get("cmd_source"),
                "command": self.sim_status.get("command"),
                "pose": self.sim_status.get("pose"),
                "joint_state_main": self.sim_status.get("joint_state_main"),
                "dock_contact": self.sim_status.get("dock_contact"),
                "is_charging": self.sim_status.get("is_charging"),
            }
        return {
            "robot": self.robot,
            "target": self.target,
            "goal": {
                "x": round(self.goal_x, 3),
                "y": round(self.goal_y, 3),
                "yaw": round(self.goal_yaw, 3),
            },
            "acknowledged": self.mission_acknowledged(),
            "autonomy_state": self.autonomy_state,
            "autonomy_status": self.autonomy_status,
            "sim_main": sim_main,
            "service_amr": service_status,
            "pose_metrics": self.pose_metrics(),
            "publish_count": self.publish_count,
            "last_publish_time": round(self.last_publish_time, 2) if self.last_publish_time > 0.0 else None,
            "dispatch_started_at": round(self.dispatch_started_at, 2) if self.dispatch_started_at > 0.0 else None,
            "mission_command_subscribers": self.pub.get_subscription_count(),
            "command_echo_count": self.command_echo_count,
            "last_command_echo": self.last_command_echo,
            "trace": list(self.trace),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether an AMR reaches a given service target live.")
    parser.add_argument("--robot", choices=("a", "b"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ack-timeout", type=float, default=20.0)
    parser.add_argument("--success-timeout", type=float, default=120.0)
    parser.add_argument("--observe-only", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = ServiceTargetChecker(args.robot, args.target)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    started_at = time.time()
    success_deadline = started_at + args.success_timeout

    success = False
    try:
        while time.time() < success_deadline:
            executor.spin_once(timeout_sec=0.2)
            node.maybe_record_trace()
            now_s = time.time()
            if not args.observe_only and not node.mission_acknowledged():
                node.publish_command_if_needed(now_s)
            if (
                not args.observe_only
                and not node.mission_acknowledged()
                and node.dispatch_started_at > 0.0
                and now_s > node.dispatch_started_at + args.ack_timeout
            ):
                break
            if node.success():
                success = True
                break
    finally:
        snapshot = node.snapshot()
        snapshot["success"] = success
        print(json.dumps(snapshot, separators=(",", ":"), default=str))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
