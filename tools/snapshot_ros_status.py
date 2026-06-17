#!/usr/bin/env python3
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class StatusSnapshot(Node):
    def __init__(self) -> None:
        super().__init__("status_snapshot")
        self.mission_status_raw = ""
        self.sim_status_raw = ""
        self.create_subscription(String, "/autonomy/mission_status", self._on_mission_status, 10)
        self.create_subscription(String, "/simulation/status", self._on_sim_status, 10)

    def _on_mission_status(self, msg: String) -> None:
        self.mission_status_raw = msg.data or ""

    def _on_sim_status(self, msg: String) -> None:
        self.sim_status_raw = msg.data or ""


def _decode(raw: str) -> object:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    rclpy.init()
    node = StatusSnapshot()
    end_time = time.time() + max(0.5, args.duration)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while time.time() < end_time:
            executor.spin_once(timeout_sec=0.2)
            if node.mission_status_raw and node.sim_status_raw:
                break
    finally:
        payload = {
            "mission_status": _decode(node.mission_status_raw),
            "simulation_status": _decode(node.sim_status_raw),
        }
        print(json.dumps(payload, separators=(",", ":"), default=str))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
