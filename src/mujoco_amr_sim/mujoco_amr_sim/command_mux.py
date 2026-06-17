from dataclasses import dataclass

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger


@dataclass
class TimedCommand:
    linear: float = 0.0
    angular: float = 0.0
    stamp_ns: int = 0


class CommandMux(Node):
    def __init__(self) -> None:
        super().__init__("command_mux")

        self.declare_parameter("command_timeout_sec", 0.45)
        self.declare_parameter("primary_source", "auto")
        self.timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        primary_source = str(self.get_parameter("primary_source").value).strip().lower()

        self.commands = {
            "manual": TimedCommand(),
            "auto": TimedCommand(),
            "nav": TimedCommand(),
            "rl": TimedCommand(),
        }
        if primary_source not in {"auto", "nav", "rl"}:
            primary_source = "auto"
        self.priority = ["manual", primary_source] + [
            source for source in ("auto", "nav", "rl") if source != primary_source
        ]
        self.emergency_stop_active = False

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.source_pub = self.create_publisher(String, "/cmd_vel_source", 10)
        self.estop_pub = self.create_publisher(Bool, "/safety/emergency_stop_active", 10)
        self.state_pub = self.create_publisher(String, "/safety/state", 10)
        self.event_pub = self.create_publisher(String, "/safety/event_log", 10)

        self.create_subscription(Twist, "/cmd_vel_manual", lambda msg: self._update("manual", msg), 10)
        self.create_subscription(Twist, "/cmd_vel_auto", lambda msg: self._update("auto", msg), 10)
        self.create_subscription(Twist, "/cmd_vel_nav", lambda msg: self._update("nav", msg), 10)
        self.create_subscription(Twist, "/cmd_vel_rl", lambda msg: self._update("rl", msg), 10)

        self.create_service(SetBool, "/safety/set_emergency_stop", self._set_emergency_stop)
        self.create_service(Trigger, "/safety/clear_commands", self._clear_commands)

        self.timer = self.create_timer(0.05, self._publish_selected)
        self._emit_event(f"Command mux initialized with priority {self.priority}")

    def _update(self, source: str, msg: Twist) -> None:
        self.commands[source] = TimedCommand(
            linear=float(msg.linear.x),
            angular=float(msg.angular.z),
            stamp_ns=self.get_clock().now().nanoseconds,
        )

    def _emit_event(self, message: str) -> None:
        self.get_logger().info(message)
        self.event_pub.publish(String(data=message))

    def _set_emergency_stop(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.emergency_stop_active = bool(request.data)
        state_text = "ACTIVE" if self.emergency_stop_active else "CLEARED"
        self._emit_event(f"Emergency stop {state_text}")
        response.success = True
        response.message = f"Emergency stop {state_text.lower()}"
        return response

    def _clear_commands(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.commands = {source: TimedCommand() for source in self.commands}
        self._emit_event("All queued velocity commands cleared")
        response.success = True
        response.message = "Velocity commands cleared"
        return response

    def _publish_selected(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        selected = "idle"
        selected_cmd = TimedCommand()

        if self.emergency_stop_active:
            msg = Twist()
            self.pub.publish(msg)
            self.source_pub.publish(String(data="emergency_stop"))
            self.estop_pub.publish(Bool(data=True))
            self.state_pub.publish(String(data="EMERGENCY_STOP"))
            return

        for source in self.priority:
            command = self.commands[source]
            age = (now_ns - command.stamp_ns) * 1e-9
            if command.stamp_ns > 0 and age <= self.timeout_sec:
                selected = source
                selected_cmd = command
                break

        msg = Twist()
        msg.linear.x = selected_cmd.linear
        msg.angular.z = selected_cmd.angular
        self.pub.publish(msg)
        self.source_pub.publish(String(data=selected))
        self.estop_pub.publish(Bool(data=False))
        self.state_pub.publish(String(data="READY" if selected != "idle" else "IDLE"))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CommandMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
