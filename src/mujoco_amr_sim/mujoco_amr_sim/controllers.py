from dataclasses import dataclass
import math
from typing import Iterable, Sequence


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def diff_drive_inverse(
    linear_velocity: float,
    angular_velocity: float,
    wheel_radius: float,
    wheel_track: float,
) -> tuple[float, float]:
    left = (linear_velocity - 0.5 * angular_velocity * wheel_track) / wheel_radius
    right = (linear_velocity + 0.5 * angular_velocity * wheel_track) / wheel_radius
    return left, right


def diff_drive_forward(
    left_wheel_angular_velocity: float,
    right_wheel_angular_velocity: float,
    wheel_radius: float,
    wheel_track: float,
) -> tuple[float, float]:
    left_linear = left_wheel_angular_velocity * wheel_radius
    right_linear = right_wheel_angular_velocity * wheel_radius
    linear = 0.5 * (left_linear + right_linear)
    angular = (right_linear - left_linear) / wheel_track
    return linear, angular


def integrate_diff_drive(
    pose_x: float,
    pose_y: float,
    yaw: float,
    left_wheel_angular_velocity: float,
    right_wheel_angular_velocity: float,
    wheel_radius: float,
    wheel_track: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    linear, angular = diff_drive_forward(
        left_wheel_angular_velocity,
        right_wheel_angular_velocity,
        wheel_radius,
        wheel_track,
    )
    if abs(angular) < 1e-6:
        next_x = pose_x + linear * math.cos(yaw) * dt
        next_y = pose_y + linear * math.sin(yaw) * dt
        next_yaw = wrap_to_pi(yaw)
        return next_x, next_y, next_yaw, linear, angular

    next_yaw = wrap_to_pi(yaw + angular * dt)
    turn_radius = linear / angular
    next_x = pose_x + turn_radius * (math.sin(next_yaw) - math.sin(yaw))
    next_y = pose_y - turn_radius * (math.cos(next_yaw) - math.cos(yaw))
    return next_x, next_y, next_yaw, linear, angular


@dataclass
class RobotCommand:
    linear: float = 0.0
    angular: float = 0.0


@dataclass
class PoseGoal:
    x: float
    y: float
    yaw: float


class PoseTrackingController:
    def __init__(
        self,
        max_linear: float = 0.7,
        max_angular: float = 1.6,
        position_tolerance: float = 0.20,
        yaw_tolerance: float = 0.20,
    ) -> None:
        self._max_linear = max_linear
        self._max_angular = max_angular
        self._position_tolerance = position_tolerance
        self._yaw_tolerance = yaw_tolerance

    @property
    def position_tolerance(self) -> float:
        return self._position_tolerance

    @property
    def yaw_tolerance(self) -> float:
        return self._yaw_tolerance

    def goal_reached(self, pose_x: float, pose_y: float, yaw: float, goal: PoseGoal) -> bool:
        return (
            math.hypot(goal.x - pose_x, goal.y - pose_y) <= self._position_tolerance
            and abs(wrap_to_pi(goal.yaw - yaw)) <= self._yaw_tolerance
        )

    def compute_command(self, pose_x: float, pose_y: float, yaw: float, goal: PoseGoal) -> RobotCommand:
        dx = goal.x - pose_x
        dy = goal.y - pose_y
        distance = math.hypot(dx, dy)
        heading_to_goal = math.atan2(dy, dx)
        heading_error = wrap_to_pi(heading_to_goal - yaw)
        yaw_error = wrap_to_pi(goal.yaw - yaw)

        linear = clamp(1.0 * distance, 0.0, self._max_linear)
        angular = clamp(2.2 * heading_error + 0.75 * yaw_error, -self._max_angular, self._max_angular)

        if abs(heading_error) > 1.10:
            linear = 0.0
        elif abs(heading_error) > 0.50:
            linear *= 0.58

        if distance < self._position_tolerance * 1.5:
            linear *= 0.45
            angular = clamp(2.4 * yaw_error, -self._max_angular, self._max_angular)

        if self.goal_reached(pose_x, pose_y, yaw, goal):
            return RobotCommand()

        return RobotCommand(linear=linear, angular=angular)


class ObstacleAwareWaypointNavigator:
    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        max_linear: float = 0.7,
        max_angular: float = 1.6,
        goal_tolerance: float = 0.26,
        stop_distance: float = 0.86,
        slow_distance: float = 1.38,
    ) -> None:
        self._waypoints = [tuple(point[:2]) for point in waypoints]
        self._base_max_linear = max_linear
        self._max_angular = max_angular
        self._goal_tolerance = goal_tolerance
        self._stop_distance = stop_distance
        self._slow_distance = slow_distance
        self._critical_stop_distance = max(0.18, stop_distance * 0.72)
        self._speed_scale = 1.0
        self._current_index = 0
        self._loop = True

    @property
    def current_goal(self) -> tuple[float, float] | None:
        if not self._waypoints:
            return None
        return self._waypoints[self._current_index]

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def waypoint_count(self) -> int:
        return len(self._waypoints)

    @property
    def speed_scale(self) -> float:
        return self._speed_scale

    def reset(self) -> None:
        self._current_index = 0

    def set_speed_scale(self, scale: float) -> None:
        self._speed_scale = clamp(scale, 0.20, 1.25)

    def set_goal_tolerance(self, tolerance: float) -> None:
        self._goal_tolerance = max(0.05, float(tolerance))

    def set_max_linear(self, max_linear: float) -> None:
        self._base_max_linear = max(0.05, float(max_linear))

    def set_waypoints(self, waypoints: Sequence[Sequence[float]], loop: bool = True) -> None:
        self._waypoints = [tuple(point[:2]) for point in waypoints]
        self._current_index = 0
        self._loop = loop

    def route_complete(self, pose_x: float, pose_y: float) -> bool:
        if not self._waypoints or self._loop:
            return False
        goal_x, goal_y = self._waypoints[min(self._current_index, len(self._waypoints) - 1)]
        return self._current_index >= len(self._waypoints) - 1 and math.hypot(goal_x - pose_x, goal_y - pose_y) < self._goal_tolerance

    def skip_to_next(self) -> tuple[float, float] | None:
        if not self._waypoints:
            return None
        if self._loop:
            self._current_index = (self._current_index + 1) % len(self._waypoints)
        else:
            self._current_index = min(self._current_index + 1, len(self._waypoints) - 1)
        return self._waypoints[self._current_index]

    def compute_command(
        self,
        pose_x: float,
        pose_y: float,
        yaw: float,
        lidar_ranges: Sequence[float],
        lidar_angles: Sequence[float],
    ) -> RobotCommand:
        if not self._waypoints:
            return RobotCommand()

        goal_x, goal_y = self._waypoints[self._current_index]
        dx = goal_x - pose_x
        dy = goal_y - pose_y
        distance = math.hypot(dx, dy)

        if distance < self._goal_tolerance:
            if not self._loop and self._current_index >= len(self._waypoints) - 1:
                return RobotCommand()
            self._current_index = (self._current_index + 1) % len(self._waypoints) if self._loop else min(
                self._current_index + 1,
                len(self._waypoints) - 1,
            )
            goal_x, goal_y = self._waypoints[self._current_index]
            dx = goal_x - pose_x
            dy = goal_y - pose_y
            distance = math.hypot(dx, dy)

        goal_heading = math.atan2(dy, dx)
        heading_error = wrap_to_pi(goal_heading - yaw)

        linear = clamp(1.05 * distance, 0.0, self._base_max_linear * self._speed_scale)
        angular = clamp(2.3 * heading_error, -self._max_angular, self._max_angular)

        if abs(heading_error) > 0.95:
            linear *= 0.42
        elif abs(heading_error) > 0.50:
            linear *= 0.74

        front_ranges = []
        front_left_ranges = []
        front_right_ranges = []
        left_ranges = []
        right_ranges = []

        for angle, distance_reading in zip(lidar_angles, lidar_ranges):
            if not math.isfinite(distance_reading):
                continue
            if abs(angle) <= math.radians(40.0):
                front_ranges.append(distance_reading)
            if math.radians(8.0) <= angle <= math.radians(75.0):
                front_left_ranges.append(distance_reading)
            if math.radians(-75.0) <= angle <= math.radians(-8.0):
                front_right_ranges.append(distance_reading)
            if math.radians(15.0) <= angle <= math.radians(100.0):
                left_ranges.append(distance_reading)
            if math.radians(-100.0) <= angle <= math.radians(-15.0):
                right_ranges.append(distance_reading)

        if front_ranges:
            closest_front = min(front_ranges)
            left_clearance = sum(left_ranges) / len(left_ranges) if left_ranges else 0.0
            right_clearance = sum(right_ranges) / len(right_ranges) if right_ranges else 0.0
            front_left_clearance = min(front_left_ranges) if front_left_ranges else left_clearance
            front_right_clearance = min(front_right_ranges) if front_right_ranges else right_clearance
            turn_direction = 1.0 if left_clearance >= right_clearance else -1.0

            if closest_front < self._critical_stop_distance:
                reverse_linear = -0.06 if abs(heading_error) < 0.65 else 0.0
                angular = turn_direction * self._max_angular
                return RobotCommand(linear=reverse_linear, angular=angular)

            if closest_front < self._stop_distance:
                linear = min(linear, 0.08)
                obstacle_bias = clamp(front_right_clearance - front_left_clearance, -1.0, 1.0)
                angular = clamp(
                    angular + turn_direction * 0.95 * self._max_angular - 0.35 * obstacle_bias,
                    -self._max_angular,
                    self._max_angular,
                )
            elif closest_front < self._slow_distance:
                slowdown_ratio = clamp(
                    (closest_front - self._stop_distance) / max(self._slow_distance - self._stop_distance, 1e-6),
                    0.18,
                    1.0,
                )
                linear *= slowdown_ratio
                obstacle_bias = clamp(front_right_clearance - front_left_clearance, -1.0, 1.0)
                angular = clamp(angular - 0.40 * obstacle_bias, -self._max_angular, self._max_angular)

        side_margin = min(left_ranges) if left_ranges else float("inf")
        opposite_side_margin = min(right_ranges) if right_ranges else float("inf")
        if side_margin < self._critical_stop_distance:
            angular = clamp(angular - 0.60, -self._max_angular, self._max_angular)
            linear *= 0.45
        if opposite_side_margin < self._critical_stop_distance:
            angular = clamp(angular + 0.60, -self._max_angular, self._max_angular)
            linear *= 0.45

        return RobotCommand(linear=linear, angular=angular)


def finite_ranges(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]
