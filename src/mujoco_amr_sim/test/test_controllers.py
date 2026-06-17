import math
import unittest

from mujoco_amr_sim.controllers import (
    diff_drive_forward,
    integrate_diff_drive,
    ObstacleAwareWaypointNavigator,
    PoseGoal,
    PoseTrackingController,
    diff_drive_inverse,
)


class DiffDriveTests(unittest.TestCase):
    def test_inverse_kinematics_straight(self):
        left, right = diff_drive_inverse(0.5, 0.0, 0.1, 0.4)
        self.assertAlmostEqual(left, 5.0)
        self.assertAlmostEqual(right, 5.0)

    def test_inverse_kinematics_turn(self):
        left, right = diff_drive_inverse(0.0, 1.0, 0.1, 0.4)
        self.assertAlmostEqual(left, -2.0)
        self.assertAlmostEqual(right, 2.0)

    def test_forward_kinematics_straight(self):
        linear, angular = diff_drive_forward(5.0, 5.0, 0.1, 0.4)
        self.assertAlmostEqual(linear, 0.5)
        self.assertAlmostEqual(angular, 0.0)

    def test_forward_kinematics_turn(self):
        linear, angular = diff_drive_forward(-2.0, 2.0, 0.1, 0.4)
        self.assertAlmostEqual(linear, 0.0)
        self.assertAlmostEqual(angular, 1.0)

    def test_integrate_diff_drive_straight(self):
        x, y, yaw, linear, angular = integrate_diff_drive(
            0.0, 0.0, 0.0, 5.0, 5.0, 0.1, 0.4, 1.0
        )
        self.assertAlmostEqual(x, 0.5, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)
        self.assertAlmostEqual(linear, 0.5, places=6)
        self.assertAlmostEqual(angular, 0.0, places=6)

    def test_integrate_diff_drive_arc(self):
        x, y, yaw, linear, angular = integrate_diff_drive(
            0.0, 0.0, 0.0, -2.0, 2.0, 0.1, 0.4, 1.0
        )
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(yaw, 1.0, places=6)
        self.assertAlmostEqual(linear, 0.0, places=6)
        self.assertAlmostEqual(angular, 1.0, places=6)


class NavigatorTests(unittest.TestCase):
    def test_waypoint_drives_forward_when_clear(self):
        navigator = ObstacleAwareWaypointNavigator([[1.0, 0.0]])
        cmd = navigator.compute_command(
            pose_x=0.0,
            pose_y=0.0,
            yaw=0.0,
            lidar_ranges=[float("inf")] * 5,
            lidar_angles=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        self.assertGreater(cmd.linear, 0.0)
        self.assertAlmostEqual(cmd.angular, 0.0, delta=1e-6)

    def test_obstacle_stops_forward_motion(self):
        navigator = ObstacleAwareWaypointNavigator([[1.0, 0.0]])
        cmd = navigator.compute_command(
            pose_x=0.0,
            pose_y=0.0,
            yaw=0.0,
            lidar_ranges=[1.5, 0.30, 0.25, 0.35, 1.6],
            lidar_angles=[-1.2, -0.2, 0.0, 0.2, 1.2],
        )
        self.assertLessEqual(cmd.linear, 0.0)
        self.assertGreater(abs(cmd.angular), 0.0)

    def test_caution_obstacle_slows_and_turns_robot(self):
        navigator = ObstacleAwareWaypointNavigator([[1.5, 0.0]])
        cmd = navigator.compute_command(
            pose_x=0.0,
            pose_y=0.0,
            yaw=0.0,
            lidar_ranges=[1.05, 0.82, 0.78, 0.88, 1.20],
            lidar_angles=[-1.0, -0.3, 0.0, 0.3, 1.0],
        )
        self.assertGreater(cmd.linear, 0.0)
        self.assertLess(cmd.linear, 0.7)
        self.assertGreater(abs(cmd.angular), 0.0)

    def test_heading_error_causes_rotation(self):
        navigator = ObstacleAwareWaypointNavigator([[0.0, 1.0]])
        cmd = navigator.compute_command(
            pose_x=0.0,
            pose_y=0.0,
            yaw=0.0,
            lidar_ranges=[float("inf")] * 7,
            lidar_angles=[math.radians(angle) for angle in (-90, -60, -30, 0, 30, 60, 90)],
        )
        self.assertGreater(cmd.angular, 0.0)


class PoseControllerTests(unittest.TestCase):
    def test_pose_controller_drives_toward_goal(self):
        controller = PoseTrackingController()
        cmd = controller.compute_command(0.0, 0.0, 0.0, PoseGoal(1.0, 0.0, 0.0))
        self.assertGreater(cmd.linear, 0.0)
        self.assertAlmostEqual(cmd.angular, 0.0, delta=1e-6)

    def test_pose_controller_stops_at_goal(self):
        controller = PoseTrackingController(position_tolerance=0.1, yaw_tolerance=0.1)
        cmd = controller.compute_command(1.0, 1.0, 0.0, PoseGoal(1.02, 1.01, 0.02))
        self.assertAlmostEqual(cmd.linear, 0.0, delta=1e-9)
        self.assertAlmostEqual(cmd.angular, 0.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
