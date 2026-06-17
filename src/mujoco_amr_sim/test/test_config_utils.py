import math
from pathlib import Path
import unittest

from mujoco_amr_sim.config_utils import (
    MAIN_DOCK_RETURN_CLEARANCE,
    SAFE_ROUTE_CLEARANCE,
    _point_is_blocked,
    build_main_dock_return_route,
    build_service_route,
    build_visibility_route,
    load_dock_config,
    normalize_service_target_key,
    table_service_goal,
)
from mujoco_amr_sim.mjcf_builder import (
    DOCK_X,
    DOCK_Y,
    HOTEL_TABLES,
    MAIN_BASE_X_RANGE,
    MAIN_BASE_Y_RANGE,
    ROBOT2_START_X,
    ROBOT2_START_Y,
    ROBOT_START_X,
    ROBOT_START_Y,
    SOFA_SPOTS,
    SERVICE_DOCK_X,
    SERVICE_DOCK_Y,
)


class TableRouteTests(unittest.TestCase):
    def test_target_aliases_normalize_to_canonical_keys(self):
        self.assertEqual(normalize_service_target_key("table 4"), "table_4")
        self.assertEqual(normalize_service_target_key("table4"), "table_4")
        self.assertEqual(normalize_service_target_key("T4"), "table_4")
        self.assertEqual(normalize_service_target_key("s1"), "sofa_1")
        self.assertEqual(normalize_service_target_key("sofa 3"), "sofa_3")

    def test_all_table_service_stop_points_are_navigable(self):
        for table_key in HOTEL_TABLES:
            for robot_variant in ("a", "b"):
                goal_x, goal_y, _ = table_service_goal(table_key, robot_variant)
                self.assertFalse(_point_is_blocked((goal_x, goal_y), SAFE_ROUTE_CLEARANCE))

    def test_main_robot_routes_end_at_safe_table_goals(self):
        for table_key, table in HOTEL_TABLES.items():
            goal_x, goal_y, _ = table_service_goal(table_key, "a")
            route = build_visibility_route((ROBOT_START_X, ROBOT_START_Y), (goal_x, goal_y))
            self.assertGreaterEqual(len(route), 1)
            self.assertEqual(route[-1], [goal_x, goal_y])
            for point in route:
                self.assertFalse(_point_is_blocked((point[0], point[1]), SAFE_ROUTE_CLEARANCE))

    def test_service_routes_stage_around_furniture_and_end_at_goal(self):
        for table_key in HOTEL_TABLES:
            route = build_service_route((ROBOT_START_X, ROBOT_START_Y), table_key, "a")
            goal_x, goal_y, _ = table_service_goal(table_key, "a")
            self.assertGreaterEqual(len(route), 1)
            self.assertEqual(route[-1], [goal_x, goal_y])
            for point in route:
                self.assertFalse(_point_is_blocked((point[0], point[1]), SAFE_ROUTE_CLEARANCE))

    def test_table_1_main_route_prefers_left_service_aisle(self):
        route = build_service_route((ROBOT_START_X, ROBOT_START_Y), "table_1", "a")
        self.assertGreaterEqual(len(route), 6)
        self.assertLess(route[0][0], -4.8)
        self.assertLess(route[0][1], -2.0)
        self.assertLess(route[1][0], -4.8)
        self.assertLess(route[1][1], -0.5)
        self.assertTrue(any(point[0] < -6.7 and point[1] < 0.2 for point in route))
        self.assertTrue(any(point[0] < -6.7 and point[1] > 1.4 for point in route))
        self.assertTrue(any(point[0] > -4.0 and point[1] > 2.4 for point in route))

    def test_service_robot_routes_end_at_safe_table_goals(self):
        for table_key, table in HOTEL_TABLES.items():
            goal_x, goal_y, _ = table_service_goal(table_key, "b")
            route = build_visibility_route((ROBOT2_START_X, ROBOT2_START_Y), (goal_x, goal_y))
            self.assertGreaterEqual(len(route), 1)
            self.assertEqual(route[-1], [goal_x, goal_y])
            for point in route:
                self.assertFalse(_point_is_blocked((point[0], point[1]), SAFE_ROUTE_CLEARANCE))

    def test_table_2_service_robot_route_prefers_right_service_aisle(self):
        route = build_service_route((ROBOT2_START_X, ROBOT2_START_Y), "table_2", "b")
        self.assertGreaterEqual(len(route), 3)
        self.assertGreater(route[0][0], 5.2)
        self.assertTrue(any(point[1] > 2.0 for point in route))

    def test_sofa_service_goals_are_navigable(self):
        for sofa_key in SOFA_SPOTS:
            for robot_variant in ("a", "b"):
                goal_x, goal_y, _ = table_service_goal(sofa_key, robot_variant)
                self.assertFalse(_point_is_blocked((goal_x, goal_y), SAFE_ROUTE_CLEARANCE))

    def test_service_robot_sofa_routes_end_at_safe_goals(self):
        for sofa_key in SOFA_SPOTS:
            goal_x, goal_y, _ = table_service_goal(sofa_key, "b")
            route = build_service_route((ROBOT2_START_X, ROBOT2_START_Y), sofa_key, "b")
            self.assertGreaterEqual(len(route), 1)
            self.assertEqual(route[-1], [goal_x, goal_y])

    def test_table_service_goals_face_the_table(self):
        for table_key, table in HOTEL_TABLES.items():
            table_x, table_y = table["pos"]
            for robot_variant in ("a", "b"):
                goal_x, goal_y, goal_yaw = table_service_goal(table_key, robot_variant)
                self.assertFalse(_point_is_blocked((goal_x, goal_y), SAFE_ROUTE_CLEARANCE))
                heading_error = math.atan2(table_y - goal_y, table_x - goal_x) - goal_yaw
                self.assertAlmostEqual(math.sin(heading_error), 0.0, delta=1e-6)
                self.assertAlmostEqual(math.cos(heading_error), 1.0, delta=1e-6)

    def test_main_dock_points_are_navigable(self):
        self.assertFalse(_point_is_blocked((DOCK_X, DOCK_Y), SAFE_ROUTE_CLEARANCE))
        route = build_visibility_route((ROBOT_START_X, ROBOT_START_Y), (DOCK_X + 1.20, DOCK_Y))
        self.assertGreaterEqual(len(route), 1)
        self.assertEqual(route[-1], [DOCK_X + 1.20, DOCK_Y])

    def test_main_base_joint_ranges_cover_service_floor(self):
        for target in list(HOTEL_TABLES.values()) + list(SOFA_SPOTS.values()):
            for point in (target["approach_a"], target["service_stop_a"]):
                qx = point[0] - ROBOT_START_X
                qy = point[1] - ROBOT_START_Y
                self.assertGreaterEqual(qx, MAIN_BASE_X_RANGE[0])
                self.assertLessEqual(qx, MAIN_BASE_X_RANGE[1])
                self.assertGreaterEqual(qy, MAIN_BASE_Y_RANGE[0])
                self.assertLessEqual(qy, MAIN_BASE_Y_RANGE[1])

    def test_main_dock_faces_outward_toward_pre_dock(self):
        dock_config = load_dock_config(
            str(Path(__file__).resolve().parents[1] / "config" / "dock_station.json")
        )
        dock_x, dock_y, dock_yaw = dock_config.dock_pose
        pre_dock_x, pre_dock_y, _ = dock_config.pre_dock_pose
        forward_progress = (
            (pre_dock_x - dock_x) * math.cos(dock_yaw)
            + (pre_dock_y - dock_y) * math.sin(dock_yaw)
        )
        self.assertGreater(forward_progress, 0.8)

    def test_main_dock_return_route_prefers_short_clear_route(self):
        dock_config = load_dock_config(
            str(Path(__file__).resolve().parents[1] / "config" / "dock_station.json")
        )
        start = (4.4, 3.4)
        route = build_main_dock_return_route(start, dock_config.pre_dock_pose[:2])
        self.assertLessEqual(len(route), 8)
        self.assertEqual(route[-1], [dock_config.pre_dock_pose[0], dock_config.pre_dock_pose[1]])
        total_distance = 0.0
        previous = start
        for point in route:
            self.assertFalse(_point_is_blocked((point[0], point[1]), MAIN_DOCK_RETURN_CLEARANCE))
            total_distance += math.hypot(point[0] - previous[0], point[1] - previous[1])
            previous = (point[0], point[1])
        self.assertLess(total_distance, 18.0)

    def test_main_dock_return_route_skips_near_start_waypoints(self):
        dock_config = load_dock_config(
            str(Path(__file__).resolve().parents[1] / "config" / "dock_station.json")
        )
        start = (-2.349, 1.75)
        route = build_main_dock_return_route(start, dock_config.pre_dock_pose[:2])
        self.assertGreaterEqual(len(route), 2)
        self.assertGreater(math.hypot(route[0][0] - start[0], route[0][1] - start[1]), 0.35)
        self.assertEqual(route[-1], [dock_config.pre_dock_pose[0], dock_config.pre_dock_pose[1]])

    def test_service_dock_points_are_navigable(self):
        self.assertFalse(_point_is_blocked((SERVICE_DOCK_X, SERVICE_DOCK_Y), SAFE_ROUTE_CLEARANCE))
        route = build_visibility_route((ROBOT2_START_X, ROBOT2_START_Y), (SERVICE_DOCK_X - 0.55, SERVICE_DOCK_Y))
        self.assertGreaterEqual(len(route), 1)
        self.assertEqual(route[-1], [SERVICE_DOCK_X - 0.55, SERVICE_DOCK_Y])

    def test_blocked_goals_are_resolved_to_nearest_safe_route_endpoint(self):
        blocked_goal = (HOTEL_TABLES["table_1"]["pos"][0], HOTEL_TABLES["table_1"]["pos"][1])
        self.assertTrue(_point_is_blocked(blocked_goal, SAFE_ROUTE_CLEARANCE))
        route = build_visibility_route((ROBOT_START_X, ROBOT_START_Y), blocked_goal)
        self.assertGreaterEqual(len(route), 1)
        self.assertNotEqual(route[-1], [blocked_goal[0], blocked_goal[1]])
        self.assertFalse(_point_is_blocked((route[-1][0], route[-1][1]), SAFE_ROUTE_CLEARANCE))


if __name__ == "__main__":
    unittest.main()
