from dataclasses import dataclass
import heapq
import json
import math

from .mjcf_builder import SERVICE_TARGETS, NAVIGATION_BOXES, WORLD_X, WORLD_Y


SAFE_ROUTE_CLEARANCE = 0.32
MAIN_DOCK_RETURN_CLEARANCE = 0.44
NAV_GRID_RESOLUTION = 0.16
TABLE_SERVICE_STANDOFF = 0.36
SERVICE_ROUTE_CLEARANCE_OVERRIDES: dict[str, float] = {
    "sofa_1": 0.32,
    "sofa_2": 0.32,
    "sofa_3": 0.32,
}
SERVICE_ROUTE_HINTS: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("table_1", "a"): [(-6.05, -4.85), (-6.20, -3.60), (-6.30, -2.20), (-6.35, -0.70), (-7.05, -0.06), (-7.05, 2.50), (-2.90, 1.94)],
    ("table_2", "a"): [(-6.05, -4.85), (-6.20, -3.60), (-6.30, -2.20), (-6.35, -0.70), (-7.05, -0.06), (-7.05, 2.50), (-1.30, 1.22), (0.54, 1.22)],
    ("table_3", "a"): [(-5.95, -4.80), (-5.90, -4.35), (-5.60, -3.35), (-4.2, -2.3), (-1.1, -1.9)],
    ("table_4", "a"): [(-5.95, -4.80), (-5.90, -4.35), (-5.60, -3.35), (-4.9, -1.8)],
    ("table_5", "a"): [(-5.95, -4.80), (-5.90, -4.35), (-5.60, -3.35), (-3.6, -2.6), (1.4, -2.9)],
    ("table_6", "a"): [(-6.05, -4.85), (-6.20, -3.60), (-6.30, -2.20), (-6.35, -0.70), (-6.50, 0.40), (-6.70, 2.55), (-3.30, 3.22), (1.00, 3.52), (3.10, 3.95)],
    ("sofa_1", "a"): [(-6.05, -4.85), (-6.20, -3.60), (-6.30, -2.20), (-6.35, -0.70), (-7.05, 0.32), (-7.05, 2.50), (-2.10, 3.46), (-0.20, 3.52)],
    ("sofa_2", "a"): [(-5.95, -4.80), (-5.90, -4.35), (-5.80, -3.35), (-6.30, -0.80), (-6.45, -0.22)],
    ("sofa_3", "a"): [(-5.95, -4.80), (-5.95, -4.40), (-5.70, -3.40), (-6.6, -2.8), (-6.3, -0.8), (-5.8, 2.8), (-3.6, 3.2), (1.2, 3.45), (4.4, 3.4)],
    ("table_1", "b"): [(6.2, -2.8), (5.8, -0.6), (5.4, 2.4), (3.58, 2.98)],
    ("table_2", "b"): [(6.2, -2.8), (5.8, -0.6), (5.4, 2.4), (3.58, 2.98), (3.15, 2.85)],
    ("table_3", "b"): [(6.2, -2.8), (4.6, -1.8), (2.8, -0.8)],
    ("table_4", "b"): [(6.2, -2.8), (4.6, -1.8), (1.0, -1.8), (-1.2, -1.5)],
    ("table_5", "b"): [(6.1, -2.8), (5.5, -1.8)],
    ("table_6", "b"): [(6.2, -2.8), (5.8, -0.6), (5.4, 2.4)],
    ("sofa_1", "b"): [(6.2, -2.8), (5.8, -0.6), (5.4, 2.4), (3.8, 3.2), (1.82, 3.62)],
    ("sofa_2", "b"): [(6.2, -2.8), (5.8, -0.6), (5.4, 2.4), (3.2, 1.2), (3.0, 0.8), (0.2, 0.8), (-2.6, 0.7)],
    ("sofa_3", "b"): [(6.2, -2.8), (5.8, -0.6), (5.4, 2.4), (7.2, 2.6)],
}


def normalize_service_target_key(target_key: str | None) -> str:
    raw_key = (target_key or "").strip().lower()
    if not raw_key:
        return ""
    compact_key = raw_key.replace("-", "_").replace(" ", "_")
    while "__" in compact_key:
        compact_key = compact_key.replace("__", "_")
    if compact_key in SERVICE_TARGETS:
        return compact_key

    alias_map: dict[str, str] = {}
    for index in range(1, 7):
        alias_map[f"t{index}"] = f"table_{index}"
        alias_map[f"table{index}"] = f"table_{index}"
        alias_map[f"table_{index}"] = f"table_{index}"
    for index in range(1, 4):
        alias_map[f"s{index}"] = f"sofa_{index}"
        alias_map[f"sofa{index}"] = f"sofa_{index}"
        alias_map[f"sofa_{index}"] = f"sofa_{index}"

    if compact_key in alias_map:
        return alias_map[compact_key]

    digits = "".join(character for character in compact_key if character.isdigit())
    if compact_key.startswith("table") and digits:
        normalized = f"table_{digits}"
        if normalized in SERVICE_TARGETS:
            return normalized
    if compact_key.startswith(("sofa", "s")) and digits:
        normalized = f"sofa_{digits}"
        if normalized in SERVICE_TARGETS:
            return normalized
    return compact_key


@dataclass
class DockStationConfig:
    dock_pose: tuple[float, float, float]
    pre_dock_pose: tuple[float, float, float]
    contact_distance: float
    charge_distance: float
    yaw_tolerance_rad: float


def load_waypoints(path: str) -> list[list[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [[float(point[0]), float(point[1])] for point in data]


def _point_is_blocked(point: tuple[float, float], clearance: float) -> bool:
    x, y = point
    boundary_margin = 0.26
    if abs(x) >= WORLD_X - boundary_margin or abs(y) >= WORLD_Y - boundary_margin:
        return True
    for box in NAVIGATION_BOXES:
        if (
            box.size[2] < 0.08
            or box.name == "dock_floor_pad"
            or box.name.startswith("dock_")
            or box.name.startswith("service_dock_")
        ):
            continue
        if abs(x - box.pos[0]) <= box.size[0] + clearance and abs(y - box.pos[1]) <= box.size[1] + clearance:
            return True
    return False


def _segment_is_clear(start: tuple[float, float], goal: tuple[float, float], clearance: float) -> bool:
    distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
    steps = max(2, int(distance / 0.05))
    for step in range(steps + 1):
        ratio = step / steps
        sample = (
            start[0] + (goal[0] - start[0]) * ratio,
            start[1] + (goal[1] - start[1]) * ratio,
        )
        if _point_is_blocked(sample, clearance):
            return False
    return True


def _visibility_anchors(clearance: float) -> list[tuple[float, float]]:
    anchors: list[tuple[float, float]] = []
    corner_margin = max(0.10, 0.5 * clearance)
    for box in NAVIGATION_BOXES:
        if (
            box.size[2] < 0.08
            or box.name == "dock_floor_pad"
            or box.name.startswith("dock_")
            or box.name.startswith("service_dock_")
        ):
            continue
        min_x = box.pos[0] - box.size[0] - clearance - corner_margin
        max_x = box.pos[0] + box.size[0] + clearance + corner_margin
        min_y = box.pos[1] - box.size[1] - clearance - corner_margin
        max_y = box.pos[1] + box.size[1] + clearance + corner_margin
        for anchor in (
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y),
        ):
            if not _point_is_blocked(anchor, clearance):
                anchors.append(anchor)
    unique_anchors: list[tuple[float, float]] = []
    for anchor in anchors:
        if any(math.hypot(anchor[0] - other[0], anchor[1] - other[1]) < 1e-6 for other in unique_anchors):
            continue
        unique_anchors.append(anchor)
    return unique_anchors


def _simplify_route(
    start: tuple[float, float],
    route: list[tuple[float, float]],
    clearance: float,
) -> list[list[float]]:
    if not route:
        return []
    simplified: list[tuple[float, float]] = []
    current = start
    index = 0
    while index < len(route):
        best_index = index
        for probe_index in range(len(route) - 1, index - 1, -1):
            if _segment_is_clear(current, route[probe_index], clearance):
                best_index = probe_index
                break
        current = route[best_index]
        simplified.append(current)
        index = best_index + 1
    return [[point[0], point[1]] for point in simplified]


def service_target_goal(target_key: str, robot_variant: str = "a") -> tuple[float, float, float]:
    normalized_target_key = normalize_service_target_key(target_key)
    target = SERVICE_TARGETS[normalized_target_key]
    table_x, table_y = (float(target["pos"][0]), float(target["pos"][1]))
    variant = str(robot_variant).strip().lower()
    stop_key = "service_stop_b" if variant == "b" else "service_stop_a"
    approach_key = "approach_b" if variant == "b" else "approach_a"
    if stop_key in target:
        goal_x, goal_y = (float(target[stop_key][0]), float(target[stop_key][1]))
    else:
        goal_x, goal_y = (float(target[approach_key][0]), float(target[approach_key][1]))
    goal_yaw = math.atan2(table_y - goal_y, table_x - goal_x)
    return goal_x, goal_y, goal_yaw


def service_approach_goal(target_key: str, robot_variant: str = "a") -> tuple[float, float, float]:
    normalized_target_key = normalize_service_target_key(target_key)
    target = SERVICE_TARGETS[normalized_target_key]
    table_x, table_y = (float(target["pos"][0]), float(target["pos"][1]))
    variant = str(robot_variant).strip().lower()
    approach_key = "approach_b" if variant == "b" else "approach_a"
    stop_key = "service_stop_b" if variant == "b" else "service_stop_a"
    if approach_key in target:
        goal_x, goal_y = (float(target[approach_key][0]), float(target[approach_key][1]))
    else:
        goal_x, goal_y = (float(target[stop_key][0]), float(target[stop_key][1]))
    goal_yaw = math.atan2(table_y - goal_y, table_x - goal_x)
    return goal_x, goal_y, goal_yaw


def build_service_route(
    start: tuple[float, float] | list[float],
    target_key: str,
    robot_variant: str = "a",
    *,
    clearance: float = 0.48,
) -> list[list[float]]:
    normalized_target_key = normalize_service_target_key(target_key)
    clearance = min(clearance, SERVICE_ROUTE_CLEARANCE_OVERRIDES.get(normalized_target_key, clearance))
    start_xy = (float(start[0]), float(start[1]))
    variant = str(robot_variant).strip().lower()
    approach_x, approach_y, _ = service_approach_goal(normalized_target_key, robot_variant)
    stop_x, stop_y, _ = service_target_goal(normalized_target_key, robot_variant)
    hint_points = SERVICE_ROUTE_HINTS.get((normalized_target_key, variant), [])
    if math.hypot(stop_x - start_xy[0], stop_y - start_xy[1]) <= 2.2:
        hint_points = []
    if hint_points:
        nearest_hint_index = min(
            range(len(hint_points)),
            key=lambda index: math.hypot(
                float(hint_points[index][0]) - start_xy[0],
                float(hint_points[index][1]) - start_xy[1],
            ),
        )
        hint_points = hint_points[nearest_hint_index:]

    merged_route: list[list[float]] = []
    segment_start: tuple[float, float] | list[float] = start_xy
    for hint_point in hint_points:
        if _segment_is_clear((float(segment_start[0]), float(segment_start[1])), hint_point, clearance):
            hint_segment = [[float(hint_point[0]), float(hint_point[1])]]
        else:
            hint_segment = build_visibility_route(segment_start, hint_point, clearance=clearance)
        for point in hint_segment:
            normalized_point = [float(point[0]), float(point[1])]
            if merged_route and math.hypot(
                merged_route[-1][0] - normalized_point[0],
                merged_route[-1][1] - normalized_point[1],
            ) < 1e-6:
                continue
            merged_route.append(normalized_point)
        segment_start = hint_point

    if _segment_is_clear((float(segment_start[0]), float(segment_start[1])), (approach_x, approach_y), clearance):
        route_to_approach = [[float(approach_x), float(approach_y)]]
    else:
        route_to_approach = build_visibility_route(segment_start, (approach_x, approach_y), clearance=clearance)
    route_start_for_stop = route_to_approach[-1] if route_to_approach else [float(segment_start[0]), float(segment_start[1])]
    if _segment_is_clear((float(route_start_for_stop[0]), float(route_start_for_stop[1])), (stop_x, stop_y), clearance):
        route_to_stop = [[float(stop_x), float(stop_y)]]
    else:
        route_to_stop = build_visibility_route(route_start_for_stop, (stop_x, stop_y), clearance=clearance)

    for point in [*route_to_approach, *route_to_stop]:
        normalized_point = [float(point[0]), float(point[1])]
        if merged_route and math.hypot(
            merged_route[-1][0] - normalized_point[0],
            merged_route[-1][1] - normalized_point[1],
        ) < 1e-6:
            continue
        merged_route.append(normalized_point)

    if not merged_route:
        merged_route.append([stop_x, stop_y])
    elif math.hypot(merged_route[-1][0] - stop_x, merged_route[-1][1] - stop_y) > 1e-6:
        merged_route.append([stop_x, stop_y])
    preserved_prefix: list[list[float]] = []
    if hint_points:
        final_hint_x, final_hint_y = (float(hint_points[-1][0]), float(hint_points[-1][1]))
        for point in merged_route:
            preserved_prefix.append([float(point[0]), float(point[1])])
            if math.hypot(point[0] - final_hint_x, point[1] - final_hint_y) < 1e-6:
                break
    if preserved_prefix:
        simplified_route = preserved_prefix[:]
        tail_start = (float(preserved_prefix[-1][0]), float(preserved_prefix[-1][1]))
        tail_route = _simplify_route(
            tail_start,
            [tuple(point[:2]) for point in merged_route[len(preserved_prefix):]],
            clearance,
        )
        for point in tail_route:
            if simplified_route and math.hypot(
                simplified_route[-1][0] - point[0],
                simplified_route[-1][1] - point[1],
            ) < 1e-6:
                continue
            simplified_route.append([float(point[0]), float(point[1])])
    else:
        simplified_route = _simplify_route(start_xy, [tuple(point[:2]) for point in merged_route], clearance)
    if not simplified_route:
        return merged_route
    if math.hypot(simplified_route[-1][0] - stop_x, simplified_route[-1][1] - stop_y) > 1e-6:
        simplified_route.append([stop_x, stop_y])
    return simplified_route


def _route_distance(start: tuple[float, float], route: list[list[float]]) -> float:
    total = 0.0
    prev_x, prev_y = float(start[0]), float(start[1])
    for point in route:
        x, y = float(point[0]), float(point[1])
        total += math.hypot(x - prev_x, y - prev_y)
        prev_x, prev_y = x, y
    return total


def build_main_dock_return_route(
    start: tuple[float, float] | list[float],
    pre_dock: tuple[float, float] | list[float],
    *,
    clearance: float = MAIN_DOCK_RETURN_CLEARANCE,
) -> list[list[float]]:
    start_xy = (float(start[0]), float(start[1]))
    pre_dock_xy = (float(pre_dock[0]), float(pre_dock[1]))

    hint_points: list[tuple[float, float]] = []
    x, y = start_xy

    if y >= 1.8:
        if x >= 2.0:
            hint_points.extend([(3.40, 3.30), (0.80, 3.30), (-2.80, 2.90), (-5.40, 2.60), (-6.70, 2.55)])
        elif x >= -2.5:
            hint_points.extend([(-3.80, 2.35), (-5.40, 2.45), (-6.70, 2.55)])
        else:
            hint_points.extend([(-4.80, 2.20), (-6.00, 2.35), (-6.85, 2.55)])
        hint_points.extend([(-6.85, 0.10), (-6.25, -2.20)])
    elif y >= -0.4:
        if x >= 1.8:
            hint_points.extend([(1.00, 0.80), (-2.40, 0.80), (-5.80, 0.10), (-6.25, -2.20)])
        elif x >= -2.5:
            hint_points.extend([(-4.00, 0.45), (-6.00, -0.10), (-6.25, -2.20)])
        else:
            hint_points.extend([(-6.25, -0.30), (-6.25, -2.20)])
    else:
        if x >= 0.8:
            hint_points.extend([(-1.20, -2.90), (-4.90, -3.00)])
        elif x >= -3.2:
            hint_points.extend([(-4.90, -3.00)])
        hint_points.extend([(-5.95, -4.20)])

    merged_route: list[list[float]] = []
    segment_start: tuple[float, float] | list[float] = start_xy
    for hint_point in hint_points:
        if _segment_is_clear((float(segment_start[0]), float(segment_start[1])), hint_point, clearance):
            hint_segment = [[float(hint_point[0]), float(hint_point[1])]]
        else:
            hint_segment = build_visibility_route(segment_start, hint_point, clearance=clearance)
        for point in hint_segment:
            normalized_point = [float(point[0]), float(point[1])]
            if merged_route and math.hypot(
                merged_route[-1][0] - normalized_point[0],
                merged_route[-1][1] - normalized_point[1],
            ) < 1e-6:
                continue
            merged_route.append(normalized_point)
        segment_start = hint_point

    if _segment_is_clear((float(segment_start[0]), float(segment_start[1])), pre_dock_xy, clearance):
        route_to_pre_dock = [[pre_dock_xy[0], pre_dock_xy[1]]]
    else:
        route_to_pre_dock = build_visibility_route(segment_start, pre_dock_xy, clearance=clearance)
    for point in route_to_pre_dock:
        normalized_point = [float(point[0]), float(point[1])]
        if merged_route and math.hypot(
            merged_route[-1][0] - normalized_point[0],
            merged_route[-1][1] - normalized_point[1],
        ) < 1e-6:
            continue
        merged_route.append(normalized_point)

    if not merged_route:
        return [[pre_dock_xy[0], pre_dock_xy[1]]]

    use_mid_dining_shortcut = 0.5 <= start_xy[0] < 2.0 and 0.4 <= start_xy[1] < 1.8
    if use_mid_dining_shortcut:
        mid_tail = [[-1.38, -1.02], [-4.90, -3.00], [-5.95, -4.20], [pre_dock_xy[0], pre_dock_xy[1]]]
        previous = start_xy
        if all(_segment_is_clear(previous if index == 0 else (mid_tail[index - 1][0], mid_tail[index - 1][1]), (point[0], point[1]), clearance) for index, point in enumerate(mid_tail)):
            merged_route = mid_tail

    direct_route = build_visibility_route(start_xy, pre_dock_xy, clearance=clearance)
    use_far_right_shortcut = start_xy[0] >= 2.0 and start_xy[1] >= 1.8
    if use_far_right_shortcut and direct_route and _route_distance(start_xy, direct_route) + 0.75 < _route_distance(start_xy, merged_route):
        merged_route = direct_route
        if use_far_right_shortcut:
            corridor_tail = [[-1.38, -1.02], [-4.90, -3.00], [-5.95, -4.20], [pre_dock_xy[0], pre_dock_xy[1]]]
            candidate_route = [list(point) for point in merged_route[:-1]]
            tail_ok = True
            for point in corridor_tail:
                if candidate_route and math.hypot(candidate_route[-1][0] - point[0], candidate_route[-1][1] - point[1]) <= 0.20:
                    continue
                previous = start_xy if not candidate_route else (candidate_route[-1][0], candidate_route[-1][1])
                if not _segment_is_clear(previous, (point[0], point[1]), clearance):
                    tail_ok = False
                    break
                candidate_route.append([float(point[0]), float(point[1])])
            if tail_ok and candidate_route:
                merged_route = candidate_route

    while len(merged_route) > 1 and math.hypot(merged_route[0][0] - start_xy[0], merged_route[0][1] - start_xy[1]) <= 0.35:
        merged_route.pop(0)
    if math.hypot(merged_route[-1][0] - pre_dock_xy[0], merged_route[-1][1] - pre_dock_xy[1]) > 1e-6:
        merged_route.append([pre_dock_xy[0], pre_dock_xy[1]])
    return merged_route


def table_service_goal(table_key: str, robot_variant: str = "a") -> tuple[float, float, float]:
    return service_target_goal(table_key, robot_variant)


def _grid_bounds(resolution: float) -> tuple[float, float, float, float]:
    boundary_margin = 0.26
    return (
        -WORLD_X + boundary_margin,
        WORLD_X - boundary_margin,
        -WORLD_Y + boundary_margin,
        WORLD_Y - boundary_margin,
    )


def _grid_index(point: tuple[float, float], resolution: float) -> tuple[int, int]:
    min_x, _, min_y, _ = _grid_bounds(resolution)
    return (
        int(round((point[0] - min_x) / resolution)),
        int(round((point[1] - min_y) / resolution)),
    )


def _grid_point(index: tuple[int, int], resolution: float) -> tuple[float, float]:
    min_x, _, min_y, _ = _grid_bounds(resolution)
    return (
        min_x + index[0] * resolution,
        min_y + index[1] * resolution,
    )


def _nearest_free_grid_index(
    point: tuple[float, float],
    resolution: float,
    clearance: float,
) -> tuple[int, int] | None:
    center = _grid_index(point, resolution)
    _, max_x, _, max_y = _grid_bounds(resolution)
    max_ix = int(round((_grid_bounds(resolution)[1] - _grid_bounds(resolution)[0]) / resolution))
    max_iy = int(round((_grid_bounds(resolution)[3] - _grid_bounds(resolution)[2]) / resolution))
    if not _point_is_blocked(_grid_point(center, resolution), clearance):
        return center
    for radius in range(1, 8):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = (center[0] + dx, center[1] + dy)
                if candidate[0] < 0 or candidate[1] < 0 or candidate[0] > max_ix or candidate[1] > max_iy:
                    continue
                candidate_point = _grid_point(candidate, resolution)
                if not _point_is_blocked(candidate_point, clearance):
                    return candidate
    return None


def _nearest_navigable_point(
    point: tuple[float, float],
    clearance: float,
    resolution: float = NAV_GRID_RESOLUTION,
) -> tuple[float, float] | None:
    candidate_index = _nearest_free_grid_index(point, resolution, clearance)
    if candidate_index is None:
        return None
    return _grid_point(candidate_index, resolution)


def _astar_route(
    start: tuple[float, float],
    goal: tuple[float, float],
    clearance: float,
    resolution: float,
) -> list[tuple[float, float]]:
    start_index = _nearest_free_grid_index(start, resolution, clearance)
    goal_index = _nearest_free_grid_index(goal, resolution, clearance)
    if start_index is None or goal_index is None:
        return []

    open_queue: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_queue, (0.0, 0.0, start_index))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost: dict[tuple[int, int], float] = {start_index: 0.0}

    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    while open_queue:
        _, cost_so_far, current = heapq.heappop(open_queue)
        if current == goal_index:
            break
        if cost_so_far > best_cost.get(current, float("inf")):
            continue

        current_point = _grid_point(current, resolution)
        for dx, dy in neighbor_offsets:
            neighbor = (current[0] + dx, current[1] + dy)
            neighbor_point = _grid_point(neighbor, resolution)
            if _point_is_blocked(neighbor_point, clearance):
                continue
            if not _segment_is_clear(current_point, neighbor_point, clearance):
                continue
            step_cost = math.hypot(neighbor_point[0] - current_point[0], neighbor_point[1] - current_point[1])
            next_cost = cost_so_far + step_cost
            if next_cost >= best_cost.get(neighbor, float("inf")):
                continue
            best_cost[neighbor] = next_cost
            came_from[neighbor] = current
            heuristic = math.hypot(goal[0] - neighbor_point[0], goal[1] - neighbor_point[1])
            heapq.heappush(open_queue, (next_cost + heuristic, next_cost, neighbor))

    if goal_index not in best_cost:
        return []

    route: list[tuple[float, float]] = []
    current = goal_index
    while current != start_index:
        route.append(_grid_point(current, resolution))
        current = came_from[current]
    route.reverse()
    return route


def build_visibility_route(
    start: tuple[float, float] | list[float],
    goal: tuple[float, float] | list[float],
    *,
    extra_anchors: tuple[tuple[float, float], ...] = (),
    clearance: float = SAFE_ROUTE_CLEARANCE,
) -> list[list[float]]:
    start_xy = (float(start[0]), float(start[1]))
    goal_xy = (float(goal[0]), float(goal[1]))
    start_route_xy = start_xy
    if _point_is_blocked(start_route_xy, clearance):
        resolved_start = _nearest_navigable_point(start_route_xy, clearance)
        if resolved_start is not None:
            start_route_xy = resolved_start

    goal_route_xy = goal_xy
    if _point_is_blocked(goal_route_xy, clearance):
        resolved_goal = _nearest_navigable_point(goal_route_xy, clearance)
        if resolved_goal is None:
            return [[goal_xy[0], goal_xy[1]]]
        goal_route_xy = resolved_goal

    direct_goal_xy = goal_xy if _segment_is_clear(start_route_xy, goal_xy, clearance) else goal_route_xy
    if _segment_is_clear(start_route_xy, direct_goal_xy, clearance):
        route: list[list[float]] = []
        if math.hypot(start_xy[0] - start_route_xy[0], start_xy[1] - start_route_xy[1]) > 1e-6:
            route.append([start_route_xy[0], start_route_xy[1]])
        route.append([direct_goal_xy[0], direct_goal_xy[1]])
        return route

    astar_path = _astar_route(start_route_xy, goal_route_xy, clearance, NAV_GRID_RESOLUTION)
    if astar_path:
        smoothed_path = _simplify_route(start_route_xy, [*astar_path, goal_route_xy], clearance)
        if smoothed_path:
            if math.hypot(goal_xy[0] - goal_route_xy[0], goal_xy[1] - goal_route_xy[1]) > 1e-6:
                smoothed_path[-1] = [goal_route_xy[0], goal_route_xy[1]]
            return smoothed_path

    nodes: list[tuple[float, float]] = [start_route_xy, goal_route_xy]
    seen: set[tuple[float, float]] = {start_route_xy, goal_route_xy}
    for anchor in (*extra_anchors, *_visibility_anchors(clearance)):
        anchor_xy = (float(anchor[0]), float(anchor[1]))
        if anchor_xy in seen or _point_is_blocked(anchor_xy, clearance):
            continue
        seen.add(anchor_xy)
        nodes.append(anchor_xy)

    graph: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for index, source in enumerate(nodes):
        for target_index in range(index + 1, len(nodes)):
            target = nodes[target_index]
            if not _segment_is_clear(source, target, clearance):
                continue
            edge_cost = math.hypot(target[0] - source[0], target[1] - source[1])
            graph[index].append((target_index, edge_cost))
            graph[target_index].append((index, edge_cost))

    queue: list[tuple[float, int, list[int]]] = [(0.0, 0, [])]
    best_cost: dict[int, float] = {}
    while queue:
        cost, node_index, path = heapq.heappop(queue)
        if node_index in best_cost and best_cost[node_index] <= cost:
            continue
        best_cost[node_index] = cost
        full_path = path + [node_index]
        if node_index == 1:
            raw_route = [nodes[index] for index in full_path[1:]]
            return _simplify_route(start_route_xy, raw_route, clearance)
        for neighbor_index, edge_cost in graph[node_index]:
            heapq.heappush(queue, (cost + edge_cost, neighbor_index, full_path))

    return [[goal_route_xy[0], goal_route_xy[1]]]


def load_dock_config(path: str) -> DockStationConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    yaw_tolerance_deg = float(data.get("yaw_tolerance_deg", 18.0))
    return DockStationConfig(
        dock_pose=tuple(float(value) for value in data["dock_pose"][:3]),
        pre_dock_pose=tuple(float(value) for value in data["pre_dock_pose"][:3]),
        contact_distance=float(data.get("contact_distance", 0.18)),
        charge_distance=float(data.get("charge_distance", 0.22)),
        yaw_tolerance_rad=yaw_tolerance_deg * 3.141592653589793 / 180.0,
    )
