from dataclasses import dataclass
import math
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


WORLD_X = 9.8
WORLD_Y = 7.2
WORLD_NAV_MARGIN = 0.26
WHEEL_RADIUS = 0.085
WHEEL_TRACK = 0.52
LIDAR_MAX_RANGE = 14.0
DOCK_X = -7.10
DOCK_Y = -5.10
SERVICE_DOCK_X = 6.95
SERVICE_DOCK_Y = -5.05
SERVICE_DOCK_YAW = 0.0
ROBOT_START_X = DOCK_X
ROBOT_START_Y = DOCK_Y
ROBOT2_START_X = SERVICE_DOCK_X
ROBOT2_START_Y = SERVICE_DOCK_Y
MAIN_BASE_X_RANGE = (
    -WORLD_X + WORLD_NAV_MARGIN - ROBOT_START_X,
    WORLD_X - WORLD_NAV_MARGIN - ROBOT_START_X,
)
MAIN_BASE_Y_RANGE = (
    -WORLD_Y + WORLD_NAV_MARGIN - ROBOT_START_Y,
    WORLD_Y - WORLD_NAV_MARGIN - ROBOT_START_Y,
)
PICK_STATION_X = 1.20
PICK_STATION_Y = -3.45
PLACE_STATION_X = 6.35
PLACE_STATION_Y = 3.90
TABLE_SERVICE_STANDOFF_VISUAL = 0.36
HOTEL_TABLES = {
    "table_1": {
        "label": "Table 1",
        "pos": (-1.20, 2.30),
        "label_pos": (-0.98, 2.18),
        "approach_a": (-2.90, 1.94),
        "approach_b": (-1.90, 3.25),
        "service_stop_a": (-2.35, 1.75),
        "service_stop_b": (-1.90, 3.25),
    },
    "table_2": {
        "label": "Table 2",
        "pos": (2.20, 2.00),
        "label_pos": (1.98, 1.82),
        "approach_a": (0.54, 1.22),
        "approach_b": (3.15, 2.85),
        "service_stop_a": (1.15, 1.25),
        "service_stop_b": (3.15, 2.85),
    },
    "table_3": {
        "label": "Table 3",
        "pos": (1.25, -0.80),
        "label_pos": (1.25, -0.62),
        "approach_a": (0.10, -1.50),
        "approach_b": (2.60, -0.80),
        "service_stop_a": (0.10, -1.50),
        "service_stop_b": (1.80, 0.09),
    },
    "table_4": {
        "label": "Table 4",
        "pos": (-3.05, -0.45),
        "label_pos": (-3.22, -0.28),
        "approach_a": (-4.20, -1.20),
        "approach_b": (-2.10, -1.30),
        "service_stop_a": (-4.20, -1.20),
        "service_stop_b": (-2.10, -1.30),
    },
    "table_5": {
        "label": "Table 5",
        "pos": (4.30, -2.10),
        "label_pos": (4.02, -2.28),
        "approach_a": (2.50, -2.85),
        "approach_b": (5.35, -1.25),
        "service_stop_a": (3.10, -2.85),
        "service_stop_b": (5.35, -1.25),
    },
    "table_6": {
        "label": "Table 6",
        "pos": (4.60, 4.10),
        "label_pos": (4.24, 4.04),
        "approach_a": (3.80, 4.00),
        "approach_b": (4.06, 2.50),
        "service_stop_a": (3.758, 4.196),
        "service_stop_b": (3.74, 3.58),
    },
}

SOFA_SPOTS = {
    "sofa_1": {
        "label": "SOFA 1",
        "pos": (0.85, 4.55),
        "label_pos": (0.62, 4.94),
        "approach_a": (-0.20, 3.92),
        "approach_b": (1.82, 3.62),
        "service_stop_a": (0.10, 3.65),
        "service_stop_b": (1.90, 3.40),
    },
    "sofa_2": {
        "label": "SOFA 2",
        "pos": (-4.95, 1.20),
        "label_pos": (-5.30, 1.58),
        "approach_a": (-6.45, -0.22),
        "approach_b": (-2.60, 0.70),
        "service_stop_a": (-6.30, -0.10),
        "service_stop_b": (-3.20, 0.60),
    },
    "sofa_3": {
        "label": "SOFA 3",
        "pos": (6.45, 4.12),
        "label_pos": (6.76, 4.70),
        "approach_a": (4.72, 3.00),
        "approach_b": (7.20, 2.60),
        "service_stop_a": (5.18, 3.14),
        "service_stop_b": (7.80, 2.60),
    },
}

SERVICE_TARGETS = {**HOTEL_TABLES, **SOFA_SPOTS}


@dataclass(frozen=True)
class SceneBox:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class LidarMount:
    name: str
    frame_id: str
    pos: tuple[float, float, float]
    yaw_rad: float


LIDAR_MOUNTS = (
    LidarMount(
        name="front_left",
        frame_id="front_left_lidar_frame",
        pos=(0.34, 0.25, 0.44),
        yaw_rad=math.radians(45.0),
    ),
    LidarMount(
        name="rear_right",
        frame_id="rear_right_lidar_frame",
        pos=(-0.28, -0.25, 0.44),
        yaw_rad=math.radians(225.0),
    ),
)


HOUSE_BOXES = (
    SceneBox("north_wall", (0.0, WORLD_Y, 0.70), (WORLD_X, 0.14, 0.70), (0.44, 0.46, 0.49, 1.0)),
    SceneBox("south_wall", (0.0, -WORLD_Y, 0.70), (WORLD_X, 0.14, 0.70), (0.44, 0.46, 0.49, 1.0)),
    SceneBox("east_wall", (WORLD_X, 0.0, 0.70), (0.14, WORLD_Y, 0.70), (0.44, 0.46, 0.49, 1.0)),
    SceneBox("west_wall", (-WORLD_X, 0.0, 0.70), (0.14, WORLD_Y, 0.70), (0.44, 0.46, 0.49, 1.0)),
    SceneBox("reception_counter", (-6.55, 5.10, 0.55), (1.30, 0.34, 0.55), (0.35, 0.25, 0.18, 1.0)),
    SceneBox("dining_table_a", (-1.20, 2.30, 0.40), (0.55, 0.55, 0.40), (0.38, 0.28, 0.18, 1.0)),
    SceneBox("dining_table_b", (2.20, 2.00, 0.40), (0.58, 0.58, 0.40), (0.38, 0.28, 0.18, 1.0)),
    SceneBox("dining_table_c", (1.25, -0.80, 0.40), (0.55, 0.55, 0.40), (0.38, 0.28, 0.18, 1.0)),
    SceneBox("dining_table_d", (-3.05, -0.45, 0.40), (0.52, 0.52, 0.40), (0.38, 0.28, 0.18, 1.0)),
    SceneBox("dining_table_e", (4.30, -2.10, 0.40), (0.52, 0.52, 0.40), (0.38, 0.28, 0.18, 1.0)),
    SceneBox("dining_table_f", (4.60, 4.10, 0.40), (0.52, 0.52, 0.40), (0.38, 0.28, 0.18, 1.0)),
    SceneBox("kitchen_counter_a", (-2.36, -4.99, 0.56), (1.65, 0.30, 0.56), (0.46, 0.48, 0.50, 1.0)),
    SceneBox("kitchen_counter_b", (0.69, -4.99, 0.56), (1.55, 0.30, 0.56), (0.46, 0.48, 0.50, 1.0)),
    SceneBox("dock_backplate", (DOCK_X - 0.72, DOCK_Y, 0.58), (0.09, 0.86, 0.58), (0.08, 0.12, 0.18, 1.0)),
    SceneBox("dock_floor_pad", (DOCK_X - 0.26, DOCK_Y, 0.035), (0.42, 0.92, 0.035), (0.12, 0.76, 0.28, 1.0)),
    SceneBox("dock_guide_left", (DOCK_X + 0.08, DOCK_Y + 0.56, 0.16), (0.24, 0.03, 0.16), (0.98, 0.84, 0.14, 1.0)),
    SceneBox("dock_guide_right", (DOCK_X + 0.08, DOCK_Y - 0.56, 0.16), (0.24, 0.03, 0.16), (0.98, 0.84, 0.14, 1.0)),
)


VISIBLE_COLLISION_BOXES = {box.name for box in HOUSE_BOXES}

CHAIR_NAV_BOXES = (
    SceneBox("chair_nav_a1", (-2.02, 2.30, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_a2", (-0.38, 2.30, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_b1", (1.26, 2.00, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_b2", (3.14, 2.00, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_c1", (0.36, -0.80, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_c2", (2.14, -0.80, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_d1", (-3.92, -0.45, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_d2", (-2.18, -0.45, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_e1", (3.44, -2.10, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_e2", (5.18, -2.10, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_f1", (4.60, 5.06, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("chair_nav_f2", (4.60, 3.14, 0.40), (0.22, 0.22, 0.40), (0.0, 0.0, 0.0, 0.0)),
)

GUEST_NAV_BOXES = ()

SOFA_NAV_BOXES = (
    SceneBox("sofa_nav_1", (0.85, 4.55, 0.42), (1.18, 0.56, 0.42), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("sofa_nav_1_table", (0.85, 3.58, 0.28), (0.40, 0.28, 0.28), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("sofa_nav_2", (-4.95, 1.20, 0.42), (1.34, 0.76, 0.42), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("sofa_nav_2_table", (-4.95, 0.18, 0.28), (0.50, 0.30, 0.28), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("sofa_nav_3", (6.75, 3.86, 0.42), (0.98, 0.84, 0.42), (0.0, 0.0, 0.0, 0.0)),
    SceneBox("sofa_nav_3_table", (6.45, 2.98, 0.28), (0.42, 0.28, 0.28), (0.0, 0.0, 0.0, 0.0)),
)

DECOR_NAV_BOXES = ()

NAVIGATION_BOXES = HOUSE_BOXES + CHAIR_NAV_BOXES + SOFA_NAV_BOXES + DECOR_NAV_BOXES


def _service_stop_pose(target_key: str, robot_variant: str = "a") -> tuple[float, float, float]:
    target = SERVICE_TARGETS[target_key]
    target_x, target_y = float(target["pos"][0]), float(target["pos"][1])
    variant = robot_variant.strip().lower()
    stop_key = "service_stop_b" if variant == "b" else "service_stop_a"
    approach_key = "approach_b" if variant == "b" else "approach_a"
    if stop_key in target:
        goal_x, goal_y = float(target[stop_key][0]), float(target[stop_key][1])
    else:
        goal_x, goal_y = float(target[approach_key][0]), float(target[approach_key][1])
    goal_yaw = math.atan2(target_y - goal_y, target_x - goal_x)
    return goal_x, goal_y, goal_yaw


def _yaw_quat(yaw: float) -> str:
    half_yaw = 0.5 * yaw
    return f"{math.cos(half_yaw):.8f} 0 0 {math.sin(half_yaw):.8f}"


def _rgba_string(rgba: tuple[float, float, float, float]) -> str:
    return " ".join(f"{channel:.3f}" for channel in rgba)


def _model_asset_path(filename: str) -> str:
    source_models = Path(__file__).resolve().parents[1] / "models" / filename
    if source_models.exists():
        return source_models.as_posix()
    if get_package_share_directory is not None:
        try:
            installed_models = Path(get_package_share_directory("mujoco_amr_sim")) / "models" / filename
            if installed_models.exists():
                return installed_models.as_posix()
        except Exception:
            pass
    return source_models.as_posix()


def _build_scene_boxes_xml() -> str:
    lines = []
    for box in HOUSE_BOXES:
        pos = " ".join(f"{value:.3f}" for value in box.pos)
        size = " ".join(f"{value:.3f}" for value in box.size)
        material = ""
        collision_flags = ""
        if "wall" in box.name:
            material = ' material="wall_mat"'
        elif "rack" in box.name:
            material = ' material="rack_blue_mat"'
        elif "dock" in box.name:
            material = ' material="dock_mat"'
            collision_flags = ' contype="0" conaffinity="0"'
        elif "fence" in box.name:
            material = ' material="safety_yellow_mat"'
        elif "pallet" in box.name:
            material = ' material="wood_mat"'
        elif "cart" in box.name:
            material = ' material="accent_blue_mat"'
        lines.append(
            "    "
            f'<geom name="{box.name}" type="box" pos="{pos}" size="{size}" rgba="{_rgba_string(box.rgba)}"{material}{collision_flags}/>'
        )
    return "\n".join(lines)


def _build_rack_visuals(prefix: str, x: float, y0: float, y1: float) -> str:
    center_y = 0.5 * (y0 + y1)
    half_span = 0.5 * abs(y1 - y0)
    shelf_y = center_y
    lines = [
        f'    <geom name="{prefix}_upright_a" type="box" pos="{x - 0.32:.2f} {y0:.2f} 1.12" size="0.05 0.05 1.12" material="rack_frame_mat" contype="0" conaffinity="0"/>',
        f'    <geom name="{prefix}_upright_b" type="box" pos="{x - 0.32:.2f} {y1:.2f} 1.12" size="0.05 0.05 1.12" material="rack_frame_mat" contype="0" conaffinity="0"/>',
        f'    <geom name="{prefix}_upright_c" type="box" pos="{x + 0.32:.2f} {y0:.2f} 1.12" size="0.05 0.05 1.12" material="rack_frame_mat" contype="0" conaffinity="0"/>',
        f'    <geom name="{prefix}_upright_d" type="box" pos="{x + 0.32:.2f} {y1:.2f} 1.12" size="0.05 0.05 1.12" material="rack_frame_mat" contype="0" conaffinity="0"/>',
    ]
    for level, z in enumerate((0.32, 0.92, 1.52), start=1):
        lines.append(
            f'    <geom name="{prefix}_shelf_{level}" type="box" pos="{x:.2f} {shelf_y:.2f} {z:.2f}" size="0.36 {half_span + 0.07:.2f} 0.03" material="rack_shelf_mat" contype="0" conaffinity="0"/>'
        )
    return "\n".join(lines)


def _build_floor_markings() -> str:
    service_highlights: list[str] = []
    highlight_rgba = {
        "table_1": {
            "a": ("0.18 0.96 0.72 0.28", "0.72 1.00 0.92 0.24"),
            "b": ("0.28 0.74 1.00 0.28", "0.76 0.92 1.00 0.24"),
        },
        "table_2": {
            "a": ("0.18 0.88 0.98 0.28", "0.76 0.96 1.00 0.24"),
            "b": ("0.36 0.78 1.00 0.28", "0.84 0.94 1.00 0.24"),
        },
        "table_3": {
            "a": ("0.20 0.92 0.76 0.28", "0.78 1.00 0.92 0.24"),
            "b": ("0.34 0.72 1.00 0.28", "0.82 0.92 1.00 0.24"),
        },
        "table_4": {
            "a": ("0.26 0.92 0.64 0.28", "0.84 1.00 0.88 0.24"),
            "b": ("0.34 0.72 1.00 0.28", "0.80 0.92 1.00 0.24"),
        },
        "table_5": {
            "a": ("0.24 0.90 0.82 0.28", "0.82 1.00 0.96 0.24"),
            "b": ("0.32 0.74 1.00 0.28", "0.82 0.94 1.00 0.24"),
        },
        "table_6": {
            "a": ("0.24 0.86 1.00 0.28", "0.80 0.96 1.00 0.24"),
            "b": ("0.34 0.76 1.00 0.28", "0.84 0.96 1.00 0.24"),
        },
    }
    for table_key in HOTEL_TABLES:
        for variant, radii in (("a", (0.34, 0.20)), ("b", (0.32, 0.18))):
            stop_x, stop_y, _ = _service_stop_pose(table_key, variant)
            outer_rgba, inner_rgba = highlight_rgba[table_key][variant]
            outer_radius, inner_radius = radii
            service_highlights.extend(
                [
                    f'    <geom name="{table_key}_service_outer_{variant}" type="cylinder" pos="{stop_x:.2f} {stop_y:.2f} 0.008" size="{outer_radius:.2f} 0.008" rgba="{outer_rgba}" contype="0" conaffinity="0"/>',
                    f'    <geom name="{table_key}_service_inner_{variant}" type="cylinder" pos="{stop_x:.2f} {stop_y:.2f} 0.009" size="{inner_radius:.2f} 0.006" rgba="{inner_rgba}" contype="0" conaffinity="0"/>',
                ]
            )
    for sofa_key, rgba_pair in {
        "sofa_1": {
            "a": ("0.88 0.78 0.18 0.28", "1.00 0.94 0.58 0.22"),
            "b": ("0.30 0.72 1.00 0.28", "0.82 0.92 1.00 0.22"),
        },
        "sofa_2": {
            "a": ("0.84 0.70 0.18 0.28", "1.00 0.90 0.54 0.22"),
            "b": ("0.28 0.70 1.00 0.28", "0.80 0.90 1.00 0.22"),
        },
        "sofa_3": {
            "a": ("0.90 0.74 0.26 0.28", "1.00 0.94 0.68 0.22"),
            "b": ("0.34 0.74 1.00 0.28", "0.84 0.94 1.00 0.22"),
        },
    }.items():
        for variant, radii in (("a", (0.34, 0.20)), ("b", (0.32, 0.18))):
            stop_x, stop_y, _ = _service_stop_pose(sofa_key, variant)
            outer_rgba, inner_rgba = rgba_pair[variant]
            outer_radius, inner_radius = radii
            service_highlights.extend(
                [
                    f'    <geom name="{sofa_key}_service_outer_{variant}" type="cylinder" pos="{stop_x:.2f} {stop_y:.2f} 0.008" size="{outer_radius:.2f} 0.008" rgba="{outer_rgba}" contype="0" conaffinity="0"/>',
                    f'    <geom name="{sofa_key}_service_inner_{variant}" type="cylinder" pos="{stop_x:.2f} {stop_y:.2f} 0.009" size="{inner_radius:.2f} 0.006" rgba="{inner_rgba}" contype="0" conaffinity="0"/>',
                ]
            )
    return f"""
    <geom name="dock_lane" type="box" pos="{DOCK_X:.2f} {DOCK_Y:.2f} 0.003" size="1.10 0.86 0.003" rgba="0.22 0.98 0.44 0.62" contype="0" conaffinity="0"/>
    <geom name="service_dock_lane" type="box" pos="{SERVICE_DOCK_X:.2f} {SERVICE_DOCK_Y:.2f} 0.003" size="1.10 0.86 0.003" rgba="0.22 0.72 1.00 0.46" contype="0" conaffinity="0"/>
    <geom name="room_drop_zone_mark" type="box" pos="{PLACE_STATION_X:.2f} {PLACE_STATION_Y:.2f} 0.003" size="1.10 0.82 0.003" rgba="0.86 0.80 0.56 0.10" contype="0" conaffinity="0"/>
{chr(10).join(service_highlights)}
    """


def _build_luxury_table_xml(name: str, x: float, y: float, radius: float, accent_rgba: str, flower_rgba: str) -> str:
    ring_radius = max(0.10, radius - 0.10)
    return f"""
    <geom name="{name}_top" type="cylinder" pos="{x:.2f} {y:.2f} 0.79" size="{radius:.2f} 0.032" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_edge" type="cylinder" pos="{x:.2f} {y:.2f} 0.75" size="{radius + 0.02:.2f} 0.020" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_pedestal" type="capsule" pos="{x:.2f} {y:.2f} 0.46" size="0.070 0.28" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_base" type="cylinder" pos="{x:.2f} {y:.2f} 0.10" size="{ring_radius:.2f} 0.06" rgba="0.22 0.18 0.16 1" contype="0" conaffinity="0"/>
    <geom name="{name}_linen" type="cylinder" pos="{x:.2f} {y:.2f} 0.822" size="{max(0.06, radius - 0.20):.2f} 0.004" material="linen_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_vase" type="capsule" pos="{x:.2f} {y:.2f} 0.90" size="0.020 0.06" material="emerald_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_flower" type="sphere" pos="{x:.2f} {y:.2f} 0.98" size="0.030" rgba="{flower_rgba}" contype="0" conaffinity="0"/>
    <geom name="{name}_accent" type="capsule" pos="{x + 0.10:.2f} {y - 0.06:.2f} 0.85" size="0.010 0.08" euler="0.18 0.10 0.54" rgba="{accent_rgba}" contype="0" conaffinity="0"/>
    """


def _build_dining_chair_xml(name: str, x: float, y: float, yaw: float, rgba: str) -> str:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    def _rotate(dx: float, dy: float) -> tuple[float, float]:
        return x + cos_yaw * dx - sin_yaw * dy, y + sin_yaw * dx + cos_yaw * dy

    back_x, back_y = _rotate(-0.15, 0.0)
    back_cushion_x, back_cushion_y = _rotate(-0.11, 0.0)
    back_post_l_x, back_post_l_y = _rotate(-0.14, 0.10)
    back_post_r_x, back_post_r_y = _rotate(-0.14, -0.10)
    leg_fl_x, leg_fl_y = _rotate(0.09, 0.09)
    leg_fr_x, leg_fr_y = _rotate(0.09, -0.09)
    leg_rl_x, leg_rl_y = _rotate(-0.08, 0.09)
    leg_rr_x, leg_rr_y = _rotate(-0.08, -0.09)
    apron_front_x, apron_front_y = _rotate(0.08, 0.0)
    apron_back_x, apron_back_y = _rotate(-0.07, 0.0)

    return f"""
    <geom name="{name}_seat_frame" type="box" pos="{x:.2f} {y:.2f} 0.36" size="0.15 0.16 0.030" euler="0 0 {yaw:.3f}" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_seat_cushion" type="box" pos="{x:.2f} {y:.2f} 0.405" size="0.13 0.14 0.028" euler="0 0 {yaw:.3f}" material="linen_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_back_panel" type="box" pos="{back_x:.2f} {back_y:.2f} 0.67" size="0.026 0.16 0.15" euler="0.16 0 {yaw:.3f}" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_back_cushion" type="box" pos="{back_cushion_x:.2f} {back_cushion_y:.2f} 0.66" size="0.018 0.13 0.13" euler="0.14 0 {yaw:.3f}" material="linen_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_back_post_l" type="capsule" pos="{back_post_l_x:.2f} {back_post_l_y:.2f} 0.34" size="0.012 0.31" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_back_post_r" type="capsule" pos="{back_post_r_x:.2f} {back_post_r_y:.2f} 0.34" size="0.012 0.31" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_leg_fl" type="capsule" pos="{leg_fl_x:.2f} {leg_fl_y:.2f} 0.18" size="0.013 0.17" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_leg_fr" type="capsule" pos="{leg_fr_x:.2f} {leg_fr_y:.2f} 0.18" size="0.013 0.17" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_leg_rl" type="capsule" pos="{leg_rl_x:.2f} {leg_rl_y:.2f} 0.18" size="0.013 0.17" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_leg_rr" type="capsule" pos="{leg_rr_x:.2f} {leg_rr_y:.2f} 0.18" size="0.013 0.17" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_apron_front" type="capsule" pos="{apron_front_x:.2f} {apron_front_y:.2f} 0.33" size="0.010 0.12" euler="0 1.57079632679 {yaw:.3f}" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="{name}_apron_back" type="capsule" pos="{apron_back_x:.2f} {apron_back_y:.2f} 0.33" size="0.010 0.12" euler="0 1.57079632679 {yaw:.3f}" material="bronze_mat" contype="0" conaffinity="0"/>
    """


def _build_dining_chair_cluster_xml() -> str:
    chairs = (
        ("chair_a1", -2.02, 2.30, 0.00, "0.34 0.27 0.20 1"),
        ("chair_a2", -0.38, 2.30, 3.14159265359, "0.34 0.27 0.20 1"),
        ("chair_b1", 1.26, 2.00, 0.00, "0.24 0.22 0.20 1"),
        ("chair_b2", 3.14, 2.00, 3.14159265359, "0.24 0.22 0.20 1"),
        ("chair_c1", 0.36, -0.80, 0.00, "0.22 0.24 0.28 1"),
        ("chair_c2", 2.14, -0.80, 3.14159265359, "0.22 0.24 0.28 1"),
        ("chair_d1", -3.92, -0.45, 0.00, "0.30 0.24 0.18 1"),
        ("chair_d2", -2.18, -0.45, 3.14159265359, "0.30 0.24 0.18 1"),
        ("chair_e1", 3.44, -2.10, 0.00, "0.20 0.24 0.22 1"),
        ("chair_e2", 5.18, -2.10, 3.14159265359, "0.20 0.24 0.22 1"),
        ("chair_f1", 4.60, 5.06, -1.57079632679, "0.28 0.24 0.18 1"),
        ("chair_f2", 4.60, 3.14, 1.57079632679, "0.28 0.24 0.18 1"),
    )
    return "\n".join(_build_dining_chair_xml(*chair) for chair in chairs)


def _build_dining_table_cluster_xml() -> str:
    tables = (
        ("table_a", -1.20, 2.30, 0.58, "0.86 0.62 0.40 1", "0.98 0.84 0.62 1"),
        ("table_b", 2.20, 2.00, 0.60, "0.74 0.72 0.40 1", "0.96 0.88 0.64 1"),
        ("table_c", 1.25, -0.80, 0.56, "0.72 0.60 0.38 1", "0.94 0.82 0.60 1"),
        ("table_d", -3.05, -0.45, 0.54, "0.56 0.74 0.42 1", "0.90 0.96 0.66 1"),
        ("table_e", 4.30, -2.10, 0.54, "0.86 0.70 0.44 1", "0.98 0.88 0.60 1"),
        ("table_f", 4.60, 4.10, 0.54, "0.66 0.78 0.96 1", "0.92 0.98 1.00 1"),
    )
    return "\n".join(_build_luxury_table_xml(*table) for table in tables)


def _build_ceiling_architecture_xml() -> str:
    return ""


def _build_wall_detail_xml() -> str:
    return f"""
    <geom name="north_wall_trim" type="box" pos="0.00 {WORLD_Y - 0.16:.2f} 0.18" size="{WORLD_X - 0.35:.2f} 0.03 0.18" rgba="0.24 0.20 0.17 1" contype="0" conaffinity="0"/>
    <geom name="south_wall_trim" type="box" pos="0.00 {-WORLD_Y + 0.16:.2f} 0.18" size="{WORLD_X - 0.35:.2f} 0.03 0.18" rgba="0.24 0.20 0.17 1" contype="0" conaffinity="0"/>
    <geom name="east_wall_trim" type="box" pos="{WORLD_X - 0.16:.2f} 0.00 0.18" size="0.03 {WORLD_Y - 0.35:.2f} 0.18" rgba="0.24 0.20 0.17 1" contype="0" conaffinity="0"/>
    <geom name="west_wall_trim" type="box" pos="{-WORLD_X + 0.16:.2f} 0.00 0.18" size="0.03 {WORLD_Y - 0.35:.2f} 0.18" rgba="0.24 0.20 0.17 1" contype="0" conaffinity="0"/>
    <geom name="north_panel_band" type="box" pos="0.00 {WORLD_Y - 0.18:.2f} 1.10" size="{WORLD_X - 0.55:.2f} 0.02 0.34" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="east_panel_band" type="box" pos="{WORLD_X - 0.18:.2f} 0.00 1.08" size="0.02 {WORLD_Y - 0.90:.2f} 0.30" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="west_panel_band" type="box" pos="{-WORLD_X + 0.18:.2f} 0.00 1.08" size="0.02 {WORLD_Y - 0.90:.2f} 0.30" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="north_window_left" type="box" pos="-4.80 {WORLD_Y - 0.11:.2f} 1.30" size="1.15 0.02 0.62" rgba="0.58 0.78 0.92 0.00" contype="0" conaffinity="0"/>
    <geom name="north_window_right" type="box" pos="4.65 {WORLD_Y - 0.11:.2f} 1.30" size="1.55 0.02 0.62" rgba="0.58 0.78 0.92 0.00" contype="0" conaffinity="0"/>
    <geom name="east_art_panel" type="box" pos="{WORLD_X - 0.12:.2f} -1.90 1.30" size="0.02 0.88 0.52" rgba="0.78 0.68 0.40 0.90" contype="0" conaffinity="0"/>
    <geom name="west_art_panel" type="box" pos="{-WORLD_X + 0.12:.2f} 1.85 1.28" size="0.02 0.76 0.48" rgba="0.30 0.52 0.64 0.90" contype="0" conaffinity="0"/>
    <geom name="north_art_sconce_left" type="sphere" pos="-4.80 {WORLD_Y - 0.20:.2f} 1.82" size="0.05" material="warm_glow_mat" contype="0" conaffinity="0"/>
    <geom name="north_art_sconce_right" type="sphere" pos="4.65 {WORLD_Y - 0.20:.2f} 1.82" size="0.05" material="warm_glow_mat" contype="0" conaffinity="0"/>
    """


def _build_guest_mesh_xml() -> str:
    return ""


def _build_decor_xml() -> str:
    guest_mesh_xml = _build_guest_mesh_xml()
    return f"""
{_build_floor_markings()}
{_build_wall_detail_xml()}
{_build_ceiling_architecture_xml()}
    <geom name="reception_monitor" type="box" pos="-6.95 5.26 1.10" size="0.10 0.03 0.16" material="screen_mat" contype="0" conaffinity="0"/>
    <geom name="reception_monitor_right" type="box" pos="-6.28 5.24 1.08" size="0.11 0.03 0.15" material="screen_mat" contype="0" conaffinity="0"/>
    <geom name="reception_terminal_base_left" type="box" pos="-6.95 5.18 0.92" size="0.11 0.08 0.02" rgba="0.20 0.22 0.24 1" contype="0" conaffinity="0"/>
    <geom name="reception_terminal_base_right" type="box" pos="-6.28 5.16 0.92" size="0.11 0.08 0.02" rgba="0.20 0.22 0.24 1" contype="0" conaffinity="0"/>
    <geom name="reception_keyboard_left" type="box" pos="-6.95 4.94 1.00" size="0.12 0.05 0.01" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>
    <geom name="reception_keyboard_right" type="box" pos="-6.28 4.92 1.00" size="0.12 0.05 0.01" rgba="0.24 0.26 0.28 1" contype="0" conaffinity="0"/>
    <geom name="reception_logo_plinth" type="box" pos="-5.85 5.25 0.95" size="0.20 0.03 0.14" rgba="0.18 0.66 0.94 1" contype="0" conaffinity="0"/>
    <geom name="reception_counter_stone_top" type="box" pos="-6.55 5.10 1.10" size="1.38 0.38 0.04" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="reception_counter_front_bronze" type="box" pos="-6.12 4.76 0.62" size="0.12 0.02 0.40" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="reception_counter_front_bronze_b" type="box" pos="-6.98 4.76 0.62" size="0.12 0.02 0.40" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="reception_counter_front_bronze_c" type="box" pos="-6.55 4.76 0.62" size="0.12 0.02 0.40" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="reception_floral_pot" type="cylinder" pos="-5.62 5.03 1.13" size="0.05 0.07" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="reception_floral_stem_a" type="capsule" pos="-5.64 5.04 1.28" size="0.018 0.10" euler="0.24 0.12 0.18" rgba="0.18 0.48 0.28 1" contype="0" conaffinity="0"/>
    <geom name="reception_floral_stem_b" type="capsule" pos="-5.59 5.00 1.26" size="0.018 0.09" euler="-0.20 -0.10 -0.16" rgba="0.30 0.62 0.36 1" contype="0" conaffinity="0"/>
    <geom name="reception_back_panel" type="box" pos="-6.55 5.66 1.42" size="1.34 0.05 0.86" material="charcoal_mat" contype="0" conaffinity="0"/>
    <geom name="reception_back_inlay" type="box" pos="-6.55 5.60 1.42" size="1.16 0.02 0.70" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="reception_back_shelf_a" type="box" pos="-6.98 5.54 1.18" size="0.24 0.10 0.02" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="reception_back_shelf_b" type="box" pos="-6.18 5.54 1.52" size="0.24 0.10 0.02" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="reception_vase_a" type="capsule" pos="-7.02 5.54 1.29" size="0.022 0.08" material="emerald_mat" contype="0" conaffinity="0"/>
    <geom name="reception_vase_b" type="capsule" pos="-6.16 5.54 1.63" size="0.022 0.08" material="emerald_mat" contype="0" conaffinity="0"/>
    <geom name="reception_flower_a" type="sphere" pos="-7.02 5.54 1.40" size="0.034" material="warm_glow_mat" contype="0" conaffinity="0"/>
    <geom name="reception_flower_b" type="sphere" pos="-6.16 5.54 1.74" size="0.034" material="warm_glow_mat" contype="0" conaffinity="0"/>
    <geom name="reception_bell" type="sphere" pos="-6.56 4.90 1.15" size="0.040" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="reception_bell_base" type="cylinder" pos="-6.56 4.90 1.10" size="0.05 0.02" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="reception_desk_lamp_stem" type="capsule" pos="-6.10 5.18 1.24" size="0.018 0.18" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="reception_desk_lamp_shade" type="capsule" pos="-6.02 5.20 1.42" size="0.06 0.10" euler="0 1.57079632679 0.16" material="linen_mat" contype="0" conaffinity="0"/>
    <geom name="reception_desk_lamp_glow" type="sphere" pos="-5.98 5.18 1.34" size="0.06" rgba="1.00 0.86 0.62 0.46" contype="0" conaffinity="0"/>
    <geom name="reception_menu_stand" type="box" pos="-6.74 4.94 1.18" size="0.05 0.02 0.12" material="linen_mat" contype="0" conaffinity="0"/>
    <geom name="reception_underglow" type="box" pos="-6.55 4.78 0.22" size="1.18 0.015 0.015" rgba="0.98 0.80 0.56 0.34" contype="0" conaffinity="0"/>
{_build_dining_table_cluster_xml()}
{_build_dining_chair_cluster_xml()}
{guest_mesh_xml}
    <geom name="kitchen_counter_top_a" type="box" pos="-2.36 -4.99 1.04" size="1.70 0.34 0.05" material="workbench_top_mat" contype="0" conaffinity="0"/>
    <geom name="kitchen_counter_top_b" type="box" pos="0.69 -4.99 1.04" size="1.60 0.34 0.05" material="workbench_top_mat" contype="0" conaffinity="0"/>
    <geom name="sofa_2_l_base_long" type="box" pos="-4.95 1.20 0.34" size="1.12 0.22 0.12" rgba="0.20 0.22 0.26 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_l_back_long" type="box" pos="-4.95 1.48 0.62" size="1.12 0.06 0.26" rgba="0.16 0.18 0.22 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_l_arm_left" type="box" pos="-6.15 1.20 0.54" size="0.08 0.22 0.20" rgba="0.16 0.18 0.22 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_l_arm_right" type="box" pos="-3.75 1.20 0.54" size="0.08 0.22 0.20" rgba="0.16 0.18 0.22 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_short_base" type="box" pos="-5.74 0.52 0.34" size="0.30 0.68 0.12" rgba="0.20 0.22 0.26 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_short_back" type="box" pos="-6.08 0.52 0.62" size="0.06 0.68 0.26" rgba="0.16 0.18 0.22 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_corner" type="box" pos="-5.46 0.96 0.34" size="0.30 0.24 0.12" rgba="0.24 0.26 0.30 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_cushion_a" type="box" pos="-5.55 1.20 0.47" size="0.26 0.18 0.03" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_cushion_b" type="box" pos="-5.05 1.20 0.47" size="0.26 0.18 0.03" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_cushion_c" type="box" pos="-4.55 1.20 0.47" size="0.26 0.18 0.03" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_cushion_d" type="box" pos="-4.05 1.20 0.47" size="0.26 0.18 0.03" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_cushion_e" type="box" pos="-5.74 0.14 0.47" size="0.18 0.18 0.03" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_cushion_f" type="box" pos="-5.74 0.54 0.47" size="0.18 0.18 0.03" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_cushion_g" type="box" pos="-5.74 0.94 0.47" size="0.18 0.18 0.03" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_table_top" type="box" pos="-4.95 0.18 0.42" size="0.44 0.24 0.03" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="sofa_2_table_edge" type="box" pos="-4.95 0.18 0.39" size="0.46 0.26 0.015" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="sofa_2_table_leg_left" type="capsule" pos="-5.08 0.18 0.22" size="0.028 0.18" euler="0 0 0" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="sofa_2_table_leg_right" type="capsule" pos="-4.82 0.18 0.22" size="0.028 0.18" euler="0 0 0" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="sofa_2_table_runner" type="box" pos="-4.95 0.18 0.455" size="0.17 0.07 0.006" material="linen_mat" contype="0" conaffinity="0"/>
    <geom name="sofa_2_table_vase" type="capsule" pos="-4.95 0.18 0.50" size="0.022 0.05" euler="0 0 0" material="emerald_mat" contype="0" conaffinity="0"/>
    <geom name="sofa_2_table_flower" type="sphere" pos="-4.95 0.18 0.56" size="0.032" material="warm_glow_mat" contype="0" conaffinity="0"/>
    <geom name="sofa_2_throw_a" type="box" pos="-5.72 0.90 0.54" size="0.12 0.10 0.015" rgba="0.56 0.46 0.28 1" contype="0" conaffinity="0"/>
    <geom name="sofa_2_throw_b" type="box" pos="-4.20 1.16 0.54" size="0.10 0.12 0.015" rgba="0.24 0.38 0.48 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_long_base" type="box" pos="0.85 4.55 0.34" size="0.94 0.22 0.12" rgba="0.22 0.24 0.28 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_long_back" type="box" pos="0.85 4.83 0.62" size="0.94 0.06 0.26" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_long_arm_left" type="box" pos="-0.17 4.55 0.54" size="0.08 0.22 0.20" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_long_arm_right" type="box" pos="1.87 4.55 0.54" size="0.08 0.22 0.20" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_short_base" type="box" pos="1.66 3.62 0.34" size="0.26 0.72 0.12" rgba="0.22 0.24 0.28 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_short_back" type="box" pos="1.96 3.62 0.62" size="0.06 0.72 0.26" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_corner_block" type="box" pos="1.48 4.20 0.34" size="0.26 0.24 0.12" rgba="0.24 0.26 0.30 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_cushion_a" type="box" pos="0.45 4.55 0.47" size="0.26 0.18 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_cushion_b" type="box" pos="0.85 4.55 0.47" size="0.26 0.18 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_cushion_c" type="box" pos="1.25 4.55 0.47" size="0.26 0.18 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_cushion_d" type="box" pos="1.66 3.34 0.47" size="0.18 0.24 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_cushion_e" type="box" pos="1.66 3.74 0.47" size="0.18 0.24 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_cushion_f" type="box" pos="1.66 4.14 0.47" size="0.18 0.24 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_table_top" type="box" pos="0.85 3.54 0.42" size="0.48 0.28 0.04" rgba="0.50 0.34 0.22 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_table_base" type="box" pos="0.85 3.54 0.22" size="0.12 0.12 0.20" rgba="0.18 0.16 0.14 1" contype="0" conaffinity="0"/>
    <geom name="sofa_1_table_decor" type="cylinder" pos="0.85 3.54 0.50" size="0.05 0.06" rgba="0.18 0.40 0.26 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_long_base" type="box" pos="6.45 4.12 0.34" size="0.94 0.22 0.12" rgba="0.22 0.24 0.28 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_long_back" type="box" pos="6.45 4.38 0.62" size="0.94 0.06 0.26" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_long_arm_left" type="box" pos="5.43 4.12 0.54" size="0.08 0.22 0.20" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_long_arm_right" type="box" pos="7.47 4.12 0.54" size="0.08 0.22 0.20" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_short_base" type="box" pos="7.26 3.18 0.34" size="0.26 0.74 0.12" rgba="0.22 0.24 0.28 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_short_back" type="box" pos="7.56 3.18 0.62" size="0.06 0.74 0.26" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_corner_block" type="box" pos="7.08 3.78 0.34" size="0.26 0.24 0.12" rgba="0.24 0.26 0.30 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_cushion_a" type="box" pos="6.05 4.12 0.47" size="0.26 0.18 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_cushion_b" type="box" pos="6.45 4.12 0.47" size="0.26 0.18 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_cushion_c" type="box" pos="6.85 4.12 0.47" size="0.26 0.18 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_cushion_d" type="box" pos="7.26 2.90 0.47" size="0.18 0.24 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_cushion_e" type="box" pos="7.26 3.30 0.47" size="0.18 0.24 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_cushion_f" type="box" pos="7.26 3.70 0.47" size="0.18 0.24 0.03" rgba="0.90 0.92 0.94 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_table_top" type="box" pos="6.45 2.98 0.42" size="0.40 0.22 0.03" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_table_edge" type="box" pos="6.45 2.98 0.39" size="0.42 0.24 0.015" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_table_leg_left" type="capsule" pos="6.34 2.98 0.22" size="0.026 0.18" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_table_leg_right" type="capsule" pos="6.56 2.98 0.22" size="0.026 0.18" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_table_runner" type="box" pos="6.45 2.98 0.455" size="0.14 0.06 0.006" material="linen_mat" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_table_vase" type="capsule" pos="6.45 2.98 0.50" size="0.020 0.05" material="emerald_mat" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_table_flower" type="sphere" pos="6.45 2.98 0.56" size="0.030" material="warm_glow_mat" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_throw_a" type="box" pos="6.12 4.08 0.54" size="0.12 0.10 0.015" rgba="0.18 0.28 0.42 1" contype="0" conaffinity="0"/>
    <geom name="vip_sofa_throw_b" type="box" pos="7.18 3.18 0.54" size="0.10 0.12 0.015" rgba="0.62 0.50 0.28 1" contype="0" conaffinity="0"/>
    <geom name="hotel_entrance_left_post" type="box" pos="8.10 -3.51 1.15" size="0.11 0.11 1.15" rgba="0.56 0.40 0.26 1" contype="0" conaffinity="0"/>
    <geom name="hotel_entrance_right_post" type="box" pos="8.10 0.19 1.15" size="0.11 0.11 1.15" rgba="0.56 0.40 0.26 1" contype="0" conaffinity="0"/>
    <geom name="hotel_entrance_header" type="box" pos="8.10 -1.66 2.18" size="0.11 1.96 0.12" rgba="0.16 0.24 0.32 1" contype="0" conaffinity="0"/>
    <geom name="hotel_entrance_sign" type="box" pos="7.94 -1.66 1.80" size="0.03 1.70 0.18" rgba="0.86 0.92 0.96 1" contype="0" conaffinity="0"/>
    <geom name="hotel_entrance_mat" type="box" pos="7.73 -1.66 0.01" size="0.58 1.72 0.01" rgba="0.56 0.46 0.18 0.32" contype="0" conaffinity="0"/>
    <geom name="entrance_top_pot" type="cylinder" pos="7.12 1.18 0.22" size="0.16 0.22" rgba="0.30 0.22 0.16 1" contype="0" conaffinity="0"/>
    <geom name="entrance_top_leaf_a" type="capsule" pos="7.06 1.18 0.64" size="0.04 0.28" euler="0.34 0.18 0.20" rgba="0.24 0.58 0.34 1" contype="0" conaffinity="0"/>
    <geom name="entrance_top_leaf_b" type="capsule" pos="7.16 1.10 0.66" size="0.04 0.24" euler="-0.28 -0.16 -0.22" rgba="0.32 0.68 0.38 1" contype="0" conaffinity="0"/>
    <geom name="entrance_top_leaf_c" type="capsule" pos="7.18 1.25 0.58" size="0.03 0.18" euler="0.20 0.26 0.34" rgba="0.18 0.48 0.28 1" contype="0" conaffinity="0"/>
    <geom name="entrance_bottom_pot" type="cylinder" pos="7.12 -4.48 0.22" size="0.16 0.22" rgba="0.30 0.22 0.16 1" contype="0" conaffinity="0"/>
    <geom name="entrance_bottom_leaf_a" type="capsule" pos="7.05 -4.46 0.64" size="0.04 0.28" euler="0.30 0.16 0.18" rgba="0.24 0.58 0.34 1" contype="0" conaffinity="0"/>
    <geom name="entrance_bottom_leaf_b" type="capsule" pos="7.17 -4.55 0.66" size="0.04 0.24" euler="-0.26 -0.18 -0.24" rgba="0.32 0.68 0.38 1" contype="0" conaffinity="0"/>
    <geom name="entrance_bottom_leaf_c" type="capsule" pos="7.18 -4.37 0.58" size="0.03 0.18" euler="0.18 0.22 0.32" rgba="0.18 0.48 0.28 1" contype="0" conaffinity="0"/>
    <geom name="entrance_top_wall_vine_stem" type="capsule" pos="9.52 1.38 1.24" size="0.02 0.56" euler="0 1.57079632679 0" rgba="0.34 0.24 0.16 1" contype="0" conaffinity="0"/>
    <geom name="entrance_top_wall_vine_a" type="capsule" pos="9.46 1.18 1.52" size="0.03 0.20" euler="0.42 0.18 0.46" rgba="0.22 0.56 0.30 1" contype="0" conaffinity="0"/>
    <geom name="entrance_top_wall_vine_b" type="capsule" pos="9.56 1.52 1.44" size="0.03 0.18" euler="-0.36 -0.16 -0.38" rgba="0.30 0.66 0.36 1" contype="0" conaffinity="0"/>
    <geom name="entrance_bottom_wall_vine_stem" type="capsule" pos="9.52 -4.72 1.24" size="0.02 0.56" euler="0 1.57079632679 0" rgba="0.34 0.24 0.16 1" contype="0" conaffinity="0"/>
    <geom name="entrance_bottom_wall_vine_a" type="capsule" pos="9.46 -4.92 1.52" size="0.03 0.20" euler="0.40 0.18 0.44" rgba="0.22 0.56 0.30 1" contype="0" conaffinity="0"/>
    <geom name="entrance_bottom_wall_vine_b" type="capsule" pos="9.56 -4.56 1.44" size="0.03 0.18" euler="-0.34 -0.16 -0.36" rgba="0.30 0.66 0.36 1" contype="0" conaffinity="0"/>
    <geom name="entry_sconce_upper_backplate" type="capsule" pos="9.58 1.36 1.55" size="0.03 0.24" euler="0 1.57079632679 1.57079632679" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="entry_sconce_upper_glow" type="sphere" pos="9.44 1.36 1.55" size="0.07" material="warm_glow_mat" contype="0" conaffinity="0"/>
    <geom name="entry_sconce_lower_backplate" type="capsule" pos="9.58 -4.56 1.55" size="0.03 0.24" euler="0 1.57079632679 1.57079632679" material="bronze_mat" contype="0" conaffinity="0"/>
    <geom name="entry_sconce_lower_glow" type="sphere" pos="9.44 -4.56 1.55" size="0.07" material="warm_glow_mat" contype="0" conaffinity="0"/>
    <geom name="dock_plinth_outer" type="box" pos="{DOCK_X - 0.28:.2f} {DOCK_Y:.2f} 0.02" size="0.62 1.04 0.02" rgba="0.10 0.22 0.14 0.28" contype="0" conaffinity="0"/>
    <geom name="dock_plinth_inner" type="box" pos="{DOCK_X - 0.24:.2f} {DOCK_Y:.2f} 0.03" size="0.48 0.88 0.01" rgba="0.20 0.96 0.40 0.18" contype="0" conaffinity="0"/>
    <geom name="dock_beacon_left" type="cylinder" pos="{DOCK_X - 0.62:.2f} {DOCK_Y + 0.50:.2f} 0.60" size="0.06 0.60" rgba="0.10 0.96 0.34 1" contype="0" conaffinity="0"/>
    <geom name="dock_beacon_right" type="cylinder" pos="{DOCK_X - 0.62:.2f} {DOCK_Y - 0.50:.2f} 0.60" size="0.06 0.60" rgba="0.10 0.96 0.34 1" contype="0" conaffinity="0"/>
    <geom name="dock_header" type="capsule" pos="{DOCK_X - 0.62:.2f} {DOCK_Y:.2f} 1.12" size="0.06 0.84" euler="1.57079632679 0 0" rgba="0.10 0.96 0.34 1" contype="0" conaffinity="0"/>
    <geom name="dock_signal" type="sphere" pos="{DOCK_X - 0.62:.2f} {DOCK_Y:.2f} 1.28" size="0.10" rgba="0.14 1.00 0.36 1" contype="0" conaffinity="0"/>
    <geom name="service_dock_backplate" type="box" pos="{SERVICE_DOCK_X + 0.72:.2f} {SERVICE_DOCK_Y:.2f} 0.58" size="0.09 0.86 0.58" rgba="0.08 0.12 0.18 1" contype="0" conaffinity="0"/>
    <geom name="service_dock_pad" type="box" pos="{SERVICE_DOCK_X - 0.26:.2f} {SERVICE_DOCK_Y:.2f} 0.035" size="0.42 0.92 0.035" rgba="0.16 0.62 0.98 0.40" contype="0" conaffinity="0"/>
    <geom name="service_dock_guide_left" type="box" pos="{SERVICE_DOCK_X + 0.10:.2f} {SERVICE_DOCK_Y + 0.50:.2f} 0.16" size="0.28 0.03 0.16" rgba="0.52 0.86 1.00 1" contype="0" conaffinity="0"/>
    <geom name="service_dock_guide_right" type="box" pos="{SERVICE_DOCK_X + 0.10:.2f} {SERVICE_DOCK_Y - 0.50:.2f} 0.16" size="0.28 0.03 0.16" rgba="0.52 0.86 1.00 1" contype="0" conaffinity="0"/>
    <geom name="service_dock_beacon_left" type="cylinder" pos="{SERVICE_DOCK_X + 0.62:.2f} {SERVICE_DOCK_Y + 0.40:.2f} 0.60" size="0.06 0.60" rgba="0.20 0.74 1.00 1" contype="0" conaffinity="0"/>
    <geom name="service_dock_beacon_right" type="cylinder" pos="{SERVICE_DOCK_X + 0.62:.2f} {SERVICE_DOCK_Y - 0.40:.2f} 0.60" size="0.06 0.60" rgba="0.20 0.74 1.00 1" contype="0" conaffinity="0"/>
    <geom name="service_dock_header" type="capsule" pos="{SERVICE_DOCK_X + 0.62:.2f} {SERVICE_DOCK_Y:.2f} 1.12" size="0.06 0.84" euler="1.57079632679 0 0" rgba="0.20 0.74 1.00 1" contype="0" conaffinity="0"/>
    <geom name="service_dock_signal" type="sphere" pos="{SERVICE_DOCK_X + 0.62:.2f} {SERVICE_DOCK_Y:.2f} 1.28" size="0.10" rgba="0.40 0.86 1.00 1" contype="0" conaffinity="0"/>
    <geom name="service_dock_glow_core" type="box" pos="{SERVICE_DOCK_X - 0.26:.2f} {SERVICE_DOCK_Y:.2f} 0.010" size="0.32 0.72 0.010" rgba="0.24 0.80 1.00 0.55" contype="0" conaffinity="0"/>
    <geom name="service_dock_glow_strip" type="capsule" pos="{SERVICE_DOCK_X + 0.30:.2f} {SERVICE_DOCK_Y:.2f} 0.10" size="0.04 0.66" euler="1.57079632679 0 0" rgba="0.52 0.92 1.00 0.88" contype="0" conaffinity="0"/>
    <geom name="service_dock_plinth_outer" type="box" pos="{SERVICE_DOCK_X - 0.24:.2f} {SERVICE_DOCK_Y:.2f} 0.02" size="0.60 1.02 0.02" rgba="0.10 0.18 0.24 0.28" contype="0" conaffinity="0"/>
    <geom name="service_dock_plinth_inner" type="box" pos="{SERVICE_DOCK_X - 0.24:.2f} {SERVICE_DOCK_Y:.2f} 0.03" size="0.46 0.84 0.01" rgba="0.34 0.82 1.00 0.16" contype="0" conaffinity="0"/>
    <geom name="lobby_service_console_top" type="box" pos="-2.55 5.05 0.84" size="0.42 0.16 0.04" material="stone_mat" contype="0" conaffinity="0"/>
    <geom name="lobby_service_console_base" type="box" pos="-2.55 5.05 0.42" size="0.30 0.12 0.42" material="walnut_mat" contype="0" conaffinity="0"/>
    <geom name="lobby_service_console_screen" type="box" pos="-2.38 5.05 1.06" size="0.02 0.12 0.12" material="screen_mat" contype="0" conaffinity="0"/>
    """


def _build_dynamic_actor_xml() -> str:
    return ""


def _build_payload_xml() -> str:
    return f"""
    <body name="payload_box" mocap="true" pos="{PICK_STATION_X + 0.18:.2f} {PICK_STATION_Y:.2f} 1.12">
      <geom name="payload_geom" type="box" pos="0 0 0" size="0.085 0.055 0.022" rgba="0.82 0.84 0.88 1"/>
      <geom name="payload_band" type="box" pos="0 0 0.018" size="0.060 0.040 0.006" rgba="0.98 0.82 0.26 1"/>
    </body>
    """


def _build_lidar_sites(lidar_beams: int, lidar_fov_deg: float) -> str:
    lines = []
    start_angle = -math.radians(lidar_fov_deg) / 2.0
    step = math.radians(lidar_fov_deg) / max(lidar_beams - 1, 1)
    for mount in LIDAR_MOUNTS:
        pos = " ".join(f"{value:.3f}" for value in mount.pos)
        for index in range(lidar_beams):
            yaw = mount.yaw_rad + start_angle + index * step
            lines.append(
                "      "
                f'<site name="{mount.name}_lidar_site_{index:03d}" pos="{pos}" size="0.004" '
                f'quat="{_yaw_quat(yaw)}" rgba="0.1 0.8 0.8 0.08"/>'
            )
    return "\n".join(lines)


def _build_lidar_sensors(lidar_beams: int) -> str:
    lines = []
    for mount in LIDAR_MOUNTS:
        for index in range(lidar_beams):
            lines.append(
                f'    <rangefinder name="{mount.name}_lidar_{index:03d}" site="{mount.name}_lidar_site_{index:03d}"/>'
            )
    return "\n".join(lines)


def build_model_xml(lidar_beams: int = 91, lidar_fov_deg: float = 200.0) -> str:
    scene_boxes = _build_scene_boxes_xml()
    decor_xml = _build_decor_xml()
    dynamic_actor_xml = _build_dynamic_actor_xml()
    payload_xml = _build_payload_xml()
    lidar_sites = _build_lidar_sites(lidar_beams, lidar_fov_deg)
    lidar_sensors = _build_lidar_sensors(lidar_beams)

    return f"""<mujoco model="hotel_service_dual_amr">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  <option timestep="0.005" integrator="RK4" solver="Newton" gravity="0 0 -9.81" iterations="100"/>
  <size njmax="12000" nconmax="4000"/>

  <visual>
    <headlight ambient="0.14 0.12 0.11" diffuse="0.40 0.38 0.34" specular="0.10 0.10 0.08"/>
    <global offwidth="2048" offheight="1536"/>
    <map znear="0.01"/>
    <rgba haze="0.36 0.30 0.26 1"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.06 0.08 0.12" rgb2="0.78 0.62 0.42" width="512" height="512"/>
    <texture name="floor_tex" type="2d" builtin="flat" rgb1="0.88 0.88 0.87" width="32" height="32"/>
    <texture name="robot_tex" type="2d" builtin="flat" rgb1="0.10 0.44 0.74" width="64" height="64"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="1 1" reflectance="0.02" shininess="0.02" specular="0.01"/>
    <material name="robot_mat" texture="robot_tex" shininess="0.22" specular="0.18"/>
    <material name="wall_mat" rgba="0.74 0.72 0.69 1" reflectance="0.08" shininess="0.10" specular="0.10"/>
    <material name="rack_blue_mat" rgba="0.22 0.30 0.40 1" specular="0.14"/>
    <material name="rack_frame_mat" rgba="0.50 0.52 0.54 1" specular="0.24"/>
    <material name="rack_shelf_mat" rgba="0.78 0.79 0.80 1" specular="0.14"/>
    <material name="workbench_top_mat" rgba="0.76 0.70 0.62 1" specular="0.16"/>
    <material name="dock_mat" rgba="0.08 0.12 0.18 1"/>
    <material name="safety_yellow_mat" rgba="0.92 0.78 0.16 1"/>
    <material name="warning_orange_mat" rgba="0.90 0.46 0.12 1"/>
    <material name="accent_blue_mat" rgba="0.18 0.46 0.84 1"/>
    <material name="green_bin_mat" rgba="0.18 0.70 0.30 1"/>
    <material name="screen_mat" rgba="0.08 0.12 0.16 1"/>
    <material name="sensor_mat" rgba="0.16 0.18 0.20 1"/>
    <material name="glass_dark_mat" rgba="0.12 0.18 0.24 0.18"/>
    <material name="accent_green_mat" rgba="0.10 0.86 0.42 1"/>
    <material name="glass_mat" rgba="0.68 0.82 0.94 0.08"/>
    <material name="tire_mat" rgba="0.08 0.08 0.08 1"/>
    <material name="aluminum_mat" rgba="0.83 0.85 0.87 1"/>
    <material name="dark_panel_mat" rgba="0.16 0.18 0.20 1"/>
    <material name="arm_white_mat" rgba="0.91 0.93 0.95 1"/>
    <material name="wood_mat" rgba="0.62 0.42 0.22 1"/>
    <material name="metal_mat" rgba="0.62 0.66 0.70 1" specular="0.28"/>
    <material name="bronze_mat" rgba="0.62 0.46 0.24 1" specular="0.32" shininess="0.26"/>
    <material name="warm_glow_mat" rgba="0.98 0.78 0.46 0.72"/>
    <material name="stone_mat" rgba="0.82 0.82 0.80 1" specular="0.22" shininess="0.18"/>
    <material name="walnut_mat" rgba="0.42 0.28 0.18 1" specular="0.14"/>
    <material name="charcoal_mat" rgba="0.12 0.13 0.15 1" specular="0.10"/>
    <material name="linen_mat" rgba="0.92 0.90 0.86 1" specular="0.06"/>
    <material name="emerald_mat" rgba="0.16 0.34 0.28 1" specular="0.10"/>
    <mesh name="guest_body_mesh" file="{_model_asset_path('guest_body_seated.obj')}"/>
    <mesh name="guest_head_mesh" file="{_model_asset_path('guest_head.obj')}"/>
    <mesh name="guest_hair_mesh" file="{_model_asset_path('guest_hair.obj')}"/>
    <mesh name="Pelvis_mesh" file="{_model_asset_path('Pelvis.stl')}" scale="0.58333333 1.03030303 1.03030303"/>
    <mesh name="LowerTrunk_mesh" file="{_model_asset_path('LowerTrunk.stl')}" scale="0.58333333 1.03030303 1.03030303"/>
    <mesh name="UpperTrunk_mesh" file="{_model_asset_path('UpperTrunk.stl')}" scale="0.58333333 1.03030303 1.03030303"/>
    <mesh name="Head_mesh" file="{_model_asset_path('Head.stl')}" scale="1.03030303 1.03030303 1.03030303"/>
    <mesh name="LeftShoulder_mesh" file="{_model_asset_path('LeftShoulder.stl')}" scale="0.64444444 0.64444444 1.03030303"/>
    <mesh name="RightShoulder_mesh" file="{_model_asset_path('RightShoulder.stl')}" scale="0.64444444 0.64444444 1.03030303"/>
    <mesh name="LeftUpperArm_mesh" file="{_model_asset_path('LeftUpperArm.stl')}" scale="0.64444444 0.64444444 1.03030303"/>
    <mesh name="RightUpperArm_mesh" file="{_model_asset_path('RightUpperArm.stl')}" scale="0.64444444 0.64444444 1.03030303"/>
    <mesh name="LeftForeArm_mesh" file="{_model_asset_path('LeftForeArm.stl')}" scale="0.61111111 0.61111111 1.03030303"/>
    <mesh name="RightForeArm_mesh" file="{_model_asset_path('RightForeArm.stl')}" scale="0.61111111 0.61111111 1.03030303"/>
    <mesh name="LeftHand_mesh" file="{_model_asset_path('LeftHand.stl')}" scale="0.9 1.03030303 0.55555556"/>
    <mesh name="RightHand_mesh" file="{_model_asset_path('RightHand.stl')}" scale="0.9 1.03030303 0.55555556"/>
    <mesh name="LeftUpperLeg_mesh" file="{_model_asset_path('LeftUpperLeg.stl')}" scale="0.83333333 0.83333333 1.03030303"/>
    <mesh name="RightUpperLeg_mesh" file="{_model_asset_path('RightUpperLeg.stl')}" scale="0.83333333 0.83333333 1.03030303"/>
    <mesh name="LeftLowerLeg_mesh" file="{_model_asset_path('LeftLowerLeg.stl')}" scale="0.7826087 0.7826087 1.03030303"/>
    <mesh name="RightLowerLeg_mesh" file="{_model_asset_path('RightLowerLeg.stl')}" scale="0.7826087 0.7826087 1.03030303"/>
    <mesh name="LeftFoot_mesh" file="{_model_asset_path('LeftFoot.stl')}" scale="1.03030303 1.03030303 1.03030303"/>
    <mesh name="RightFoot_mesh" file="{_model_asset_path('RightFoot.stl')}" scale="1.03030303 1.03030303 1.03030303"/>
  </asset>

  <default>
    <geom condim="3" conaffinity="1" contype="1" friction="1.1 0.02 0.001" solref="0.01 1"/>
    <joint damping="0.15" armature="0.01"/>
  </default>

  <worldbody>
    <light name="ambient_house" pos="0 0 9" dir="0.10 0.0 -1" directional="true" diffuse="0.26 0.24 0.22"/>
    <light name="reception_light" pos="-6.0 4.8 3.8" diffuse="0.50 0.42 0.32"/>
    <light name="dining_light" pos="1.6 1.6 3.8" diffuse="0.54 0.44 0.34"/>
    <light name="kitchen_light" pos="-1.2 -4.6 3.8" diffuse="0.42 0.44 0.46"/>
    <light name="corridor_light" pos="6.4 3.8 3.8" diffuse="0.48 0.42 0.34"/>
    <light name="dock_light" pos="{DOCK_X:.2f} {DOCK_Y:.2f} 2.6" diffuse="0.16 0.44 0.20"/>
    <light name="service_dock_light" pos="{SERVICE_DOCK_X:.2f} {SERVICE_DOCK_Y:.2f} 2.5" diffuse="0.18 0.52 0.68"/>
    <light name="vip_light" pos="6.25 3.45 3.2" diffuse="0.74 0.60 0.42"/>
    <light name="lounge_light" pos="-5.15 0.75 3.2" diffuse="0.72 0.58 0.40"/>
    <light name="entry_light_upper" pos="8.05 1.35 2.9" diffuse="0.70 0.62 0.46"/>
    <light name="entry_light_lower" pos="8.05 -4.55 2.9" diffuse="0.68 0.60 0.44"/>
    <light name="ceiling_center_light" pos="0.60 0.10 2.64" diffuse="0.48 0.42 0.34"/>
    <light name="ceiling_left_light" pos="-4.90 1.10 2.62" diffuse="0.40 0.36 0.30"/>
    <light name="ceiling_right_light" pos="6.45 3.72 2.62" diffuse="0.42 0.36 0.32"/>
    <light name="reception_accent_light" pos="-6.52 4.88 1.86" diffuse="0.88 0.72 0.50"/>
    <light name="reception_backwash_light" pos="-6.55 5.56 2.06" diffuse="0.62 0.52 0.40"/>
    <light name="vip_fill_light" pos="6.54 3.18 2.20" diffuse="0.74 0.62 0.42"/>
    <light name="lounge_fill_light" pos="-5.16 0.28 2.18" diffuse="0.70 0.58 0.40"/>
    <light name="dining_fill_light" pos="1.20 0.38 2.18" diffuse="0.76 0.62 0.44"/>
    <camera name="overview_cam" pos="0 0 18" xyaxes="1 0 0 0 1 0"/>

    <geom name="floor" type="plane" size="0 0 0.1" material="floor_mat" contype="0" conaffinity="0"/>
    <geom name="north_window_band" type="box" pos="0 7.08 1.90" size="8.90 0.02 0.58" rgba="0.68 0.82 0.94 0.00" contype="0" conaffinity="0"/>
    <geom name="lobby_glass_panel" type="box" pos="-9.66 3.80 1.65" size="0.02 1.30 0.56" rgba="0.68 0.82 0.94 0.00" contype="0" conaffinity="0"/>
{scene_boxes}
{decor_xml}
{dynamic_actor_xml}
{payload_xml}
    <body name="service_bot" mocap="true" pos="{ROBOT2_START_X:.2f} {ROBOT2_START_Y:.2f} 0.11">
      <geom name="service_bot_base" type="box" pos="0 0 0.15" size="0.40 0.28 0.15" rgba="0.24 0.25 0.27 1"/>
      <geom name="service_bot_lower_skirt" type="box" pos="0.01 0 0.09" size="0.42 0.30 0.03" rgba="0.18 0.19 0.21 1" contype="0" conaffinity="0"/>
      <geom name="service_bot_upper_shell" type="capsule" pos="0.06 0 0.30" size="0.20 0.24" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_rear_shell" type="capsule" pos="-0.12 0 0.22" size="0.10 0.14" euler="0 1.57079632679 0" material="charcoal_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_side_fairing_left" type="box" pos="0.02 0.25 0.18" size="0.34 0.03 0.10" material="charcoal_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_side_fairing_right" type="box" pos="0.02 -0.25 0.18" size="0.34 0.03 0.10" material="charcoal_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_body" type="capsule" pos="0.04 0 0.37" size="0.18 0.19" euler="0 1.57079632679 0" rgba="0.98 0.98 0.99 1"/>
      <geom name="service_bot_body_shoulder" type="capsule" pos="0.04 0 0.49" size="0.12 0.14" euler="0 1.57079632679 0" rgba="0.98 0.98 0.99 1" contype="0" conaffinity="0"/>
      <geom name="service_bot_canopy" type="capsule" pos="0.00 0 0.43" size="0.15 0.20" euler="0 1.57079632679 0" material="linen_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_side_shell_left" type="capsule" pos="-0.02 0.24 0.22" size="0.020 0.26" euler="0 0 1.57079632679" material="charcoal_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_side_shell_right" type="capsule" pos="-0.02 -0.24 0.22" size="0.020 0.26" euler="0 0 1.57079632679" material="charcoal_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_front_trim" type="capsule" pos="0.34 0 0.18" size="0.016 0.16" euler="0 1.57079632679 0" material="bronze_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_nose_shell" type="capsule" pos="0.28 0 0.22" size="0.08 0.12" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_screen" type="box" pos="0.20 0 0.39" size="0.020 0.08 0.06" material="glass_dark_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_service_tray" type="capsule" pos="0.03 0 0.53" size="0.13 0.18" euler="0 1.57079632679 0" rgba="0.98 0.98 0.99 1"/>
      <geom name="service_bot_tray_lip_front" type="capsule" pos="0.18 0 0.57" size="0.014 0.14" euler="0 1.57079632679 0" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_tray_lip_left" type="capsule" pos="0.01 0.14 0.57" size="0.014 0.14" euler="0 0 1.57079632679" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_tray_lip_right" type="capsule" pos="0.01 -0.14 0.57" size="0.014 0.14" euler="0 0 1.57079632679" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_top_cowl" type="capsule" pos="0.05 0 0.60" size="0.09 0.10" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_lidar_pod" type="capsule" pos="0.18 0 0.62" size="0.058 0.050" euler="0 1.57079632679 0" rgba="0.08 0.08 0.09 1" contype="0" conaffinity="0"/>
      <geom name="service_bot_lidar_glass" type="capsule" pos="0.18 0 0.65" size="0.060 0.050" euler="0 1.57079632679 0" material="glass_dark_mat" contype="0" conaffinity="0"/>
      <geom name="service_bot_led_bar" type="box" pos="0.25 0 0.25" size="0.018 0.20 0.018" rgba="0.94 0.95 0.96 1" contype="0" conaffinity="0"/>
      <geom name="service_bot_front_bumper" type="box" pos="0.39 0 0.10" size="0.022 0.24 0.05" rgba="0.16 0.17 0.19 1" contype="0" conaffinity="0"/>
      <geom name="service_bot_side_guard_left" type="capsule" pos="0.02 0.29 0.22" size="0.016 0.26" euler="0 0 1.57079632679" rgba="0.18 0.19 0.21 1" contype="0" conaffinity="0"/>
      <geom name="service_bot_side_guard_right" type="capsule" pos="0.02 -0.29 0.22" size="0.016 0.26" euler="0 0 1.57079632679" rgba="0.18 0.19 0.21 1" contype="0" conaffinity="0"/>
      <geom name="service_bot_rear_trim" type="box" pos="-0.21 0 0.22" size="0.04 0.18 0.08" rgba="0.24 0.25 0.27 1" contype="0" conaffinity="0"/>
      <geom name="service_bot_left_wheel" type="cylinder" pos="0 0.29 0.02" size="0.07 0.035" euler="1.57079632679 0 0" rgba="0.08 0.08 0.08 1"/>
      <geom name="service_bot_right_wheel" type="cylinder" pos="0 -0.29 0.02" size="0.07 0.035" euler="1.57079632679 0 0" rgba="0.08 0.08 0.08 1"/>
      <geom name="service_bot_rear_caster" type="sphere" pos="-0.26 0 -0.01" size="0.045" material="tire_mat" contype="0" conaffinity="0"/>
      <site name="service_bot_center_site" pos="0 0 0.30" size="0.02" rgba="0.20 0.78 0.98 0.9"/>
    </body>
    <site name="dock_target" pos="{DOCK_X:.2f} {DOCK_Y:.2f} 0.09" size="0.03" rgba="0.0 1.0 0.0 0.5"/>
    <site name="dock_charge_zone" pos="{DOCK_X:.2f} {DOCK_Y:.2f} 0.05" size="0.58 0.98 0.02" rgba="0.10 0.95 0.30 0.30"/>
    <site name="service_dock_target" pos="{SERVICE_DOCK_X:.2f} {SERVICE_DOCK_Y:.2f} 0.09" size="0.03" rgba="0.2 0.78 1.0 0.6"/>

    <body name="base" pos="{ROBOT_START_X:.2f} {ROBOT_START_Y:.2f} 0.11">
      <joint name="base_x_joint" type="slide" axis="1 0 0" range="{MAIN_BASE_X_RANGE[0]:.2f} {MAIN_BASE_X_RANGE[1]:.2f}" damping="40"/>
      <joint name="base_y_joint" type="slide" axis="0 1 0" range="{MAIN_BASE_Y_RANGE[0]:.2f} {MAIN_BASE_Y_RANGE[1]:.2f}" damping="40"/>
      <joint name="base_yaw_joint" type="hinge" axis="0 0 1" damping="18"/>
      <inertial pos="0 0 0" mass="21.0" diaginertia="0.23 0.27 0.36"/>

      <geom name="collision_shell" type="box" pos="0 0 0.20" size="0.46 0.30 0.18" rgba="0 0 0 0"/>
      <geom name="base_floor" type="box" pos="0 0 0.06" size="0.45 0.28 0.025" material="dark_panel_mat" contype="0" conaffinity="0"/>
      <geom name="top_deck" type="capsule" pos="0.02 0 0.45" size="0.18 0.26" euler="0 1.57079632679 0" rgba="0.92 0.93 0.94 1" contype="0" conaffinity="0"/>
      <geom name="front_nose_panel" type="box" pos="0.43 0.0 0.23" size="0.06 0.30 0.18" rgba="0.90 0.91 0.92 1" contype="0" conaffinity="0"/>
      <geom name="rear_panel" type="box" pos="-0.43 0.0 0.20" size="0.05 0.29 0.16" rgba="0.18 0.20 0.22 1" contype="0" conaffinity="0"/>
      <geom name="side_panel_left" type="box" pos="0.0 0.28 0.20" size="0.38 0.02 0.16" rgba="0.10 0.12 0.14 0.75" contype="0" conaffinity="0"/>
      <geom name="side_panel_right" type="box" pos="0.0 -0.28 0.20" size="0.38 0.02 0.16" rgba="0.10 0.12 0.14 0.75" contype="0" conaffinity="0"/>
      <geom name="side_fairing_left" type="capsule" pos="0.05 0.30 0.22" size="0.026 0.36" euler="0 0 1.57079632679" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="side_fairing_right" type="capsule" pos="0.05 -0.30 0.22" size="0.026 0.36" euler="0 0 1.57079632679" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="nose_shell_upper" type="capsule" pos="0.34 0.0 0.30" size="0.12 0.14" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="rear_shell_lower" type="capsule" pos="-0.28 0.0 0.22" size="0.09 0.12" euler="0 1.57079632679 0" material="charcoal_mat" contype="0" conaffinity="0"/>
      <geom name="body_canopy" type="capsule" pos="0.06 0.0 0.39" size="0.19 0.24" euler="0 1.57079632679 0" material="linen_mat" contype="0" conaffinity="0"/>
      <geom name="beltline_trim" type="capsule" pos="0.08 0.0 0.27" size="0.014 0.28" euler="0 1.57079632679 0" material="bronze_mat" contype="0" conaffinity="0"/>
      <geom name="rear_shell_cap" type="capsule" pos="-0.23 0.0 0.28" size="0.10 0.12" euler="0 1.57079632679 0" material="charcoal_mat" contype="0" conaffinity="0"/>
      <geom name="roof_panel" type="capsule" pos="0.04 0.0 0.56" size="0.12 0.16" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="nose_glow_trim" type="capsule" pos="0.44 0.0 0.28" size="0.012 0.14" euler="0 1.57079632679 0" material="warm_glow_mat" contype="0" conaffinity="0"/>
      <geom name="cargo_side_panel_left" type="capsule" pos="-0.02 0.19 0.45" size="0.014 0.22" euler="0 0 1.57079632679" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="cargo_side_panel_right" type="capsule" pos="-0.02 -0.19 0.45" size="0.014 0.22" euler="0 0 1.57079632679" material="arm_white_mat" contype="0" conaffinity="0"/>
      <geom name="top_shell_spine" type="capsule" pos="0.02 0.0 0.52" size="0.018 0.26" euler="0 1.57079632679 0" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="top_shell_shoulder_left" type="capsule" pos="0.00 0.18 0.46" size="0.016 0.18" euler="0 0 1.57079632679" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="top_shell_shoulder_right" type="capsule" pos="0.00 -0.18 0.46" size="0.016 0.18" euler="0 0 1.57079632679" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="center_battery_module" type="box" pos="0.00 0.00 0.18" size="0.16 0.16 0.11" material="dark_panel_mat" contype="0" conaffinity="0"/>
      <geom name="rear_compute_box" type="box" pos="-0.12 0.00 0.28" size="0.11 0.10 0.08" rgba="0.12 0.14 0.16 1" contype="0" conaffinity="0"/>
      <geom name="status_light" type="sphere" pos="0.34 0.0 0.26" size="0.026" rgba="0.10 0.95 0.40 1" contype="0" conaffinity="0"/>
      <geom name="front_light_bar" type="box" pos="0.41 0.0 0.34" size="0.02 0.18 0.016" rgba="0.20 0.92 0.48 1" contype="0" conaffinity="0"/>
      <geom name="rear_light_bar" type="box" pos="-0.39 0.0 0.30" size="0.015 0.16 0.014" rgba="0.98 0.36 0.14 1" contype="0" conaffinity="0"/>
      <geom name="cargo_rail_left" type="capsule" pos="0.00 0.17 0.56" size="0.014 0.22" euler="0 1.57079632679 0" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="cargo_rail_right" type="capsule" pos="0.00 -0.17 0.56" size="0.014 0.22" euler="0 1.57079632679 0" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="cargo_rail_rear" type="capsule" pos="-0.23 0.0 0.56" size="0.014 0.16" euler="0 0 1.57079632679" material="aluminum_mat" contype="0" conaffinity="0"/>
      <site name="robot_center_site" pos="0 0 0.26" size="0.025" rgba="1.0 1.0 1.0 0.85"/>

      <geom name="front_left_lidar_geom" type="cylinder" pos="0.34 0.25 0.44" size="0.050 0.034" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
      <geom name="rear_right_lidar_geom" type="cylinder" pos="-0.28 -0.25 0.44" size="0.050 0.034" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
      <geom name="front_left_lidar_cap" type="cylinder" pos="0.34 0.25 0.478" size="0.053 0.008" material="glass_dark_mat" contype="0" conaffinity="0"/>
      <geom name="rear_right_lidar_cap" type="cylinder" pos="-0.28 -0.25 0.478" size="0.053 0.008" material="glass_dark_mat" contype="0" conaffinity="0"/>
      <geom name="front_left_lidar_guard" type="capsule" pos="0.30 0.21 0.44" size="0.014 0.09" euler="0 0 0.78539816339" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
      <geom name="rear_right_lidar_guard" type="capsule" pos="-0.24 -0.21 0.44" size="0.014 0.09" euler="0 0 0.78539816339" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>

      <geom name="sensor_mast" type="capsule" pos="0.17 0.0 0.70" size="0.020 0.14" euler="0 0 0" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="sensor_crossbar" type="capsule" pos="0.17 0.0 0.86" size="0.014 0.07" euler="0 1.57079632679 0" material="aluminum_mat" contype="0" conaffinity="0"/>
      <geom name="camera_head" type="capsule" pos="0.27 0.0 0.85" size="0.055 0.050" euler="0 1.57079632679 0" material="sensor_mat" contype="0" conaffinity="0"/>
      <geom name="camera_lens" type="capsule" pos="0.32 0.0 0.85" size="0.014 0.022" euler="0 1.57079632679 0" rgba="0.08 0.22 0.34 1" contype="0" conaffinity="0"/>
      <geom name="camera_glass" type="capsule" pos="0.315 0.0 0.85" size="0.010 0.024" euler="0 1.57079632679 0" material="glass_dark_mat" contype="0" conaffinity="0"/>
      <geom name="camera_status_ring" type="capsule" pos="0.24 0.0 0.90" size="0.010 0.035" euler="0 1.57079632679 0" material="accent_green_mat" contype="0" conaffinity="0"/>
      <geom name="front_bumper" type="box" pos="0.45 0.0 0.10" size="0.025 0.23 0.06" rgba="0.96 0.64 0.12 1" contype="0" conaffinity="0"/>

      <geom name="front_skid" type="sphere" pos="0.26 0 -0.055" size="0.032" material="tire_mat" contype="0" conaffinity="0"/>
      <geom name="rear_left_skid" type="sphere" pos="-0.26 0.13 -0.055" size="0.028" material="tire_mat" contype="0" conaffinity="0"/>
      <geom name="rear_right_skid" type="sphere" pos="-0.26 -0.13 -0.055" size="0.028" material="tire_mat" contype="0" conaffinity="0"/>

      <site name="imu_site" pos="0 0 0.30" size="0.01"/>
{lidar_sites}

      <camera name="front_cam" pos="0.34 0 0.88" xyaxes="0 1 0 -0.05 0 1" fovy="58"/>
      <camera name="left_cam" pos="0.10 0.02 0.58" xyaxes="-1 0 0 0 0 1" fovy="68"/>
      <camera name="right_cam" pos="0.10 -0.02 0.58" xyaxes="1 0 0 0 0 1" fovy="68"/>
      <camera name="rear_cam" pos="-0.32 0.00 0.54" xyaxes="0 -1 0 0 0 1" fovy="68"/>
      <camera name="top_cam" pos="0 0 9" xyaxes="1 0 0 0 1 0"/>

      <body name="arm_base" pos="-0.04 0.00 0.48">
        <joint name="arm_base_joint" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.0"/>
        <geom name="arm_base_pedestal" type="cylinder" pos="0 0 0.08" size="0.07 0.08" material="dark_panel_mat" contype="0" conaffinity="0"/>
        <body name="shoulder_link" pos="0 0 0.16">
          <joint name="shoulder_lift_joint" type="hinge" axis="0 1 0" range="-2.0 1.4" damping="0.8"/>
          <geom name="shoulder_housing" type="capsule" pos="0.04 0 0.08" size="0.06 0.12" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
          <body name="upper_arm_link" pos="0.08 0 0.12">
            <geom name="upper_arm_geom" type="capsule" pos="0.18 0 0.00" size="0.04 0.18" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
            <body name="elbow_link" pos="0.36 0 0.00">
              <joint name="elbow_joint" type="hinge" axis="0 1 0" range="-2.2 2.2" damping="0.7"/>
              <geom name="elbow_housing" type="sphere" pos="0 0 0" size="0.055" material="dark_panel_mat" contype="0" conaffinity="0"/>
              <body name="forearm_link" pos="0.04 0 0">
                <geom name="forearm_geom" type="capsule" pos="0.16 0 0.00" size="0.032 0.16" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
                <body name="wrist_pitch_link" pos="0.32 0 0.00">
                  <joint name="wrist_pitch_joint" type="hinge" axis="0 1 0" range="-2.0 2.0" damping="0.4"/>
                  <geom name="wrist_pitch_housing" type="sphere" pos="0 0 0" size="0.045" material="sensor_mat" contype="0" conaffinity="0"/>
                  <body name="wrist_roll_link" pos="0.06 0 0.00">
                    <joint name="wrist_roll_joint" type="hinge" axis="1 0 0" range="-3.14 3.14" damping="0.3"/>
                    <geom name="wrist_roll_geom" type="capsule" pos="0.07 0 0" size="0.025 0.07" euler="0 1.57079632679 0" material="arm_white_mat" contype="0" conaffinity="0"/>
                    <body name="gripper_body" pos="0.14 0 0">
                      <joint name="gripper_joint" type="slide" axis="0 1 0" range="0 0.04" damping="0.1"/>
                      <geom name="gripper_palm" type="box" pos="0.05 0 0" size="0.04 0.03 0.03" material="dark_panel_mat" contype="0" conaffinity="0"/>
                      <geom name="gripper_finger_left" type="box" pos="0.12 0.035 0" size="0.05 0.008 0.012" material="metal_mat" contype="0" conaffinity="0"/>
                      <geom name="gripper_finger_right" type="box" pos="0.12 -0.035 0" size="0.05 0.008 0.012" material="metal_mat" contype="0" conaffinity="0"/>
                      <site name="tool_site" pos="0.16 0 0" size="0.015" rgba="1 0.4 0 0.85"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <body name="left_wheel" pos="0 0.26 0">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0" damping="0.2" armature="0.03"/>
        <geom name="left_wheel_geom" type="cylinder" size="{WHEEL_RADIUS} 0.035" euler="1.57079632679 0 0" material="tire_mat" contype="0" conaffinity="0"/>
      </body>

      <body name="right_wheel" pos="0 -0.26 0">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0" damping="0.2" armature="0.03"/>
        <geom name="right_wheel_geom" type="cylinder" size="{WHEEL_RADIUS} 0.035" euler="1.57079632679 0 0" material="tire_mat" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>

  <sensor>
    <accelerometer name="imu_accel" site="imu_site"/>
    <gyro name="imu_gyro" site="imu_site"/>
    <framequat name="base_quat" objtype="body" objname="base"/>
{lidar_sensors}
  </sensor>

  <actuator>
    <velocity name="base_x_motor" joint="base_x_joint" kv="35" ctrlrange="-1.2 1.2"/>
    <velocity name="base_y_motor" joint="base_y_joint" kv="35" ctrlrange="-1.2 1.2"/>
    <velocity name="base_yaw_motor" joint="base_yaw_joint" kv="28" ctrlrange="-1.8 1.8"/>
  </actuator>
</mujoco>
"""
