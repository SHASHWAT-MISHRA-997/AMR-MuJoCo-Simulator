import json
import math
import os
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

try:
    import tkinter as tk
except ImportError:
    tk = None

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, TransformStamped, Twist
import mujoco
import mujoco.viewer
from nav_msgs.msg import Odometry
import numpy as np
try:
    from PIL import Image as PilImage
    from PIL import ImageTk
    from PIL import ImageDraw
except ImportError:
    PilImage = None
    ImageTk = None
    ImageDraw = None
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, CameraInfo, Image, Imu, JointState, LaserScan, PointCloud2, PointField
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .config_utils import (
    build_visibility_route,
    build_service_route,
    load_dock_config,
    load_waypoints,
    normalize_service_target_key,
    service_target_goal,
    table_service_goal,
)
from .controllers import (
    ObstacleAwareWaypointNavigator,
    RobotCommand,
    clamp,
    diff_drive_forward,
    diff_drive_inverse,
    wrap_to_pi,
)
from .mjcf_builder import (
    DOCK_X,
    DOCK_Y,
    HOTEL_TABLES,
    HOUSE_BOXES,
    NAVIGATION_BOXES,
    SERVICE_TARGETS,
    SOFA_SPOTS,
    LIDAR_MAX_RANGE,
    LIDAR_MOUNTS,
    PICK_STATION_X,
    PICK_STATION_Y,
    PLACE_STATION_X,
    PLACE_STATION_Y,
    ROBOT_START_X,
    ROBOT_START_Y,
    ROBOT2_START_X,
    ROBOT2_START_Y,
    SERVICE_DOCK_X,
    SERVICE_DOCK_Y,
    SERVICE_DOCK_YAW,
    WORLD_X,
    WORLD_Y,
    WHEEL_RADIUS,
    WHEEL_TRACK,
    build_model_xml,
)


DYNAMIC_OBSTACLE_BODIES = (
    "dynamic_actor_a",
    "dynamic_actor_b",
    "dynamic_actor_c",
)
SERVICE_MISSION_KEYS = tuple(SERVICE_TARGETS.keys())
TABLE_MISSION_KEYS = tuple(HOTEL_TABLES.keys())
BRAND_WINDOW_TITLE = "AMR TWINFLOW"
BRAND_HERO_TITLE = "AMR TWINFLOW"
BRAND_HERO_BADGE = "Dual AMR Ops"
BRAND_HERO_BADGE_ALT = "Restaurant Robotics"
BRAND_SUBTITLE = "Dual-AMR service control for dining, docking,\nand safe floor delivery."
BRAND_PREVIEW_TITLE = "SERVICE FLOOR PREVIEW"
BRAND_POWER_TITLE = "POWER + DOCK HEALTH"
BRAND_CONTROL_TITLE = "SERVICE CONTROL"
BRAND_MONITORING_TITLE = "LIVE AMR STATUS"
BRAND_DETAIL_TITLE = "SERVICE INSIGHTS"
BRAND_EVENT_TITLE = "MISSION TIMELINE"
BRAND_MAP_TITLE = "LIVE AMR TWINFLOW"
LINKEDIN_URL = "https://www.linkedin.com/in/sm980/"
LIGHTING_MODE_KEYS = ("Day", "Evening", "Night", "Spotlight", "Cinema")


@dataclass
class TaskRecord:
    task_id: str
    robot: str
    mission: str
    priority: int
    status: str
    created_at: float
    retries_remaining: int = 1
    started_at: float | None = None
    finished_at: float | None = None
    last_note: str = ""


def quat_wxyz_to_ros_xyzw(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]


def yaw_from_wxyz(quat_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def local_point_to_world(
    base_x: float,
    base_y: float,
    base_yaw: float,
    local_x: float,
    local_y: float,
    local_z: float,
) -> tuple[float, float, float]:
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    return (
        base_x + cos_yaw * local_x - sin_yaw * local_y,
        base_y + sin_yaw * local_x + cos_yaw * local_y,
        local_z,
    )


def mission_display_name(mission_mode: str) -> str:
    mission_key = normalize_service_target_key((mission_mode or "").strip().lower())
    if mission_key == "idle":
        return "Waiting"
    if mission_key in SERVICE_TARGETS:
        return str(SERVICE_TARGETS[mission_key]["label"])
    return mission_key.replace("_", " ").title() if mission_key else "N/A"


class OverviewWindow:
    def __init__(self, width: int = 430, height: int = 560) -> None:
        if tk is None or PilImage is None or ImageTk is None or ImageDraw is None:
            raise RuntimeError("tkinter/Pillow preview dependencies are unavailable")
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title("AMR 360 Follow View")
        self.root.configure(bg="#11161d")
        self.root.resizable(False, False)

        screen_width = self.root.winfo_screenwidth()
        x_pos = max(20, screen_width - width - 80)
        self.root.geometry(f"{width}x{height + 76}+{x_pos}+80")

        header = tk.Label(
            self.root,
            text="AMR 360 / ORBIT + FRONT + LEFT + RIGHT",
            bg="#11161d",
            fg="#8ee6c4",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
            padx=10,
            pady=8,
        )
        header.pack(fill="x")

        self.image_label = tk.Label(self.root, bg="#0b0f14", bd=0, highlightthickness=0)
        self.image_label.pack(fill="both", expand=False, padx=8, pady=6)

        self.info_label = tk.Label(
            self.root,
            text="Waiting for first frame...",
            bg="#11161d",
            fg="#dce7f2",
            font=("Consolas", 10),
            justify="left",
            anchor="w",
            padx=10,
            pady=4,
        )
        self.info_label.pack(fill="x")
        self._photo = None
        self.tile_width = max(180, (self.width - 24) // 2)
        self.tile_height = max(150, (self.height - 74) // 2)

    def update_image(
        self,
        view_images: dict[str, np.ndarray],
        battery_pct: float,
        autonomy_state: str,
        is_charging: bool,
    ) -> None:
        mosaic_width = self.tile_width * 2 + 8
        mosaic_height = self.tile_height * 2 + 8
        image = PilImage.new("RGB", (mosaic_width, mosaic_height), color=(12, 16, 22))
        draw = ImageDraw.Draw(image)

        layout = [
            ("360 ORBIT", "orbit"),
            ("FRONT FOLLOW", "front"),
            ("LEFT VIEW", "left"),
            ("RIGHT VIEW", "right"),
        ]
        for index, (label, key) in enumerate(layout):
            row = index // 2
            col = index % 2
            x0 = col * (self.tile_width + 8)
            y0 = row * (self.tile_height + 8)
            tile = PilImage.fromarray(view_images[key]).resize(
                (self.tile_width, self.tile_height),
                PilImage.Resampling.BILINEAR,
            )
            image.paste(tile, (x0, y0))
            draw.rectangle(
                [(x0, y0), (x0 + self.tile_width - 1, y0 + self.tile_height - 1)],
                outline=(86, 206, 180),
                width=2,
            )
            draw.rectangle([(x0 + 6, y0 + 6), (x0 + 92, y0 + 26)], fill=(8, 14, 20))
            draw.text((x0 + 12, y0 + 10), label, fill=(194, 243, 225))

        self._photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self._photo)
        self.info_label.configure(
            text=(
                f"Battery: {100.0 * battery_pct:5.1f}%   "
                f"Autonomy: {autonomy_state}   "
                f"Charging: {'YES' if is_charging else 'NO'}"
            )
        )

    def pump_events(self) -> None:
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass


class StatusWindow:
    def __init__(self, width: int = 1680, height: int = 920) -> None:
        if tk is None:
            raise RuntimeError("tkinter preview dependencies are unavailable")
        self.root = tk.Tk()
        self.root.title(BRAND_WINDOW_TITLE)
        self.root.configure(bg="#10161f")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        screen_width = max(1280, self.root.winfo_screenwidth())
        screen_height = max(720, self.root.winfo_screenheight())
        window_width = min(width, max(1240, screen_width - 60))
        window_height = min(height, max(760, screen_height - 120))
        x_pos = max(10, (screen_width - window_width) // 2)
        y_pos = max(10, (screen_height - window_height) // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(1200, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()
        except tk.TclError:
            pass

        self.mission_callback = None
        self.speed_callback = None
        self.robot_b_mission_callback = None
        self.robot_b_speed_callback = None
        self.return_home_callback = None
        self.return_home_b_callback = None
        self.lighting_callback = None
        self._button_colors = {}
        self.metric_vars: dict[str, tk.StringVar] = {}
        self.mission_var = tk.StringVar(value="table_1")
        self.speed_var = tk.DoubleVar(value=100.0)
        self.robot_b_mission_var = tk.StringVar(value="table_2")
        self.robot_b_speed_var = tk.DoubleVar(value=100.0)
        self.light_mode_var = tk.StringVar(value="Cinema")
        self.camera_azimuth_var = tk.DoubleVar(value=180.0)
        self.camera_elevation_var = tk.DoubleVar(value=-48.0)
        self.camera_distance_var = tk.DoubleVar(value=24.0)
        self.camera_focus_var = tk.StringVar(value="Center")
        self.graph_canvas = None
        self.graph_hint = None
        self.preview_label = None
        self._photo = None
        self._preview_photo = None
        self._theme_phase = 0.0
        self.speed_var.trace_add("write", self._speed_var_changed)
        self.robot_b_speed_var.trace_add("write", self._speed_var_b_changed)
        self._last_apply_action_at = 0.0

        shell = tk.Frame(self.root, bg="#0d131a")
        shell.pack(fill="both", expand=True)

        self.accent_canvas = tk.Canvas(shell, height=6, bg="#0d131a", highlightthickness=0)
        self.accent_canvas.pack(fill="x", side="top")
        self.accent_line = self.accent_canvas.create_rectangle(0, 0, 2000, 6, fill="#18d3ff", outline="")

        self.panel_collapsed = False
        self.left_outer = tk.Frame(shell, bg="#10161f", width=520)
        self.left_outer.pack(side="left", fill="y")
        self.left_outer.pack_propagate(False)

        self.left_scrollbar = tk.Scrollbar(self.left_outer, orient="vertical", troughcolor="#0c1219", bg="#182433")
        self.left_scrollbar.pack(side="right", fill="y")
        self.left_canvas = tk.Canvas(
            self.left_outer,
            bg="#10161f",
            highlightthickness=0,
            yscrollcommand=self.left_scrollbar.set,
        )
        self.left_canvas.pack(side="left", fill="both", expand=True)
        self.left_scrollbar.configure(command=self.left_canvas.yview)

        left = tk.Frame(self.left_canvas, bg="#10161f")
        self.left_window = self.left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", self._sync_left_scroll_region)
        self.left_canvas.bind("<Configure>", self._sync_left_scroll_width)
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.left_canvas.bind_all("<Button-4>", self._on_mousewheel_linux_up, add="+")
        self.left_canvas.bind_all("<Button-5>", self._on_mousewheel_linux_down, add="+")

        self.right = tk.Frame(shell, bg="#0b0f14")
        self.right.pack(side="right", fill="both", expand=True)

        hero = tk.Frame(left, bg="#111926", highlightbackground="#26435d", highlightthickness=1)
        hero.pack(fill="x", padx=16, pady=(12, 6))

        self.hero_glow = tk.Canvas(hero, height=6, bg="#111926", highlightthickness=0, bd=0)
        self.hero_glow.pack(fill="x", side="top")
        self.hero_glow_left = self.hero_glow.create_rectangle(0, 0, 420, 6, fill="#2ce2ff", outline="")
        self.hero_glow_right = self.hero_glow.create_rectangle(420, 0, 900, 6, fill="#7cf7c7", outline="")

        hero_inner = tk.Frame(hero, bg="#111926")
        hero_inner.pack(fill="x", padx=14, pady=12)

        self.logo_canvas = tk.Canvas(hero_inner, width=86, height=86, bg="#111926", highlightthickness=0)
        self.logo_canvas.pack(side="left", padx=(0, 14))
        self._build_brand_mark()

        hero_text = tk.Frame(hero_inner, bg="#111926")
        hero_text.pack(side="left", fill="x", expand=True)

        self.title_label = tk.Label(
            hero_text,
            text=BRAND_HERO_TITLE,
            bg="#111926",
            fg="#7cf7c7",
            font=("Segoe UI Semibold", 15, "bold"),
            anchor="w",
            pady=4,
            wraplength=320,
            justify="left",
        )
        self.title_label.pack(fill="x")

        developer_row = tk.Frame(hero_text, bg="#111926")
        developer_row.pack(fill="x")
        self.developer_label = tk.Label(
            developer_row,
            text=BRAND_HERO_BADGE,
            bg="#1a2638",
            fg="#dff7ff",
            font=("Segoe UI", 7, "bold"),
            anchor="w",
            pady=4,
            padx=8,
        )
        self.developer_label.pack(side="left")
        self.linkedin_label = tk.Label(
            developer_row,
            text=f"  {BRAND_HERO_BADGE_ALT}",
            bg="#0f3240",
            fg="#8ef0ff",
            font=("Segoe UI", 7, "bold"),
            anchor="w",
            pady=4,
            padx=8,
        )
        self.linkedin_label.pack(side="left", padx=(8, 0))

        credit_row = tk.Frame(hero_text, bg="#111926")
        credit_row.pack(fill="x", pady=(6, 0))
        self.made_by_label = tk.Label(
            credit_row,
            text="Made by SHASHWAT MISHRA",
            bg="#173049",
            fg="#f4fbff",
            font=("Segoe UI", 8, "bold"),
            anchor="w",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        self.made_by_label.pack(side="left")
        self.made_by_label.bind("<Button-1>", lambda _event: self._open_linkedin())
        self.linkedin_credit_label = tk.Label(
            credit_row,
            text="LinkedIn",
            bg="#0f4963",
            fg="#bff5ff",
            font=("Segoe UI", 8, "underline"),
            anchor="w",
            cursor="hand2",
            padx=10,
            pady=4,
        )
        self.linkedin_credit_label.pack(side="left", padx=(8, 0))
        self.linkedin_credit_label.bind("<Button-1>", lambda _event: self._open_linkedin())

        subtitle = tk.Label(
            hero_text,
            text=BRAND_SUBTITLE,
            bg="#111926",
            fg="#94abc3",
            font=("Segoe UI", 8),
            anchor="w",
            pady=6,
            wraplength=300,
            justify="left",
        )
        subtitle.pack(fill="x")

        preview_card = tk.Frame(left, bg="#16202b", highlightbackground="#35506a", highlightthickness=1)
        preview_card.pack(fill="x", padx=16, pady=12)
        self._decorate_card(preview_card, "#2ce2ff", "#7cf7c7")

        preview_title = tk.Label(
            preview_card,
            text=BRAND_PREVIEW_TITLE,
            bg="#16202b",
            fg="#8ee6c4",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=10,
            pady=8,
        )
        preview_title.pack(fill="x")

        self.preview_label = tk.Label(
            preview_card,
            bg="#0b0f14",
            bd=0,
            highlightthickness=0,
            text="Loading dual-robot preview...",
            fg="#7c93a9",
            font=("Segoe UI", 10, "bold"),
        )
        self.preview_label.pack(fill="x", padx=10, pady=10)

        self.camera_value_labels: dict[str, tk.Label] = {}
        try:
            camera_card = tk.Frame(left, bg="#16202b", highlightbackground="#35506a", highlightthickness=1)
            camera_card.pack(fill="x", padx=16, pady=8)
            self._decorate_card(camera_card, "#41c7ff", "#ffd66f")

            camera_title = tk.Label(
                camera_card,
                text="SIMULATION VIEW CONTROLS",
                bg="#16202b",
                fg="#8ee6c4",
                font=("Segoe UI", 11, "bold"),
                anchor="w",
                padx=10,
                pady=8,
            )
            camera_title.pack(fill="x")

            for label_text, variable, key, from_value, to_value, command in (
                ("Rotate", self.camera_azimuth_var, "azimuth", 40, 240, self._on_camera_slide),
                ("Tilt", self.camera_elevation_var, "elevation", -75, -8, self._on_camera_slide),
                ("Zoom +/-", self.camera_distance_var, "distance", 11, 28, self._on_camera_slide),
            ):
                row = tk.Frame(camera_card, bg="#16202b")
                row.pack(fill="x", padx=10, pady=5)
                tk.Label(
                    row,
                    text=label_text,
                    bg="#16202b",
                    fg="#dce7f2",
                    font=("Segoe UI", 10, "bold"),
                    width=7,
                    anchor="w",
                ).pack(side="left")
                scale = tk.Scale(
                    row,
                    from_=from_value,
                    to=to_value,
                    orient="horizontal",
                    resolution=1,
                    showvalue=False,
                    variable=variable,
                    bg="#16202b",
                    fg="#8ee6c4",
                    troughcolor="#09111a",
                    activebackground="#12d8ff",
                    highlightthickness=0,
                    sliderrelief="flat",
                    length=180,
                    command=command,
                )
                scale.pack(side="left", fill="x", expand=True, padx=(0, 8))
                value_label = tk.Label(
                    row,
                    text="--",
                    bg="#16202b",
                    fg="#f1f7ff",
                    font=("Segoe UI", 10, "bold"),
                    width=7,
                )
                value_label.pack(side="left")
                self.camera_value_labels[key] = value_label

            reset_camera_button = tk.Button(
                camera_card,
                text="Reset View",
                bg="#13384d",
                fg="#f3fbff",
                activebackground="#00d0ff",
                activeforeground="#04131d",
                relief="flat",
                font=("Segoe UI", 10, "bold"),
                padx=10,
                pady=5,
                command=self._reset_camera_controls,
            )
            reset_camera_button.pack(anchor="e", padx=10, pady=(2, 10))
            self._add_hover_style(reset_camera_button, "#13384d", "#1b6786")

            preset_row = tk.Frame(camera_card, bg="#16202b")
            preset_row.pack(fill="x", padx=10, pady=(0, 8))
            tk.Label(
                preset_row,
                text="Shots",
                bg="#16202b",
                fg="#dce7f2",
                font=("Segoe UI", 10, "bold"),
                width=7,
                anchor="w",
            ).pack(side="left")
            preset_specs = (
                ("Dock", "dock"),
                ("Dining", "dining"),
                ("VIP", "vip"),
                ("Entry", "entrance"),
                ("AMR-A", "amr_a"),
                ("AMR-B", "amr_b"),
            )
            for label, preset_key in preset_specs:
                button = tk.Button(
                    preset_row,
                    text=label,
                    bg="#1f2d3c",
                    fg="#f3fbff",
                    activebackground="#7cf7c7",
                    activeforeground="#081017",
                    relief="flat",
                    font=("Segoe UI", 8, "bold"),
                    padx=8,
                    pady=4,
                    command=lambda key=preset_key: self._apply_named_camera_preset(key),
                )
                button.pack(side="left", padx=(0, 4))
                self._add_hover_style(button, "#1f2d3c", "#2f6d62")

            lighting_row = tk.Frame(camera_card, bg="#16202b")
            lighting_row.pack(fill="x", padx=10, pady=(0, 10))
            tk.Label(
                lighting_row,
                text="Lights",
                bg="#16202b",
                fg="#dce7f2",
                font=("Segoe UI", 10, "bold"),
                width=7,
                anchor="w",
            ).pack(side="left")
            lighting_menu = tk.OptionMenu(
                lighting_row,
                self.light_mode_var,
                *LIGHTING_MODE_KEYS,
            )
            lighting_menu.configure(
                bg="#0f1b27",
                fg="#e7f3ff",
                activebackground="#2a4f7b",
                activeforeground="#ffffff",
                relief="flat",
                highlightthickness=0,
                width=14,
                font=("Segoe UI", 10),
            )
            lighting_menu["menu"].configure(
                bg="#132131",
                fg="#e7f3ff",
                activebackground="#23536d",
                activeforeground="#ffffff",
            )
            lighting_menu.pack(side="left", padx=(0, 8))
            lighting_menu.configure(takefocus=1)
            lighting_apply = tk.Button(
                lighting_row,
                text="Apply Lights",
                bg="#23364f",
                fg="#f3fbff",
                activebackground="#66c2ff",
                activeforeground="#04131d",
                relief="flat",
                font=("Segoe UI", 10, "bold"),
                padx=10,
                pady=5,
                command=self._apply_lighting_mode,
            )
            lighting_apply.pack(side="left")
            self._add_hover_style(lighting_apply, "#23364f", "#2f6fb0")

        except Exception:
            self.camera_value_labels = {}

        battery_card = tk.Frame(left, bg="#16202b", highlightbackground="#35506a", highlightthickness=1)
        battery_card.pack(fill="x", padx=16, pady=8)
        self._decorate_card(battery_card, "#7cf7c7", "#2ce2ff")

        battery_title = tk.Label(
            battery_card,
            text=BRAND_POWER_TITLE,
            bg="#16202b",
            fg="#8ee6c4",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=10,
            pady=8,
        )
        battery_title.pack(fill="x")

        self.battery_canvas_a = tk.Canvas(battery_card, height=24, bg="#0f1721", highlightthickness=0)
        self.battery_canvas_a.pack(fill="x", padx=10, pady=4)
        self.battery_canvas_b = tk.Canvas(battery_card, height=24, bg="#0f1721", highlightthickness=0)
        self.battery_canvas_b.pack(fill="x", padx=10, pady=(4, 10))
        self.battery_bar_a = self.battery_canvas_a.create_rectangle(4, 4, 100, 20, fill="#24d66b", outline="")
        self.battery_text_a = self.battery_canvas_a.create_text(12, 12, text="AMR-A 100%", anchor="w", fill="#ffffff")
        self.battery_bar_b = self.battery_canvas_b.create_rectangle(4, 4, 100, 20, fill="#29a3ff", outline="")
        self.battery_text_b = self.battery_canvas_b.create_text(12, 12, text="AMR-B 100%", anchor="w", fill="#ffffff")

        control_card = tk.Frame(left, bg="#16202b", highlightbackground="#35506a", highlightthickness=1)
        control_card.pack(fill="x", padx=16, pady=8)
        self._decorate_card(control_card, "#ffd66f", "#ff9867")

        control_title = tk.Label(
            control_card,
            text=BRAND_CONTROL_TITLE,
            bg="#16202b",
            fg="#8ee6c4",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=10,
            pady=8,
        )
        control_title.pack(fill="x")

        mission_row = tk.Frame(control_card, bg="#16202b")
        mission_row.pack(fill="x", padx=10, pady=6)

        mission_label = tk.Label(
            mission_row,
            text="AMR-A",
            bg="#16202b",
            fg="#dce7f2",
            font=("Segoe UI", 10, "bold"),
            width=10,
            anchor="w",
        )
        mission_label.pack(side="left")

        mission_menu = tk.OptionMenu(
            mission_row,
            self.mission_var,
            *SERVICE_MISSION_KEYS,
        )
        mission_menu.configure(
            bg="#0f1b27",
            fg="#e7f3ff",
            activebackground="#2a4f7b",
            activeforeground="#ffffff",
            relief="flat",
            highlightthickness=0,
            width=18,
            font=("Segoe UI", 10),
        )
        mission_menu["menu"].configure(
            bg="#132131",
            fg="#e7f3ff",
            activebackground="#23536d",
            activeforeground="#ffffff",
        )
        mission_menu.pack(side="left", padx=(0, 8))
        mission_menu.configure(takefocus=1)

        self.mission_apply_button = tk.Button(
            mission_row,
            text="Apply",
            bg="#13384d",
            fg="#f3fbff",
            activebackground="#00d0ff",
            activeforeground="#04131d",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=5,
            command=self._apply_mission,
        )
        self.mission_apply_button.pack(side="left")
        self._add_hover_style(self.mission_apply_button, "#13384d", "#1b6786")

        speed_row = tk.Frame(control_card, bg="#16202b")
        speed_row.pack(fill="x", padx=10, pady=6)

        speed_label = tk.Label(
            speed_row,
            text="AMR-A Speed",
            bg="#16202b",
            fg="#dce7f2",
            font=("Segoe UI", 10, "bold"),
            width=10,
            anchor="w",
        )
        speed_label.pack(side="left")

        self.speed_scale = tk.Scale(
            speed_row,
            from_=30,
            to=125,
            orient="horizontal",
            resolution=5,
            variable=self.speed_var,
            bg="#16202b",
            fg="#8ee6c4",
            troughcolor="#09111a",
            activebackground="#12d8ff",
            highlightthickness=0,
            sliderrelief="flat",
            length=170,
            command=self._on_speed_slide,
            showvalue=True,
        )
        self.speed_scale.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.speed_scale.bind("<B1-Motion>", lambda _event: self._speed_var_changed())
        self.speed_scale.bind("<ButtonRelease-1>", lambda _event: self._apply_speed())

        self.speed_value_label = tk.Label(
            speed_row,
            text="100%",
            bg="#16202b",
            fg="#f1f7ff",
            font=("Segoe UI", 10, "bold"),
            width=6,
        )
        self.speed_value_label.pack(side="left", padx=(0, 6))

        self.speed_apply_button = tk.Button(
            speed_row,
            text="Set",
            bg="#2f1944",
            fg="#f8f0ff",
            activebackground="#ff5df4",
            activeforeground="#140019",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=5,
            command=self._apply_speed,
        )
        self.speed_apply_button.pack(side="left")
        self._add_hover_style(self.speed_apply_button, "#2f1944", "#6a2e92")

        home_row = tk.Frame(control_card, bg="#16202b")
        home_row.pack(fill="x", padx=10, pady=(2, 8))

        self.return_home_button = tk.Button(
            home_row,
            text="AMR-A Return Home",
            bg="#24492f",
            fg="#f4fff5",
            activebackground="#51d17a",
            activeforeground="#07130b",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=5,
            command=self._return_home,
        )
        self.return_home_button.pack(side="left")
        self._add_hover_style(self.return_home_button, "#24492f", "#2f6d45")

        mission_row_b = tk.Frame(control_card, bg="#16202b")
        mission_row_b.pack(fill="x", padx=10, pady=6)

        mission_label_b = tk.Label(
            mission_row_b,
            text="AMR-B",
            bg="#16202b",
            fg="#dce7f2",
            font=("Segoe UI", 10, "bold"),
            width=10,
            anchor="w",
        )
        mission_label_b.pack(side="left")

        mission_menu_b = tk.OptionMenu(
            mission_row_b,
            self.robot_b_mission_var,
            *SERVICE_MISSION_KEYS,
        )
        mission_menu_b.configure(
            bg="#0f1b27",
            fg="#e7f3ff",
            activebackground="#2a4f7b",
            activeforeground="#ffffff",
            relief="flat",
            highlightthickness=0,
            width=18,
            font=("Segoe UI", 10),
        )
        mission_menu_b["menu"].configure(
            bg="#132131",
            fg="#e7f3ff",
            activebackground="#23536d",
            activeforeground="#ffffff",
        )
        mission_menu_b.pack(side="left", padx=(0, 8))
        mission_menu_b.configure(takefocus=1)

        self.mission_b_apply_button = tk.Button(
            mission_row_b,
            text="Apply",
            bg="#24354f",
            fg="#f3fbff",
            activebackground="#59a6ff",
            activeforeground="#04131d",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=5,
            command=self._apply_robot_b_mission,
        )
        self.mission_b_apply_button.pack(side="left")
        self._add_hover_style(self.mission_b_apply_button, "#24354f", "#2f6fb0")

        speed_row_b = tk.Frame(control_card, bg="#16202b")
        speed_row_b.pack(fill="x", padx=10, pady=6)

        speed_label_b = tk.Label(
            speed_row_b,
            text="AMR-B Speed",
            bg="#16202b",
            fg="#dce7f2",
            font=("Segoe UI", 10, "bold"),
            width=10,
            anchor="w",
        )
        speed_label_b.pack(side="left")

        self.speed_scale_b = tk.Scale(
            speed_row_b,
            from_=30,
            to=125,
            orient="horizontal",
            resolution=5,
            variable=self.robot_b_speed_var,
            bg="#16202b",
            fg="#8ee6c4",
            troughcolor="#09111a",
            activebackground="#12d8ff",
            highlightthickness=0,
            sliderrelief="flat",
            length=170,
            command=self._on_speed_slide_b,
            showvalue=True,
        )
        self.speed_scale_b.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.speed_scale_b.bind("<B1-Motion>", lambda _event: self._speed_var_b_changed())
        self.speed_scale_b.bind("<ButtonRelease-1>", lambda _event: self._apply_robot_b_speed())

        self.speed_value_label_b = tk.Label(
            speed_row_b,
            text="95%",
            bg="#16202b",
            fg="#f1f7ff",
            font=("Segoe UI", 10, "bold"),
            width=6,
        )
        self.speed_value_label_b.pack(side="left", padx=(0, 6))

        self.speed_b_apply_button = tk.Button(
            speed_row_b,
            text="Set",
            bg="#143a32",
            fg="#f3fbff",
            activebackground="#57f0c8",
            activeforeground="#04131d",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=5,
            command=self._apply_robot_b_speed,
        )
        self.speed_b_apply_button.pack(side="left")
        self._add_hover_style(self.speed_b_apply_button, "#143a32", "#1d6a5d")

        home_row_b = tk.Frame(control_card, bg="#16202b")
        home_row_b.pack(fill="x", padx=10, pady=(2, 8))

        self.return_home_button_b = tk.Button(
            home_row_b,
            text="AMR-B Return Home",
            bg="#24492f",
            fg="#f4fff5",
            activebackground="#51d17a",
            activeforeground="#07130b",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=5,
            command=self._return_home_b,
        )
        self.return_home_button_b.pack(side="left")
        self._add_hover_style(self.return_home_button_b, "#24492f", "#2f6d45")

        metric_card = tk.Frame(left, bg="#16202b", highlightbackground="#35506a", highlightthickness=1)
        metric_card.pack(fill="x", padx=16, pady=8)
        self._decorate_card(metric_card, "#8ef0ff", "#7cf7c7")

        metric_title = tk.Label(
            metric_card,
            text=BRAND_MONITORING_TITLE,
            bg="#16202b",
            fg="#8ee6c4",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=10,
            pady=8,
        )
        metric_title.pack(fill="x")

        metric_grid = tk.Frame(metric_card, bg="#16202b")
        metric_grid.pack(fill="x", padx=10, pady=6)
        metric_fields = [
            ("AMR-A Battery", "battery"),
            ("AMR-A Target", "mission"),
            ("AMR-A State", "autonomy"),
            ("AMR-A Dock", "dock"),
            ("AMR-A Speed", "speed"),
            ("AMR-A Goal", "goal"),
            ("AMR-B Battery", "robot_b_battery"),
            ("AMR-B Target", "robot_b_mission"),
            ("AMR-B State", "robot_b_state"),
            ("AMR-B Dock", "robot_b_dock"),
            ("AMR-B Speed", "robot_b_speed"),
            ("AMR-B Goal", "robot_b_goal"),
        ]
        for index, (label_text, key) in enumerate(metric_fields):
            card = tk.Frame(metric_grid, bg="#0f1721", highlightbackground="#25364a", highlightthickness=1)
            card.grid(row=index // 2, column=index % 2, padx=4, pady=4, sticky="nsew")
            label = tk.Label(
                card,
                text=label_text.upper(),
                bg="#0f1721",
                fg="#7b95b1",
                font=("Segoe UI", 8, "bold"),
                anchor="w",
                padx=8,
                pady=4,
            )
            label.pack(fill="x")
            value_var = tk.StringVar(value="--")
            value = tk.Label(
                card,
                textvariable=value_var,
                bg="#0f1721",
                fg="#f2f7ff",
                font=("Segoe UI", 10, "bold"),
                anchor="w",
                padx=8,
                pady=6,
            )
            value.pack(fill="x")
            self.metric_vars[key] = value_var
        metric_grid.grid_columnconfigure(0, weight=1)
        metric_grid.grid_columnconfigure(1, weight=1)

        lower_section = tk.Frame(left, bg="#10161f")
        lower_section.pack(fill="both", expand=True, padx=16, pady=10)

        detail_card = tk.Frame(lower_section, bg="#16202b", highlightbackground="#35506a", highlightthickness=1)
        detail_card.pack(fill="both", expand=True, side="top")
        self._decorate_card(detail_card, "#2ce2ff", "#7cf7c7")

        detail_title = tk.Label(
            detail_card,
            text=BRAND_DETAIL_TITLE,
            bg="#16202b",
            fg="#8ee6c4",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=10,
            pady=8,
        )
        detail_title.pack(fill="x")

        self.body = tk.Text(
            detail_card,
            bg="#0d131b",
            fg="#dce7f2",
            relief="flat",
            highlightthickness=0,
            wrap="word",
            font=("Consolas", 10),
            height=12,
            padx=10,
            pady=10,
        )
        self.body.pack(fill="both", expand=True, padx=8, pady=8)
        self.body.insert("1.0", "Waiting for simulator...")
        self.body.configure(state="disabled")

        event_card = tk.Frame(lower_section, bg="#16202b", highlightbackground="#35506a", highlightthickness=1)
        event_card.pack(fill="both", expand=False, side="bottom", pady=8)
        self._decorate_card(event_card, "#ffd66f", "#2ce2ff")

        event_title = tk.Label(
            event_card,
            text=BRAND_EVENT_TITLE,
            bg="#16202b",
            fg="#8ee6c4",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=10,
            pady=8,
        )
        event_title.pack(fill="x")

        self.event_text = tk.Text(
            event_card,
            bg="#0d131b",
            fg="#dce7f2",
            relief="flat",
            highlightthickness=0,
            wrap="word",
            font=("Consolas", 10),
            height=7,
            padx=10,
            pady=10,
        )
        self.event_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.event_text.insert("1.0", "AMR TwinFlow command deck online.")
        self.event_text.configure(state="disabled")

        map_header = tk.Frame(self.right, bg="#0d131a")
        map_header.pack(fill="x")

        map_header_row = tk.Frame(map_header, bg="#0d131a")
        map_header_row.pack(fill="x")

        sim_title = tk.Label(
            map_header_row,
            text=BRAND_MAP_TITLE,
            bg="#0d131a",
            fg="#dce7f2",
            font=("Segoe UI Semibold", 12, "bold"),
            anchor="w",
            padx=14,
            pady=10,
        )
        sim_title.pack(side="left", fill="x", expand=True)

        self.panel_toggle_button = tk.Button(
            map_header_row,
            text="Hide Panel",
            command=self._toggle_left_panel,
            bg="#163044",
            fg="#dff7ff",
            activebackground="#1c4a67",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            cursor="hand2",
        )
        self.panel_toggle_button.pack(side="right", padx=(8, 12), pady=8)

        self.map_glow = tk.Canvas(map_header, height=4, bg="#0d131a", highlightthickness=0, bd=0)
        self.map_glow.pack(fill="x")
        self.map_glow_left = self.map_glow.create_rectangle(0, 0, 780, 4, fill="#2ce2ff", outline="")
        self.map_glow_right = self.map_glow.create_rectangle(780, 0, 1600, 4, fill="#7cf7c7", outline="")

        self.image_label = tk.Label(
            self.right,
            bg="#0b0f14",
            bd=0,
            highlightthickness=0,
            text="Initializing AMR TwinFlow live renderer...",
            fg="#8aa9c6",
            font=("Segoe UI", 12, "bold"),
        )
        self.image_label.pack(fill="both", expand=True, padx=14, pady=14)
        self._photo = None
        self._preview_photo = None
        self._reset_camera_controls()

    def _decorate_card(self, card, accent_left: str, accent_right: str) -> None:
        accent = tk.Canvas(card, height=5, bg=card.cget("bg"), highlightthickness=0, bd=0)
        accent.pack(fill="x", side="top")
        accent.create_rectangle(0, 0, 170, 5, fill=accent_left, outline="")
        accent.create_rectangle(170, 0, 520, 5, fill=accent_right, outline="")

    def _build_brand_mark(self) -> None:
        self.logo_orbit = self.logo_canvas.create_oval(10, 10, 76, 76, outline="#2ce2ff", width=3)
        self.logo_lane = self.logo_canvas.create_oval(20, 20, 66, 66, outline="#7cf7c7", width=2)
        self.logo_robot_a = self.logo_canvas.create_rectangle(20, 30, 36, 56, fill="#14d1ff", outline="")
        self.logo_robot_b = self.logo_canvas.create_rectangle(50, 30, 66, 56, fill="#7cf7c7", outline="")
        self.logo_robot_a_cap = self.logo_canvas.create_oval(19, 21, 37, 35, fill="#eef6ff", outline="")
        self.logo_robot_b_cap = self.logo_canvas.create_oval(49, 21, 67, 35, fill="#eef6ff", outline="")
        self.logo_bridge = self.logo_canvas.create_line(36, 43, 50, 43, fill="#ffd66f", width=4, smooth=True)
        self.logo_signal_a = self.logo_canvas.create_oval(26, 16, 30, 20, fill="#2ce2ff", outline="")
        self.logo_signal_b = self.logo_canvas.create_oval(56, 16, 60, 20, fill="#7cf7c7", outline="")
        self.logo_items = [
            self.logo_orbit,
            self.logo_lane,
            self.logo_robot_a,
            self.logo_robot_b,
            self.logo_robot_a_cap,
            self.logo_robot_b_cap,
            self.logo_bridge,
            self.logo_signal_a,
            self.logo_signal_b,
        ]

    def _sync_left_scroll_region(self, _event=None) -> None:
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _sync_left_scroll_width(self, event) -> None:
        self.left_canvas.itemconfigure(self.left_window, width=event.width)

    def _toggle_left_panel(self) -> None:
        if self.panel_collapsed:
            self.left_outer.pack(side="left", fill="y", before=self.right)
            self.panel_toggle_button.configure(text="Hide Panel")
            self.panel_collapsed = False
        else:
            self.left_outer.pack_forget()
            self.panel_toggle_button.configure(text="Show Panel")
            self.panel_collapsed = True

    def _on_mousewheel(self, event) -> None:
        if not self.left_canvas.winfo_exists():
            return
        self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux_up(self, _event) -> None:
        if self.left_canvas.winfo_exists():
            self.left_canvas.yview_scroll(-1, "units")

    def _on_mousewheel_linux_down(self, _event) -> None:
        if self.left_canvas.winfo_exists():
            self.left_canvas.yview_scroll(1, "units")

    def _fit_image_to_widget(
        self,
        rgb_image: np.ndarray,
        widget,
        min_width: int,
        min_height: int,
        mode: str = "contain",
    ) -> "PilImage.Image | None":
        if PilImage is None:
            return None
        widget.update_idletasks()
        target_width = max(min_width, int(widget.winfo_width() or min_width))
        target_height = max(min_height, int(widget.winfo_height() or min_height))
        source = PilImage.fromarray(rgb_image)
        if mode == "fit_width":
            scale = target_width / max(source.width, 1)
        else:
            scale = min(target_width / max(source.width, 1), target_height / max(source.height, 1))
        scaled_width = max(1, int(source.width * scale))
        scaled_height = max(1, int(source.height * scale))
        resized = source.resize((scaled_width, scaled_height), PilImage.Resampling.BILINEAR)
        canvas = PilImage.new("RGB", (target_width, target_height), color=(9, 13, 18))
        offset_x = max(0, (target_width - scaled_width) // 2)
        if mode == "fit_width" and scaled_height > target_height:
            crop_top = max(0, (scaled_height - target_height) // 2)
            resized = resized.crop((0, crop_top, scaled_width, crop_top + target_height))
            scaled_height = target_height
        offset_y = max(0, (target_height - scaled_height) // 2)
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    def _add_hover_style(self, button: tk.Button, base_bg: str, hover_bg: str) -> None:
        self._button_colors[button] = (base_bg, hover_bg)

        def _enter(_event):
            button.configure(bg=hover_bg)

        def _leave(_event):
            button.configure(bg=base_bg)

        button.bind("<Enter>", _enter)
        button.bind("<Leave>", _leave)

    def _open_linkedin(self) -> None:
        url = LINKEDIN_URL
        open_commands = [
            ["powershell.exe", "-Command", f"Start-Process '{url}'"],
            ["cmd.exe", "/c", "start", "", url],
            ["xdg-open", url],
            ["gio", "open", url],
        ]
        for command in open_commands:
            try:
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
        try:
            if webbrowser.open(url, new=2):
                return
        except Exception:
            pass
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(url)
            self.root.update_idletasks()
            self.linkedin_credit_label.configure(text="LinkedIn URL copied")
            self.root.after(1800, lambda: self.linkedin_credit_label.configure(text="LinkedIn"))
        except Exception:
            pass

    def set_mission_callback(self, callback) -> None:
        self.mission_callback = callback

    def set_speed_callback(self, callback) -> None:
        self.speed_callback = callback

    def set_robot_b_mission_callback(self, callback) -> None:
        self.robot_b_mission_callback = callback

    def set_robot_b_speed_callback(self, callback) -> None:
        self.robot_b_speed_callback = callback

    def set_return_home_callback(self, callback) -> None:
        self.return_home_callback = callback

    def set_return_home_b_callback(self, callback) -> None:
        self.return_home_b_callback = callback

    def set_lighting_callback(self, callback) -> None:
        self.lighting_callback = callback

    def _apply_mission(self) -> None:
        self._mark_interaction()
        if self.mission_callback is not None:
            self.mission_callback(str(self.mission_var.get()))

    def _speed_var_changed(self, *_args) -> None:
        self.speed_value_label.configure(text=f"{int(round(self.speed_var.get()))}%")

    def _on_speed_slide(self, value: str) -> None:
        self.speed_value_label.configure(text=f"{int(float(value))}%")

    def _apply_speed(self) -> None:
        self._mark_interaction()
        self.speed_value_label.configure(text=f"{int(self.speed_var.get())}%")
        if self.speed_callback is not None:
            self.speed_callback(float(self.speed_var.get()) / 100.0)

    def _apply_robot_b_mission(self) -> None:
        self._mark_interaction()
        if self.robot_b_mission_callback is not None:
            self.robot_b_mission_callback(str(self.robot_b_mission_var.get()))

    def _speed_var_b_changed(self, *_args) -> None:
        self.speed_value_label_b.configure(text=f"{int(round(self.robot_b_speed_var.get()))}%")

    def _on_speed_slide_b(self, value: str) -> None:
        self.speed_value_label_b.configure(text=f"{int(float(value))}%")

    def _apply_robot_b_speed(self) -> None:
        self._mark_interaction()
        self.speed_value_label_b.configure(text=f"{int(self.robot_b_speed_var.get())}%")
        if self.robot_b_speed_callback is not None:
            self.robot_b_speed_callback(float(self.robot_b_speed_var.get()) / 100.0)

    def _return_home(self) -> None:
        self._mark_interaction()
        if self.return_home_callback is not None:
            self.return_home_callback()

    def _return_home_b(self) -> None:
        self._mark_interaction()
        if self.return_home_b_callback is not None:
            self.return_home_b_callback()

    def _apply_lighting_mode(self) -> None:
        self._mark_interaction()
        if self.lighting_callback is not None:
            self.lighting_callback(str(self.light_mode_var.get()))

    def _mark_interaction(self) -> None:
        self._last_apply_action_at = time.perf_counter()
        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def _reset_camera_controls(self) -> None:
        self.camera_focus_var.set("Center")
        self._apply_camera_preset(180.0, -48.0, 24.0)

    def _apply_camera_preset(self, azimuth: float, elevation: float, distance: float) -> None:
        self.camera_azimuth_var.set(azimuth)
        self.camera_elevation_var.set(elevation)
        self.camera_distance_var.set(distance)
        self._refresh_camera_labels()

    def _apply_named_camera_preset(self, preset_key: str) -> None:
        presets = {
            "dock": (150.0, -34.0, 10.8, "Dock"),
            "dining": (164.0, -46.0, 12.8, "Dining"),
            "vip": (118.0, -30.0, 8.8, "VIP"),
            "entrance": (208.0, -28.0, 9.6, "Entrance"),
            "amr_a": (136.0, -24.0, 6.4, "AMR-A"),
            "amr_b": (58.0, -18.0, 5.2, "AMR-B"),
        }
        azimuth, elevation, distance, focus_key = presets.get(preset_key, (180.0, -48.0, 24.0, "Center"))
        self.camera_focus_var.set(focus_key)
        self._apply_camera_preset(azimuth, elevation, distance)

    def _on_camera_slide(self, _value: str) -> None:
        self._refresh_camera_labels()

    def _refresh_camera_labels(self) -> None:
        if not self.camera_value_labels:
            return
        self.camera_value_labels["azimuth"].configure(text=f"{int(round(self.camera_azimuth_var.get()))} deg")
        self.camera_value_labels["elevation"].configure(text=f"{int(round(self.camera_elevation_var.get()))} deg")
        self.camera_value_labels["distance"].configure(text=f"{self.camera_distance_var.get():.1f}x")

    def get_camera_config(self) -> tuple[float, float, float, str]:
        return (
            float(self.camera_azimuth_var.get()),
            float(self.camera_elevation_var.get()),
            float(self.camera_distance_var.get()),
            str(self.camera_focus_var.get()),
        )

    def update_image(self, rgb_image: np.ndarray) -> None:
        if PilImage is None or ImageTk is None:
            return
        view = self._fit_image_to_widget(rgb_image, self.image_label, 1120, 760, mode="fit_width")
        if view is None:
            return
        self._photo = ImageTk.PhotoImage(image=view)
        self.image_label.configure(image=self._photo, text="")

    def update_preview_image(self, rgb_image: np.ndarray) -> None:
        if PilImage is None or ImageTk is None:
            return
        preview = self._fit_image_to_widget(rgb_image, self.preview_label, 404, 286)
        if preview is None:
            return
        self._preview_photo = ImageTk.PhotoImage(image=preview)
        self.preview_label.configure(image=self._preview_photo, text="")

    def update_text(self, text: str) -> None:
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", text)
        self.body.configure(state="disabled")

    def update_dashboard(self, data: dict[str, object]) -> None:
        for key in (
            "battery",
            "mission",
            "autonomy",
            "dock",
            "speed",
            "goal",
            "robot_b_battery",
            "robot_b_mission",
            "robot_b_state",
            "robot_b_dock",
            "robot_b_speed",
            "robot_b_goal",
        ):
            if key in self.metric_vars:
                self.metric_vars[key].set(str(data.get(key, "--")))
        speed_pct = int(round(100.0 * float(data.get("speed_scale", 0.85))))
        self.speed_value_label.configure(text=f"{max(30, min(125, speed_pct))}%")
        speed_b_pct = int(round(100.0 * float(data.get("robot_b_speed_scale", 0.80))))
        self.speed_value_label_b.configure(text=f"{max(30, min(125, speed_b_pct))}%")
        self._update_battery_canvas(
            self.battery_canvas_a,
            self.battery_bar_a,
            self.battery_text_a,
            float(data.get("battery_ratio", 1.0)),
            f"AMR-A  {data.get('battery', '--')}  {data.get('dock', '')}",
            charging=bool(data.get("charging_a", False)),
            accent="#2ad96b",
        )
        self._update_battery_canvas(
            self.battery_canvas_b,
            self.battery_bar_b,
            self.battery_text_b,
            float(data.get("robot_b_battery_ratio", 1.0)),
            f"AMR-B  {data.get('robot_b_battery', '--')}  {data.get('robot_b_dock', '')}",
            charging=bool(data.get("charging_b", False)),
            accent="#39a8ff",
        )

        self.update_text(str(data.get("details_text", "")))
        events = list(data.get("events", []))
        self.event_text.configure(state="normal")
        self.event_text.delete("1.0", "end")
        self.event_text.insert("1.0", "\n".join(events) if events else "No recent mission events.")
        self.event_text.configure(state="disabled")
        history_a = self._coerce_battery_history(data.get("battery_history_a", []), data.get("battery_ratio", 1.0))
        history_b = self._coerce_battery_history(data.get("battery_history_b", []), data.get("robot_b_battery_ratio", 1.0))
        try:
            if history_a or history_b:
                self._update_status_graph(history_a, history_b)
            else:
                self._draw_graph_placeholder("Waiting for fleet battery history...")
        except Exception:
            self._draw_graph_placeholder("AMR graph unavailable")
        if self.graph_hint is not None:
            self.graph_hint.configure(
                text=(
                    f"Battery trend: AMR-A {data.get('battery', '--')}, "
                    f"AMR-B {data.get('robot_b_battery', '--')}"
                )
            )
        if self._photo is None:
            self.image_label.configure(text=str(data.get("renderer_status", "Live simulation renderer unavailable.")))
        if self.preview_label is not None and self._preview_photo is None:
            self.preview_label.configure(text=str(data.get("preview_status", "Dual-robot preview unavailable.")))

    def _coerce_battery_history(self, values, fallback_ratio: object) -> list[float]:
        history: list[float] = []
        if isinstance(values, list):
            for value in values[-80:]:
                try:
                    history.append(max(0.0, min(1.0, float(value))))
                except (TypeError, ValueError):
                    continue
        if history:
            return history
        try:
            ratio = max(0.0, min(1.0, float(fallback_ratio)))
        except (TypeError, ValueError):
            ratio = 0.0
        return [ratio]

    def _update_battery_canvas(self, canvas, bar_id, text_id, ratio: float, text: str, charging: bool, accent: str) -> None:
        canvas.update_idletasks()
        width = max(220, canvas.winfo_width() or 260)
        ratio = max(0.0, min(1.0, ratio))
        fill_width = 4 + ratio * (width - 8)
        canvas.coords(bar_id, 4, 4, fill_width, 20)
        fill = "#2df27c" if charging else accent
        canvas.itemconfigure(bar_id, fill=fill)
        canvas.coords(text_id, 12, 12)
        canvas.itemconfigure(text_id, text=text)

    def _draw_graph_placeholder(self, message: str) -> None:
        if self.graph_canvas is None:
            return
        canvas = self.graph_canvas
        canvas.delete("all")
        canvas.configure(bg="#0c1420")
        canvas.update_idletasks()
        width = max(260, int(canvas.winfo_width() or 420))
        height = max(120, int(canvas.winfo_height() or 146))
        canvas.create_rectangle(0, 0, width, height, fill="#0c1420", outline="")
        canvas.create_rectangle(34, 12, width - 12, height - 22, outline="#3d5974", width=2)
        canvas.create_text(width / 2, height / 2, text=message, fill="#9bb4cf", font=("Segoe UI", 10, "bold"))

    def _update_status_graph(self, history_a: list[float], history_b: list[float]) -> None:
        if self.graph_canvas is None:
            return
        canvas = self.graph_canvas
        canvas.configure(bg="#0c1420")
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(260, int(canvas.winfo_width() or 420))
        height = max(120, int(canvas.winfo_height() or 146))
        left_pad = 34
        right_pad = 12
        top_pad = 12
        bottom_pad = 22
        plot_w = max(10, width - left_pad - right_pad)
        plot_h = max(10, height - top_pad - bottom_pad)
        canvas.create_rectangle(0, 0, width, height, fill="#0c1420", outline="")
        canvas.create_rectangle(left_pad, top_pad, left_pad + plot_w, top_pad + plot_h, outline="#3d5974", width=2)
        for percent in (0, 25, 50, 75, 100):
            y = top_pad + plot_h - int((percent / 100.0) * plot_h)
            canvas.create_line(left_pad, y, left_pad + plot_w, y, fill="#223347")
            canvas.create_text(18, y, text=f"{percent}", fill="#9bb4cf", font=("Segoe UI", 8))
        canvas.create_text(left_pad + plot_w - 10, top_pad + plot_h + 12, text="Time", fill="#9bb4cf", font=("Segoe UI", 8), anchor="e")

        def _draw_series(values: list[float], color: str) -> None:
            if not values:
                return
            points = []
            denom = max(1, len(values) - 1)
            for index, ratio in enumerate(values):
                x = left_pad + int((index / denom) * plot_w)
                y = top_pad + plot_h - int(max(0.0, min(1.0, float(ratio))) * plot_h)
                points.extend((x, y))
            if len(values) == 1:
                x = left_pad + plot_w - 8
                y = top_pad + plot_h - int(max(0.0, min(1.0, float(values[0]))) * plot_h)
                canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="")
                return
            canvas.create_line(*points, fill=color, width=3, smooth=True)
            canvas.create_oval(points[-2] - 4, points[-1] - 4, points[-2] + 4, points[-1] + 4, fill=color, outline="")

        _draw_series(history_a, "#2ad96b")
        _draw_series(history_b, "#39a8ff")
        canvas.create_rectangle(left_pad + 6, top_pad + 6, left_pad + 18, top_pad + 18, fill="#2ad96b", outline="")
        canvas.create_text(left_pad + 24, top_pad + 12, text="AMR-A", fill="#eef6ff", font=("Segoe UI", 8, "bold"), anchor="w")
        canvas.create_rectangle(left_pad + 94, top_pad + 6, left_pad + 106, top_pad + 18, fill="#39a8ff", outline="")
        canvas.create_text(left_pad + 112, top_pad + 12, text="AMR-B", fill="#eef6ff", font=("Segoe UI", 8, "bold"), anchor="w")
        current_a = max(0.0, min(1.0, float(history_a[-1] if history_a else 0.0)))
        current_b = max(0.0, min(1.0, float(history_b[-1] if history_b else 0.0)))
        bar_left = left_pad + 12
        bar_right = left_pad + plot_w - 12
        bar_width = max(10, bar_right - bar_left)
        bar_y_a = top_pad + plot_h - 26
        bar_y_b = top_pad + plot_h - 12
        canvas.create_rectangle(bar_left, bar_y_a, bar_right, bar_y_a + 7, fill="#112231", outline="")
        canvas.create_rectangle(bar_left, bar_y_b, bar_right, bar_y_b + 7, fill="#112231", outline="")
        canvas.create_rectangle(bar_left, bar_y_a, bar_left + int(bar_width * current_a), bar_y_a + 7, fill="#2ad96b", outline="")
        canvas.create_rectangle(bar_left, bar_y_b, bar_left + int(bar_width * current_b), bar_y_b + 7, fill="#39a8ff", outline="")
        canvas.create_text(bar_right, bar_y_a + 3, text=f"{100.0 * current_a:4.1f}%", fill="#eef6ff", font=("Segoe UI", 8, "bold"), anchor="e")
        canvas.create_text(bar_right, bar_y_b + 3, text=f"{100.0 * current_b:4.1f}%", fill="#eef6ff", font=("Segoe UI", 8, "bold"), anchor="e")

    def _animate_theme(self) -> None:
        self._theme_phase += 0.10
        r = int(40 + 60 * (0.5 + 0.5 * math.sin(self._theme_phase)))
        g = int(160 + 70 * (0.5 + 0.5 * math.sin(self._theme_phase + 1.9)))
        b = int(180 + 60 * (0.5 + 0.5 * math.sin(self._theme_phase + 3.4)))
        color = f"#{r:02x}{g:02x}{b:02x}"
        alt_color = f"#{min(255, g + 18):02x}{min(255, b + 10):02x}{min(255, r + 22):02x}"
        self.accent_canvas.itemconfigure(self.accent_line, fill=color)
        self.title_label.configure(fg=color)
        self.developer_label.configure(bg=f"#{max(18, r // 2):02x}{max(26, g // 6):02x}{max(38, b // 5):02x}")
        self.linkedin_label.configure(bg=f"#{max(14, b // 6):02x}{max(34, g // 4):02x}{max(46, r // 3):02x}")
        self.made_by_label.configure(bg=f"#{max(16, r // 3):02x}{max(38, g // 4):02x}{max(62, b // 4):02x}", fg="#f6fbff")
        self.linkedin_credit_label.configure(bg=f"#{max(12, b // 4):02x}{max(58, g // 3):02x}{max(72, r // 4):02x}", fg="#d8fbff")
        self.logo_canvas.itemconfigure(self.logo_orbit, outline=color)
        self.logo_canvas.itemconfigure(self.logo_lane, outline=f"#{b:02x}{r:02x}{g:02x}")
        self.logo_canvas.itemconfigure(self.logo_robot_a, fill=f"#{min(255, r + 10):02x}{min(255, g + 18):02x}{b:02x}")
        self.logo_canvas.itemconfigure(self.logo_robot_b, fill=f"#{min(255, g + 12):02x}{min(255, b + 10):02x}{min(255, r + 18):02x}")
        if hasattr(self, "hero_glow"):
            self.hero_glow.itemconfigure(self.hero_glow_left, fill=color)
            self.hero_glow.itemconfigure(self.hero_glow_right, fill=alt_color)
        if hasattr(self, "map_glow"):
            self.map_glow.itemconfigure(self.map_glow_left, fill=color)
            self.map_glow.itemconfigure(self.map_glow_right, fill=alt_color)

    def pump_events(self) -> None:
        self._animate_theme()
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        self.closed = True
        try:
            self.left_canvas.unbind_all("<MouseWheel>")
            self.left_canvas.unbind_all("<Button-4>")
            self.left_canvas.unbind_all("<Button-5>")
        except Exception:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


class MujocoAmrSimNode(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_amr_sim")

        package_share = Path(get_package_share_directory("mujoco_amr_sim"))

        self.declare_parameter("use_viewer", True)
        self.declare_parameter("auto_mode", True)
        self.declare_parameter("sim_rate_hz", 200.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("render_rate_hz", 8.0)
        self.declare_parameter("real_time_factor", 1.0)
        self.declare_parameter("cmd_vel_timeout_sec", 0.8)
        self.declare_parameter("lidar_beams", 91)
        self.declare_parameter("lidar_fov_deg", 200.0)
        self.declare_parameter("combined_lidar_beams", 181)
        self.declare_parameter("waypoints_file", str(package_share / "config" / "waypoints.json"))
        self.declare_parameter("dock_config_file", str(package_share / "config" / "dock_station.json"))
        self.declare_parameter("publish_odom_tf", True)
        self.declare_parameter("publish_depth_camera", True)
        self.declare_parameter("publish_pointcloud", True)
        self.declare_parameter("show_overview_window", False)
        self.declare_parameter("show_status_window", True)
        self.declare_parameter("enable_dynamic_obstacles", False)
        self.declare_parameter("dynamic_obstacle_speed_scale", 1.0)
        self.declare_parameter("overview_window_width", 430)
        self.declare_parameter("overview_window_height", 560)
        self.declare_parameter("camera_width", 320)
        self.declare_parameter("camera_height", 240)
        self.declare_parameter("sensor_random_seed", 7)
        self.declare_parameter("lidar_noise_stddev", 0.012)
        self.declare_parameter("imu_accel_noise_stddev", 0.04)
        self.declare_parameter("imu_gyro_noise_stddev", 0.008)
        self.declare_parameter("odom_xy_noise_stddev", 0.008)
        self.declare_parameter("odom_yaw_noise_stddev", 0.006)
        self.declare_parameter("odom_linear_velocity_noise_stddev", 0.012)
        self.declare_parameter("odom_angular_velocity_noise_stddev", 0.012)
        self.declare_parameter("battery_noise_stddev", 0.002)
        self.declare_parameter("initial_battery_percentage", 1.0)
        self.declare_parameter("battery_discharge_idle_per_sec", 0.00025)
        self.declare_parameter("battery_discharge_motion_per_sec", 0.0016)
        self.declare_parameter("battery_charge_per_sec", 0.018)
        self.declare_parameter("startup_mission_a", "")
        self.declare_parameter("startup_mission_b", "")
        self.declare_parameter("startup_mission_delay_sec", 1.0)

        self.use_viewer = bool(self.get_parameter("use_viewer").value)
        self.auto_mode = bool(self.get_parameter("auto_mode").value)
        self.sim_rate_hz = float(self.get_parameter("sim_rate_hz").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.render_rate_hz = float(self.get_parameter("render_rate_hz").value)
        self.real_time_factor = max(0.01, float(self.get_parameter("real_time_factor").value))
        self.cmd_vel_timeout_sec = float(self.get_parameter("cmd_vel_timeout_sec").value)
        self.lidar_beams = int(self.get_parameter("lidar_beams").value)
        self.lidar_fov_deg = float(self.get_parameter("lidar_fov_deg").value)
        self.combined_lidar_beams = int(self.get_parameter("combined_lidar_beams").value)
        self.waypoints_file = str(self.get_parameter("waypoints_file").value)
        self.dock_config_file = str(self.get_parameter("dock_config_file").value)
        self.publish_odom_tf = bool(self.get_parameter("publish_odom_tf").value)
        self.publish_depth_camera = bool(self.get_parameter("publish_depth_camera").value)
        self.publish_pointcloud = bool(self.get_parameter("publish_pointcloud").value)
        self.show_overview_window = bool(self.get_parameter("show_overview_window").value)
        self.show_status_window = bool(self.get_parameter("show_status_window").value)
        self.enable_dynamic_obstacles = bool(self.get_parameter("enable_dynamic_obstacles").value)
        self.dynamic_obstacle_speed_scale = max(0.0, float(self.get_parameter("dynamic_obstacle_speed_scale").value))
        self.overview_window_width = int(self.get_parameter("overview_window_width").value)
        self.overview_window_height = int(self.get_parameter("overview_window_height").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.sensor_random_seed = int(self.get_parameter("sensor_random_seed").value)
        self.lidar_noise_stddev = max(0.0, float(self.get_parameter("lidar_noise_stddev").value))
        self.imu_accel_noise_stddev = max(0.0, float(self.get_parameter("imu_accel_noise_stddev").value))
        self.imu_gyro_noise_stddev = max(0.0, float(self.get_parameter("imu_gyro_noise_stddev").value))
        self.odom_xy_noise_stddev = max(0.0, float(self.get_parameter("odom_xy_noise_stddev").value))
        self.odom_yaw_noise_stddev = max(0.0, float(self.get_parameter("odom_yaw_noise_stddev").value))
        self.odom_linear_velocity_noise_stddev = max(
            0.0, float(self.get_parameter("odom_linear_velocity_noise_stddev").value)
        )
        self.odom_angular_velocity_noise_stddev = max(
            0.0, float(self.get_parameter("odom_angular_velocity_noise_stddev").value)
        )
        self.battery_noise_stddev = max(0.0, float(self.get_parameter("battery_noise_stddev").value))
        self.battery_pct = max(0.0, min(1.0, float(self.get_parameter("initial_battery_percentage").value)))
        self.battery_idle_draw = float(self.get_parameter("battery_discharge_idle_per_sec").value)
        self.battery_motion_draw = float(self.get_parameter("battery_discharge_motion_per_sec").value)
        self.battery_charge_rate = float(self.get_parameter("battery_charge_per_sec").value)
        self.startup_mission_a = normalize_service_target_key(str(self.get_parameter("startup_mission_a").value or ""))
        self.startup_mission_b = normalize_service_target_key(str(self.get_parameter("startup_mission_b").value or ""))
        self.startup_mission_delay_sec = max(0.0, float(self.get_parameter("startup_mission_delay_sec").value))
        self.rng = np.random.default_rng(self.sensor_random_seed)

        self.mount_relative_angles = self._build_lidar_angles(self.lidar_beams, self.lidar_fov_deg)
        self.combined_lidar_angles = self._build_lidar_angles(self.combined_lidar_beams, 360.0)

        self.dock = load_dock_config(self.dock_config_file)
        self.waypoints = load_waypoints(self.waypoints_file)
        self.navigator = ObstacleAwareWaypointNavigator(self.waypoints)

        xml = build_model_xml(lidar_beams=self.lidar_beams, lidar_fov_deg=self.lidar_fov_deg)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.lidar_sensor_ids_by_mount = {
            mount.name: [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{mount.name}_lidar_{index:03d}")
                for index in range(self.lidar_beams)
            ]
            for mount in LIDAR_MOUNTS
        }
        self.imu_accel_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        self.imu_gyro_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.base_quat_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "base_quat")
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.front_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")
        self.overview_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_cam")
        self.left_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_cam")
        self.right_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_cam")
        self.rear_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rear_cam")
        self.scene_light_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, name)
            for name in (
                "ambient_house",
                "reception_light",
                "dining_light",
                "kitchen_light",
                "corridor_light",
                "dock_light",
                "service_dock_light",
                "vip_light",
                "lounge_light",
                "entry_light_upper",
                "entry_light_lower",
                "ceiling_center_light",
                "ceiling_left_light",
                "ceiling_right_light",
                "reception_accent_light",
                "reception_backwash_light",
                "vip_fill_light",
                "lounge_fill_light",
                "dining_fill_light",
            )
        }
        self.dynamic_obstacles = self._init_dynamic_obstacles()

        self.left_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        self.right_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
        self.base_x_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_x_joint")
        self.base_y_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_y_joint")
        self.base_yaw_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw_joint")
        self.base_x_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_x_motor")
        self.base_y_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_y_motor")
        self.base_yaw_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_yaw_motor")

        self.base_x_qpos_adr = self.model.jnt_qposadr[self.base_x_joint_id]
        self.base_y_qpos_adr = self.model.jnt_qposadr[self.base_y_joint_id]
        self.base_yaw_qpos_adr = self.model.jnt_qposadr[self.base_yaw_joint_id]
        self.base_x_qvel_adr = self.model.jnt_dofadr[self.base_x_joint_id]
        self.base_y_qvel_adr = self.model.jnt_dofadr[self.base_y_joint_id]
        self.base_yaw_qvel_adr = self.model.jnt_dofadr[self.base_yaw_joint_id]
        self.left_qpos_adr = self.model.jnt_qposadr[self.left_joint_id]
        self.right_qpos_adr = self.model.jnt_qposadr[self.right_joint_id]
        self.left_qvel_adr = self.model.jnt_dofadr[self.left_joint_id]
        self.right_qvel_adr = self.model.jnt_dofadr[self.right_joint_id]
        self.base_yaw_world_offset = 0.0
        self.data.qpos[self.base_x_qpos_adr] = 0.0
        self.data.qpos[self.base_y_qpos_adr] = 0.0
        self.data.qpos[self.base_yaw_qpos_adr] = self._world_yaw_to_joint_yaw(float(self.dock.dock_pose[2]))
        self.light_mode = "cinema"
        self._apply_scene_lighting()
        mujoco.mj_forward(self.model, self.data)
        self.arm_joint_names = [
            "arm_base_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_pitch_joint",
            "wrist_roll_joint",
            "gripper_joint",
        ]
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in self.arm_joint_names
        ]
        self.arm_qpos_adrs = [self.model.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids]
        self.arm_qvel_adrs = [self.model.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids]
        self.tool_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
        self.payload_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload_box")
        self.payload_mocap_id = int(self.model.body_mocapid[self.payload_body_id])
        self.service_bot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "service_bot")
        self.service_bot_mocap_id = int(self.model.body_mocapid[self.service_bot_body_id]) if self.service_bot_body_id >= 0 else -1

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.gt_odom_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.mount_scan_pubs = {
            "front_left": self.create_publisher(LaserScan, "/scan/front_left", 10),
            "rear_right": self.create_publisher(LaserScan, "/scan/rear_right", 10),
        }
        self.imu_pub = self.create_publisher(Imu, "/imu/data", 10)
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.battery_pub = self.create_publisher(BatteryState, "/battery_state", 10)
        self.dock_contact_pub = self.create_publisher(Bool, "/dock/in_contact", 10)
        self.dock_charging_pub = self.create_publisher(Bool, "/dock/is_charging", 10)
        self.dock_state_pub = self.create_publisher(String, "/dock/state", 10)
        self.rgb_pub = self.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.rgb_info_pub = self.create_publisher(CameraInfo, "/camera/rgb/camera_info", 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info", 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, "/camera/depth/points", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/monitoring/markers", 10)
        self.status_pub = self.create_publisher(String, "/simulation/status", 10)
        self.event_pub = self.create_publisher(String, "/simulation/event_log", 10)
        self.mission_command_pub = self.create_publisher(String, "/autonomy/mission_command", 10)
        self.speed_limit_pub = self.create_publisher(Float32, "/autonomy/speed_limit", 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)
        self.cmd_auto_sub = self.create_subscription(Twist, "/cmd_vel_auto", self._cmd_vel_auto_callback, 10)
        self.operator_mission_sub = self.create_subscription(
            String,
            "/operator/amr_a_mission_command",
            self._operator_mission_callback,
            10,
        )
        self.service_bot_mission_sub = self.create_subscription(
            String, "/service_amr/mission_command", self._service_bot_mission_callback, 10
        )
        self.service_bot_speed_sub = self.create_subscription(
            Float32, "/service_amr/speed_limit", self._service_bot_speed_callback, 10
        )
        self.autonomy_state_sub = self.create_subscription(
            String, "/autonomy/state", self._autonomy_state_callback, 10
        )
        self.mission_status_sub = self.create_subscription(
            String, "/autonomy/mission_status", self._mission_status_callback, 10
        )
        self.autonomy_event_sub = self.create_subscription(
            String, "/autonomy/event_log", self._autonomy_event_callback, 10
        )
        self.cmd_source_sub = self.create_subscription(String, "/cmd_vel_source", self._cmd_source_callback, 10)
        self.estop_sub = self.create_subscription(
            Bool, "/safety/emergency_stop_active", self._estop_callback, 10
        )

        self.latest_manual_cmd = RobotCommand()
        self.latest_auto_cmd = RobotCommand()
        self.last_manual_cmd_time = self.get_clock().now()
        self.last_auto_cmd_time = self.get_clock().now()
        self.applied_cmd = RobotCommand()
        self.latest_autonomy_state = "N/A"
        self.latest_cmd_source = "idle"
        self.latest_mission_status: dict[str, object] = {}
        self.current_mission_mode = "idle"
        self.speed_limit_scale = 1.00
        self._apply_main_navigation_profile()
        self.operator_message = "Console ready"
        self.recent_events = ["Console ready"]
        self.task_queue_a: list[TaskRecord] = []
        self.task_queue_b: list[TaskRecord] = []
        self.active_task_a: TaskRecord | None = None
        self.active_task_b: TaskRecord | None = None
        self.deferred_mission_a: str | None = None
        self.deferred_mission_b: str | None = None
        self.completed_tasks: list[TaskRecord] = []
        self.task_counter = 0
        self.service_bot_pose = np.array([ROBOT2_START_X, ROBOT2_START_Y], dtype=float)
        self.service_bot_yaw = SERVICE_DOCK_YAW
        self.service_bot_battery_pct = 1.0
        self.service_bot_is_charging = False
        self.service_bot_mission_mode = "idle"
        self.service_bot_saved_mission_mode = "idle"
        self.service_bot_speed_scale = 1.00
        self.service_bot_goal_index = 0
        self.service_bot_active_path: list[tuple[float, float]] = []
        self.service_bot_payload_state = "EMPTY"
        self.service_bot_state = "IDLE"
        self.service_bot_dock_contact = False
        self.service_bot_waiting_for_task = True
        self.startup_mission_a_dispatched = False
        self.startup_mission_b_dispatched = False
        self.service_bot_last_switch_time = 0.0
        self.service_bot_last_progress_time = 0.0
        self.service_bot_last_goal_distance = float("inf")
        self.service_bot_last_goal_index = 0
        self.service_bot_return_to_charge = False
        self.service_bot_table_hold_until: float | None = None
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.target_left_wheel_speed = 0.0
        self.target_right_wheel_speed = 0.0
        self.dock_in_contact = False
        self.is_charging = False
        self.emergency_stop_active = False
        self.last_pose_for_odom = None
        self.last_publish_sim_time = None
        self.last_wheel_angles_for_odom: tuple[float, float] | None = None
        self.last_render_sim_time = -1e9
        self.last_overview_render_sim_time = -1e9
        self.last_status_render_sim_time = -1e9
        self.path_history: list[tuple[float, float]] = []
        self.battery_history_a: list[float] = [self.battery_pct]
        self.battery_history_b: list[float] = [self.service_bot_battery_pct]
        self.last_battery_history_time = -1e9
        self.arm_named_poses = {"removed": np.zeros((6,), dtype=float)}
        self.arm_pose = self.arm_named_poses["removed"].copy()
        self.arm_target_pose = self.arm_named_poses["removed"].copy()
        self.arm_state = "REMOVED"
        self.payload_attached = False
        self.payload_location = "disabled"
        self.pick_station = np.array([PICK_STATION_X + 0.18, PICK_STATION_Y, 1.12], dtype=float)
        self.place_station = np.array([PLACE_STATION_X - 0.08, PLACE_STATION_Y, 1.12], dtype=float)
        self.last_pick_place_action_time = 0.0
        self.data.mocap_quat[self.payload_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self.service_bot_mocap_id >= 0:
            self.data.mocap_pos[self.service_bot_mocap_id] = np.array(
                [self.service_bot_pose[0], self.service_bot_pose[1], 0.11],
                dtype=float,
            )
            self.data.mocap_quat[self.service_bot_mocap_id] = np.array(
                [math.cos(0.5 * self.service_bot_yaw), 0.0, 0.0, math.sin(0.5 * self.service_bot_yaw)],
                dtype=float,
            )
        self._snap_main_robot_to_dock_pose()
        self._snap_service_bot_to_dock_pose()

        self.rgb_renderer = None
        self.depth_renderer = None
        self.overview_renderer = None
        self.status_renderer = None
        self.status_preview_renderer = None
        self.offscreen_gl_context = None
        self.status_render_ready = False
        self.status_render_mode = "pending"
        self.status_render_error = "Renderer boot pending"
        self.last_status_ui_update_sim_time = -1e9
        self.last_status_event_pump_wall_time = 0.0
        self.overview_window = None
        self.status_window = None
        self.force_dock_client = self.create_client(Trigger, "/autonomy/force_dock")
        self.resume_patrol_client = self.create_client(Trigger, "/autonomy/resume_patrol")
        self.skip_waypoint_client = self.create_client(Trigger, "/autonomy/skip_waypoint")
        self.reload_mission_client = self.create_client(Trigger, "/autonomy/reload_mission")
        self.patrol_toggle_client = self.create_client(SetBool, "/autonomy/set_patrol_enabled")
        self.estop_client = self.create_client(SetBool, "/safety/set_emergency_stop")
        self._init_renderers()
        self._init_overview_window()
        self._init_status_window()
        self._disable_hidden_manipulator_visuals()
        self._emit_event(
            "Simulation ready with "
            f"{len(self.dynamic_obstacles)} dynamic obstacles and "
            f"{'overview panel enabled' if self.show_overview_window else 'overview panel disabled'}"
        )

        self.get_logger().info(
            f"Simulation ready: viewer={self.use_viewer}, auto_mode={self.auto_mode}, battery={self.battery_pct:.2f}"
        )

    def _build_lidar_angles(self, beams: int, fov_deg: float) -> list[float]:
        if beams <= 1:
            return [0.0]
        start = -math.radians(fov_deg) / 2.0
        step = math.radians(fov_deg) / (beams - 1)
        return [start + index * step for index in range(beams)]

    def _sample_noise(self, stddev: float) -> float:
        if stddev <= 0.0:
            return 0.0
        return float(self.rng.normal(0.0, stddev))

    def _push_recent_event(self, message: str) -> None:
        if not message:
            return
        self.recent_events.append(f"[{self.data.time:5.1f}s] {message}")
        self.recent_events = self.recent_events[-10:]

    def _emit_event(self, message: str) -> None:
        self.get_logger().info(message)
        self._push_recent_event(message)
        self.event_pub.publish(String(data=message))

    def _init_dynamic_obstacles(self) -> list[dict[str, float | int | str]]:
        if not self.enable_dynamic_obstacles:
            return []

        obstacles = []
        for body_name in DYNAMIC_OBSTACLE_BODIES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                continue
            obstacles.append(
                {
                    "name": body_name,
                    "body_id": body_id,
                    "mocap_id": mocap_id,
                }
            )
        return obstacles

    def _disable_hidden_manipulator_visuals(self) -> None:
        hidden_geoms = [
            "arm_base_pedestal",
            "shoulder_housing",
            "upper_arm_geom",
            "elbow_housing",
            "forearm_geom",
            "wrist_pitch_housing",
            "wrist_roll_geom",
            "gripper_palm",
            "gripper_finger_left",
            "gripper_finger_right",
            "payload_geom",
            "payload_band",
            "sensor_mast",
            "sensor_crossbar",
            "camera_head",
            "camera_lens",
        ]
        for geom_name in hidden_geoms:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                self.model.geom_rgba[geom_id, 3] = 0.0
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0

    def _estop_callback(self, msg: Bool) -> None:
        new_value = bool(msg.data)
        if new_value != self.emergency_stop_active:
            self._emit_event(f"Emergency stop {'enabled' if new_value else 'cleared'}")
        self.emergency_stop_active = new_value

    def _apply_main_navigation_profile(self) -> None:
        service_route_active = normalize_service_target_key(getattr(self, "current_mission_mode", "idle")) in SERVICE_TARGETS
        self.navigator.set_goal_tolerance(0.12 if service_route_active else 0.22)
        self.navigator.set_max_linear(0.58 if service_route_active else 0.70)
        speed_scale = max(0.20, min(1.25, float(getattr(self, "speed_limit_scale", 1.0))))
        if service_route_active:
            speed_scale = min(speed_scale, 0.78)
        self.navigator.set_speed_scale(speed_scale)

    def _mission_status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.latest_mission_status = payload
            mission_mode = payload.get("mission_mode")
            if isinstance(mission_mode, str) and mission_mode:
                self.current_mission_mode = mission_mode
            speed_limit = payload.get("speed_limit_scale")
            if isinstance(speed_limit, (float, int)):
                self.speed_limit_scale = float(speed_limit)
            self._apply_main_navigation_profile()

    def _autonomy_event_callback(self, msg: String) -> None:
        text = str(msg.data).strip()
        if text:
            self._push_recent_event(text)

    def _dynamic_obstacle_pose(self, name: str, sim_time: float) -> tuple[float, float, float]:
        t = sim_time * self.dynamic_obstacle_speed_scale
        if name == "dynamic_actor_a":
            x = -1.65
            y = -4.20 + 8.20 * (0.5 + 0.5 * math.sin(0.16 * t))
            yaw = math.pi / 2.0 if math.cos(0.16 * t) >= 0.0 else -math.pi / 2.0
            return x, y, yaw
        if name == "dynamic_actor_b":
            x = 2.75 + 2.10 * math.sin(0.12 * t + 0.9)
            y = 5.05
            yaw = 0.0 if math.cos(0.18 * t + 0.9) >= 0.0 else math.pi
            return x, y, yaw
        x = 7.35
        y = -2.40 + 3.25 * math.sin(0.14 * t + 0.4)
        yaw = math.pi / 2.0 if math.cos(0.14 * t + 0.4) >= 0.0 else -math.pi / 2.0
        return x, y, yaw

    def _update_dynamic_obstacles(self) -> None:
        if not self.dynamic_obstacles:
            return
        for obstacle in self.dynamic_obstacles:
            x, y, yaw = self._dynamic_obstacle_pose(str(obstacle["name"]), self.data.time)
            mocap_id = int(obstacle["mocap_id"])
            self.data.mocap_pos[mocap_id] = np.array([x, y, 0.0], dtype=float)
            self.data.mocap_quat[mocap_id] = np.array(
                [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
                dtype=float,
            )

    def _service_bot_table_route(self, table_key: str) -> list[tuple[float, float]]:
        start_xy = (float(self.service_bot_pose[0]), float(self.service_bot_pose[1]))
        return [
            (float(point[0]), float(point[1]))
            for point in build_service_route(start_xy, table_key, "b")
        ]

    def _service_bot_charge_route(self) -> list[tuple[float, float]]:
        start_xy = (float(self.service_bot_pose[0]), float(self.service_bot_pose[1]))
        service_pre_dock = (SERVICE_DOCK_X - 0.55, SERVICE_DOCK_Y)
        route = [
            (float(point[0]), float(point[1]))
            for point in build_visibility_route(start_xy, service_pre_dock)
        ]
        if not route or route[-1] != service_pre_dock:
            route.append(service_pre_dock)
        route.append((SERVICE_DOCK_X, SERVICE_DOCK_Y))
        return route

    def _service_bot_goal_pose(self) -> tuple[float, float, float] | None:
        if self.service_bot_mission_mode in SERVICE_TARGETS:
            return service_target_goal(self.service_bot_mission_mode, "b")
        if self.service_bot_mission_mode == "charge_return":
            return SERVICE_DOCK_X, SERVICE_DOCK_Y, SERVICE_DOCK_YAW
        return None

    def _snap_service_bot_to_dock_pose(self) -> None:
        self.service_bot_pose = np.array([SERVICE_DOCK_X, SERVICE_DOCK_Y], dtype=float)
        self.service_bot_yaw = SERVICE_DOCK_YAW
        self.service_bot_dock_contact = True
        self.service_bot_is_charging = True
        self.service_bot_state = "CHARGING"
        self.service_bot_mission_mode = "idle"
        self.service_bot_payload_state = "EMPTY"
        self.service_bot_waiting_for_task = True
        self.service_bot_return_to_charge = False
        self.service_bot_goal_index = 0
        self.service_bot_active_path = []

    def _snap_service_bot_to_pose(self, goal_x: float, goal_y: float, goal_yaw: float) -> None:
        self.service_bot_pose = np.array([goal_x, goal_y], dtype=float)
        self.service_bot_yaw = goal_yaw

    def _service_bot_mission_paths(self) -> dict[str, list[tuple[float, float]]]:
        paths = {
            "idle": [],
            "charge_return": self._service_bot_charge_route(),
            "lobby_assist": [
                (ROBOT2_START_X, ROBOT2_START_Y),
                (3.20, -3.20),
                (0.80, -1.80),
                (-2.40, 0.40),
                (-4.60, 2.40),
                (-6.00, 4.60),
            ],
        }
        for table_key in SERVICE_TARGETS:
            paths[table_key] = self._service_bot_table_route(table_key)
        paths["table_service"] = list(paths["table_2"])
        paths["guest_delivery"] = list(paths["table_6"])
        return paths

    def _reset_service_bot_active_path(self) -> None:
        mission_paths = self._service_bot_mission_paths()
        path = mission_paths.get(self.service_bot_mission_mode, mission_paths["table_2"])
        self.service_bot_active_path = [tuple(point[:2]) for point in path]
        self.service_bot_goal_index = 0
        self.service_bot_last_progress_time = float(self.data.time)
        self.service_bot_last_goal_distance = float("inf")
        self.service_bot_last_goal_index = 0

    def _robot_clearance_blocked(
        self,
        x: float,
        y: float,
        *,
        against_main_robot: bool = False,
        against_service_robot: bool = False,
        clearance: float = 0.54,
    ) -> bool:
        robot_radius = 0.60 + clearance
        if against_main_robot:
            base_x, base_y, _, _ = self._read_pose()
            if math.hypot(x - base_x, y - base_y) <= robot_radius:
                return True
        if against_service_robot:
            if math.hypot(x - float(self.service_bot_pose[0]), y - float(self.service_bot_pose[1])) <= robot_radius:
                return True
        return False

    def _turn_away_direction(
        self,
        pose_x: float,
        pose_y: float,
        yaw: float,
        obstacle_x: float,
        obstacle_y: float,
    ) -> float:
        relative_angle = wrap_to_pi(math.atan2(obstacle_y - pose_y, obstacle_x - pose_x) - yaw)
        return -1.0 if relative_angle >= 0.0 else 1.0

    def _update_service_bot(self, sim_dt: float) -> None:
        if self.service_bot_mocap_id < 0:
            return

        if self.service_bot_mission_mode == "idle":
            dock_xy = np.array([SERVICE_DOCK_X, SERVICE_DOCK_Y], dtype=float)
            dock_distance = float(np.linalg.norm(self.service_bot_pose - dock_xy))
            self.service_bot_dock_contact = dock_distance < 0.40
            self.service_bot_is_charging = dock_distance < 0.34
            self.service_bot_state = "CHARGING" if self.service_bot_is_charging else "WAITING_TASK"
            self.service_bot_waiting_for_task = True
            return

        if not self.service_bot_active_path:
            self._reset_service_bot_active_path()
        path = self.service_bot_active_path
        if not path:
            return

        dock_xy = np.array([SERVICE_DOCK_X, SERVICE_DOCK_Y], dtype=float)
        self.service_bot_dock_contact = False
        if self.service_bot_mission_mode == "charge_return":
            dock_distance = float(np.linalg.norm(self.service_bot_pose - dock_xy))
            dock_yaw_error = abs(wrap_to_pi(SERVICE_DOCK_YAW - self.service_bot_yaw))
            self.service_bot_dock_contact = dock_distance < 0.28 and dock_yaw_error < 0.40
            if (
                (dock_distance < 0.08 and dock_yaw_error < 0.08)
                or (dock_distance < 0.36 and dock_yaw_error < 1.30)
            ):
                self._snap_service_bot_to_dock_pose()
                self.service_bot_state = "CHARGING"
                self.service_bot_battery_pct = min(1.0, self.service_bot_battery_pct + 0.030 * sim_dt)
            else:
                self.service_bot_is_charging = False
        else:
            self.service_bot_is_charging = False

        if self.service_bot_waiting_for_task and not self.service_bot_is_charging and self.service_bot_mission_mode == "charge_return":
            self.service_bot_state = "WAITING_TASK"
        elif not self.service_bot_is_charging:
            goal_x, goal_y = path[self.service_bot_goal_index % len(path)]
            goal = np.array([goal_x, goal_y], dtype=float)
            delta = goal - self.service_bot_pose
            distance = float(np.linalg.norm(delta))
            final_table_goal = self.service_bot_goal_index >= len(path) - 1 and self.service_bot_mission_mode in SERVICE_TARGETS
            final_charge_goal = self.service_bot_goal_index >= len(path) - 1 and self.service_bot_mission_mode == "charge_return"
            goal_pose = self._service_bot_goal_pose()
            final_goal_yaw = self.service_bot_yaw if goal_pose is None else float(goal_pose[2])
            distance_to_service_goal = float("inf")
            if goal_pose is not None and self.service_bot_mission_mode in SERVICE_TARGETS:
                service_goal = np.array([float(goal_pose[0]), float(goal_pose[1])], dtype=float)
                distance_to_service_goal = float(np.linalg.norm(service_goal - self.service_bot_pose))
                if self.service_bot_goal_index >= len(path) - 2 or distance_to_service_goal <= 1.45:
                    goal = service_goal
                    goal_x, goal_y = float(service_goal[0]), float(service_goal[1])
                    delta = goal - self.service_bot_pose
                    distance = distance_to_service_goal
                    final_table_goal = True
            desired_yaw = math.atan2(delta[1], delta[0]) if distance > 1e-6 else final_goal_yaw
            if (final_table_goal or final_charge_goal) and distance < 0.32:
                desired_yaw = final_goal_yaw
            heading_error = wrap_to_pi(desired_yaw - self.service_bot_yaw)
            final_yaw_error = wrap_to_pi(final_goal_yaw - self.service_bot_yaw)
            turn_rate = max(-1.3, min(1.3, 2.0 * heading_error))
            self.service_bot_yaw = wrap_to_pi(self.service_bot_yaw + turn_rate * sim_dt)

            max_linear = 0.84 * self.service_bot_speed_scale
            linear = min(max_linear, 0.95 * distance)
            if final_table_goal:
                linear = min(linear, 0.42 if distance > 1.0 else 0.32)
            if final_charge_goal:
                linear = min(linear, 0.18)
            if abs(heading_error) > 0.7:
                linear *= 0.45 if final_table_goal else 0.24
            elif abs(heading_error) > 0.35:
                linear *= 0.72 if final_table_goal else 0.52
            if final_table_goal and distance < 0.48:
                if abs(heading_error) > 0.32:
                    linear = 0.0
                else:
                    linear = min(0.22, 0.90 * distance)
                    if abs(final_yaw_error) > 0.65:
                        linear *= 0.45
            elif (final_table_goal or final_charge_goal) and distance < 0.18 and abs(final_yaw_error) > 0.18:
                linear = 0.0

            base_x, base_y, _, _ = self._read_pose()
            vector_to_a = np.array([base_x, base_y], dtype=float) - self.service_bot_pose
            separation = float(np.linalg.norm(vector_to_a))
            if separation < 1.80:
                turn_direction = self._turn_away_direction(
                    float(self.service_bot_pose[0]),
                    float(self.service_bot_pose[1]),
                    self.service_bot_yaw,
                    base_x,
                    base_y,
                )
                linear = -0.16 if separation < 1.15 else 0.0
                self.service_bot_yaw = wrap_to_pi(self.service_bot_yaw + turn_direction * 1.1 * sim_dt)
                self.service_bot_state = "AVOID_AMR_A"
            else:
                self.service_bot_state = "EXECUTING"

            predicted_lookahead = min(
                max(0.18 if (final_table_goal or final_charge_goal) else 0.30, linear * 0.85),
                max(0.14, distance),
            )
            predicted_pose = self.service_bot_pose + np.array(
                [math.cos(self.service_bot_yaw), math.sin(self.service_bot_yaw)],
                dtype=float,
            ) * predicted_lookahead
            blocked_ahead = self._position_blocked(
                float(predicted_pose[0]),
                float(predicted_pose[1]),
                clearance=0.32,
                allow_main_dock=False,
                allow_service_dock=self.service_bot_mission_mode == "charge_return",
                avoid_main_robot=True,
            )
            if blocked_ahead:
                if (
                    (final_table_goal and distance < 0.55)
                    or (self.service_bot_mission_mode in SERVICE_TARGETS and distance < 0.65)
                ):
                    blocked_ahead = False
                else:
                    linear = 0.0
                    self.service_bot_yaw = wrap_to_pi(self.service_bot_yaw + (0.95 if heading_error >= 0.0 else -0.95) * sim_dt)
                    self.service_bot_state = "AVOID_STATIC"
            if blocked_ahead:
                linear = 0.0
                self.service_bot_state = "AVOID_STATIC"

            move = np.array([math.cos(self.service_bot_yaw), math.sin(self.service_bot_yaw)], dtype=float) * linear * sim_dt
            self.service_bot_pose = self.service_bot_pose + move
            if abs(linear) > 0.04:
                self.service_bot_last_progress_time = float(self.data.time)
            if self.service_bot_goal_index != self.service_bot_last_goal_index:
                self.service_bot_last_progress_time = float(self.data.time)
                self.service_bot_last_goal_index = int(self.service_bot_goal_index)
                self.service_bot_last_goal_distance = float("inf")
            elif distance < self.service_bot_last_goal_distance - 0.04:
                self.service_bot_last_progress_time = float(self.data.time)
            self.service_bot_last_goal_distance = min(self.service_bot_last_goal_distance, distance)
            if abs(linear) <= 0.04 and self.data.time - self.service_bot_last_progress_time > 3.5:
                self._reset_service_bot_active_path()
                self.service_bot_state = "REPLAN"
                self.service_bot_yaw = wrap_to_pi(self.service_bot_yaw + 0.8 * sim_dt)
                return
            self.service_bot_battery_pct = max(0.0, self.service_bot_battery_pct - (0.00035 + 0.0016 * abs(linear)) * sim_dt)

            updated_distance = float(np.linalg.norm(goal - self.service_bot_pose))
            updated_final_yaw_error = abs(wrap_to_pi(final_goal_yaw - self.service_bot_yaw))
            goal_reached_radius = 0.32 if final_table_goal else (0.08 if final_charge_goal else 0.55)
            goal_orientation_tolerance = 0.62 if final_table_goal else (0.14 if final_charge_goal else math.pi)
            if updated_distance < goal_reached_radius and updated_final_yaw_error <= goal_orientation_tolerance:
                if final_table_goal:
                    if self.service_bot_table_hold_until is None:
                        self._snap_service_bot_to_pose(goal_x, goal_y, final_goal_yaw)
                        self.service_bot_table_hold_until = self.data.time + 5.0
                        self.service_bot_state = "SERVING"
                        self._push_recent_event(f"AMR-B serving {mission_display_name(self.service_bot_mission_mode)} for 5 seconds")
                    elif self.data.time >= self.service_bot_table_hold_until:
                        completed_mission = self.service_bot_mission_mode
                        self.service_bot_mission_mode = "charge_return"
                        self.service_bot_return_to_charge = True
                        self.service_bot_waiting_for_task = False
                        self.service_bot_table_hold_until = None
                        self.service_bot_state = "RETURNING_DOCK"
                        self.service_bot_payload_state = "EMPTY"
                        self._reset_service_bot_active_path()
                        self._push_recent_event(f"AMR-B completed {mission_display_name(completed_mission)} and is returning to auto dock")
                    else:
                        self.service_bot_state = "SERVING"
                elif final_charge_goal:
                    self._snap_service_bot_to_dock_pose()
                    self.service_bot_state = "CHARGING"
                    self.service_bot_payload_state = "DOCKED"
                    self.service_bot_battery_pct = min(1.0, self.service_bot_battery_pct + 0.030 * sim_dt)
                else:
                    self.service_bot_table_hold_until = None
                    self.service_bot_goal_index = min(self.service_bot_goal_index + 1, len(path) - 1)
                    self.service_bot_last_progress_time = float(self.data.time)
                    if self.service_bot_mission_mode.startswith("table_") or self.service_bot_mission_mode == "table_service":
                        self.service_bot_payload_state = "TRAY" if self.service_bot_payload_state == "EMPTY" else "EMPTY"
                    elif self.service_bot_mission_mode == "guest_delivery":
                        self.service_bot_payload_state = "ROOM"
                    elif self.service_bot_mission_mode == "lobby_assist":
                        self.service_bot_payload_state = "GUIDE"
        else:
            self.service_bot_payload_state = "DOCKED"

        self.data.mocap_pos[self.service_bot_mocap_id] = np.array(
            [self.service_bot_pose[0], self.service_bot_pose[1], 0.11],
            dtype=float,
        )
        self.data.mocap_quat[self.service_bot_mocap_id] = np.array(
            [math.cos(0.5 * self.service_bot_yaw), 0.0, 0.0, math.sin(0.5 * self.service_bot_yaw)],
            dtype=float,
        )

    def _near_station(self, target_xy: np.ndarray, threshold: float = 0.78) -> bool:
        x, y, _, _ = self._read_pose()
        return math.hypot(target_xy[0] - x, target_xy[1] - y) <= threshold

    def _update_arm_and_payload(self, sim_dt: float) -> None:
        del sim_dt
        self.arm_state = "REMOVED"
        self.payload_attached = False
        self.payload_location = "disabled"
        self.arm_pose = self.arm_named_poses["removed"].copy()
        self.arm_target_pose = self.arm_named_poses["removed"].copy()
        for qpos_adr, qvel_adr in zip(self.arm_qpos_adrs, self.arm_qvel_adrs, strict=False):
            self.data.qpos[qpos_adr] = 0.0
            self.data.qvel[qvel_adr] = 0.0
        self.data.mocap_pos[self.payload_mocap_id] = np.array([100.0, 100.0, -10.0], dtype=float)
        self.data.mocap_quat[self.payload_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _cmd_vel_callback(self, msg: Twist) -> None:
        self.latest_manual_cmd = RobotCommand(linear=float(msg.linear.x), angular=float(msg.angular.z))
        self.last_manual_cmd_time = self.get_clock().now()

    def _cmd_vel_auto_callback(self, msg: Twist) -> None:
        self.latest_auto_cmd = RobotCommand(linear=float(msg.linear.x), angular=float(msg.angular.z))
        self.last_auto_cmd_time = self.get_clock().now()

    def _autonomy_state_callback(self, msg: String) -> None:
        self.latest_autonomy_state = str(msg.data) if msg.data else "N/A"

    def _cmd_source_callback(self, msg: String) -> None:
        self.latest_cmd_source = str(msg.data) if msg.data else "idle"

    def _service_bot_mission_callback(self, msg: String) -> None:
        self._dispatch_robot_b_mission(str(msg.data).strip())

    def _operator_mission_callback(self, msg: String) -> None:
        self.get_logger().info(f"Operator mission topic received: {msg.data}")
        self._dispatch_mission_command(str(msg.data).strip())

    def _service_bot_speed_callback(self, msg: Float32) -> None:
        self._dispatch_robot_b_speed(float(msg.data))

    def _init_renderers(self) -> None:
        if self.publish_depth_camera:
            try:
                self._ensure_offscreen_context()
                self.rgb_renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
                self.depth_renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
                self.depth_renderer.enable_depth_rendering()
            except Exception as exc:
                self.rgb_renderer = None
                self.depth_renderer = None
                self.publish_depth_camera = False
                self.publish_pointcloud = False
                self.get_logger().warning(
                    "Camera rendering disabled because OpenGL context creation failed. "
                    f"DISPLAY={os.environ.get('DISPLAY', '')!r}, "
                    f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')!r}, "
                    f"XDG_RUNTIME_DIR={os.environ.get('XDG_RUNTIME_DIR', '')!r}. "
                    f"Original error: {exc}"
                )

        if self.show_status_window and self.use_viewer:
            self._create_status_renderers()

        if self.show_overview_window and self.use_viewer:
            try:
                self._ensure_offscreen_context()
                self.overview_renderer = mujoco.Renderer(
                    self.model,
                    height=max(110, (self.overview_window_height - 74) // 2),
                    width=max(140, (self.overview_window_width - 24) // 2),
                )
            except Exception as exc:
                self.overview_renderer = None
                self.show_overview_window = False
                self.get_logger().warning(f"Overview preview renderer disabled: {exc}")

    def _ensure_offscreen_context(self) -> None:
        gl_context_cls = getattr(mujoco, "GLContext", None)
        if gl_context_cls is None:
            return
        if self.offscreen_gl_context is None:
            self.offscreen_gl_context = gl_context_cls(2048, 1536)
        make_current = getattr(self.offscreen_gl_context, "make_current", None)
        if callable(make_current):
            make_current()

    def _create_status_renderers(self) -> None:
        try:
            if self.status_renderer is not None:
                self.status_renderer.close()
            if self.status_preview_renderer is not None:
                self.status_preview_renderer.close()
        except Exception:
            pass
        self.status_renderer = None
        self.status_preview_renderer = None
        try:
            self._ensure_offscreen_context()
            self.status_renderer = mujoco.Renderer(self.model, height=760, width=1120)
            self.status_preview_renderer = mujoco.Renderer(self.model, height=286, width=404)
            self.status_render_mode = "opengl"
            self.status_render_error = ""
        except Exception as exc:
            self.status_renderer = None
            self.status_preview_renderer = None
            self.status_render_ready = False
            self.status_render_mode = "unavailable"
            self.status_render_error = f"Renderer unavailable: {exc}"
            self.get_logger().warning(f"Status preview renderer disabled: {exc}")

    def _init_overview_window(self) -> None:
        if not (self.use_viewer and self.show_overview_window):
            return
        if tk is None or PilImage is None or ImageTk is None or ImageDraw is None:
            self.show_overview_window = False
            self.get_logger().warning("Overview preview window disabled because tkinter/Pillow is unavailable.")
            return
        try:
            self.overview_window = OverviewWindow(
                width=self.overview_window_width,
                height=self.overview_window_height,
            )
        except Exception as exc:
            self.overview_window = None
            self.show_overview_window = False
            self.get_logger().warning(f"Overview preview window disabled: {exc}")

    def _init_status_window(self) -> None:
        if not (self.use_viewer and self.show_status_window):
            return
        if tk is None:
            self.show_status_window = False
            self.get_logger().warning("Status window disabled because tkinter is unavailable.")
            return
        created_window = None
        try:
            created_window = StatusWindow()
            self.status_window = created_window
            self.get_logger().info(f"{BRAND_WINDOW_TITLE} window created")
            self.status_window.update_text("Initializing simulator...")
            self.status_window.set_mission_callback(self._dispatch_mission_command)
            self.status_window.set_speed_callback(self._dispatch_speed_limit)
            self.status_window.set_robot_b_mission_callback(self._dispatch_robot_b_mission)
            self.status_window.set_robot_b_speed_callback(self._dispatch_robot_b_speed)
            self.status_window.set_return_home_callback(self._dispatch_return_home_a)
            self.status_window.set_return_home_b_callback(self._dispatch_return_home_b)
            self.status_window.set_lighting_callback(self._dispatch_lighting_mode)
        except Exception as exc:
            if created_window is not None:
                try:
                    created_window.close()
                except Exception:
                    pass
            self.status_window = None
            self.show_status_window = False
            self.get_logger().warning(f"Status window disabled: {exc}")

    def _make_world_camera(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
        lookat_xy: tuple[float, float] | None = None,
        lookat_z: float = 0.72,
    ) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        lookat_x = 0.0 if lookat_xy is None else float(lookat_xy[0])
        lookat_y = 0.0 if lookat_xy is None else float(lookat_xy[1])
        camera.lookat[0] = lookat_x
        camera.lookat[1] = lookat_y
        camera.lookat[2] = lookat_z
        camera.distance = distance
        camera.azimuth = azimuth_deg
        camera.elevation = elevation_deg
        return camera

    def _make_follow_camera(self, azimuth_deg: float, elevation_deg: float, distance: float) -> mujoco.MjvCamera:
        x, y, _, _ = self._read_pose()
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[0] = x
        camera.lookat[1] = y
        camera.lookat[2] = 0.52
        camera.distance = distance
        camera.azimuth = azimuth_deg
        camera.elevation = elevation_deg
        return camera

    def _table_target_xy(self, mission_mode: str) -> tuple[float, float] | None:
        mission_key = (mission_mode or "").strip().lower()
        table = SERVICE_TARGETS.get(mission_key)
        if table is None:
            return None
        table_x, table_y = table["pos"]
        return float(table_x), float(table_y)

    def _main_robot_dock_state(self) -> str:
        if self.is_charging or self.dock_in_contact or self.latest_autonomy_state == "WAITING_TASK":
            return "Service Dock"
        if self.battery_pct <= 0.30 or self.latest_autonomy_state in {"SEEK_DOCK", "FINAL_DOCKING", "CHARGING"}:
            return "Returning Dock"
        return "Serving Floor"

    def _service_robot_dock_state(self) -> str:
        if self.service_bot_is_charging:
            return "Service Dock"
        if self.service_bot_dock_contact:
            return "Dock Contact"
        if self.service_bot_waiting_for_task:
            return "Home Waiting"
        if self.service_bot_return_to_charge or self.service_bot_mission_mode == "charge_return":
            return "Returning Dock"
        return "Serving Floor"

    def _robot_b_goal_text(self) -> str:
        path = self.service_bot_active_path
        if not path:
            return "N/A"
        goal_x, goal_y = path[self.service_bot_goal_index % len(path)]
        return f"{goal_x:.1f}, {goal_y:.1f}"

    def _build_status_panel_text(self) -> str:
        current_goal = self.latest_mission_status.get("current_goal") if isinstance(self.latest_mission_status, dict) else None
        current_goal_text = "N/A"
        if isinstance(current_goal, list) and len(current_goal) >= 2:
            current_goal_text = f"{current_goal[0]:.2f}, {current_goal[1]:.2f}"
        amr_a_table = mission_display_name(self.current_mission_mode)
        amr_b_table = mission_display_name(self.service_bot_mission_mode)
        if self.status_render_ready and self.status_render_mode == "fallback":
            renderer_state = "LIVE (2D FALLBACK)"
        elif self.status_render_ready:
            renderer_state = "LIVE (OPENGL)"
        else:
            renderer_state = self.status_render_error or "UNAVAILABLE"
        return "\n".join(
            [
                "DUAL AMR LIVE OPERATIONS",
                "------------------------",
                f"Renderer : {renderer_state}",
                f"Sim Time : {self.data.time:5.1f}s",
                f"AMR-A    : {amr_a_table} | {100.0 * self.battery_pct:5.1f}% | {self._main_robot_dock_state()} | {self.latest_autonomy_state}",
                f"AMR-A Goal: {current_goal_text} | Speed {self.speed_limit_scale:.2f}x | Charging {'YES' if self.is_charging else 'NO'}",
                f"AMR-B    : {amr_b_table} | {100.0 * self.service_bot_battery_pct:5.1f}% | {self._service_robot_dock_state()} | {self.service_bot_state}",
                f"AMR-B Goal: {self._robot_b_goal_text()} | Speed {self.service_bot_speed_scale:.2f}x | Charging {'YES' if self.service_bot_is_charging else 'NO'}",
                f"Operator : {self.operator_message}",
            ]
        )

    def _build_dashboard_state(self) -> dict[str, object]:
        current_goal = self.latest_mission_status.get("current_goal") if isinstance(self.latest_mission_status, dict) else None
        goal_text = "N/A"
        if isinstance(current_goal, list) and len(current_goal) >= 2:
            goal_text = f"{current_goal[0]:.1f}, {current_goal[1]:.1f}"
        return {
            "battery": f"{100.0 * self.battery_pct:4.1f}%",
            "mission": mission_display_name(self.current_mission_mode),
            "autonomy": self.latest_autonomy_state,
            "dock": self._main_robot_dock_state(),
            "speed": f"{self.speed_limit_scale:.2f}x",
            "goal": goal_text,
            "robot_b_battery": f"{100.0 * self.service_bot_battery_pct:4.1f}%",
            "robot_b_mission": mission_display_name(self.service_bot_mission_mode),
            "robot_b_state": "Charging" if self.service_bot_is_charging else self.service_bot_state,
            "robot_b_dock": self._service_robot_dock_state(),
            "robot_b_speed": f"{self.service_bot_speed_scale:.2f}x",
            "robot_b_goal": self._robot_b_goal_text(),
            "selected_mission": self.current_mission_mode,
            "selected_robot_b_mission": self.service_bot_mission_mode,
            "speed_scale": self.speed_limit_scale,
            "robot_b_speed_scale": self.service_bot_speed_scale,
            "battery_ratio": float(self.battery_pct),
            "robot_b_battery_ratio": float(self.service_bot_battery_pct),
            "charging_a": self.is_charging,
            "charging_b": self.service_bot_is_charging,
            "renderer_status": (
                "Live simulation feed ready (OpenGL)"
                if self.status_render_ready and self.status_render_mode == "opengl"
                else (
                    "Live 2D hotel operations map active"
                    if self.status_render_ready and self.status_render_mode == "fallback"
                    else (self.status_render_error or "Live simulation feed unavailable on current graphics context")
                )
            ),
            "preview_status": (
                "Dual-robot preview ready"
                if self.status_render_ready
                else (self.status_render_error or "Dual-robot preview unavailable on current graphics context")
            ),
            "details_text": self._build_status_panel_text(),
            "events": list(reversed(self.recent_events[-8:])),
            "battery_history_a": list(self.battery_history_a),
            "battery_history_b": list(self.battery_history_b),
        }

    def _dashboard_camera_config(self) -> tuple[float, float, float, str]:
        if self.status_window is None:
            return 180.0, -48.0, 24.0, "Center"
        try:
            return self.status_window.get_camera_config()
        except Exception:
            return 180.0, -48.0, 24.0, "Center"

    def _camera_focus_target(self, focus_key: str) -> tuple[tuple[float, float], float]:
        focus = (focus_key or "Center").strip().lower()
        if focus == "dock":
            return (DOCK_X + 0.55, DOCK_Y + 0.10), 0.72
        if focus == "dining":
            return (1.10, 0.80), 0.72
        if focus == "vip":
            return (6.10, 3.55), 0.72
        if focus == "entrance":
            return (7.85, -4.15), 0.72
        if focus == "amr-a":
            x, y, _, _ = self._read_pose()
            return (float(x), float(y)), 0.56
        if focus == "amr-b":
            return (float(self.service_bot_pose[0]) - 0.18, float(self.service_bot_pose[1]) + 0.08), 0.46
        return (0.20, -0.40), 0.64

    def _record_battery_history(self) -> None:
        if self.data.time - self.last_battery_history_time < 0.5:
            return
        self.last_battery_history_time = self.data.time
        self.battery_history_a.append(float(self.battery_pct))
        self.battery_history_b.append(float(self.service_bot_battery_pct))
        self.battery_history_a = self.battery_history_a[-80:]
        self.battery_history_b = self.battery_history_b[-80:]

    def _dispatch_startup_missions_if_ready(self) -> None:
        if self.data.time < self.startup_mission_delay_sec:
            return
        if (
            not self.startup_mission_a_dispatched
            and self.startup_mission_a
            and self.current_mission_mode == "idle"
            and self.latest_autonomy_state not in {"N/A", "WAITING_FOR_ODOM"}
        ):
            self.startup_mission_a_dispatched = True
            self.get_logger().info(f"Dispatching startup mission for AMR-A: {self.startup_mission_a}")
            self._dispatch_mission_command(self.startup_mission_a)
        if not self.startup_mission_b_dispatched and self.startup_mission_b and self.service_bot_mission_mode == "idle":
            self.startup_mission_b_dispatched = True
            self.get_logger().info(f"Dispatching startup mission for AMR-B: {self.startup_mission_b}")
            self._dispatch_robot_b_mission(self.startup_mission_b)

    def _dispatch_trigger_request(self, client, label: str, pending_message: str) -> None:
        self.operator_message = pending_message
        if not client.wait_for_service(timeout_sec=0.05):
            self.operator_message = f"{label}: service unavailable"
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda fut, op=label: self._handle_service_response(op, fut))

    def _dispatch_set_bool_request(self, client, value: bool, label: str, pending_message: str) -> None:
        self.operator_message = pending_message
        if not client.wait_for_service(timeout_sec=0.05):
            self.operator_message = f"{label}: service unavailable"
            return
        request = SetBool.Request()
        request.data = bool(value)
        future = client.call_async(request)
        future.add_done_callback(lambda fut, op=label: self._handle_service_response(op, fut))

    def _dispatch_mission_command(self, mission_mode: str) -> None:
        mission_mode = normalize_service_target_key((mission_mode or "").strip().lower())
        if not mission_mode:
            self.operator_message = "Mission Apply: empty mission"
            return
        if mission_mode in SERVICE_TARGETS and self._service_target_conflicts("AMR-A", mission_mode):
            self.deferred_mission_a = mission_mode
            self.operator_message = f"AMR-A Wait [SAFE]: {mission_display_name(mission_mode)} busy"
            self._push_recent_event(
                f"AMR-A waiting for target clearance: {mission_display_name(mission_mode)} already assigned to AMR-B"
            )
            return
        if self.active_task_a is not None and self.active_task_a.mission != mission_mode:
            self.active_task_a.status = "PREEMPTED"
            self.active_task_a.finished_at = float(self.data.time)
            self.active_task_a.last_note = "Operator changed mission"
            self.completed_tasks.append(self.active_task_a)
        self.deferred_mission_a = None
        self.mission_command_pub.publish(String(data=mission_mode))
        self.current_mission_mode = mission_mode
        self.is_charging = False
        self._apply_main_navigation_profile()
        self.active_task_a = TaskRecord(
            task_id=f"AMR-A-DIRECT-{int(self.data.time * 10):04d}",
            robot="AMR-A",
            mission=mission_mode,
            priority=self._task_priority("AMR-A", mission_mode),
            status="RUNNING",
            created_at=float(self.data.time),
            started_at=float(self.data.time),
            last_note=f"Serving {mission_display_name(mission_mode)}",
        )
        self.operator_message = f"AMR-A Apply [OK]: {mission_display_name(mission_mode)}"
        self._push_recent_event(f"Operator mission request AMR-A: {mission_display_name(mission_mode)}")

    def _dispatch_speed_limit(self, speed_scale: float) -> None:
        speed_scale = float(max(0.20, min(1.25, speed_scale)))
        self.speed_limit_pub.publish(Float32(data=speed_scale))
        self.speed_limit_scale = speed_scale
        self._apply_main_navigation_profile()
        self.operator_message = f"Speed Apply [OK]: {speed_scale:.2f}x"
        self._push_recent_event(f"Operator speed request: {speed_scale:.2f}x")

    def _dispatch_robot_b_mission(self, mission_mode: str) -> None:
        mission_mode = normalize_service_target_key((mission_mode or "").strip().lower())
        if not mission_mode:
            self.operator_message = "AMR-B Mission: empty mission"
            return
        if mission_mode in SERVICE_TARGETS and self._service_target_conflicts("AMR-B", mission_mode):
            self.deferred_mission_b = mission_mode
            self.operator_message = f"AMR-B Wait [SAFE]: {mission_display_name(mission_mode)} busy"
            self._push_recent_event(
                f"AMR-B waiting for target clearance: {mission_display_name(mission_mode)} already assigned to AMR-A"
            )
            return
        if self.active_task_b is not None and self.active_task_b.mission != mission_mode:
            self.active_task_b.status = "PREEMPTED"
            self.active_task_b.finished_at = float(self.data.time)
            self.active_task_b.last_note = "Operator changed AMR-B mission"
            self.completed_tasks.append(self.active_task_b)
        self.deferred_mission_b = None
        self.service_bot_mission_mode = mission_mode
        self.service_bot_saved_mission_mode = mission_mode if mission_mode != "charge_return" else self.service_bot_saved_mission_mode
        self.service_bot_goal_index = 0
        self.service_bot_return_to_charge = mission_mode == "charge_return"
        self.service_bot_waiting_for_task = False
        self.service_bot_table_hold_until = None
        self.service_bot_is_charging = False
        self._reset_service_bot_active_path()
        self.active_task_b = TaskRecord(
            task_id=f"AMR-B-DIRECT-{int(self.data.time * 10):04d}",
            robot="AMR-B",
            mission=mission_mode,
            priority=self._task_priority("AMR-B", mission_mode),
            status="RUNNING",
            created_at=float(self.data.time),
            started_at=float(self.data.time),
            last_note=f"Serving {mission_display_name(mission_mode)}",
        )
        self.operator_message = f"AMR-B Apply [OK]: {mission_display_name(mission_mode)}"
        self._push_recent_event(f"Operator mission request AMR-B: {mission_display_name(mission_mode)}")

    def _service_target_conflicts(self, requesting_robot: str, mission_mode: str) -> bool:
        mission_key = normalize_service_target_key((mission_mode or "").strip().lower())
        if mission_key not in SERVICE_TARGETS:
            return False
        if requesting_robot == "AMR-A":
            return bool(
                self.active_task_b is not None
                and self.active_task_b.status == "RUNNING"
                and self.active_task_b.mission == mission_key
            )
        return bool(
            self.active_task_a is not None
            and self.active_task_a.status == "RUNNING"
            and self.active_task_a.mission == mission_key
        )

    def _dispatch_robot_b_speed(self, speed_scale: float) -> None:
        self.service_bot_speed_scale = float(max(0.20, min(1.25, speed_scale)))
        self.operator_message = f"AMR-B Speed [OK]: {self.service_bot_speed_scale:.2f}x"
        self._push_recent_event(f"Operator speed request AMR-B: {self.service_bot_speed_scale:.2f}x")

    def _dispatch_return_home_a(self) -> None:
        self._dispatch_mission_command("dock_return")

    def _dispatch_return_home_b(self) -> None:
        self._dispatch_robot_b_mission("charge_return")

    def _apply_scene_lighting(self) -> None:
        profiles = {
            "day": {
                "headlight_ambient": (0.14, 0.14, 0.13),
                "headlight_diffuse": (0.36, 0.34, 0.32),
                "haze": (0.34, 0.32, 0.28, 1.0),
                "lights": {
                    "ambient_house": (0.36, 0.34, 0.30),
                    "reception_light": (0.62, 0.52, 0.38),
                    "dining_light": (0.68, 0.58, 0.42),
                    "kitchen_light": (0.48, 0.50, 0.54),
                    "corridor_light": (0.60, 0.52, 0.40),
                    "dock_light": (0.20, 0.52, 0.24),
                    "service_dock_light": (0.24, 0.58, 0.74),
                    "vip_light": (0.68, 0.56, 0.40),
                    "lounge_light": (0.64, 0.54, 0.38),
                    "entry_light_upper": (0.62, 0.56, 0.40),
                    "entry_light_lower": (0.60, 0.54, 0.38),
                    "ceiling_center_light": (0.44, 0.38, 0.30),
                    "ceiling_left_light": (0.34, 0.30, 0.26),
                    "ceiling_right_light": (0.36, 0.30, 0.26),
                    "reception_accent_light": (0.78, 0.64, 0.46),
                    "reception_backwash_light": (0.52, 0.44, 0.34),
                    "vip_fill_light": (0.64, 0.54, 0.38),
                    "lounge_fill_light": (0.60, 0.50, 0.36),
                    "dining_fill_light": (0.66, 0.56, 0.40),
                },
            },
            "evening": {
                "headlight_ambient": (0.12, 0.10, 0.10),
                "headlight_diffuse": (0.32, 0.28, 0.24),
                "haze": (0.28, 0.20, 0.18, 1.0),
                "lights": {
                    "ambient_house": (0.22, 0.18, 0.16),
                    "reception_light": (0.84, 0.62, 0.42),
                    "dining_light": (0.90, 0.66, 0.42),
                    "kitchen_light": (0.54, 0.56, 0.64),
                    "corridor_light": (0.78, 0.58, 0.34),
                    "dock_light": (0.24, 0.60, 0.28),
                    "service_dock_light": (0.28, 0.66, 0.84),
                    "vip_light": (0.90, 0.70, 0.44),
                    "lounge_light": (0.84, 0.66, 0.42),
                    "entry_light_upper": (0.88, 0.72, 0.48),
                    "entry_light_lower": (0.84, 0.68, 0.44),
                    "ceiling_center_light": (0.62, 0.50, 0.36),
                    "ceiling_left_light": (0.50, 0.42, 0.32),
                    "ceiling_right_light": (0.52, 0.42, 0.32),
                    "reception_accent_light": (0.98, 0.76, 0.50),
                    "reception_backwash_light": (0.72, 0.56, 0.40),
                    "vip_fill_light": (0.82, 0.66, 0.42),
                    "lounge_fill_light": (0.76, 0.62, 0.40),
                    "dining_fill_light": (0.84, 0.68, 0.44),
                },
            },
            "night": {
                "headlight_ambient": (0.07, 0.08, 0.10),
                "headlight_diffuse": (0.18, 0.20, 0.24),
                "haze": (0.14, 0.16, 0.20, 1.0),
                "lights": {
                    "ambient_house": (0.10, 0.12, 0.16),
                    "reception_light": (0.62, 0.48, 0.34),
                    "dining_light": (0.76, 0.58, 0.38),
                    "kitchen_light": (0.42, 0.46, 0.54),
                    "corridor_light": (0.62, 0.48, 0.30),
                    "dock_light": (0.28, 0.74, 0.34),
                    "service_dock_light": (0.34, 0.76, 0.94),
                    "vip_light": (0.78, 0.62, 0.40),
                    "lounge_light": (0.72, 0.58, 0.38),
                    "entry_light_upper": (0.70, 0.62, 0.40),
                    "entry_light_lower": (0.68, 0.60, 0.38),
                    "ceiling_center_light": (0.44, 0.36, 0.28),
                    "ceiling_left_light": (0.34, 0.28, 0.24),
                    "ceiling_right_light": (0.36, 0.28, 0.24),
                    "reception_accent_light": (0.84, 0.66, 0.44),
                    "reception_backwash_light": (0.56, 0.44, 0.32),
                    "vip_fill_light": (0.68, 0.56, 0.38),
                    "lounge_fill_light": (0.64, 0.52, 0.36),
                    "dining_fill_light": (0.72, 0.58, 0.40),
                },
            },
            "spotlight": {
                "headlight_ambient": (0.10, 0.10, 0.10),
                "headlight_diffuse": (0.24, 0.24, 0.22),
                "haze": (0.18, 0.16, 0.14, 1.0),
                "lights": {
                    "ambient_house": (0.08, 0.08, 0.08),
                    "reception_light": (0.92, 0.70, 0.46),
                    "dining_light": (1.00, 0.76, 0.48),
                    "kitchen_light": (0.42, 0.44, 0.50),
                    "corridor_light": (0.86, 0.62, 0.36),
                    "dock_light": (0.32, 0.86, 0.38),
                    "service_dock_light": (0.36, 0.84, 1.00),
                    "vip_light": (1.00, 0.76, 0.48),
                    "lounge_light": (0.92, 0.70, 0.44),
                    "entry_light_upper": (0.96, 0.80, 0.52),
                    "entry_light_lower": (0.92, 0.76, 0.48),
                    "ceiling_center_light": (0.68, 0.54, 0.38),
                    "ceiling_left_light": (0.52, 0.42, 0.32),
                    "ceiling_right_light": (0.54, 0.42, 0.32),
                    "reception_accent_light": (1.00, 0.82, 0.54),
                    "reception_backwash_light": (0.74, 0.58, 0.40),
                    "vip_fill_light": (0.88, 0.70, 0.44),
                    "lounge_fill_light": (0.82, 0.66, 0.42),
                    "dining_fill_light": (0.90, 0.72, 0.46),
                },
            },
            "cinema": {
                "headlight_ambient": (0.08, 0.07, 0.08),
                "headlight_diffuse": (0.16, 0.16, 0.18),
                "haze": (0.12, 0.10, 0.12, 1.0),
                "lights": {
                    "ambient_house": (0.06, 0.06, 0.07),
                    "reception_light": (1.00, 0.76, 0.46),
                    "dining_light": (1.00, 0.80, 0.48),
                    "kitchen_light": (0.28, 0.32, 0.42),
                    "corridor_light": (0.96, 0.68, 0.38),
                    "dock_light": (0.22, 0.72, 0.32),
                    "service_dock_light": (0.28, 0.76, 0.92),
                    "vip_light": (1.00, 0.84, 0.54),
                    "lounge_light": (0.96, 0.74, 0.46),
                    "entry_light_upper": (1.00, 0.84, 0.56),
                    "entry_light_lower": (0.98, 0.82, 0.54),
                    "ceiling_center_light": (0.74, 0.58, 0.40),
                    "ceiling_left_light": (0.56, 0.44, 0.32),
                    "ceiling_right_light": (0.58, 0.44, 0.32),
                    "reception_accent_light": (1.00, 0.88, 0.60),
                    "reception_backwash_light": (0.82, 0.64, 0.44),
                    "vip_fill_light": (0.92, 0.74, 0.46),
                    "lounge_fill_light": (0.88, 0.70, 0.44),
                    "dining_fill_light": (0.98, 0.78, 0.48),
                },
            },
        }
        profile = profiles.get(self.light_mode, profiles["day"])
        self.model.vis.headlight.ambient[:] = np.array(profile["headlight_ambient"], dtype=float)
        self.model.vis.headlight.diffuse[:] = np.array(profile["headlight_diffuse"], dtype=float)
        self.model.vis.rgba.haze[:] = np.array(profile["haze"], dtype=float)
        for light_name, diffuse in profile["lights"].items():
            light_id = self.scene_light_ids.get(light_name, -1)
            if light_id is not None and light_id >= 0:
                self.model.light_diffuse[light_id, :] = np.array(diffuse, dtype=float)

    def _dispatch_lighting_mode(self, mode: str) -> None:
        normalized = (mode or "Day").strip().lower()
        if normalized not in {"day", "evening", "night", "spotlight", "cinema"}:
            normalized = "day"
        self.light_mode = normalized
        self._apply_scene_lighting()
        self.operator_message = f"Lighting [OK]: {normalized.title()}"
        self._push_recent_event(f"Scene lighting set to {normalized.title()}")

    def _task_priority(self, robot: str, mission_mode: str) -> int:
        mission = (mission_mode or "").strip().lower()
        if mission in SERVICE_TARGETS:
            return 85 if robot == "AMR-A" else 82
        if mission in {"dock_return", "charge_return"}:
            return 100
        if mission in {"room_delivery", "table_service", "guest_delivery"}:
            return 80
        if mission in {"lobby_patrol", "lobby_assist"}:
            return 60
        return 40 if robot == "AMR-A" else 35

    def _format_task_label(self, task: TaskRecord) -> str:
        suffix = f"P{task.priority}"
        return f"{task.task_id} | {task.mission} | {task.status} | {suffix}"

    def _create_task(self, robot: str, mission_mode: str) -> TaskRecord:
        self.task_counter += 1
        return TaskRecord(
            task_id=f"{robot}-{self.task_counter:03d}",
            robot=robot,
            mission=mission_mode,
            priority=self._task_priority(robot, mission_mode),
            status="QUEUED",
            created_at=float(self.data.time),
            retries_remaining=1,
        )

    def _sort_task_queue(self, queue: list[TaskRecord]) -> None:
        queue.sort(key=lambda task: (-task.priority, task.created_at))

    def _complete_task(self, task: TaskRecord | None, note: str) -> None:
        if task is None:
            return
        task.status = "COMPLETED"
        task.finished_at = float(self.data.time)
        task.last_note = note
        self.completed_tasks.append(task)
        self.completed_tasks = self.completed_tasks[-12:]
        self._push_recent_event(f"{task.task_id} completed: {note}")

    def _activate_task_a(self, task: TaskRecord) -> None:
        task.status = "RUNNING"
        task.started_at = float(self.data.time)
        task.last_note = "Dispatched to autonomy manager"
        self.active_task_a = task
        self._dispatch_mission_command(task.mission)
        self._push_recent_event(f"{task.task_id} dispatched to AMR-A")

    def _activate_task_b(self, task: TaskRecord) -> None:
        task.status = "RUNNING"
        task.started_at = float(self.data.time)
        task.last_note = "Dispatched to service robot"
        self.active_task_b = task
        self._dispatch_robot_b_mission(task.mission)
        self._push_recent_event(f"{task.task_id} dispatched to AMR-B")

    def _queue_mission_a(self, mission_mode: str) -> None:
        mission_mode = (mission_mode or "").strip().lower()
        if not mission_mode:
            self.operator_message = "AMR-A Queue: empty mission"
            return
        task = self._create_task("AMR-A", mission_mode)
        self.task_queue_a.append(task)
        self._sort_task_queue(self.task_queue_a)
        self.operator_message = f"AMR-A Queue [OK]: {task.task_id}"
        self._push_recent_event(f"Queued AMR-A mission: {task.task_id} -> {mission_mode}")

    def _queue_mission_b(self, mission_mode: str) -> None:
        mission_mode = (mission_mode or "").strip().lower()
        if not mission_mode:
            self.operator_message = "AMR-B Queue: empty mission"
            return
        task = self._create_task("AMR-B", mission_mode)
        self.task_queue_b.append(task)
        self._sort_task_queue(self.task_queue_b)
        self.operator_message = f"AMR-B Queue [OK]: {task.task_id}"
        self._push_recent_event(f"Queued AMR-B mission: {task.task_id} -> {mission_mode}")

    def _run_next_task_a(self) -> None:
        if not self.task_queue_a:
            self.operator_message = "AMR-A Queue: no mission pending"
            return
        if self.active_task_a is not None and self.active_task_a.status == "RUNNING":
            self.operator_message = f"AMR-A Busy: {self.active_task_a.task_id}"
            return
        task = self.task_queue_a.pop(0)
        self._activate_task_a(task)

    def _run_next_task_b(self) -> None:
        if not self.task_queue_b:
            self.operator_message = "AMR-B Queue: no mission pending"
            return
        if self.active_task_b is not None and self.active_task_b.status == "RUNNING":
            self.operator_message = f"AMR-B Busy: {self.active_task_b.task_id}"
            return
        task = self.task_queue_b.pop(0)
        self._activate_task_b(task)

    def _clear_task_queue_a(self) -> None:
        self.task_queue_a.clear()
        self.operator_message = "AMR-A Queue [OK]: cleared"
        self._push_recent_event("Cleared AMR-A mission queue")

    def _clear_task_queue_b(self) -> None:
        self.task_queue_b.clear()
        self.operator_message = "AMR-B Queue [OK]: cleared"
        self._push_recent_event("Cleared AMR-B mission queue")

    def _update_task_scheduler(self) -> None:
        if self.active_task_a is not None and self.active_task_a.status == "RUNNING":
            if self.active_task_a.mission == "dock_return" and self.is_charging:
                self._complete_task(self.active_task_a, "dock contact achieved")
                self.active_task_a = None
            elif self.active_task_a.mission in SERVICE_TARGETS:
                if (
                    self.is_charging
                    and isinstance(self.latest_mission_status, dict)
                    and str(self.latest_mission_status.get("mission_mode", "")).lower() == "dock_return"
                ):
                    self._complete_task(self.active_task_a, f"{mission_display_name(self.active_task_a.mission)} served and docked")
                    self.active_task_a = None
                elif self.latest_autonomy_state == "WAITING_TASK" and self.active_task_a.started_at is not None:
                    self._complete_task(self.active_task_a, f"{mission_display_name(self.active_task_a.mission)} served")
                    self.active_task_a = None
            elif (
                self.active_task_a.mission == "room_delivery"
                and self._near_station(np.array([PLACE_STATION_X, PLACE_STATION_Y], dtype=float), threshold=1.10)
                and self.active_task_a.started_at is not None
                and self.data.time - self.active_task_a.started_at > 8.0
            ):
                self._complete_task(self.active_task_a, "guest delivery checkpoint completed")
                self.active_task_a = None
            elif self.active_task_a.mission in {"lobby_patrol", "table_service", "room_delivery"} and self.latest_autonomy_state == "PATROL":
                self.active_task_a.last_note = f"Waypoint {self.latest_mission_status.get('waypoint_index', 'n/a')}"

        if self.active_task_b is not None and self.active_task_b.status == "RUNNING":
            if self.active_task_b.mission == "charge_return" and self.service_bot_is_charging:
                self._complete_task(self.active_task_b, "service dock reached")
                self.active_task_b = None
            elif self.active_task_b.mission in SERVICE_TARGETS:
                if self.service_bot_waiting_for_task and self.active_task_b.started_at is not None:
                    self._complete_task(self.active_task_b, f"{mission_display_name(self.active_task_b.mission)} served")
                    self.active_task_b = None
            elif self.active_task_b.mission in {"table_service", "guest_delivery"} and self.service_bot_payload_state in {"TRAY", "ROOM"}:
                self.active_task_b.last_note = f"Payload state: {self.service_bot_payload_state}"
            elif self.active_task_b.mission == "lobby_assist" and self.service_bot_payload_state == "GUIDE":
                self.active_task_b.last_note = "Guiding guests across the lobby route"

        if self.active_task_a is None and self.task_queue_a and not self.is_charging:
            self._activate_task_a(self.task_queue_a.pop(0))

        if self.active_task_b is None and self.task_queue_b and not self.service_bot_is_charging:
            self._activate_task_b(self.task_queue_b.pop(0))

        if (
            self.deferred_mission_a
            and self.active_task_a is None
            and not self.is_charging
            and not self._service_target_conflicts("AMR-A", self.deferred_mission_a)
        ):
            pending_mission = self.deferred_mission_a
            self.deferred_mission_a = None
            self._push_recent_event(f"AMR-A target cleared: dispatching {mission_display_name(pending_mission)}")
            self._dispatch_mission_command(pending_mission)

        if (
            self.deferred_mission_b
            and self.active_task_b is None
            and not self.service_bot_is_charging
            and not self._service_target_conflicts("AMR-B", self.deferred_mission_b)
        ):
            pending_mission = self.deferred_mission_b
            self.deferred_mission_b = None
            self._push_recent_event(f"AMR-B target cleared: dispatching {mission_display_name(pending_mission)}")
            self._dispatch_robot_b_mission(pending_mission)

    def _handle_service_response(self, operation: str, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.operator_message = f"{operation}: failed ({exc})"
            return
        if response is None:
            self.operator_message = f"{operation}: no response"
            return
        state = "OK" if getattr(response, "success", False) else "WARN"
        detail = getattr(response, "message", "") or "request processed"
        self.operator_message = f"{operation} [{state}]: {detail}"
        self._push_recent_event(self.operator_message)

    def _render_status_preview(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.status_renderer is None or self.status_preview_renderer is None:
            if self.show_status_window and self.use_viewer:
                self._create_status_renderers()
        if self.status_renderer is None or self.status_preview_renderer is None:
            return self._render_fallback_status_preview()
        if self.data.time - self.last_status_render_sim_time < 1.0 / max(self.render_rate_hz, 1e-3):
            return None, None
        base_x, base_y, _, _ = self._read_pose()
        center_x = clamp(0.5 * (base_x + float(self.service_bot_pose[0])), -WORLD_X * 0.55, WORLD_X * 0.55)
        center_y = clamp(0.5 * (base_y + float(self.service_bot_pose[1])), -WORLD_Y * 0.55, WORLD_Y * 0.55)
        user_azimuth, user_elevation, user_distance, focus_key = self._dashboard_camera_config()
        focus_xy, focus_z = self._camera_focus_target(focus_key)
        main_camera = self._make_world_camera(
            user_azimuth,
            user_elevation,
            user_distance,
            lookat_xy=focus_xy,
            lookat_z=focus_z,
        )
        preview_camera = self._make_world_camera(
            clamp(user_azimuth - 6.0, 35.0, 245.0),
            clamp(user_elevation + 3.0, -60.0, -6.0),
            clamp(user_distance * 0.55, 8.2, 13.2),
            lookat_xy=(center_x, center_y),
            lookat_z=0.46,
        )
        try:
            self._ensure_offscreen_context()
            self.status_renderer.update_scene(self.data, camera=main_camera)
            self.status_preview_renderer.update_scene(self.data, camera=preview_camera)
            main_image = np.array(self.status_renderer.render(), copy=True)
            preview_image = np.array(self.status_preview_renderer.render(), copy=True)
        except Exception as exc:
            self.get_logger().warning(f"Status renderer frame dropped: {exc}")
            self.status_renderer = None
            self.status_preview_renderer = None
            self.status_render_ready = False
            self.status_render_mode = "unavailable"
            self.status_render_error = f"OpenGL frame render failed: {exc}"
            return self._render_fallback_status_preview()
        if float(np.mean(main_image)) < 1.0 and float(np.mean(preview_image)) < 1.0:
            self.get_logger().warning("Status renderer returned blank frames, using fallback map.")
            self.status_renderer = None
            self.status_preview_renderer = None
            self.status_render_ready = False
            self.status_render_mode = "unavailable"
            self.status_render_error = "OpenGL renderer returned blank frames on current graphics context"
            return self._render_fallback_status_preview()
        self.status_render_ready = True
        self.status_render_mode = "opengl"
        self.status_render_error = ""
        self.last_status_render_sim_time = self.data.time
        main_image = self._overlay_service_scene_annotations(
            main_image,
            azimuth_deg=user_azimuth,
            elevation_deg=user_elevation,
            distance=user_distance,
            lookat_xy=focus_xy,
            lookat_z=focus_z,
            compact=False,
        )
        preview_image = self._overlay_service_scene_annotations(
            preview_image,
            azimuth_deg=clamp(user_azimuth - 6.0, 35.0, 245.0),
            elevation_deg=clamp(user_elevation + 3.0, -60.0, -6.0),
            distance=clamp(user_distance * 0.55, 8.2, 13.2),
            lookat_xy=(center_x, center_y),
            lookat_z=0.46,
            compact=True,
        )
        return main_image, preview_image

    def _world_to_canvas(self, x: float, y: float, width: int, height: int, pad: int = 24) -> tuple[int, int]:
        usable_w = max(1, width - 2 * pad)
        usable_h = max(1, height - 2 * pad)
        px = pad + int(((x + WORLD_X) / (2.0 * WORLD_X)) * usable_w)
        py = height - pad - int(((y + WORLD_Y) / (2.0 * WORLD_Y)) * usable_h)
        return px, py

    def _render_fallback_status_preview(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if PilImage is None or ImageDraw is None:
            return None, None
        self.status_render_ready = True
        self.status_render_mode = "fallback"
        self.status_render_error = "OpenGL unavailable, showing truthful 2D hotel map"
        self.last_status_render_sim_time = self.data.time
        return (
            self._render_fallback_map_image(1024, 760, include_labels=True),
            self._render_fallback_map_image(460, 320, include_labels=False),
        )

    def _render_fallback_map_image(self, width: int, height: int, include_labels: bool) -> np.ndarray:
        image = PilImage.new("RGB", (width, height), color=(10, 14, 19))
        draw = ImageDraw.Draw(image)

        floor_margin = 22
        draw.rectangle(
            [(floor_margin, floor_margin), (width - floor_margin, height - floor_margin)],
            fill=(26, 31, 37),
            outline=(63, 86, 108),
            width=2,
        )

        carpet_color = (103, 59, 120)
        for zone_x, zone_y, zone_w, zone_h in (
            (-5.9, 4.5, 2.2, 1.3),
            (1.8, 1.4, 3.2, 2.7),
            (-1.7, -4.4, 3.5, 1.0),
            (6.0, 4.3, 2.2, 1.6),
        ):
            x0, y0 = self._world_to_canvas(zone_x - zone_w, zone_y + zone_h, width, height)
            x1, y1 = self._world_to_canvas(zone_x + zone_w, zone_y - zone_h, width, height)
            draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=18, fill=carpet_color)

        for box in NAVIGATION_BOXES:
            x0, y0 = self._world_to_canvas(box.pos[0] - box.size[0], box.pos[1] + box.size[1], width, height)
            x1, y1 = self._world_to_canvas(box.pos[0] + box.size[0], box.pos[1] - box.size[1], width, height)
            fill = tuple(int(max(0, min(255, channel * 255))) for channel in box.rgba[:3])
            draw.rectangle([(x0, y0), (x1, y1)], fill=fill, outline=(16, 20, 24))

        for table_key, table in HOTEL_TABLES.items():
            table_x, table_y = table["pos"]
            cx, cy = self._world_to_canvas(float(table_x), float(table_y), width, height)
            draw.ellipse([(cx - 18, cy - 18), (cx + 18, cy + 18)], fill=(206, 154, 92), outline=(255, 236, 200), width=2)
            if include_labels:
                label_x, label_y = table.get("label_pos", table["pos"])
                lx, ly = self._world_to_canvas(float(label_x), float(label_y), width, height)
                draw.text((lx - 8, ly - 8), f"T{list(HOTEL_TABLES.keys()).index(table_key) + 1}", fill=(18, 20, 24))

        for sofa_key, sofa in SOFA_SPOTS.items():
            sofa_x, sofa_y = sofa["pos"]
            cx, cy = self._world_to_canvas(float(sofa_x), float(sofa_y), width, height)
            draw.rounded_rectangle([(cx - 22, cy - 14), (cx + 22, cy + 14)], radius=6, fill=(70, 76, 86), outline=(240, 244, 248), width=2)
            if include_labels:
                label_x, label_y = sofa.get("label_pos", sofa["pos"])
                lx, ly = self._world_to_canvas(float(label_x), float(label_y), width, height)
                short_label = f"S{sofa_key.rsplit('_', 1)[-1]}"
                draw.text((lx - 10, ly - 8), short_label, fill=(232, 238, 244))

        for zone_x, zone_y, zone_w, zone_h, color in (
            (DOCK_X, DOCK_Y, 0.62, 0.96, (42, 220, 95)),
            (SERVICE_DOCK_X, SERVICE_DOCK_Y, 0.56, 0.82, (56, 168, 255)),
            (PICK_STATION_X, PICK_STATION_Y, 1.15, 0.90, (42, 120, 230)),
            (PLACE_STATION_X, PLACE_STATION_Y, 1.20, 0.96, (58, 210, 110)),
        ):
            x0, y0 = self._world_to_canvas(zone_x - zone_w, zone_y + zone_h, width, height)
            x1, y1 = self._world_to_canvas(zone_x + zone_w, zone_y - zone_h, width, height)
            draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=3)

        base_x, base_y, base_yaw, _ = self._read_pose()
        self._draw_robot_marker(draw, width, height, base_x, base_y, base_yaw, (35, 228, 108), "A", include_labels)
        self._draw_robot_marker(
            draw,
            width,
            height,
            float(self.service_bot_pose[0]),
            float(self.service_bot_pose[1]),
            float(self.service_bot_yaw),
            (66, 172, 255),
            "B",
            include_labels,
        )

        if include_labels:
            draw.text((26, 20), f"2D {BRAND_MAP_TITLE}", fill=(170, 242, 219))
            draw.text((26, 42), f"Mode: {self.current_mission_mode}", fill=(202, 214, 226))
            draw.text((26, 62), f"Render: {self.status_render_error}", fill=(120, 152, 184))

        return np.array(image, dtype=np.uint8)

    def _project_world_to_image(
        self,
        point_xyz: tuple[float, float, float],
        width: int,
        height: int,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
        lookat_xy: tuple[float, float],
        lookat_z: float,
        fovy_deg: float = 45.0,
    ) -> tuple[int, int] | None:
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        lookat = np.array([lookat_xy[0], lookat_xy[1], lookat_z], dtype=float)
        forward = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=float,
        )
        camera_pos = lookat - forward * distance
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-6:
            return None
        right /= right_norm
        up = np.cross(right, forward)
        point = np.array(point_xyz, dtype=float)
        delta = point - camera_pos
        depth = float(np.dot(delta, forward))
        if depth <= 0.05:
            return None
        x_cam = float(np.dot(delta, right))
        y_cam = float(np.dot(delta, up))
        focal = (height * 0.5) / math.tan(math.radians(fovy_deg) * 0.5)
        px = int(width * 0.5 + (x_cam / depth) * focal)
        py = int(height * 0.5 - (y_cam / depth) * focal)
        if px < -40 or px > width + 40 or py < -40 or py > height + 40:
            return None
        return px, py

    def _overlay_service_scene_annotations(
        self,
        image: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
        lookat_xy: tuple[float, float],
        lookat_z: float,
        compact: bool,
    ) -> np.ndarray:
        if PilImage is None or ImageDraw is None:
            return image
        pil_image = PilImage.fromarray(image)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        width, height = pil_image.size
        for index, table in enumerate(HOTEL_TABLES.values(), start=1):
            label_x, label_y = table.get("label_pos", table["pos"])
            projected = self._project_world_to_image(
                (float(label_x), float(label_y), 0.81),
                width,
                height,
                azimuth_deg,
                elevation_deg,
                distance,
                lookat_xy,
                lookat_z,
            )
            if projected is None:
                continue
            label = f"{index}"
            tw = 14 if compact else 18
            th = 11
            x0 = projected[0] - tw // 2
            y0 = projected[1] - th // 2
            draw.rounded_rectangle((x0, y0, x0 + tw, y0 + th), radius=6, fill=(18, 20, 24, 170), outline=(244, 205, 94, 210), width=1)
            draw.text((x0 + 4, y0 - 1), label, fill=(255, 243, 214, 255))

        for sofa_key, sofa in SOFA_SPOTS.items():
            label_x, label_y = sofa.get("label_pos", sofa["pos"])
            projected = self._project_world_to_image(
                (float(label_x), float(label_y), 0.81),
                width,
                height,
                azimuth_deg,
                elevation_deg,
                distance,
                lookat_xy,
                lookat_z,
            )
            if projected is None:
                continue
            label = f"S{sofa_key.rsplit('_', 1)[-1]}"
            tw = 18 if compact else 24
            th = 11
            x0 = projected[0] - tw // 2
            y0 = projected[1] - th // 2
            draw.rounded_rectangle((x0, y0, x0 + tw, y0 + th), radius=6, fill=(24, 28, 34, 178), outline=(232, 238, 244, 210), width=1)
            draw.text((x0 + 4, y0 - 1), label, fill=(245, 248, 252, 255))

        dock_labels = [
            ((DOCK_X, DOCK_Y, 1.28), "AUTO DOCK A", (56, 232, 114)),
            ((SERVICE_DOCK_X, SERVICE_DOCK_Y, 1.28), "AUTO DOCK B", (76, 180, 255)),
        ]
        for point, text, color in dock_labels:
            projected = self._project_world_to_image(
                point,
                width,
                height,
                azimuth_deg,
                elevation_deg,
                distance,
                lookat_xy,
                lookat_z,
            )
            if projected is None:
                continue
            tw = 46 if compact else 96
            x0 = projected[0] - tw // 2
            y0 = projected[1] - 12
            draw.rounded_rectangle((x0, y0, x0 + tw, y0 + 18), radius=8, fill=(9, 14, 18, 190), outline=(*color, 240), width=2)
            draw.text((x0 + 6, y0 + 2), text if not compact else text[-1], fill=(236, 244, 248, 255))

        entrance_projected = self._project_world_to_image(
            (7.94, -1.66, 2.30),
            width,
            height,
            azimuth_deg,
            elevation_deg,
            distance,
            lookat_xy,
            lookat_z,
        )
        if entrance_projected is not None:
            label_text = "WELCOME" if not compact else "IN"
            tw = 108 if not compact else 42
            x0 = entrance_projected[0] - tw // 2
            y0 = entrance_projected[1] - 10
            draw.rounded_rectangle((x0, y0, x0 + tw, y0 + 18), radius=8, fill=(9, 14, 18, 190), outline=(255, 226, 146, 240), width=2)
            draw.text((x0 + 7, y0 + 2), label_text, fill=(255, 244, 214, 255))

        base_x, base_y, base_yaw, _ = self._read_pose()
        robot_labels = [
            ((base_x, base_y, 0.92), "AMR-A", (110, 255, 150)),
            ((float(self.service_bot_pose[0]), float(self.service_bot_pose[1]), 0.84), "AMR-B", (120, 208, 255)),
        ]
        for point, text, color in robot_labels:
            projected = self._project_world_to_image(point, width, height, azimuth_deg, elevation_deg, distance, lookat_xy, lookat_z)
            if projected is None:
                continue
            draw.text((projected[0] + 10, projected[1] - 12), text, fill=(*color, 255))

        return np.array(pil_image, dtype=np.uint8)

    def _draw_robot_marker(
        self,
        draw,
        width: int,
        height: int,
        x: float,
        y: float,
        yaw: float,
        color: tuple[int, int, int],
        label: str,
        include_labels: bool,
    ) -> None:
        cx, cy = self._world_to_canvas(x, y, width, height)
        radius = 12
        draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=color, outline=(245, 250, 255), width=2)
        tip_x = cx + int(math.cos(yaw) * 18)
        tip_y = cy - int(math.sin(yaw) * 18)
        draw.line([(cx, cy), (tip_x, tip_y)], fill=(255, 245, 160), width=3)
        if include_labels:
            draw.text((cx + 14, cy - 18), f"AMR-{label}", fill=(240, 244, 248))

    def _read_lidar_mount(self, mount_name: str) -> list[float]:
        ranges = []
        for sensor_id in self.lidar_sensor_ids_by_mount[mount_name]:
            start = self.model.sensor_adr[sensor_id]
            value = float(self.data.sensordata[start])
            if value < 0.0:
                ranges.append(float("inf"))
            else:
                noisy_value = min(value, LIDAR_MAX_RANGE) + self._sample_noise(self.lidar_noise_stddev)
                ranges.append(clamp(noisy_value, 0.03, LIDAR_MAX_RANGE))
        return ranges

    def _read_all_lidar_mounts(self) -> dict[str, list[float]]:
        return {mount.name: self._read_lidar_mount(mount.name) for mount in LIDAR_MOUNTS}

    def _fuse_lidar_scans(self, mount_scans: dict[str, list[float]]) -> list[float]:
        fused = [float("inf")] * len(self.combined_lidar_angles)

        for mount in LIDAR_MOUNTS:
            ranges = mount_scans[mount.name]
            sensor_x, sensor_y, _ = mount.pos
            for relative_angle, distance in zip(self.mount_relative_angles, ranges):
                if not math.isfinite(distance):
                    continue
                world_angle = mount.yaw_rad + relative_angle
                hit_x = sensor_x + distance * math.cos(world_angle)
                hit_y = sensor_y + distance * math.sin(world_angle)
                fused_angle = math.atan2(hit_y, hit_x)
                fused_distance = math.hypot(hit_x, hit_y)
                normalized = (fused_angle + math.pi) / (2.0 * math.pi)
                bin_index = int(round(normalized * (len(fused) - 1)))
                bin_index = max(0, min(len(fused) - 1, bin_index))
                fused[bin_index] = min(fused[bin_index], fused_distance)

        return fused

    def _read_pose(self) -> tuple[float, float, float, tuple[float, float, float, float]]:
        x = float(self.data.xpos[self.base_body_id][0])
        y = float(self.data.xpos[self.base_body_id][1])
        quat_start = self.model.sensor_adr[self.base_quat_sensor]
        quat_wxyz = (
            float(self.data.sensordata[quat_start + 0]),
            float(self.data.sensordata[quat_start + 1]),
            float(self.data.sensordata[quat_start + 2]),
            float(self.data.sensordata[quat_start + 3]),
        )
        yaw = self._joint_yaw_to_world_yaw(float(self.data.qpos[self.base_yaw_qpos_adr]))
        return x, y, yaw, quat_wxyz

    def _joint_yaw_to_world_yaw(self, joint_yaw: float) -> float:
        return wrap_to_pi(joint_yaw + self.base_yaw_world_offset)

    def _world_yaw_to_joint_yaw(self, world_yaw: float) -> float:
        return wrap_to_pi(world_yaw - self.base_yaw_world_offset)

    def _select_command(self, lidar_ranges: list[float]) -> RobotCommand:
        now = self.get_clock().now()
        manual_age = (now - self.last_manual_cmd_time).nanoseconds * 1e-9
        if manual_age <= self.cmd_vel_timeout_sec:
            return self.latest_manual_cmd
        auto_age = (now - self.last_auto_cmd_time).nanoseconds * 1e-9
        if auto_age <= self.cmd_vel_timeout_sec:
            return self.latest_auto_cmd
        if not self.auto_mode:
            return RobotCommand()

        x, y, yaw, _ = self._read_pose()
        return self.navigator.compute_command(
            pose_x=x,
            pose_y=y,
            yaw=yaw,
            lidar_ranges=lidar_ranges,
            lidar_angles=self.combined_lidar_angles,
        )

    def _estimate_sector_clearance(self, lidar_ranges: list[float], angle_min_deg: float, angle_max_deg: float) -> float:
        readings = [
            float(distance)
            for angle, distance in zip(self.combined_lidar_angles, lidar_ranges, strict=False)
            if math.isfinite(distance) and math.radians(angle_min_deg) <= angle <= math.radians(angle_max_deg)
        ]
        if not readings:
            return float("inf")
        return min(readings)

    def _guided_main_dock_exit_command(
        self,
        pose_x: float,
        pose_y: float,
        yaw: float,
        goal_x: float,
        goal_y: float,
    ) -> RobotCommand:
        dx = goal_x - pose_x
        dy = goal_y - pose_y
        distance = math.hypot(dx, dy)
        if distance <= 0.12:
            return RobotCommand()
        heading_error = wrap_to_pi(math.atan2(dy, dx) - yaw)
        linear = min(0.36, 0.92 * distance)
        angular = clamp(2.4 * heading_error, -1.25, 1.25)
        if abs(heading_error) > 0.75:
            linear *= 0.20
        elif abs(heading_error) > 0.35:
            linear *= 0.55
        return RobotCommand(linear=linear, angular=angular)

    def _guided_main_service_waypoint_command(
        self,
        pose_x: float,
        pose_y: float,
        yaw: float,
        goal_x: float,
        goal_y: float,
        *,
        max_linear: float,
    ) -> RobotCommand:
        dx = goal_x - pose_x
        dy = goal_y - pose_y
        distance = math.hypot(dx, dy)
        if distance <= 0.16:
            return RobotCommand()
        heading_error = wrap_to_pi(math.atan2(dy, dx) - yaw)
        linear = min(max_linear, 0.82 * distance)
        angular = clamp(1.45 * heading_error, -0.80, 0.80)
        if distance < 1.15 and abs(heading_error) > 0.95:
            linear = 0.0
            angular = clamp(1.85 * heading_error, -0.95, 0.95)
            return RobotCommand(linear=linear, angular=angular)
        if distance < 0.70 and abs(heading_error) > 0.55:
            linear *= 0.12
            angular = clamp(1.75 * heading_error, -0.90, 0.90)
            return RobotCommand(linear=linear, angular=angular)
        if abs(heading_error) > 1.00:
            linear *= 0.12
        elif abs(heading_error) > 0.60:
            linear *= 0.55
        elif abs(heading_error) > 0.28:
            linear *= 0.86
        elif distance < 0.20:
            linear *= 0.82
        return RobotCommand(linear=linear, angular=angular)

    def _guided_main_dock_return_command(
        self,
        pose_x: float,
        pose_y: float,
        yaw: float,
        goal_x: float,
        goal_y: float,
    ) -> RobotCommand:
        dx = goal_x - pose_x
        dy = goal_y - pose_y
        distance = math.hypot(dx, dy)
        if distance <= 0.18:
            return RobotCommand()
        heading_error = wrap_to_pi(math.atan2(dy, dx) - yaw)
        angular = clamp(1.75 * heading_error, -1.05, 1.05)
        linear = min(0.72, 0.92 * distance)
        if abs(heading_error) > 1.70:
            linear = 0.0
        elif abs(heading_error) > 1.20:
            linear = min(linear, 0.08)
        elif abs(heading_error) > 0.85:
            linear *= 0.22
        elif abs(heading_error) > 0.55:
            linear *= 0.34
        elif abs(heading_error) > 0.30:
            linear *= 0.70
        return RobotCommand(linear=linear, angular=angular)

    def _guided_main_precise_service_command(
        self,
        pose_x: float,
        pose_y: float,
        yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
    ) -> RobotCommand:
        dx = goal_x - pose_x
        dy = goal_y - pose_y
        distance = math.hypot(dx, dy)
        if distance <= 0.08 and abs(wrap_to_pi(goal_yaw - yaw)) <= 0.12:
            return RobotCommand()

        desired_yaw = math.atan2(dy, dx) if distance > 0.30 else goal_yaw
        heading_error = wrap_to_pi(desired_yaw - yaw)
        final_yaw_error = wrap_to_pi(goal_yaw - yaw)

        linear = min(0.60 if distance > 2.0 else 0.52 if distance > 1.2 else 0.34, 1.08 * distance)
        angular = clamp(1.95 * heading_error + 0.20 * final_yaw_error, -1.12, 1.12)
        if distance < 0.35:
            angular = clamp(2.45 * final_yaw_error, -1.14, 1.14)

        if abs(heading_error) > 0.85:
            linear *= 0.18
        elif abs(heading_error) > 0.45:
            linear *= 0.52

        if distance < 0.18 and abs(final_yaw_error) > 0.16:
            linear = 0.0
        elif distance < 0.24:
            linear = min(linear, 0.14)

        return RobotCommand(linear=linear, angular=angular)

    def _apply_main_robot_kinematics(
        self,
        sim_dt: float,
        pose_x: float,
        pose_y: float,
        yaw: float,
        left_wheel_speed: float,
        right_wheel_speed: float,
    ) -> None:
        linear, angular = diff_drive_forward(
            left_wheel_speed,
            right_wheel_speed,
            WHEEL_RADIUS,
            WHEEL_TRACK,
        )
        # Keep AMR-A body propagation visually consistent with AMR-B:
        # rotate first, then advance along the updated heading.
        next_yaw = wrap_to_pi(yaw + angular * sim_dt)
        next_x = pose_x + linear * math.cos(next_yaw) * sim_dt
        next_y = pose_y + linear * math.sin(next_yaw) * sim_dt
        world_vx = linear * math.cos(next_yaw)
        world_vy = linear * math.sin(next_yaw)
        self.data.qpos[self.base_x_qpos_adr] = float(next_x - ROBOT_START_X)
        self.data.qpos[self.base_y_qpos_adr] = float(next_y - ROBOT_START_Y)
        self.data.qpos[self.base_yaw_qpos_adr] = self._world_yaw_to_joint_yaw(float(next_yaw))
        self.data.qvel[self.base_x_qvel_adr] = float(world_vx)
        self.data.qvel[self.base_y_qvel_adr] = float(world_vy)
        self.data.qvel[self.base_yaw_qvel_adr] = float(angular)
        self.data.ctrl[self.base_x_actuator_id] = 0.0
        self.data.ctrl[self.base_y_actuator_id] = 0.0
        self.data.ctrl[self.base_yaw_actuator_id] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _position_blocked(
        self,
        x: float,
        y: float,
        clearance: float = 0.46,
        allow_main_dock: bool = False,
        allow_service_dock: bool = False,
        avoid_main_robot: bool = False,
        avoid_service_robot: bool = False,
    ) -> bool:
        boundary_margin = 0.26
        if abs(x) >= WORLD_X - boundary_margin or abs(y) >= WORLD_Y - boundary_margin:
            return True

        for box in HOUSE_BOXES:
            if box.size[2] < 0.08:
                continue
            if box.name == "dock_floor_pad":
                continue
            if allow_main_dock and box.name.startswith("dock_"):
                continue
            if allow_service_dock and box.name.startswith("service_"):
                continue
            if abs(x - box.pos[0]) <= box.size[0] + clearance and abs(y - box.pos[1]) <= box.size[1] + clearance:
                return True
        if self._robot_clearance_blocked(
            x,
            y,
            against_main_robot=avoid_main_robot,
            against_service_robot=avoid_service_robot,
            clearance=clearance,
        ):
            return True
        return False

    def _resolve_blocked_motion(
        self,
        pose_x: float,
        pose_y: float,
        yaw: float,
        linear: float,
        angular: float,
        lidar_ranges: list[float],
        *,
        allow_main_dock: bool = False,
    ) -> tuple[float, float]:
        if linear <= 0.0:
            return linear, angular

        dock_return_active = self.current_mission_mode == "dock_return"
        lookahead_time = 0.85 if dock_return_active else 0.50
        collision_clearance = 0.44 if dock_return_active else 0.32
        predicted_x = pose_x + math.cos(yaw) * linear * lookahead_time
        predicted_y = pose_y + math.sin(yaw) * linear * lookahead_time

        if dock_return_active:
            front_clearance = self._estimate_sector_clearance(lidar_ranges, -28.0, 28.0)
            if front_clearance < 0.68:
                left_clearance = self._estimate_sector_clearance(lidar_ranges, 20.0, 115.0)
                right_clearance = self._estimate_sector_clearance(lidar_ranges, -115.0, -20.0)
                requested_turn = 1.0 if angular >= 0.0 else -1.0
                safest_turn = 1.0 if left_clearance >= right_clearance else -1.0
                requested_clearance = left_clearance if requested_turn > 0.0 else right_clearance
                opposite_clearance = right_clearance if requested_turn > 0.0 else left_clearance
                if requested_clearance + 0.18 >= opposite_clearance:
                    turn_direction = requested_turn
                else:
                    turn_direction = safest_turn
                if front_clearance < 0.34:
                    return 0.0, turn_direction * max(0.95, abs(angular))
                return min(linear, 0.055), turn_direction * max(0.72, abs(angular))

        if self._robot_clearance_blocked(
            predicted_x,
            predicted_y,
            against_service_robot=True,
            clearance=0.46,
        ):
            turn_direction = self._turn_away_direction(
                pose_x,
                pose_y,
                yaw,
                float(self.service_bot_pose[0]),
                float(self.service_bot_pose[1]),
            )
            return 0.0, turn_direction * max(1.0, abs(angular))

        mission_key = normalize_service_target_key(self.current_mission_mode)
        if mission_key in SERVICE_TARGETS and self.latest_autonomy_state == "TABLE_APPROACH":
            goal_x, goal_y, _goal_yaw = service_target_goal(mission_key, "a")
            if math.hypot(goal_x - pose_x, goal_y - pose_y) <= 0.64:
                front_clearance = self._estimate_sector_clearance(lidar_ranges, -18.0, 18.0)
                if front_clearance > 0.18:
                    return linear, angular

        if not self._position_blocked(
            predicted_x,
            predicted_y,
            clearance=collision_clearance,
            allow_main_dock=allow_main_dock,
            avoid_service_robot=True,
        ):
            return linear, angular

        left_clearance = self._estimate_sector_clearance(lidar_ranges, 20.0, 115.0)
        right_clearance = self._estimate_sector_clearance(lidar_ranges, -115.0, -20.0)
        safest_turn = 1.0 if left_clearance >= right_clearance else -1.0
        if dock_return_active and abs(angular) > 0.12:
            requested_turn = 1.0 if angular >= 0.0 else -1.0
            requested_clearance = left_clearance if requested_turn > 0.0 else right_clearance
            opposite_clearance = right_clearance if requested_turn > 0.0 else left_clearance
            turn_direction = requested_turn if requested_clearance + 0.18 >= opposite_clearance else safest_turn
        else:
            turn_direction = safest_turn
        return 0.0, turn_direction * max(0.95, abs(angular))

    def _main_robot_departing_dock(self, pose_x: float, pose_y: float) -> bool:
        if self.current_mission_mode in {"idle", "dock_return"}:
            return False
        dock_x, dock_y, _dock_yaw = self.dock.dock_pose
        near_main_dock = math.hypot(dock_x - pose_x, dock_y - pose_y) <= 1.35
        return near_main_dock

    def _main_robot_ready_for_dock_capture(self, pose_x: float, pose_y: float, yaw: float) -> bool:
        if self.current_mission_mode != "dock_return":
            return False
        dock_x, dock_y, dock_yaw = self.dock.dock_pose
        distance = math.hypot(dock_x - pose_x, dock_y - pose_y)
        dx = dock_x - pose_x
        dy = dock_y - pose_y
        local_x = math.cos(dock_yaw) * dx + math.sin(dock_yaw) * dy
        local_y = -math.sin(dock_yaw) * dx + math.cos(dock_yaw) * dy
        yaw_error = abs(wrap_to_pi(dock_yaw - yaw))
        if distance <= 0.18 and yaw_error <= 0.28:
            return True
        final_docking_active = self.latest_autonomy_state == "FINAL_DOCKING"
        if final_docking_active and distance <= 0.54 and -0.30 <= local_x <= 0.56 and abs(local_y) <= 0.34 and yaw_error <= 0.55:
            return True
        return final_docking_active and -0.15 <= local_x <= 0.34 and abs(local_y) <= 0.20 and yaw_error <= 0.32

    def _snap_main_robot_to_dock_pose(self) -> None:
        dock_x, dock_y, dock_yaw = self.dock.dock_pose
        self._snap_main_robot_to_pose(dock_x, dock_y, dock_yaw)
        self.dock_in_contact = True
        self.is_charging = True

    def _snap_main_robot_to_pose(self, goal_x: float, goal_y: float, goal_yaw: float) -> None:
        self.data.qpos[self.base_x_qpos_adr] = float(goal_x - ROBOT_START_X)
        self.data.qpos[self.base_y_qpos_adr] = float(goal_y - ROBOT_START_Y)
        self.data.qpos[self.base_yaw_qpos_adr] = self._world_yaw_to_joint_yaw(float(goal_yaw))
        self.data.qvel[self.base_x_qvel_adr] = 0.0
        self.data.qvel[self.base_y_qvel_adr] = 0.0
        self.data.qvel[self.base_yaw_qvel_adr] = 0.0
        self.data.ctrl[self.base_x_actuator_id] = 0.0
        self.data.ctrl[self.base_y_actuator_id] = 0.0
        self.data.ctrl[self.base_yaw_actuator_id] = 0.0
        self.applied_cmd = RobotCommand()
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.target_left_wheel_speed = 0.0
        self.target_right_wheel_speed = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _align_main_robot_for_undock(self) -> None:
        dock_yaw = float(self.dock.dock_pose[2])
        self.data.qpos[self.base_yaw_qpos_adr] = self._world_yaw_to_joint_yaw(dock_yaw)
        self.data.qvel[self.base_yaw_qvel_adr] = 0.0
        self.data.ctrl[self.base_yaw_actuator_id] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _enforce_precise_service_stop(self) -> bool:
        mission_key = normalize_service_target_key(self.current_mission_mode)
        if mission_key not in SERVICE_TARGETS:
            return False
        pose_x, pose_y, yaw, _ = self._read_pose()
        goal_x, goal_y, goal_yaw = service_target_goal(mission_key, "a")
        distance = math.hypot(goal_x - pose_x, goal_y - pose_y)
        yaw_error = abs(wrap_to_pi(goal_yaw - yaw))
        table_approach_active = self.latest_autonomy_state == "TABLE_APPROACH"
        snap_distance = 0.30 if table_approach_active else 0.14
        snap_yaw_error = 0.38 if table_approach_active else 0.24
        max_linear = 0.22 if table_approach_active else 0.26
        max_angular = 1.15 if table_approach_active else 0.90
        if distance > snap_distance or yaw_error > snap_yaw_error:
            return False
        if abs(self.applied_cmd.linear) > max_linear or abs(self.applied_cmd.angular) > max_angular:
            return False
        self._snap_main_robot_to_pose(goal_x, goal_y, goal_yaw)
        return True

    def _slew_limit(self, current: float, target: float, step_limit: float, decel_limit: float | None = None) -> float:
        active_limit = step_limit
        if decel_limit is not None and (current * target < 0.0 or abs(target) < abs(current)):
            active_limit = decel_limit
        if target > current + active_limit:
            return current + active_limit
        if target < current - active_limit:
            return current - active_limit
        return target

    def _apply_main_robot_wheel_motion(
        self,
        sim_dt: float,
        pose_x: float,
        pose_y: float,
        yaw: float,
        desired_linear: float,
        desired_angular: float,
    ) -> None:
        target_left, target_right = diff_drive_inverse(
            linear_velocity=desired_linear,
            angular_velocity=desired_angular,
            wheel_radius=WHEEL_RADIUS,
            wheel_track=WHEEL_TRACK,
        )
        wheel_accel_limit = 24.0 * sim_dt
        wheel_decel_limit = 40.0 * sim_dt
        self.target_left_wheel_speed = target_left
        self.target_right_wheel_speed = target_right
        self.left_wheel_speed = self._slew_limit(
            self.left_wheel_speed,
            self.target_left_wheel_speed,
            wheel_accel_limit,
            wheel_decel_limit,
        )
        self.right_wheel_speed = self._slew_limit(
            self.right_wheel_speed,
            self.target_right_wheel_speed,
            wheel_accel_limit,
            wheel_decel_limit,
        )
        actual_linear, actual_angular = diff_drive_forward(
            self.left_wheel_speed,
            self.right_wheel_speed,
            WHEEL_RADIUS,
            WHEEL_TRACK,
        )
        self.applied_cmd = RobotCommand(linear=actual_linear, angular=actual_angular)
        self._apply_main_robot_kinematics(
            sim_dt,
            pose_x,
            pose_y,
            yaw,
            self.left_wheel_speed,
            self.right_wheel_speed,
        )

    def _apply_control(self, sim_dt: float, lidar_ranges: list[float]) -> None:
        mission_key = normalize_service_target_key(self.current_mission_mode)
        desired_cmd = self._select_command(lidar_ranges)
        if mission_key in SERVICE_TARGETS and self.latest_autonomy_state == "PATROL":
            current_goal = self.latest_mission_status.get("current_goal") if isinstance(self.latest_mission_status, dict) else None
            if isinstance(current_goal, list) and len(current_goal) >= 2:
                pose_x0, pose_y0, yaw0, _ = self._read_pose()
                desired_cmd = self._guided_main_service_waypoint_command(
                    pose_x0,
                    pose_y0,
                    yaw0,
                    float(current_goal[0]),
                    float(current_goal[1]),
                    max_linear=0.44 if self._main_robot_departing_dock(pose_x0, pose_y0) else 0.72,
                )
        elif mission_key in SERVICE_TARGETS and self.latest_autonomy_state == "TABLE_APPROACH":
            pose_x0, pose_y0, yaw0, _ = self._read_pose()
            goal_x, goal_y, goal_yaw = service_target_goal(mission_key, "a")
            desired_cmd = self._guided_main_precise_service_command(
                pose_x0,
                pose_y0,
                yaw0,
                goal_x,
                goal_y,
                goal_yaw,
            )
        elif self.current_mission_mode == "dock_return":
            if desired_cmd.linear < -0.02 and self.latest_autonomy_state == "SEEK_DOCK":
                pass
            else:
                pose_x0, pose_y0, yaw0, _ = self._read_pose()
                dock_x, dock_y, dock_yaw = self.dock.dock_pose
                pre_dock_x, pre_dock_y, pre_dock_yaw = self.dock.pre_dock_pose
                if self.latest_autonomy_state != "FINAL_DOCKING":
                    current_goal = self.latest_mission_status.get("current_goal") if isinstance(self.latest_mission_status, dict) else None
                    if isinstance(current_goal, list) and len(current_goal) >= 2:
                        goal_x = float(current_goal[0])
                        goal_y = float(current_goal[1])
                        close_to_pre_dock = math.hypot(goal_x - float(pre_dock_x), goal_y - float(pre_dock_y)) <= 0.18
                        if close_to_pre_dock:
                            desired_cmd = self._guided_main_precise_service_command(
                                pose_x0,
                                pose_y0,
                                yaw0,
                                float(pre_dock_x),
                                float(pre_dock_y),
                                float(pre_dock_yaw),
                            )
                        else:
                            desired_cmd = self._guided_main_dock_return_command(
                                pose_x0,
                                pose_y0,
                                yaw0,
                                goal_x,
                                goal_y,
                            )
                    else:
                        desired_cmd = self._guided_main_dock_return_command(
                            pose_x0,
                            pose_y0,
                            yaw0,
                            float(dock_x),
                            float(dock_y),
                        )
        if abs(desired_cmd.linear) < 1e-4 and abs(desired_cmd.angular) < 1e-4:
            linear = 0.0
            angular = 0.0
        else:
            linear = self._slew_limit(self.applied_cmd.linear, desired_cmd.linear, 1.8 * sim_dt, 10.0 * sim_dt)
            angular = self._slew_limit(self.applied_cmd.angular, desired_cmd.angular, 3.1 * sim_dt, 12.0 * sim_dt)
        pose_x, pose_y, yaw, _ = self._read_pose()
        if self.current_mission_mode == "idle" and self.latest_autonomy_state in {"WAITING_FOR_ODOM", "WAITING_TASK", "CHARGING"}:
            self._snap_main_robot_to_dock_pose()
            return
        if self.latest_autonomy_state == "UNDOCKING" and self._main_robot_departing_dock(pose_x, pose_y):
            pre_dock_x, pre_dock_y, pre_dock_yaw = self.dock.pre_dock_pose
            self.dock_in_contact = False
            self.is_charging = False
            self._snap_main_robot_to_pose(pre_dock_x, pre_dock_y, pre_dock_yaw)
            return
        if self._main_robot_ready_for_dock_capture(pose_x, pose_y, yaw):
            self._snap_main_robot_to_dock_pose()
            return
        if (
            normalize_service_target_key(self.current_mission_mode) in SERVICE_TARGETS
            and self.latest_autonomy_state == "PATROL"
            and self._main_robot_departing_dock(pose_x, pose_y)
        ):
            current_goal = self.latest_mission_status.get("current_goal") if isinstance(self.latest_mission_status, dict) else None
            if isinstance(current_goal, list) and len(current_goal) >= 2:
                guided_cmd = self._guided_main_dock_exit_command(
                    pose_x,
                    pose_y,
                    yaw,
                    float(current_goal[0]),
                    float(current_goal[1]),
                )
                linear, angular = self._resolve_blocked_motion(
                    pose_x,
                    pose_y,
                    yaw,
                    guided_cmd.linear,
                    guided_cmd.angular,
                    lidar_ranges,
                    allow_main_dock=True,
                )
                self._apply_main_robot_wheel_motion(sim_dt, pose_x, pose_y, yaw, linear, angular)
                return
        allow_main_dock = self.current_mission_mode == "dock_return" or self.latest_autonomy_state in {
            "SEEK_DOCK",
            "LOW_BATTERY",
            "DOCKING",
        } or self._main_robot_departing_dock(pose_x, pose_y)
        linear, angular = self._resolve_blocked_motion(
            pose_x,
            pose_y,
            yaw,
            linear,
            angular,
            lidar_ranges,
            allow_main_dock=allow_main_dock,
        )
        self._apply_main_robot_wheel_motion(sim_dt, pose_x, pose_y, yaw, linear, angular)
        if self._enforce_precise_service_stop():
            return

    def _update_visual_wheels(self, sim_dt: float) -> None:
        left_angle = float(self.data.qpos[self.left_qpos_adr]) + self.left_wheel_speed * sim_dt
        right_angle = float(self.data.qpos[self.right_qpos_adr]) + self.right_wheel_speed * sim_dt
        self.data.qpos[self.left_qpos_adr] = math.atan2(math.sin(left_angle), math.cos(left_angle))
        self.data.qpos[self.right_qpos_adr] = math.atan2(math.sin(right_angle), math.cos(right_angle))
        self.data.qvel[self.left_qvel_adr] = self.left_wheel_speed
        self.data.qvel[self.right_qvel_adr] = self.right_wheel_speed
        mujoco.mj_forward(self.model, self.data)

    def _update_battery_and_dock(self, sim_dt: float) -> None:
        x, y, yaw, _ = self._read_pose()
        dock_x, dock_y, dock_yaw = self.dock.dock_pose
        distance = math.hypot(dock_x - x, dock_y - y)
        yaw_error = abs(wrap_to_pi(dock_yaw - yaw))
        robot_is_still = abs(self.applied_cmd.linear) < 0.05 and abs(self.applied_cmd.angular) < 0.12
        waiting_for_task = self.current_mission_mode == "idle" or self.latest_autonomy_state == "WAITING_TASK"
        relaxed_dock_alignment = waiting_for_task
        charging_allowed = waiting_for_task or self.current_mission_mode == "dock_return"
        precise_contact = distance <= self.dock.contact_distance and (
            yaw_error <= max(self.dock.yaw_tolerance_rad, 0.12) or relaxed_dock_alignment
        )
        dock_zone_contact = distance <= max(self.dock.charge_distance, 0.28) and (
            yaw_error <= max(self.dock.yaw_tolerance_rad, 0.22) or relaxed_dock_alignment
        )
        if self.current_mission_mode == "dock_return" and distance <= 0.18 and yaw_error <= 0.28:
            self._snap_main_robot_to_dock_pose()
            x, y, yaw, _ = self._read_pose()
            distance = math.hypot(dock_x - x, dock_y - y)
            yaw_error = abs(wrap_to_pi(dock_yaw - yaw))
            precise_contact = True
            dock_zone_contact = True
            robot_is_still = True
        elif self.current_mission_mode == "dock_return" and dock_zone_contact and robot_is_still:
            self._snap_main_robot_to_dock_pose()
            x, y, yaw, _ = self._read_pose()
            distance = math.hypot(dock_x - x, dock_y - y)
            yaw_error = abs(wrap_to_pi(dock_yaw - yaw))
            precise_contact = True
            dock_zone_contact = True
            robot_is_still = True
        has_effective_drive_request = self.latest_cmd_source not in {"idle", "auto"} or not robot_is_still
        is_idle_at_home = waiting_for_task and not has_effective_drive_request
        if is_idle_at_home and dock_zone_contact and robot_is_still:
            self._snap_main_robot_to_dock_pose()
            x, y, yaw, _ = self._read_pose()
            distance = math.hypot(dock_x - x, dock_y - y)
            yaw_error = abs(wrap_to_pi(dock_yaw - yaw))
            precise_contact = distance <= self.dock.contact_distance
            dock_zone_contact = distance <= max(self.dock.charge_distance, 0.34)
            robot_is_still = True

        self.dock_in_contact = precise_contact or (dock_zone_contact and robot_is_still)
        self.is_charging = (
            charging_allowed
            and (
                precise_contact
                or (dock_zone_contact and is_idle_at_home and robot_is_still)
            )
        )

        if self.is_charging:
            self.battery_pct = min(1.0, self.battery_pct + self.battery_charge_rate * sim_dt)
        else:
            has_active_user_request = not waiting_for_task or has_effective_drive_request
            robot_is_moving = abs(self.applied_cmd.linear) > 0.01 or abs(self.applied_cmd.angular) > 0.05
            if not has_active_user_request and not robot_is_moving:
                self._record_battery_history()
                return
            motion_factor = abs(self.applied_cmd.linear) / 0.7 + 0.5 * abs(self.applied_cmd.angular) / 1.5
            self.battery_pct = max(
                0.0,
                self.battery_pct - (self.battery_idle_draw + self.battery_motion_draw * motion_factor) * sim_dt,
            )
        self._record_battery_history()

    def _publish_scan_msg(
        self,
        stamp,
        frame_id: str,
        angles: list[float],
        ranges: list[float],
        publisher,
    ) -> None:
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.angle_min = angles[0]
        msg.angle_max = angles[-1]
        msg.angle_increment = ((angles[-1] - angles[0]) / (len(angles) - 1)) if len(angles) > 1 else 0.0
        msg.scan_time = 1.0 / self.publish_rate_hz
        msg.range_min = 0.03
        msg.range_max = LIDAR_MAX_RANGE
        msg.ranges = ranges
        publisher.publish(msg)

    def _publish_imu(self, stamp, quat_wxyz: tuple[float, float, float, float]) -> None:
        accel_start = self.model.sensor_adr[self.imu_accel_sensor]
        gyro_start = self.model.sensor_adr[self.imu_gyro_sensor]
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = "imu_link"
        quat_xyzw = quat_wxyz_to_ros_xyzw(quat_wxyz)
        msg.orientation.x = quat_xyzw[0]
        msg.orientation.y = quat_xyzw[1]
        msg.orientation.z = quat_xyzw[2]
        msg.orientation.w = quat_xyzw[3]
        msg.angular_velocity.x = float(self.data.sensordata[gyro_start + 0]) + self._sample_noise(self.imu_gyro_noise_stddev)
        msg.angular_velocity.y = float(self.data.sensordata[gyro_start + 1]) + self._sample_noise(self.imu_gyro_noise_stddev)
        msg.angular_velocity.z = float(self.data.sensordata[gyro_start + 2]) + self._sample_noise(self.imu_gyro_noise_stddev)
        msg.linear_acceleration.x = float(self.data.sensordata[accel_start + 0]) + self._sample_noise(self.imu_accel_noise_stddev)
        msg.linear_acceleration.y = float(self.data.sensordata[accel_start + 1]) + self._sample_noise(self.imu_accel_noise_stddev)
        msg.linear_acceleration.z = float(self.data.sensordata[accel_start + 2]) + self._sample_noise(self.imu_accel_noise_stddev)
        msg.orientation_covariance = [0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.04]
        msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.02]
        msg.linear_acceleration_covariance = [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.08]
        self.imu_pub.publish(msg)

    def _publish_joint_states(self, stamp) -> None:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = ["left_wheel_joint", "right_wheel_joint"] + self.arm_joint_names
        msg.position = [
            float(self.data.qpos[self.left_qpos_adr]),
            float(self.data.qpos[self.right_qpos_adr]),
        ] + [float(self.data.qpos[qpos_adr]) for qpos_adr in self.arm_qpos_adrs]
        msg.velocity = [
            float(self.data.qvel[self.left_qvel_adr]),
            float(self.data.qvel[self.right_qvel_adr]),
        ] + [float(self.data.qvel[qvel_adr]) for qvel_adr in self.arm_qvel_adrs]
        self.joint_state_pub.publish(msg)

    def _make_odom(
        self,
        stamp,
        frame_id: str,
        child_frame_id: str,
        x: float,
        y: float,
        quat_xyzw: tuple[float, float, float, float],
        vx_body: float,
        vy_body: float,
        wz: float,
    ) -> Odometry:
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = frame_id
        odom.child_frame_id = child_frame_id
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.x = quat_xyzw[0]
        odom.pose.pose.orientation.y = quat_xyzw[1]
        odom.pose.pose.orientation.z = quat_xyzw[2]
        odom.pose.pose.orientation.w = quat_xyzw[3]
        odom.twist.twist.linear.x = vx_body
        odom.twist.twist.linear.y = vy_body
        odom.twist.twist.angular.z = wz
        odom.pose.covariance = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
        odom.twist.covariance = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08]
        return odom

    def _publish_odom_and_tf(self, stamp, x: float, y: float, yaw: float, quat_wxyz: tuple[float, float, float, float]) -> None:
        current_left_angle = float(self.data.qpos[self.left_qpos_adr])
        current_right_angle = float(self.data.qpos[self.right_qpos_adr])
        if self.last_wheel_angles_for_odom is None or self.last_publish_sim_time is None:
            vx_body = 0.0
            vy_body = 0.0
            wz = 0.0
        else:
            dt = max(self.data.time - self.last_publish_sim_time, 1e-6)
            prev_left_angle, prev_right_angle = self.last_wheel_angles_for_odom
            delta_left = wrap_to_pi(current_left_angle - prev_left_angle) * WHEEL_RADIUS
            delta_right = wrap_to_pi(current_right_angle - prev_right_angle) * WHEEL_RADIUS
            delta_s = 0.5 * (delta_left + delta_right)
            delta_yaw = (delta_right - delta_left) / WHEEL_TRACK
            vx_body = delta_s / dt
            vy_body = 0.0
            wz = delta_yaw / dt

        quat_xyzw = quat_wxyz_to_ros_xyzw(quat_wxyz)
        raw_yaw = wrap_to_pi(yaw + self._sample_noise(self.odom_yaw_noise_stddev))
        raw_quat_xyzw = euler_to_quat_xyzw(0.0, 0.0, raw_yaw)
        raw_odom = self._make_odom(
            stamp,
            "odom",
            "base_link",
            x + self._sample_noise(self.odom_xy_noise_stddev),
            y + self._sample_noise(self.odom_xy_noise_stddev),
            raw_quat_xyzw,
            vx_body + self._sample_noise(self.odom_linear_velocity_noise_stddev),
            vy_body + self._sample_noise(self.odom_linear_velocity_noise_stddev),
            wz + self._sample_noise(self.odom_angular_velocity_noise_stddev),
        )
        gt_odom = self._make_odom(stamp, "odom", "base_link", x, y, quat_xyzw, vx_body, vy_body, wz)
        self.odom_pub.publish(raw_odom)
        self.gt_odom_pub.publish(gt_odom)

        if self.publish_odom_tf:
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "odom"
            transform.child_frame_id = "base_link"
            transform.transform.translation.x = raw_odom.pose.pose.position.x
            transform.transform.translation.y = raw_odom.pose.pose.position.y
            transform.transform.rotation.x = raw_quat_xyzw[0]
            transform.transform.rotation.y = raw_quat_xyzw[1]
            transform.transform.rotation.z = raw_quat_xyzw[2]
            transform.transform.rotation.w = raw_quat_xyzw[3]
            self.tf_broadcaster.sendTransform(transform)

        self.last_pose_for_odom = (x, y, yaw)
        self.last_wheel_angles_for_odom = (current_left_angle, current_right_angle)
        self.last_publish_sim_time = self.data.time

    def _publish_battery_and_dock(self, stamp) -> None:
        msg = BatteryState()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        reported_pct = clamp(self.battery_pct + self._sample_noise(self.battery_noise_stddev), 0.0, 1.0)
        msg.voltage = 24.0 * (0.80 + 0.20 * reported_pct)
        msg.current = (6.5 if self.is_charging else -1.8) + self._sample_noise(0.05)
        msg.percentage = reported_pct
        msg.power_supply_status = (
            BatteryState.POWER_SUPPLY_STATUS_CHARGING if self.is_charging else BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        )
        msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
        self.battery_pub.publish(msg)
        self.dock_contact_pub.publish(Bool(data=self.dock_in_contact))
        self.dock_charging_pub.publish(Bool(data=self.is_charging))
        dock_state = "CHARGING" if self.is_charging else ("CONTACT" if self.dock_in_contact else "FREE")
        self.dock_state_pub.publish(String(data=dock_state))

    def _publish_runtime_status(self, x: float, y: float, yaw: float) -> None:
        status = {
            "sim_time_sec": round(float(self.data.time), 3),
            "battery_pct": round(float(self.battery_pct), 4),
            "dock_contact": self.dock_in_contact,
            "is_charging": self.is_charging,
            "autonomy_state": self.latest_autonomy_state,
            "mission_mode": self.current_mission_mode,
            "cmd_source": self.latest_cmd_source,
            "speed_limit_scale": round(float(self.speed_limit_scale), 3),
            "emergency_stop_active": self.emergency_stop_active,
            "dynamic_obstacle_count": len(self.dynamic_obstacles),
            "pose": {
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw": round(yaw, 3),
            },
            "joint_state_main": {
                "base_x_qpos": round(float(self.data.qpos[self.base_x_qpos_adr]), 4),
                "base_y_qpos": round(float(self.data.qpos[self.base_y_qpos_adr]), 4),
                "base_yaw_qpos": round(float(self.data.qpos[self.base_yaw_qpos_adr]), 4),
                "base_x_qvel": round(float(self.data.qvel[self.base_x_qvel_adr]), 4),
                "base_y_qvel": round(float(self.data.qvel[self.base_y_qvel_adr]), 4),
                "base_yaw_qvel": round(float(self.data.qvel[self.base_yaw_qvel_adr]), 4),
            },
            "command": {
                "linear": round(float(self.applied_cmd.linear), 3),
                "angular": round(float(self.applied_cmd.angular), 3),
            },
            "manipulator": {
                "state": self.arm_state,
                "payload_attached": self.payload_attached,
                "payload_location": self.payload_location,
            },
            "scheduler": {
                "active_task_a": None if self.active_task_a is None else self.active_task_a.task_id,
                "active_task_b": None if self.active_task_b is None else self.active_task_b.task_id,
                "queue_a": [task.task_id for task in self.task_queue_a],
                "queue_b": [task.task_id for task in self.task_queue_b],
            },
                "service_amr": {
                    "mission_mode": self.service_bot_mission_mode,
                    "battery_pct": round(float(self.service_bot_battery_pct), 4),
                    "is_charging": self.service_bot_is_charging,
                    "state": self.service_bot_state,
                    "payload_state": self.service_bot_payload_state,
                    "goal_index": int(self.service_bot_goal_index),
                    "goal_count": len(self.service_bot_active_path),
                    "speed_scale": round(float(self.service_bot_speed_scale), 3),
                    "current_goal": None
                    if not self.service_bot_active_path
                    else [
                        round(float(self.service_bot_active_path[self.service_bot_goal_index % len(self.service_bot_active_path)][0]), 3),
                        round(float(self.service_bot_active_path[self.service_bot_goal_index % len(self.service_bot_active_path)][1]), 3),
                    ],
                    "pose": {
                        "x": round(float(self.service_bot_pose[0]), 3),
                        "y": round(float(self.service_bot_pose[1]), 3),
                    "yaw": round(float(self.service_bot_yaw), 3),
                },
            },
        }
        self.status_pub.publish(String(data=json.dumps(status, separators=(",", ":"))))

    def _build_camera_info(self, stamp, frame_id: str) -> CameraInfo:
        fovy = math.radians(float(self.model.cam_fovy[self.front_cam_id]))
        fy = 0.5 * self.camera_height / math.tan(0.5 * fovy)
        fx = fy
        cx = (self.camera_width - 1) * 0.5
        cy = (self.camera_height - 1) * 0.5
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.width = self.camera_width
        msg.height = self.camera_height
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return msg

    def _make_image_msg(self, stamp, frame_id: str, encoding: str, array: np.ndarray) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(array.shape[0])
        msg.width = int(array.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = int(array.strides[0])
        msg.data = array.tobytes()
        return msg

    def _make_pointcloud_msg(self, stamp, frame_id: str, depth: np.ndarray) -> PointCloud2:
        sample = depth[::4, ::4]
        fovy = math.radians(float(self.model.cam_fovy[self.front_cam_id]))
        fy = 0.5 * self.camera_height / math.tan(0.5 * fovy)
        fx = fy
        cx = (self.camera_width - 1) * 0.5
        cy = (self.camera_height - 1) * 0.5

        rows, cols = np.indices(sample.shape, dtype=np.float32)
        rows *= 4.0
        cols *= 4.0
        z = sample.astype(np.float32)
        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        points = points.astype(np.float32)

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = points.tobytes()
        return msg

    def _publish_camera(self, stamp) -> None:
        if not self.publish_depth_camera or self.rgb_renderer is None or self.depth_renderer is None:
            return
        if self.data.time - self.last_render_sim_time < 1.0 / max(self.render_rate_hz, 1e-3):
            return

        self.rgb_renderer.update_scene(self.data, camera="front_cam")
        self.depth_renderer.update_scene(self.data, camera="front_cam")
        rgb = self.rgb_renderer.render()
        depth = self.depth_renderer.render().astype(np.float32)

        rgb_msg = self._make_image_msg(stamp, "camera_rgb_optical_frame", "rgb8", rgb)
        depth_msg = self._make_image_msg(stamp, "camera_depth_optical_frame", "32FC1", depth)
        rgb_info = self._build_camera_info(stamp, "camera_rgb_optical_frame")
        depth_info = self._build_camera_info(stamp, "camera_depth_optical_frame")

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.rgb_info_pub.publish(rgb_info)
        self.depth_info_pub.publish(depth_info)

        if self.publish_pointcloud:
            self.pointcloud_pub.publish(self._make_pointcloud_msg(stamp, "camera_depth_optical_frame", depth))

        self.last_render_sim_time = self.data.time

    def _update_overview_window(self) -> None:
        if self.overview_window is None or self.overview_renderer is None:
            return
        if self.data.time - self.last_overview_render_sim_time < 1.0 / max(self.render_rate_hz, 1e-3):
            try:
                self.overview_window.pump_events()
            except tk.TclError:
                self.overview_window = None
            return

        try:
            _, _, yaw, _ = self._read_pose()
            orbit_azimuth = math.degrees(yaw) + (self.data.time * 35.0)
            view_images = {}
            for key, camera in (
                ("orbit", self._make_follow_camera(orbit_azimuth, -28.0, 2.45)),
                ("front", self._make_follow_camera(math.degrees(yaw), -18.0, 1.95)),
                ("left", self._make_follow_camera(math.degrees(yaw) + 90.0, -18.0, 2.05)),
                ("right", self._make_follow_camera(math.degrees(yaw) - 90.0, -18.0, 2.05)),
            ):
                self.overview_renderer.update_scene(self.data, camera=camera)
                view_images[key] = np.array(self.overview_renderer.render(), copy=True)
            self.overview_window.update_image(
                view_images,
                battery_pct=self.battery_pct,
                autonomy_state=self.latest_autonomy_state,
                is_charging=self.is_charging,
            )
            self.last_overview_render_sim_time = self.data.time
            self.overview_window.pump_events()
        except tk.TclError:
            self.overview_window = None

    def _update_status_window(self) -> None:
        if self.status_window is None:
            return
        try:
            now_wall = time.perf_counter()
            if now_wall - self.last_status_event_pump_wall_time >= (1.0 / 24.0):
                self.status_window.pump_events()
                self.last_status_event_pump_wall_time = now_wall
            if self.data.time - self.last_status_ui_update_sim_time < 1.0 / max(12.0, self.render_rate_hz):
                return
            self.last_status_ui_update_sim_time = self.data.time
            main_image, preview_image = self._render_status_preview()
            if main_image is not None:
                try:
                    self.status_window.update_image(main_image)
                except Exception as exc:
                    self.get_logger().warning(f"Status window main image update skipped: {exc}")
            if preview_image is not None:
                try:
                    self.status_window.update_preview_image(preview_image)
                except Exception as exc:
                    self.get_logger().warning(f"Status window preview update skipped: {exc}")
            try:
                self.status_window.update_dashboard(self._build_dashboard_state())
            except Exception as exc:
                self.get_logger().warning(f"Status window dashboard update skipped: {exc}")
        except Exception as exc:
            try:
                self.get_logger().warning(f"Status window disabled after runtime failure: {exc}")
            except Exception:
                pass
            try:
                self.status_window.close()
            except Exception:
                pass
            self.status_window = None

    def _append_world_marker(
        self,
        markers: MarkerArray,
        stamp,
        ns: str,
        marker_id: int,
        marker_type: int,
        position: tuple[float, float, float],
        scale: tuple[float, float, float],
        color: tuple[float, float, float, float],
        rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        text: str = "",
    ) -> None:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "odom"
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        qx, qy, qz, qw = euler_to_quat_xyzw(*rpy)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.text = text
        markers.markers.append(marker)

    def _append_local_marker(
        self,
        markers: MarkerArray,
        stamp,
        ns: str,
        marker_id: int,
        marker_type: int,
        base_x: float,
        base_y: float,
        base_yaw: float,
        local_position: tuple[float, float, float],
        scale: tuple[float, float, float],
        color: tuple[float, float, float, float],
        local_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        text: str = "",
    ) -> None:
        world_x, world_y, world_z = local_point_to_world(
            base_x,
            base_y,
            base_yaw,
            local_position[0],
            local_position[1],
            local_position[2],
        )
        self._append_world_marker(
            markers,
            stamp,
            ns,
            marker_id,
            marker_type,
            (world_x, world_y, world_z),
            scale,
            color,
            (local_rpy[0], local_rpy[1], base_yaw + local_rpy[2]),
            text=text,
        )

    def _append_main_robot_scene_markers(
        self,
        markers: MarkerArray,
        stamp,
        base_x: float,
        base_y: float,
        base_yaw: float,
        start_id: int,
    ) -> None:
        dark = (0.16, 0.18, 0.20, 1.0)
        darker = (0.10, 0.12, 0.14, 0.95)
        shell = (0.92, 0.93, 0.95, 1.0)
        metal = (0.83, 0.85, 0.87, 1.0)
        sensor = (0.12, 0.14, 0.16, 1.0)
        orange = (0.96, 0.64, 0.12, 1.0)
        tire = (0.08, 0.08, 0.08, 1.0)
        glow = (0.18, 0.92, 0.48, 0.95)
        parts = [
            (Marker.CUBE, (0.00, 0.00, 0.06), (0.90, 0.56, 0.05), dark, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.00, 0.00, 0.18), (0.32, 0.32, 0.22), dark, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (-0.12, 0.00, 0.28), (0.22, 0.20, 0.16), darker, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.43, 0.00, 0.23), (0.12, 0.60, 0.36), shell, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.00, 0.00, 0.45), (0.88, 0.56, 0.05), dark, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.45, 0.00, 0.10), (0.05, 0.46, 0.12), orange, (0.0, 0.0, 0.0)),
            (Marker.CYLINDER, (0.34, 0.24, 0.44), (0.072, 0.072, 0.056), sensor, (0.0, 0.0, 0.0)),
            (Marker.CYLINDER, (-0.28, -0.24, 0.44), (0.072, 0.072, 0.056), sensor, (0.0, 0.0, 0.0)),
            (Marker.CYLINDER, (0.00, 0.26, 0.00), (0.17, 0.07, 0.07), tire, (math.pi / 2.0, 0.0, 0.0)),
            (Marker.CYLINDER, (0.00, -0.26, 0.00), (0.17, 0.07, 0.07), tire, (math.pi / 2.0, 0.0, 0.0)),
            (Marker.SPHERE, (0.34, 0.00, 0.26), (0.052, 0.052, 0.052), glow, (0.0, 0.0, 0.0)),
        ]
        for offset, (marker_type, local_position, scale, color, local_rpy) in enumerate(parts):
            self._append_local_marker(
                markers,
                stamp,
                "scene_amr_a",
                start_id + offset,
                marker_type,
                base_x,
                base_y,
                base_yaw,
                local_position,
                scale,
                color,
                local_rpy=local_rpy,
            )
        self._append_local_marker(
            markers,
            stamp,
            "scene_amr_a",
            start_id + 40,
            Marker.TEXT_VIEW_FACING,
            base_x,
            base_y,
            base_yaw,
            (0.0, 0.0, 1.24),
            (0.0, 0.0, 0.16),
            (0.96, 0.98, 1.0, 0.96),
            text="AMR-A",
        )

    def _append_service_robot_scene_markers(
        self,
        markers: MarkerArray,
        stamp,
        base_x: float,
        base_y: float,
        base_yaw: float,
        start_id: int,
    ) -> None:
        dark = (0.24, 0.25, 0.27, 1.0)
        darker = (0.18, 0.19, 0.21, 1.0)
        shell = (0.98, 0.98, 0.99, 1.0)
        linen = (0.95, 0.96, 0.97, 0.95)
        metal = (0.83, 0.85, 0.87, 1.0)
        sensor = (0.08, 0.08, 0.09, 1.0)
        blue = (0.20, 0.74, 1.00, 0.95)
        tire = (0.08, 0.08, 0.08, 1.0)
        parts = [
            (Marker.CUBE, (0.00, 0.00, 0.15), (0.80, 0.56, 0.30), dark, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.01, 0.00, 0.09), (0.84, 0.60, 0.06), darker, (0.0, 0.0, 0.0)),
            (Marker.CYLINDER, (0.05, 0.00, 0.34), (0.34, 0.34, 0.34), shell, (0.0, math.pi / 2.0, 0.0)),
            (Marker.CUBE, (0.03, 0.00, 0.53), (0.30, 0.24, 0.04), shell, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.17, 0.00, 0.57), (0.22, 0.02, 0.02), metal, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.01, 0.14, 0.57), (0.02, 0.22, 0.02), metal, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.01, -0.14, 0.57), (0.02, 0.22, 0.02), metal, (0.0, 0.0, 0.0)),
            (Marker.CYLINDER, (0.18, 0.00, 0.64), (0.12, 0.12, 0.10), sensor, (0.0, math.pi / 2.0, 0.0)),
            (Marker.CUBE, (0.20, 0.00, 0.39), (0.04, 0.16, 0.12), darker, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.39, 0.00, 0.10), (0.044, 0.48, 0.10), darker, (0.0, 0.0, 0.0)),
            (Marker.CUBE, (0.25, 0.00, 0.25), (0.036, 0.40, 0.036), blue, (0.0, 0.0, 0.0)),
            (Marker.CYLINDER, (0.00, 0.29, 0.02), (0.14, 0.07, 0.07), tire, (math.pi / 2.0, 0.0, 0.0)),
            (Marker.CYLINDER, (0.00, -0.29, 0.02), (0.14, 0.07, 0.07), tire, (math.pi / 2.0, 0.0, 0.0)),
            (Marker.SPHERE, (-0.26, 0.00, -0.01), (0.09, 0.09, 0.09), tire, (0.0, 0.0, 0.0)),
            (Marker.CYLINDER, (0.00, 0.00, 0.43), (0.30, 0.30, 0.18), linen, (0.0, math.pi / 2.0, 0.0)),
        ]
        for offset, (marker_type, local_position, scale, color, local_rpy) in enumerate(parts):
            self._append_local_marker(
                markers,
                stamp,
                "scene_amr_b",
                start_id + offset,
                marker_type,
                base_x,
                base_y,
                base_yaw,
                local_position,
                scale,
                color,
                local_rpy=local_rpy,
            )
        self._append_local_marker(
            markers,
            stamp,
            "scene_amr_b",
            start_id + 40,
            Marker.TEXT_VIEW_FACING,
            base_x,
            base_y,
            base_yaw,
            (0.0, 0.0, 1.05),
            (0.0, 0.0, 0.16),
            (0.96, 0.98, 1.0, 0.96),
            text="AMR-B",
        )

    def _append_scene_decor_markers(self, markers: MarkerArray, stamp) -> None:
        self._append_world_marker(
            markers,
            stamp,
            "scene_floor",
            0,
            Marker.CUBE,
            (0.0, 0.0, -0.01),
            (19.2, 13.8, 0.02),
            (0.22, 0.23, 0.25, 1.0),
        )
        self._append_world_marker(
            markers,
            stamp,
            "scene_service_dock",
            1,
            Marker.CUBE,
            (SERVICE_DOCK_X - 0.26, SERVICE_DOCK_Y, 0.035),
            (0.84, 1.84, 0.07),
            (0.16, 0.62, 0.98, 0.40),
        )
        self._append_world_marker(
            markers,
            stamp,
            "scene_service_dock",
            2,
            Marker.CUBE,
            (SERVICE_DOCK_X + 0.72, SERVICE_DOCK_Y, 0.58),
            (0.18, 1.72, 1.16),
            (0.08, 0.12, 0.18, 0.88),
        )
        self._append_world_marker(
            markers,
            stamp,
            "scene_service_dock",
            3,
            Marker.TEXT_VIEW_FACING,
            (SERVICE_DOCK_X - 0.08, SERVICE_DOCK_Y, 1.05),
            (0.0, 0.0, 0.20),
            (0.75, 0.92, 1.0, 0.96),
            text="SERVICE DOCK",
        )
        for index, table in enumerate(HOTEL_TABLES.values(), start=10):
            table_x = float(table["pos"][0])
            table_y = float(table["pos"][1])
            self._append_world_marker(
                markers,
                stamp,
                "scene_tables",
                index,
                Marker.CYLINDER,
                (table_x, table_y, 0.79),
                (1.08, 1.08, 0.064),
                (0.52, 0.42, 0.30, 0.96),
            )
            self._append_world_marker(
                markers,
                stamp,
                "scene_tables",
                index + 20,
                Marker.CYLINDER,
                (table_x, table_y, 0.46),
                (0.14, 0.14, 0.56),
                (0.64, 0.52, 0.28, 0.96),
            )
        sofa_boxes = [
            ((0.85, 4.55, 0.34), (1.88, 0.44, 0.24)),
            ((1.66, 3.62, 0.34), (0.52, 1.44, 0.24)),
            ((-4.95, 1.20, 0.34), (2.24, 0.44, 0.24)),
            ((-5.74, 0.52, 0.34), (0.60, 1.36, 0.24)),
            ((6.45, 4.12, 0.34), (1.88, 0.44, 0.24)),
            ((7.26, 3.18, 0.34), (0.52, 1.48, 0.24)),
        ]
        for index, (position, scale) in enumerate(sofa_boxes, start=60):
            self._append_world_marker(
                markers,
                stamp,
                "scene_sofas",
                index,
                Marker.CUBE,
                position,
                scale,
                (0.22, 0.24, 0.28, 0.96),
            )

    def _publish_visual_markers(self, stamp, x: float, y: float, yaw: float) -> None:
        markers = MarkerArray()
        clear_all = Marker()
        clear_all.header.stamp = stamp
        clear_all.header.frame_id = "odom"
        clear_all.action = Marker.DELETEALL
        markers.markers.append(clear_all)
        self._append_scene_decor_markers(markers, stamp)
        self._append_main_robot_scene_markers(markers, stamp, x, y, yaw, 300)
        self._append_service_robot_scene_markers(
            markers,
            stamp,
            float(self.service_bot_pose[0]),
            float(self.service_bot_pose[1]),
            self.service_bot_yaw,
            400,
        )
        self._append_world_marker(
            markers,
            stamp,
            "scene_status",
            500,
            Marker.TEXT_VIEW_FACING,
            (x, y, 1.48),
            (0.0, 0.0, 0.18),
            (1.0, 1.0, 1.0, 0.96),
            text=f"AMR-A | {100.0 * self.battery_pct:5.1f}% | {self.latest_autonomy_state}",
        )
        self._append_world_marker(
            markers,
            stamp,
            "scene_status",
            501,
            Marker.TEXT_VIEW_FACING,
            (float(self.service_bot_pose[0]), float(self.service_bot_pose[1]), 1.32),
            (0.0, 0.0, 0.18),
            (1.0, 1.0, 1.0, 0.96),
            text=f"AMR-B | {100.0 * self.service_bot_battery_pct:4.1f}% | {self.service_bot_mission_mode}",
        )

        self.marker_pub.publish(markers)

    def _publish_state(self) -> None:
        mount_scans = self._read_all_lidar_mounts()
        fused_scan = self._fuse_lidar_scans(mount_scans)
        x, y, yaw, quat_wxyz = self._read_pose()
        stamp = self.get_clock().now().to_msg()
        if not self.path_history or math.hypot(x - self.path_history[-1][0], y - self.path_history[-1][1]) > 0.10:
            self.path_history.append((x, y))
            if len(self.path_history) > 240:
                self.path_history = self.path_history[-240:]

        self._publish_scan_msg(stamp, "laser_frame", self.combined_lidar_angles, fused_scan, self.scan_pub)
        self._publish_scan_msg(
            stamp,
            "front_left_lidar_frame",
            self.mount_relative_angles,
            mount_scans["front_left"],
            self.mount_scan_pubs["front_left"],
        )
        self._publish_scan_msg(
            stamp,
            "rear_right_lidar_frame",
            self.mount_relative_angles,
            mount_scans["rear_right"],
            self.mount_scan_pubs["rear_right"],
        )
        self._publish_imu(stamp, quat_wxyz)
        self._publish_joint_states(stamp)
        self._publish_odom_and_tf(stamp, x, y, yaw, quat_wxyz)
        self._publish_battery_and_dock(stamp)
        self._publish_runtime_status(x, y, yaw)
        self._publish_camera(stamp)
        self._publish_visual_markers(stamp, x, y, yaw)

    def _configure_viewer(self, viewer) -> None:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[0] = 0.0
            viewer.cam.lookat[1] = 0.0
            viewer.cam.lookat[2] = 0.96
            viewer.cam.distance = 11.2
            viewer.cam.azimuth = 138.0
            viewer.cam.elevation = -58.0
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
            for flag in (
                mujoco.mjtVisFlag.mjVIS_CONVEXHULL,
                mujoco.mjtVisFlag.mjVIS_JOINT,
                mujoco.mjtVisFlag.mjVIS_CAMERA,
                mujoco.mjtVisFlag.mjVIS_LIGHT,
                mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
                mujoco.mjtVisFlag.mjVIS_INERTIA,
                mujoco.mjtVisFlag.mjVIS_SCLINERTIA,
                mujoco.mjtVisFlag.mjVIS_PERTFORCE,
                mujoco.mjtVisFlag.mjVIS_PERTOBJ,
                mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
                mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
                mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT,
                mujoco.mjtVisFlag.mjVIS_TRANSPARENT,
                mujoco.mjtVisFlag.mjVIS_COM,
            ):
                viewer.opt.flags[flag] = False

            if viewer.user_scn is not None:
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = False

    def _update_viewer_overlay(self, viewer) -> None:
        viewer.set_texts((None, None, "", ""))

    def _drain_ros_callbacks(self, max_callbacks: int = 8) -> None:
        for _ in range(max(1, int(max_callbacks))):
            rclpy.spin_once(self, timeout_sec=0.0)

    def _run_loop(self, viewer=None) -> None:
        sim_dt = 1.0 / self.sim_rate_hz
        publish_dt = 1.0 / self.publish_rate_hz
        next_wall_time = time.perf_counter()
        next_publish_sim_time = 0.0

        while rclpy.ok() and (viewer is None or viewer.is_running()):
            if self.status_window is not None and getattr(self.status_window, "closed", False):
                break
            self._drain_ros_callbacks()
            self._update_dynamic_obstacles()
            self._update_service_bot(sim_dt)
            mujoco.mj_forward(self.model, self.data)
            current_scans = self._read_all_lidar_mounts()
            fused_scan = self._fuse_lidar_scans(current_scans)
            self._apply_control(sim_dt, fused_scan)
            mujoco.mj_step(self.model, self.data)
            self._update_arm_and_payload(sim_dt)
            self._update_visual_wheels(sim_dt)
            self._update_battery_and_dock(sim_dt)
            self._update_task_scheduler()
            self._dispatch_startup_missions_if_ready()

            if self.data.time >= next_publish_sim_time:
                self._publish_state()
                next_publish_sim_time += publish_dt

            if viewer is not None:
                self._update_viewer_overlay(viewer)
                viewer.sync()
                self._update_overview_window()

            self._update_status_window()

            next_wall_time += sim_dt / self.real_time_factor
            sleep_time = next_wall_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_wall_time = time.perf_counter()

    def run(self) -> None:
        if self.status_window is not None:
            self._run_loop(viewer=None)
        elif self.use_viewer:
            with mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            ) as viewer:
                self._configure_viewer(viewer)
                self._run_loop(viewer=viewer)
        else:
            self._run_loop(viewer=None)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = MujocoAmrSimNode()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None and getattr(node, "overview_window", None) is not None:
            node.overview_window.close()
        if node is not None and getattr(node, "status_window", None) is not None:
            node.status_window.close()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
