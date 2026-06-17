import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description_path = os.path.join(get_package_share_directory("mujoco_amr_sim"), "urdf", "amr.urdf")
    with open(robot_description_path, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()

    default_waypoints = PathJoinSubstitution(
        [FindPackageShare("mujoco_amr_sim"), "config", "waypoints.json"]
    )
    default_dock = PathJoinSubstitution(
        [FindPackageShare("mujoco_amr_sim"), "config", "dock_station.json"]
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("DISPLAY", ":0"),
            SetEnvironmentVariable("WAYLAND_DISPLAY", "wayland-0"),
            SetEnvironmentVariable("XDG_RUNTIME_DIR", "/mnt/wslg/runtime-dir"),
            SetEnvironmentVariable("PULSE_SERVER", "unix:/mnt/wslg/PulseServer"),
            DeclareLaunchArgument("use_viewer", default_value="true"),
            DeclareLaunchArgument("auto_mode", default_value="true"),
            DeclareLaunchArgument("show_overview_window", default_value="false"),
            DeclareLaunchArgument("show_status_window", default_value="true"),
            DeclareLaunchArgument("enable_dynamic_obstacles", default_value="false"),
            DeclareLaunchArgument("dynamic_obstacle_speed_scale", default_value="1.0"),
            DeclareLaunchArgument("sim_rate_hz", default_value="150.0"),
            DeclareLaunchArgument("publish_rate_hz", default_value="12.0"),
            DeclareLaunchArgument("render_rate_hz", default_value="5.0"),
            DeclareLaunchArgument("camera_width", default_value="256"),
            DeclareLaunchArgument("camera_height", default_value="192"),
            DeclareLaunchArgument("real_time_factor", default_value="1.0"),
            DeclareLaunchArgument("waypoints_file", default_value=default_waypoints),
            DeclareLaunchArgument("dock_config_file", default_value=default_dock),
            DeclareLaunchArgument("initial_battery_percentage", default_value="1.0"),
            DeclareLaunchArgument("battery_discharge_idle_per_sec", default_value="0.0006"),
            DeclareLaunchArgument("battery_discharge_motion_per_sec", default_value="0.0045"),
            DeclareLaunchArgument("battery_charge_per_sec", default_value="0.018"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"publish_frequency": 30.0, "robot_description": robot_description}],
            ),
            Node(
                package="mujoco_amr_sim",
                executable="mujoco_amr_sim_node",
                name="mujoco_amr_sim",
                output="screen",
                parameters=[
                    {
                        "use_viewer": LaunchConfiguration("use_viewer"),
                        "auto_mode": LaunchConfiguration("auto_mode"),
                        "show_overview_window": LaunchConfiguration("show_overview_window"),
                        "show_status_window": LaunchConfiguration("show_status_window"),
                        "enable_dynamic_obstacles": LaunchConfiguration("enable_dynamic_obstacles"),
                        "dynamic_obstacle_speed_scale": LaunchConfiguration("dynamic_obstacle_speed_scale"),
                        "sim_rate_hz": LaunchConfiguration("sim_rate_hz"),
                        "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
                        "render_rate_hz": LaunchConfiguration("render_rate_hz"),
                        "camera_width": LaunchConfiguration("camera_width"),
                        "camera_height": LaunchConfiguration("camera_height"),
                        "real_time_factor": LaunchConfiguration("real_time_factor"),
                        "waypoints_file": LaunchConfiguration("waypoints_file"),
                        "dock_config_file": LaunchConfiguration("dock_config_file"),
                        "initial_battery_percentage": LaunchConfiguration("initial_battery_percentage"),
                        "battery_discharge_idle_per_sec": LaunchConfiguration("battery_discharge_idle_per_sec"),
                        "battery_discharge_motion_per_sec": LaunchConfiguration("battery_discharge_motion_per_sec"),
                        "battery_charge_per_sec": LaunchConfiguration("battery_charge_per_sec"),
                        "publish_depth_camera": True,
                        "publish_pointcloud": True,
                        "publish_odom_tf": True,
                    }
                ],
            ),
        ]
    )
