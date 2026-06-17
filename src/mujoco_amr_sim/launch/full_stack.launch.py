import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description_path = os.path.join(get_package_share_directory("mujoco_amr_sim"), "urdf", "amr.urdf")
    with open(robot_description_path, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()

    package_share = FindPackageShare("mujoco_amr_sim")
    nav2_share = FindPackageShare("nav2_bringup")
    slam_share = FindPackageShare("slam_toolbox")

    default_waypoints = PathJoinSubstitution([package_share, "config", "waypoints.json"])
    default_dock = PathJoinSubstitution([package_share, "config", "dock_station.json"])
    default_ekf = PathJoinSubstitution([package_share, "config", "ekf.yaml"])
    default_slam = PathJoinSubstitution([package_share, "config", "slam_toolbox.yaml"])
    default_nav2_params = PathJoinSubstitution([package_share, "config", "nav2_params.yaml"])
    default_rl_model = PathJoinSubstitution([package_share, "models", "mujoco_amr_ppo.zip"])
    default_rviz = PathJoinSubstitution([package_share, "rviz", "monitoring.rviz"])
    selected_odom_topic = PythonExpression(["'/odometry/filtered' if '", LaunchConfiguration("use_ekf"), "' == 'true' else '/odom'"])

    publish_odom_tf = PythonExpression(["'false' if '", LaunchConfiguration("use_ekf"), "' == 'true' else 'true'"])
    primary_source = PythonExpression(
        [
            "'rl' if '", LaunchConfiguration("use_rl"), "' == 'true' else "
            "('nav' if '", LaunchConfiguration("use_nav2"), "' == 'true' else 'auto')",
        ]
    )
    nav2_enabled = PythonExpression(
        ["'", LaunchConfiguration("use_nav2"), "' == 'true' and '", LaunchConfiguration("use_rl"), "' != 'true'"]
    )
    local_autonomy_enabled = PythonExpression(["'", LaunchConfiguration("use_nav2"), "' != 'true'"])

    sim_node = Node(
        package="mujoco_amr_sim",
        executable="mujoco_amr_sim_node",
        name="mujoco_amr_sim",
        output="screen",
        parameters=[
            {
                "use_viewer": LaunchConfiguration("use_viewer"),
                "auto_mode": False,
                "sim_rate_hz": LaunchConfiguration("sim_rate_hz"),
                "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
                "render_rate_hz": LaunchConfiguration("render_rate_hz"),
                "real_time_factor": LaunchConfiguration("real_time_factor"),
                "publish_odom_tf": publish_odom_tf,
                "publish_depth_camera": True,
                "publish_pointcloud": True,
                "show_overview_window": LaunchConfiguration("show_overview_window"),
                "show_status_window": LaunchConfiguration("show_status_window"),
                "enable_dynamic_obstacles": LaunchConfiguration("enable_dynamic_obstacles"),
                "dynamic_obstacle_speed_scale": LaunchConfiguration("dynamic_obstacle_speed_scale"),
                "camera_width": LaunchConfiguration("camera_width"),
                "camera_height": LaunchConfiguration("camera_height"),
                "dock_config_file": LaunchConfiguration("dock_config_file"),
                "initial_battery_percentage": LaunchConfiguration("initial_battery_percentage"),
                "battery_discharge_idle_per_sec": LaunchConfiguration("battery_discharge_idle_per_sec"),
                "battery_discharge_motion_per_sec": LaunchConfiguration("battery_discharge_motion_per_sec"),
                "battery_charge_per_sec": LaunchConfiguration("battery_charge_per_sec"),
                "startup_mission_a": LaunchConfiguration("startup_mission_a"),
                "startup_mission_b": LaunchConfiguration("startup_mission_b"),
                "startup_mission_delay_sec": LaunchConfiguration("startup_mission_delay_sec"),
            }
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"publish_frequency": 15.0, "robot_description": robot_description}],
    )

    mux_node = Node(
        package="mujoco_amr_sim",
        executable="command_mux",
        name="command_mux",
        output="screen",
        parameters=[{"primary_source": primary_source}],
    )

    autonomy_node = Node(
        condition=IfCondition(local_autonomy_enabled),
        package="mujoco_amr_sim",
        executable="autonomy_manager",
        name="autonomy_manager",
        output="screen",
        parameters=[
            {
                "patrol_enabled": LaunchConfiguration("patrol_enabled"),
                "battery_dock_threshold": LaunchConfiguration("battery_dock_threshold"),
                "battery_resume_threshold": LaunchConfiguration("battery_resume_threshold"),
                "odom_topic": selected_odom_topic,
                "waypoints_file": LaunchConfiguration("waypoints_file"),
                "dock_config_file": LaunchConfiguration("dock_config_file"),
            }
        ],
    )

    nav2_bridge_node = Node(
        condition=IfCondition(nav2_enabled),
        package="mujoco_amr_sim",
        executable="nav2_mission_bridge",
        name="nav2_mission_bridge",
        output="screen",
        parameters=[
            {
                "odom_topic": selected_odom_topic,
                "dock_config_file": LaunchConfiguration("dock_config_file"),
            }
        ],
    )

    ekf_node = Node(
        condition=IfCondition(LaunchConfiguration("use_ekf")),
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[LaunchConfiguration("ekf_params_file")],
        remappings=[("odometry/filtered", "/odometry/filtered")],
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([slam_share, "launch", "online_async_launch.py"])
        ),
        condition=IfCondition(LaunchConfiguration("use_slam")),
        launch_arguments={
            "use_sim_time": "false",
            "slam_params_file": LaunchConfiguration("slam_params_file"),
        }.items(),
    )

    nav2_group = GroupAction(
        condition=IfCondition(nav2_enabled),
        actions=[
            SetRemap(src="/cmd_vel", dst="/cmd_vel_nav"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([nav2_share, "launch", "navigation_launch.py"])
                ),
                launch_arguments={
                    "use_sim_time": "false",
                    "params_file": LaunchConfiguration("nav2_params_file"),
                    "autostart": "true",
                    "use_composition": "False",
                    "use_respawn": "False",
                    "container_name": "nav2_container",
                }.items(),
            ),
        ],
    )

    delayed_nav2 = TimerAction(period=5.0, actions=[nav2_group])

    rl_node = Node(
        condition=IfCondition(LaunchConfiguration("use_rl")),
        package="mujoco_amr_sim",
        executable="rl_policy_node",
        name="rl_policy_node",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("rl_model_path"),
                "odom_topic": selected_odom_topic,
                "dock_config_file": LaunchConfiguration("dock_config_file"),
                "waypoints_file": LaunchConfiguration("waypoints_file"),
            }
        ],
    )

    monitoring_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([package_share, "launch", "monitoring.launch.py"])),
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        launch_arguments={"rviz_config_file": LaunchConfiguration("rviz_config_file")}.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_viewer", default_value="true"),
            DeclareLaunchArgument("use_nav2", default_value="false"),
            DeclareLaunchArgument("use_slam", default_value="false"),
            DeclareLaunchArgument("use_ekf", default_value="false"),
            DeclareLaunchArgument("use_rl", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("show_overview_window", default_value="false"),
            DeclareLaunchArgument("show_status_window", default_value="true"),
            DeclareLaunchArgument("enable_dynamic_obstacles", default_value="false"),
            DeclareLaunchArgument("dynamic_obstacle_speed_scale", default_value="1.0"),
            DeclareLaunchArgument("sim_rate_hz", default_value="150.0"),
            DeclareLaunchArgument("publish_rate_hz", default_value="12.0"),
            DeclareLaunchArgument("render_rate_hz", default_value="5.0"),
            DeclareLaunchArgument("real_time_factor", default_value="1.0"),
            DeclareLaunchArgument("display_env", default_value=":0"),
            DeclareLaunchArgument("wayland_display_env", default_value="wayland-0"),
            DeclareLaunchArgument("xdg_runtime_dir_env", default_value="/mnt/wslg/runtime-dir"),
            DeclareLaunchArgument("pulse_server_env", default_value="unix:/mnt/wslg/PulseServer"),
            DeclareLaunchArgument("camera_width", default_value="256"),
            DeclareLaunchArgument("camera_height", default_value="192"),
            DeclareLaunchArgument("patrol_enabled", default_value="true"),
            DeclareLaunchArgument("battery_dock_threshold", default_value="0.30"),
            DeclareLaunchArgument("battery_resume_threshold", default_value="0.98"),
            DeclareLaunchArgument("initial_battery_percentage", default_value="1.0"),
            DeclareLaunchArgument("battery_discharge_idle_per_sec", default_value="0.0006"),
            DeclareLaunchArgument("battery_discharge_motion_per_sec", default_value="0.0045"),
            DeclareLaunchArgument("battery_charge_per_sec", default_value="0.018"),
            DeclareLaunchArgument("startup_mission_a", default_value=""),
            DeclareLaunchArgument("startup_mission_b", default_value=""),
            DeclareLaunchArgument("startup_mission_delay_sec", default_value="1.5"),
            DeclareLaunchArgument("waypoints_file", default_value=default_waypoints),
            DeclareLaunchArgument("dock_config_file", default_value=default_dock),
            DeclareLaunchArgument("ekf_params_file", default_value=default_ekf),
            DeclareLaunchArgument("slam_params_file", default_value=default_slam),
            DeclareLaunchArgument("nav2_params_file", default_value=default_nav2_params),
            DeclareLaunchArgument("rl_model_path", default_value=default_rl_model),
            DeclareLaunchArgument("rviz_config_file", default_value=default_rviz),
            SetEnvironmentVariable("DISPLAY", LaunchConfiguration("display_env")),
            SetEnvironmentVariable("WAYLAND_DISPLAY", LaunchConfiguration("wayland_display_env")),
            SetEnvironmentVariable("XDG_RUNTIME_DIR", LaunchConfiguration("xdg_runtime_dir_env")),
            SetEnvironmentVariable("PULSE_SERVER", LaunchConfiguration("pulse_server_env")),
            robot_state_publisher,
            sim_node,
            mux_node,
            autonomy_node,
            nav2_bridge_node,
            rl_node,
            ekf_node,
            slam_launch,
            delayed_nav2,
            monitoring_launch,
        ]
    )
