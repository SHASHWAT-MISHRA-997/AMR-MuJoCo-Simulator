from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_rviz = PathJoinSubstitution([FindPackageShare("mujoco_amr_sim"), "rviz", "monitoring_clean.rviz"])

    return LaunchDescription(
        [
            SetEnvironmentVariable("DISPLAY", ":0"),
            SetEnvironmentVariable("WAYLAND_DISPLAY", "wayland-0"),
            SetEnvironmentVariable("XDG_RUNTIME_DIR", "/mnt/wslg/runtime-dir"),
            SetEnvironmentVariable("PULSE_SERVER", "unix:/mnt/wslg/PulseServer"),
            DeclareLaunchArgument("rviz_config_file", default_value=default_rviz),
            Node(
                package="rviz2",
                executable="rviz2",
                name="amr_monitoring",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config_file")],
            ),
        ]
    )
