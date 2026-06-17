import os
from glob import glob

from setuptools import find_packages, setup


package_name = "mujoco_amr_sim"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.json") + glob("config/*.yaml")),
        (os.path.join("share", package_name, "docs"), glob("docs/*")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.urdf")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "mujoco",
        "glfw",
        "PyOpenGL",
        "Pillow",
        "absl-py",
        "etils",
    ],
    zip_safe=True,
    maintainer="shash",
    maintainer_email="shash@example.com",
    description="MuJoCo-based autonomous mobile robot simulation for ROS 2 Jazzy.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_amr_sim_node = mujoco_amr_sim.sim_node:main",
            "autonomy_manager = mujoco_amr_sim.autonomy_manager:main",
            "nav2_mission_bridge = mujoco_amr_sim.nav2_mission_bridge:main",
            "command_mux = mujoco_amr_sim.command_mux:main",
            "train_rl_policy = mujoco_amr_sim.train_rl:main",
            "rl_policy_node = mujoco_amr_sim.rl_policy_node:main",
        ],
    },
)
