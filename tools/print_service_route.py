#!/usr/bin/env python3
import argparse
import json

from mujoco_amr_sim.config_utils import build_service_route, service_target_goal
from mujoco_amr_sim.mjcf_builder import DOCK_X, DOCK_Y, SERVICE_DOCK_X, SERVICE_DOCK_Y


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a generated service route for AMR service targets.")
    parser.add_argument("target", help="Target key like table_1 or sofa_2")
    parser.add_argument("--robot", choices=("a", "b"), default="a", help="Robot variant")
    parser.add_argument("--start-x", type=float, help="Override start X")
    parser.add_argument("--start-y", type=float, help="Override start Y")
    args = parser.parse_args()

    if args.start_x is not None and args.start_y is not None:
        start = (args.start_x, args.start_y)
    elif args.robot == "b":
        start = (SERVICE_DOCK_X, SERVICE_DOCK_Y)
    else:
        start = (DOCK_X, DOCK_Y)

    route = build_service_route(start, args.target, args.robot)
    goal = service_target_goal(args.target, args.robot)
    print(
        json.dumps(
            {
                "robot": args.robot,
                "target": args.target,
                "start": [round(start[0], 3), round(start[1], 3)],
                "goal": [round(goal[0], 3), round(goal[1], 3), round(goal[2], 3)],
                "route": [[round(point[0], 3), round(point[1], 3)] for point in route],
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()

