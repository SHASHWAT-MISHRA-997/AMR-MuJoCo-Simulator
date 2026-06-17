#!/usr/bin/env bash
set -eo pipefail

ROBOT="${1:-}"
TARGET="${2:-}"
SUCCESS_TIMEOUT="${3:-120}"

if [[ -z "$ROBOT" || -z "$TARGET" ]]; then
  echo "usage: $0 <a|b> <target> [success-timeout-seconds]" >&2
  exit 2
fi

cd "/mnt/c/Users/shash/OneDrive/Desktop/AMR Simulation MuJoCo/ros2_ws"
TMP_OUTPUT="$(mktemp)"
set +e
bash tools/snapshot_service_target.sh "$ROBOT" "$TARGET" "$SUCCESS_TIMEOUT" >"$TMP_OUTPUT" 2>&1
SNAPSHOT_EXIT=$?
set -e

STATUS_JSON="$(awk 'BEGIN{seen=0} /^STATUS_JSON$/{seen=1; next} seen && /^\{/{print; exit}' "$TMP_OUTPUT")"
if [[ -z "$STATUS_JSON" ]]; then
  STATUS_JSON='{}'
fi

RESULT_JSON="$(
  PYTHONPATH=src/mujoco_amr_sim python3 - "$ROBOT" "$TARGET" "$STATUS_JSON" "$TMP_OUTPUT" <<'PY'
import json
import math
import pathlib
import sys

from mujoco_amr_sim.config_utils import service_target_goal
from mujoco_amr_sim.mjcf_builder import SERVICE_TARGETS


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


robot = sys.argv[1]
target = sys.argv[2]
status_payload = json.loads(sys.argv[3] or "{}")
output_path = pathlib.Path(sys.argv[4])
output_text = output_path.read_text(encoding="utf-8", errors="replace")

mission_status = status_payload.get("mission_status") or {}
sim_status = status_payload.get("simulation_status") or {}
goal_x, goal_y, goal_yaw = service_target_goal(target, robot)

if robot == "a":
    pose = (sim_status.get("pose") or {}) if isinstance(sim_status, dict) else {}
    if not pose and isinstance(mission_status, dict):
        pose = mission_status.get("pose") or {}
    state = str((mission_status or {}).get("state", ""))
    mission_mode = str((mission_status or {}).get("mission_mode", ""))
    dock_contact = bool((mission_status or {}).get("dock_contact", False))
    is_charging = bool((mission_status or {}).get("is_charging", False))
    pose_x = float(pose.get("x", 0.0))
    pose_y = float(pose.get("y", 0.0))
    pose_yaw = float(pose.get("yaw", 0.0))
    distance = math.hypot(goal_x - pose_x, goal_y - pose_y)
    yaw_error = abs(wrap(goal_yaw - pose_yaw))
    label = str(SERVICE_TARGETS[target]["label"])
    reached_log = f"{label} reached; holding service position" in output_text
    served_log = f"{label} served; returning to auto dock" in output_text
    success = (
        state == "CHARGING"
        and mission_mode == "idle"
        and dock_contact
        and is_charging
        and reached_log
        and served_log
    )
    result = {
        "robot": robot,
        "target": target,
        "success": success,
        "state": state,
        "mission_mode": mission_mode,
        "dock_request_reason": mission_status.get("dock_request_reason"),
        "dock_contact": dock_contact,
        "is_charging": is_charging,
        "pose_metrics": {
            "x": round(pose_x, 3),
            "y": round(pose_y, 3),
            "yaw": round(pose_yaw, 3),
            "distance_to_goal": round(distance, 3),
            "yaw_error": round(yaw_error, 3),
        },
    }
else:
    service = (sim_status.get("service_amr") or {}) if isinstance(sim_status, dict) else {}
    pose = (service.get("pose") or {}) if isinstance(service, dict) else {}
    state = str(service.get("state", ""))
    mission_mode = str(service.get("mission_mode", ""))
    payload_state = str(service.get("payload_state", ""))
    is_charging = bool(service.get("is_charging", False))
    pose_x = float(pose.get("x", 0.0))
    pose_y = float(pose.get("y", 0.0))
    pose_yaw = float(pose.get("yaw", 0.0))
    distance = math.hypot(goal_x - pose_x, goal_y - pose_y)
    yaw_error = abs(wrap(goal_yaw - pose_yaw))
    success = (
        state == "CHARGING"
        and mission_mode == "idle"
        and is_charging
        and payload_state in {"EMPTY", "DOCKED"}
    )
    result = {
        "robot": robot,
        "target": target,
        "success": success,
        "state": state,
        "mission_mode": mission_mode,
        "payload_state": payload_state,
        "is_charging": is_charging,
        "pose_metrics": {
            "x": round(pose_x, 3),
            "y": round(pose_y, 3),
            "yaw": round(pose_yaw, 3),
            "distance_to_goal": round(distance, 3),
            "yaw_error": round(yaw_error, 3),
        },
    }

print(json.dumps(result, separators=(",", ":")))
PY
)"

printf '%s\n' "$RESULT_JSON"
printf '\n--- snapshot output ---\n'
cat "$TMP_OUTPUT"
rm -f "$TMP_OUTPUT"

python3 - "$RESULT_JSON" <<'PY'
import json
import sys

raise SystemExit(0 if json.loads(sys.argv[1]).get("success") else 1)
PY

