#!/usr/bin/env bash
set -eo pipefail

ROBOT="${1:-a}"
TARGET="${2:-table_1}"
WAIT_SECONDS="${3:-70}"
VERIFY_RTF="${VERIFY_RTF_OVERRIDE:-12.0}"

cd "/mnt/c/Users/shash/OneDrive/Desktop/AMR Simulation MuJoCo/ros2_ws"
source /opt/ros/jazzy/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID_OVERRIDE:-$((70 + RANDOM % 80))}"
mkdir -p .verify_logs
LOG=".verify_logs/snapshot_${ROBOT}_${TARGET}.log"

ros2 launch mujoco_amr_sim full_stack.launch.py \
  use_viewer:=false \
  show_overview_window:=false \
  show_status_window:=false \
  sim_rate_hz:=100.0 \
  publish_rate_hz:=8.0 \
  render_rate_hz:=2.0 \
  real_time_factor:="$VERIFY_RTF" \
  >"$LOG" 2>&1 &
LAUNCH_PID=$!

cleanup() {
  kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true
  kill "$LAUNCH_PID" >/dev/null 2>&1 || true
  wait "$LAUNCH_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

heartbeat() {
  local started_at
  started_at="$(date +%s)"
  while kill -0 "$LAUNCH_PID" >/dev/null 2>&1; do
    sleep 15
    local elapsed
    elapsed=$(( $(date +%s) - started_at ))
    printf '[snapshot] %s %s at %ss\n' "$ROBOT" "$TARGET" "$elapsed"
  done
}

heartbeat &
HEARTBEAT_PID=$!

for _ in $(seq 1 60); do
  if ros2 node list 2>/dev/null | grep -q "/autonomy_manager"; then
    break
  fi
  sleep 1
done

sleep 5
if [[ "$ROBOT" == "a" ]]; then
  ros2 topic pub --once /autonomy/mission_command std_msgs/msg/String "{data: $TARGET}" >/dev/null
else
  for _ in 1 2 3; do
    ros2 topic pub --once /service_amr/mission_command std_msgs/msg/String "{data: $TARGET}" >/dev/null
    sleep 2
  done
fi

LAST_STATUS_JSON='{}'
deadline=$(( $(date +%s) + WAIT_SECONDS ))
while (( $(date +%s) < deadline )); do
  sleep 5
  LAST_STATUS_JSON="$(python3 tools/snapshot_ros_status.py --duration 4.0 2>/dev/null || echo '{}')"
  if python3 - "$ROBOT" "$LAST_STATUS_JSON" <<'PY'
import json
import sys

robot = sys.argv[1]
payload = json.loads(sys.argv[2] or "{}")
mission = payload.get("mission_status") or {}
sim = payload.get("simulation_status") or {}

if robot == "a":
    ok = (
        str(mission.get("state", "")) == "CHARGING"
        and str(mission.get("mission_mode", "")) == "idle"
        and bool(mission.get("dock_contact", False))
        and bool(mission.get("is_charging", False))
    )
else:
    service = sim.get("service_amr") or {}
    ok = (
        str(service.get("state", "")) == "CHARGING"
        and str(service.get("mission_mode", "")) == "idle"
        and bool(service.get("is_charging", False))
        and str(service.get("payload_state", "")) == "EMPTY"
    )

raise SystemExit(0 if ok else 1)
PY
  then
    break
  fi
done

status_missing_sim() {
  python3 - "$1" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1] or "{}")
sim = payload.get("simulation_status")
raise SystemExit(0 if sim in (None, {}, "") else 1)
PY
}

if status_missing_sim "$LAST_STATUS_JSON"; then
  for _ in 1 2 3; do
    sleep 2
    RETRY_STATUS_JSON="$(python3 tools/snapshot_ros_status.py --duration 5.0 2>/dev/null || echo '{}')"
    if ! status_missing_sim "$RETRY_STATUS_JSON"; then
      LAST_STATUS_JSON="$RETRY_STATUS_JSON"
      break
    fi
  done
fi

echo "STATUS_JSON"
if [[ "$LAST_STATUS_JSON" == '{}' ]]; then
  python3 tools/snapshot_ros_status.py --duration 5.0 || true
else
  printf '%s\n' "$LAST_STATUS_JSON"
fi
echo "LOGTAIL"
tail -n 120 "$LOG"

