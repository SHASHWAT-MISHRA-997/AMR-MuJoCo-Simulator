#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/shash/OneDrive/Desktop/AMR Simulation MuJoCo/ros2_ws"

TIMEOUT_SECS="${1:-180}"
RESULT_DIR="${2:-/mnt/c/Users/shash/OneDrive/Desktop/AMR Simulation MuJoCo/ros2_ws/.verify_logs/matrix}"
ROBOT_FILTER="${3:-all}"
targets=(table_1 table_2 table_3 table_4 table_5 table_6 sofa_1 sofa_2 sofa_3)
robots=(a b)

mkdir -p "$RESULT_DIR"
overall_ok=0
for robot in "${robots[@]}"; do
  if [[ "$ROBOT_FILTER" != "all" && "$ROBOT_FILTER" != "$robot" ]]; then
    continue
  fi
  for target in "${targets[@]}"; do
    result_file="${RESULT_DIR}/${robot}_${target}.log"
    echo "=== VERIFY ${robot^^} ${target} ==="
    ./verify_service_target.sh "$robot" "$target" "$TIMEOUT_SECS" >"$result_file" 2>&1 || true
    json_line="$(grep -m1 '^{' "$result_file" || true)"
    if [[ -z "$json_line" ]]; then
      echo "RESULT FAIL missing-json"
      overall_ok=1
      echo "LOG $result_file"
      echo
      continue
    fi
    if python3 - "$json_line" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
raise SystemExit(0 if payload.get("success") else 1)
PY
    then
      python3 - "$json_line" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
pose = payload.get("pose_metrics") or {}
print(
    f"RESULT PASS dist={pose.get('distance_to_goal')} yaw={pose.get('yaw_error')} "
    f"state={payload.get('state')}"
)
PY
    else
      python3 - "$json_line" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
pose = payload.get("pose_metrics") or {}
print(
    f"RESULT FAIL dist={pose.get('distance_to_goal')} yaw={pose.get('yaw_error')} "
    f"state={payload.get('state')}"
)
PY
      overall_ok=1
    fi
    echo "LOG $result_file"
    echo
  done
done

exit "$overall_ok"

