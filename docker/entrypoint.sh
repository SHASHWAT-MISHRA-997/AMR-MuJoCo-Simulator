#!/usr/bin/env bash
set -euo pipefail

export ROS_WS="${ROS_WS:-/opt/ros2_ws}"
export DISPLAY="${APP_DISPLAY:-:1}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-glx}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export PYTHONPATH="${PYTHONPATH:-/opt/venv/lib/python3.12/site-packages}"
unset WAYLAND_DISPLAY
unset XDG_RUNTIME_DIR

export USE_VIEWER="${USE_VIEWER:-true}"
export SHOW_OVERVIEW_WINDOW="${SHOW_OVERVIEW_WINDOW:-false}"
export SHOW_STATUS_WINDOW="${SHOW_STATUS_WINDOW:-true}"
export SIM_RATE_HZ="${SIM_RATE_HZ:-100.0}"
export PUBLISH_RATE_HZ="${PUBLISH_RATE_HZ:-8.0}"
export RENDER_RATE_HZ="${RENDER_RATE_HZ:-12.0}"
export REAL_TIME_FACTOR="${REAL_TIME_FACTOR:-1.0}"

DISPLAY_NUM="${DISPLAY#:}"
DISPLAY_LOCK="/tmp/.X${DISPLAY_NUM}-lock"
DISPLAY_SOCKET="/tmp/.X11-unix/X${DISPLAY_NUM}"

focus_main_window() {
  if ! command -v wmctrl >/dev/null 2>&1; then
    return 0
  fi

  for _ in $(seq 1 60); do
    local win_id
    win_id="$(wmctrl -l 2>/dev/null | awk '/AMR TwinFlow Command Deck/ {print $1; exit}')"
    if [[ -n "${win_id}" ]]; then
      wmctrl -i -r "${win_id}" -b add,maximized_vert,maximized_horz >/dev/null 2>&1 || true
      wmctrl -i -a "${win_id}" >/dev/null 2>&1 || true
      return 0
    fi
    sleep 1
  done
}

brand_novnc() {
  install -m 0644 /opt/amr-favicon.svg /usr/share/novnc/app/images/amr-favicon.svg

  for icon_path in \
    /usr/share/novnc/app/images/novnc-icon.svg \
    /usr/share/novnc/app/images/novnc-icon-sm.svg
  do
    if [[ -f "${icon_path}" ]]; then
      cp /opt/amr-favicon.svg "${icon_path}"
    fi
  done

  python3 <<'PY'
from pathlib import Path

vnc_path = Path("/usr/share/novnc/vnc.html")
html = vnc_path.read_text(encoding="utf-8")

if "AMR TwinFlow Command Deck" not in html or "AMR TWINFLOW" not in html:
    html = html.replace(
        "<title>noVNC</title>",
        "<title>AMR TwinFlow Command Deck</title>\n"
        '    <link rel="icon" type="image/svg+xml" href="/app/images/amr-favicon.svg">\n'
        '    <meta name="theme-color" content="#09131c">\n'
        '    <script>\n'
        '      (function () {\n'
        '        const targetTitle = "AMR TWINFLOW";\n'
        '        const applyTitle = () => {\n'
        '          if (document.title !== targetTitle) {\n'
        '            document.title = targetTitle;\n'
        '          }\n'
        '        };\n'
        '        applyTitle();\n'
        '        window.addEventListener("load", applyTitle);\n'
        '        setInterval(applyTitle, 1000);\n'
        '      })();\n'
        '    </script>'
    )
    vnc_path.write_text(html, encoding="utf-8")
PY

  cat >/usr/share/novnc/index.html <<'HTML'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AMR TWINFLOW</title>
    <link rel="icon" type="image/svg+xml" href="/app/images/amr-favicon.svg">
    <meta name="theme-color" content="#000000">
    <style>
      html, body {
        margin: 0;
        height: 100%;
        background: #000;
        overflow: hidden;
      }
      iframe {
        display: block;
        width: 100vw;
        height: 100vh;
        border: 0;
        background: #000;
      }
      .credit-link {
        position: fixed;
        top: 12px;
        right: 14px;
        z-index: 10;
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 8px 12px;
        border-radius: 6px;
        background: rgba(4, 14, 24, 0.88);
        color: #dff7ff;
        font: 600 13px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        text-decoration: none;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
      }
      .credit-link span:last-child {
        color: #8ef0ff;
        text-decoration: underline;
      }
      .credit-link:hover {
        background: rgba(8, 33, 50, 0.96);
      }
    </style>
  </head>
  <body>
    <a class="credit-link" href="https://www.linkedin.com/in/sm980/" target="_blank" rel="noopener noreferrer" title="Open Shashwat Mishra on LinkedIn">
      <span>Made by SHASHWAT MISHRA</span>
      <span>LinkedIn</span>
    </a>
    <iframe src="/vnc.html?autoconnect=1&resize=scale" title="AMR TwinFlow Console"></iframe>
  </body>
</html>
HTML

  cp /usr/share/novnc/index.html /usr/share/novnc/app/index.html
}

cleanup() {
  kill "${WATCHDOG_PID:-}" >/dev/null 2>&1 || true
  kill "${LAUNCH_PID:-}" >/dev/null 2>&1 || true
  kill "${NOVNC_PID:-}" >/dev/null 2>&1 || true
  kill "${VNC_PID:-}" >/dev/null 2>&1 || true
  kill "${OPENBOX_PID:-}" >/dev/null 2>&1 || true
  kill "${XVFB_PID:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

display_is_live() {
  ps -eo args | grep -F "Xvfb ${DISPLAY}" | grep -v grep >/dev/null 2>&1
}

process_is_live() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1
}

prepare_display() {
  if [[ -f "${DISPLAY_LOCK}" || -S "${DISPLAY_SOCKET}" ]]; then
    if display_is_live; then
      echo "Reusing existing Xvfb session on ${DISPLAY}"
      return 0
    fi

    echo "Removing stale X display artifacts for ${DISPLAY}"
    rm -f "${DISPLAY_LOCK}"
    rm -f "${DISPLAY_SOCKET}"
  fi
}

start_xvfb() {
  if display_is_live; then
    XVFB_PID=""
    return 0
  fi

  Xvfb "${DISPLAY}" -screen 0 1600x900x24 -ac +extension GLX +render -noreset &
  XVFB_PID=$!

  for _ in $(seq 1 20); do
    if [[ -S "${DISPLAY_SOCKET}" ]] && display_is_live; then
      return 0
    fi
    sleep 0.5
  done

  echo "Xvfb failed to become ready on ${DISPLAY}" >&2
  exit 1
}

start_openbox() {
  if process_is_live "${OPENBOX_PID:-}"; then
    return 0
  fi

  openbox >/tmp/openbox.log 2>&1 &
  OPENBOX_PID=$!
}

start_x11vnc() {
  if process_is_live "${VNC_PID:-}"; then
    return 0
  fi

  x11vnc -display "${DISPLAY}" -forever -shared -nopw -xkb -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
  VNC_PID=$!

  for _ in $(seq 1 20); do
    if process_is_live "${VNC_PID}"; then
      return 0
    fi
    sleep 0.5
  done

  echo "x11vnc failed to become ready on ${DISPLAY}" >&2
  return 1
}

start_novnc() {
  if process_is_live "${NOVNC_PID:-}"; then
    return 0
  fi

  websockify --web=/usr/share/novnc/ 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
  NOVNC_PID=$!

  for _ in $(seq 1 20); do
    if process_is_live "${NOVNC_PID}"; then
      return 0
    fi
    sleep 0.5
  done

  echo "websockify failed to become ready on port 6080" >&2
  return 1
}

fatal_restart() {
  local reason="${1:-container watchdog triggered}"
  echo "${reason}" >&2
  kill -TERM "${LAUNCH_PID:-}" >/dev/null 2>&1 || true
  kill -TERM "$$" >/dev/null 2>&1 || exit 1
}

watchdog() {
  while true; do
    if [[ ! -S "${DISPLAY_SOCKET}" ]] || ! display_is_live; then
      fatal_restart "X display backend lost on ${DISPLAY}; restarting container"
    fi

    start_openbox || fatal_restart "Openbox failed to stay alive; restarting container"
    start_x11vnc || fatal_restart "x11vnc failed to stay alive; restarting container"
    start_novnc || fatal_restart "websockify failed to stay alive; restarting container"
    sleep 5
  done
}

prepare_display
brand_novnc
start_xvfb

start_openbox
start_x11vnc
start_novnc

cd "${ROS_WS}"
set +u
. /opt/ros/jazzy/setup.sh
. "${ROS_WS}/install/setup.sh"
set -u

ros2 launch mujoco_amr_sim full_stack.launch.py \
  use_viewer:="${USE_VIEWER}" \
  show_overview_window:="${SHOW_OVERVIEW_WINDOW}" \
  show_status_window:="${SHOW_STATUS_WINDOW}" \
  sim_rate_hz:="${SIM_RATE_HZ}" \
  publish_rate_hz:="${PUBLISH_RATE_HZ}" \
  render_rate_hz:="${RENDER_RATE_HZ}" \
  real_time_factor:="${REAL_TIME_FACTOR}" \
  display_env:="${DISPLAY}" \
  wayland_display_env:="disabled" \
  xdg_runtime_dir_env:="/tmp" \
  pulse_server_env:="disabled" &
LAUNCH_PID=$!

focus_main_window &
watchdog &
WATCHDOG_PID=$!

wait "${LAUNCH_PID}"
