#!/bin/bash

set -euo pipefail

APP_DIR="/Users/mcummins/Library/CloudStorage/Dropbox/coding_projects/biomarker_studio"
PYTHON_BIN="$APP_DIR/.venv/bin/python3"
APP_FILE="$APP_DIR/app.py"
URL="http://localhost:8501"
LOG_DIR="$APP_DIR/.launcher"
LOG_FILE="$LOG_DIR/streamlit.log"
PID_FILE="$LOG_DIR/streamlit.pid"
ARCH_BIN="/usr/bin/arch"
VERSION_FILE="$LOG_DIR/launcher-version"
LAUNCHER_VERSION="5"

mkdir -p "$LOG_DIR"

is_pid_running() {
    local pid="$1"
    if [[ -z "$pid" ]]; then
        return 1
    fi

    /bin/kill -0 "$pid" >/dev/null 2>&1
}

stop_pid() {
    local pid="$1"
    if [[ -z "$pid" ]]; then
        return 0
    fi

    /bin/kill "$pid" >/dev/null 2>&1 || true

    for _ in $(/usr/bin/seq 1 10); do
        if ! is_pid_running "$pid"; then
            return 0
        fi
        /bin/sleep 1
    done

    /bin/kill -9 "$pid" >/dev/null 2>&1 || true
}

server_responding() {
    /usr/bin/curl --silent --head --max-time 2 "$URL" >/dev/null 2>&1
}

open_app_url() {
    /usr/bin/open "$URL" >/dev/null 2>&1 || /usr/bin/osascript -e "open location \"$URL\"" >/dev/null 2>&1
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    /usr/bin/osascript -e 'display alert "Biomarker Studio" message "Could not find .venv/bin/python3. Recreate the virtual environment or update the launcher." as critical'
    exit 1
fi

if [[ ! -x "$ARCH_BIN" ]]; then
    /usr/bin/osascript -e 'display alert "Biomarker Studio" message "Could not find /usr/bin/arch, which is needed to start Python in Apple Silicon mode." as critical'
    exit 1
fi

if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE")"
    if ! is_pid_running "$existing_pid"; then
        /bin/rm -f "$PID_FILE"
    fi
fi

if [[ -f "$PID_FILE" ]]; then
    existing_version=""
    if [[ -f "$VERSION_FILE" ]]; then
        existing_version="$(cat "$VERSION_FILE")"
    fi

    if [[ "$existing_version" != "$LAUNCHER_VERSION" ]]; then
        stop_pid "$(cat "$PID_FILE")"
        /bin/rm -f "$PID_FILE"
    fi
fi

if [[ -f "$PID_FILE" ]]; then
    open_app_url
    exit 0
fi

(
    cd "$APP_DIR"
    nohup "$ARCH_BIN" -arm64 "$PYTHON_BIN" -m streamlit run "$APP_FILE" --server.address 127.0.0.1 >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "$LAUNCHER_VERSION" >"$VERSION_FILE"
) >/dev/null 2>&1

new_pid="$(cat "$PID_FILE")"
/bin/sleep 2

if ! is_pid_running "$new_pid"; then
    /bin/rm -f "$PID_FILE"
    /usr/bin/osascript -e 'display alert "Biomarker Studio" message "Streamlit exited before it finished starting. Check .launcher/streamlit.log in the project folder for details." as critical'
    exit 1
fi

for _ in $(/usr/bin/seq 1 30); do
    if server_responding; then
        open_app_url
        exit 0
    fi
    /bin/sleep 1
done

open_app_url
exit 0
