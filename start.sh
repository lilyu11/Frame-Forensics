#!/usr/bin/env bash
# FrameForensics launcher (macOS / Linux): run `./start.sh` (macOS: or double-click start.command).
# First run creates frameforensics-backend/.venv and installs dependencies (a few minutes).
set -euo pipefail
cd "$(dirname "$0")/frameforensics-backend"

VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "Looking for Python 3.10 - 3.13 ..."
    PYCMD=""
    for cand in python3.12 python3.11 python3.10 python3.13 python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c \
            "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" 2>/dev/null; then
            PYCMD="$cand"
            break
        fi
    done
    if [ -z "$PYCMD" ]; then
        echo "[ERROR] Python 3.10 - 3.13 was not found."
        echo "Install Python 3.12 from https://www.python.org/downloads/ and run this script again."
        exit 1
    fi
    echo "Creating virtual environment with: $PYCMD"
    "$PYCMD" -m venv .venv
fi

# Reinstall only when requirements-app.txt changed since the last install.
if ! cmp -s requirements-app.txt .venv/installed-requirements.txt; then
    echo "Installing dependencies, this can take a few minutes the first time ..."
    "$VENV_PY" -m pip install --upgrade pip --quiet --disable-pip-version-check
    "$VENV_PY" -m pip install -r requirements-app.txt --disable-pip-version-check
    cp requirements-app.txt .venv/installed-requirements.txt
fi

echo
echo "Starting FrameForensics at http://127.0.0.1:7860  (Ctrl+C to stop)"
echo
exec "$VENV_PY" app/app.py
