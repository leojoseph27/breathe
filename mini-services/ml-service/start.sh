#!/usr/bin/env bash
# Start the breathe ML mini-service on port 5001.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv at .venv ..."
    /home/z/.venv/bin/python3 -m venv .venv
fi
source .venv/bin/activate
python app.py
