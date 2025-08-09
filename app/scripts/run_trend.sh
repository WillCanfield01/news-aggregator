#!/usr/bin/env bash
set -euo pipefail

echo "== Refreshing site =="
curl -fsS https://news-aggregator-vmz9.onrender.com/refresh

echo "== Running trend bot =="

# Always operate from repo root on Render
cd /opt/render/project/src

# Use the job's python on PATH (Render puts the venv there)
PY=python

# Sanity info (shows which interpreter is used)
which $PY || true
$PY -V || true

# Ensure deps are present
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt --no-cache-dir

# Run the bot
export PYTHONPATH=/opt/render/project/src
$PY -m app.trend_hijack_x