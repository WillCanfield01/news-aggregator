#!/usr/bin/env bash
set -euo pipefail

echo "== Refreshing site =="
curl -fsS https://news-aggregator-vmz9.onrender.com/refresh

echo "== Running trend bot =="
cd /opt/render/project/src
export PYTHONPATH=/opt/render/project/src
/opt/render/project/src/.venv/bin/python -m app.trend_hijack_x