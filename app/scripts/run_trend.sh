#!/usr/bin/env bash
set -euo pipefail

echo "== Refreshing site =="
curl -fsS https://news-aggregator-vmz9.onrender.com/refresh

echo "== Running trend bot =="
cd /opt/render/project/src
export PYTHONPATH=/opt/render/project/src

# use the job's venv python
if command -v python >/dev/null 2>&1; then
  PY=python
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PY="$VIRTUAL_ENV/bin/python"
else
  PY="/opt/render/project/src/.venv/bin/python"  # last resort
fi

$PY -m app.trend_hijack_x