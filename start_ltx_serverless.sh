#!/usr/bin/env bash
set -euo pipefail

python /usr/local/bin/bootstrap_ltx_models.py
exec /start.sh

