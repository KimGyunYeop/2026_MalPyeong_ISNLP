#!/usr/bin/env bash
set -euo pipefail

exec python -m uvicorn main_code_relonation.serve:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info
