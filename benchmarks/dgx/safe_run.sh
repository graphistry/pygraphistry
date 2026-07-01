#!/bin/bash
# Enforced safety wrapper for dgx GB10 GPU/big runs. PROVEN (2026-07-01):
#  - docker --memory is TRANSPARENT to cudf/unified allocs (A2) -> useless.
#  - RMM LimitingResourceAdaptor caps cudf AND cugraph cleanly (A3/A3b) -> the real cap,
#    injected non-invasively via sitecustomize + GFQL_RMM_LIMIT_GB.
#  - preflight refuses obviously-oversized runs; host watchdog force-kills on low RAM;
#    hard timeout + docker kill -s KILL so a hung container can't wedge the box.
# Usage:
#   safe_run.sh --name N --est-edges E [--rmm-gb 80] [--floor-gb 20] [--timeout 3600] \
#     [--pythonpath /opt/pygraphistry] -- <extra docker args e.g. -v ...> IMAGE <cmd...>
set -uo pipefail
NAME="dgxsafe_$$"; EST_EDGES=0; RMM_GB=80; FLOOR_GB=20; TIMEOUT=3600; PYPATH="/opt/pygraphistry"
GUARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ $# -gt 0 ]]; do case "$1" in
  --name) NAME="$2"; shift 2;; --est-edges) EST_EDGES="$2"; shift 2;;
  --rmm-gb) RMM_GB="$2"; shift 2;; --floor-gb) FLOOR_GB="$2"; shift 2;;
  --timeout) TIMEOUT="$2"; shift 2;; --pythonpath) PYPATH="$2"; shift 2;;
  --) shift; break;; *) echo "unknown arg $1"; exit 2;; esac; done

if [[ "$EST_EDGES" -gt 0 ]]; then
  if ! python3 "$GUARD_DIR/preflight.py" "$EST_EDGES"; then
    echo "[safe_run] REFUSED: estimated peak > budget for $EST_EDGES edges."; exit 3; fi
fi
echo "[safe_run] name=$NAME rmm=${RMM_GB}GB floor=${FLOOR_GB}GB timeout=${TIMEOUT}s"

( while true; do
    avail=$(free -g | awk '/Mem:/{print $7}')
    echo "[watchdog $(date +%H:%M:%S)] host avail=${avail}GB"
    if [[ "${avail:-999}" -lt "$FLOOR_GB" ]]; then
      echo "[watchdog] avail ${avail}GB < floor ${FLOOR_GB}GB -> KILL $NAME"
      docker kill -s KILL "$NAME" >/dev/null 2>&1; break; fi
    sleep 5
  done ) & WD=$!

timeout -s KILL "$TIMEOUT" docker run --rm --name "$NAME" --gpus all \
  -e GFQL_RMM_LIMIT_GB="$RMM_GB" -e PYTHONPATH="/dgx-guard:${PYPATH}" \
  -v "$GUARD_DIR:/dgx-guard:ro" "$@"
rc=$?
docker kill -s KILL "$NAME" >/dev/null 2>&1; kill "$WD" >/dev/null 2>&1
echo "[safe_run] exit=$rc"
exit $rc
