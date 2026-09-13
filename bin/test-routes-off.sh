#!/usr/bin/env bash
# Replay the GFQL suites with each hot path declined (GFQL_ROUTES_OFF, see graphistry/tests/conftest.py).
# Engagement pins carry the route_engaged marker and are skipped, so every remaining failure is a
# route-vs-general result divergence. Fail on any failed replay; retain per-mode logs and id lists.
set -uo pipefail
cd "$(dirname "$0")/.."
# One registry drives forcing and replay: new routes automatically join all-off.
ALL_ROUTES=$(python -c 'from graphistry.tests.compute.gfql.routes.switch import ROUTES; print(",".join(ROUTES))') || exit $?
MODES=${MODES:-${ALL_ROUTES//,/ } all-off}
SUITES=${SUITES:-graphistry/tests/compute/test_chain.py graphistry/tests/compute/test_hop.py graphistry/tests/compute/test_gfql.py graphistry/tests/compute/gfql}
OUT=${OUT:-build/routes-off}
mkdir -p "$OUT"
status=0
for mode in $MODES; do
  if [ "$mode" = all-off ]; then routes=$ALL_ROUTES; else routes=$mode; fi
  GFQL_ROUTES_OFF=$routes python -m pytest $SUITES -q -p no:cacheprovider -o addopts="" -rfE > "$OUT/$mode.log" 2>&1 || status=1
  grep -E "^(FAILED|ERROR) " "$OUT/$mode.log" | sed 's/ - .*//' | sort -u > "$OUT/$mode.divergences"
  echo "$mode: $(wc -l < "$OUT/$mode.divergences") divergence id(s); $(tail -1 "$OUT/$mode.log")"
done
exit "$status"
