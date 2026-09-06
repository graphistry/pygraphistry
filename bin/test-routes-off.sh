#!/usr/bin/env bash
# Replay the GFQL suites with each hot path declined (GFQL_ROUTES_OFF, see graphistry/tests/conftest.py).
# Engagement pins carry the route_engaged marker and are skipped, so every remaining failure is a
# route-vs-general result divergence. Non-blocking ledger: always exits 0; per-mode logs + id lists in $OUT.
set -uo pipefail
cd "$(dirname "$0")/.."
MODES=${MODES:-native-fast polars-seeded polars-plain index-hop indexed-kernel cypher-fast all-off}
SUITES=${SUITES:-graphistry/tests/compute/test_chain.py graphistry/tests/compute/test_hop.py graphistry/tests/compute/test_gfql.py graphistry/tests/compute/gfql}
OUT=${OUT:-build/routes-off}
mkdir -p "$OUT"
for mode in $MODES; do
  if [ "$mode" = all-off ]; then routes=native-fast,polars-seeded,polars-plain,index-hop,indexed-kernel,cypher-fast; else routes=$mode; fi
  GFQL_ROUTES_OFF=$routes python -m pytest $SUITES -q -p no:cacheprovider -o addopts="" -rfE > "$OUT/$mode.log" 2>&1
  grep -E "^(FAILED|ERROR) " "$OUT/$mode.log" | sed 's/ - .*//' | sort -u > "$OUT/$mode.divergences"
  echo "$mode: $(wc -l < "$OUT/$mode.divergences") divergence id(s); $(tail -1 "$OUT/$mode.log")"
done
exit 0
