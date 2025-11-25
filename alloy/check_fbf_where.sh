#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALS="/work/gfql_fbf_where.als"
IMAGE="local/alloy6:latest"
FULL=${FULL:-0}
MULTI=${MULTI:-0}

# Build image if missing
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" "$HERE"
fi

if [ "$FULL" = "1" ]; then
  docker run --rm -v "$HERE":/work "$IMAGE" exec -c SpecNoWhereEqAlgoNoWhere -o - "$ALS"
  docker run --rm -v "$HERE":/work "$IMAGE" exec -c SpecWhereEqAlgoLowered -o - "$ALS"
else
  docker run --rm -v "$HERE":/work "$IMAGE" exec -c SpecNoWhereEqAlgoNoWhereSmall -o - "$ALS"
  docker run --rm -v "$HERE":/work "$IMAGE" exec -c SpecWhereEqAlgoLoweredSmall -o - "$ALS"
fi

# Scenario coverage + additional scopes (fixed small scopes inside .als)
for ASSERT in SpecNoWhereEqAlgoNoWhereMultiChain SpecWhereEqAlgoLoweredMultiChain SpecWhereEqAlgoLoweredFan SpecWhereEqAlgoLoweredCycle SpecWhereEqAlgoLoweredParallel SpecWhereEqAlgoLoweredDisconnected SpecWhereEqAlgoLoweredAliasWhere SpecWhereEqAlgoLoweredMixedWhere SpecWhereEqAlgoLoweredFilterMix; do
  docker run --rm -v "$HERE":/work "$IMAGE" exec -c "$ASSERT" -o - "$ALS"
done

if [ "$MULTI" = "1" ]; then
  docker run --rm -v "$HERE":/work "$IMAGE" exec -c SpecNoWhereEqAlgoNoWhereMultiChainFull -o - "$ALS"
  docker run --rm -v "$HERE":/work "$IMAGE" exec -c SpecWhereEqAlgoLoweredMultiChainFull -o - "$ALS"
fi
