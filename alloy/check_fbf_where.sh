#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALS="/work/gfql_fbf_where.als"
IMAGE="${ALLOY_IMAGE:-ghcr.io/graphistry/alloy6:6.2.0}"
LOCAL_FALLBACK_IMAGE="${ALLOY_FALLBACK_IMAGE:-local/alloy6:latest}"
FULL=${FULL:-0}
MULTI=${MULTI:-0}
PUSH=${ALLOY_PUSH:-0}

# Resolve image: pull ghcr if possible, otherwise build local; optionally push built image to ghcr for caching
resolve_image() {
  local img="$IMAGE"
  if docker image inspect "$img" >/dev/null 2>&1; then
    IMAGE="$img"
    return
  fi

  if docker pull "$img" >/dev/null 2>&1; then
    IMAGE="$img"
    return
  fi

  # Fall back to local build
  if ! docker image inspect "$LOCAL_FALLBACK_IMAGE" >/dev/null 2>&1; then
    docker build -t "$LOCAL_FALLBACK_IMAGE" "$HERE"
  fi

  # Optionally publish to ghcr for future pulls
  if [ "$PUSH" = "1" ]; then
    docker tag "$LOCAL_FALLBACK_IMAGE" "$img"
    docker push "$img" || true
    IMAGE="$img"
  else
    IMAGE="$LOCAL_FALLBACK_IMAGE"
  fi
}

resolve_image

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
