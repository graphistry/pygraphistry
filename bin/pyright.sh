#!/bin/bash
set -euo pipefail

# Run pyright over graphistry/ at the version the ratchet baseline was built against.
#
# The pin is load-bearing: bin/ci_pyright_baseline.json holds per-file finding counts,
# and upstream adds and retunes rules between releases, so an unpinned tool would move
# the baseline under us. Bump PYRIGHT_VERSION and --update-baseline together.
#
#   ./bin/pyright.sh                       # check graphistry/
#   ./bin/pyright.sh --outputjson          # machine-readable, used by ci_pyright_guard.py
#   ./bin/pyright.sh graphistry/compute    # narrow the target
#   PYRIGHT_EXTRA_ARGS="--stats" ./bin/pyright.sh
#
# See DEVELOP.md "Pyright ratchet".

PYRIGHT_VERSION="${PYRIGHT_VERSION:-1.1.414}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Prefer an already-installed pyright, but only when it is the pinned version:
# a mismatched local install silently produces a different finding set.
PYRIGHT_CMD=()
if command -v pyright >/dev/null 2>&1; then
  LOCAL_VERSION="$(pyright --version 2>/dev/null | awk '{print $2}' || true)"
  if [ "${LOCAL_VERSION:-}" = "$PYRIGHT_VERSION" ]; then
    PYRIGHT_CMD=(pyright)
  else
    echo "bin/pyright.sh: local pyright is ${LOCAL_VERSION:-unknown}, need ${PYRIGHT_VERSION}; fetching the pinned build instead." >&2
  fi
fi
if [ "${#PYRIGHT_CMD[@]}" -eq 0 ]; then
  if command -v uvx >/dev/null 2>&1; then
    PYRIGHT_CMD=(uvx --from "pyright==${PYRIGHT_VERSION}" pyright)
  elif command -v npx >/dev/null 2>&1; then
    PYRIGHT_CMD=(npx -y "pyright@${PYRIGHT_VERSION}")
  else
    echo "bin/pyright.sh: pyright ${PYRIGHT_VERSION} not found, and neither uvx nor npx is available to fetch it." >&2
    exit 1
  fi
fi

# --version answers about the tool, not the tree, so it takes no config or targets.
if [ "${1:-}" = "--version" ]; then
  exec "${PYRIGHT_CMD[@]}" --version
fi

EXTRA_ARGS=()
if [ -n "${PYRIGHT_EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA_ARGS <<< "$PYRIGHT_EXTRA_ARGS"
fi

ARGS=("$@")
HAS_TARGET=0
for arg in ${ARGS[@]+"${ARGS[@]}"}; do
  case "$arg" in
    -*) ;;
    *) HAS_TARGET=1 ;;
  esac
done
if [ "$HAS_TARGET" -eq 0 ]; then
  ARGS+=(graphistry)
fi

exec "${PYRIGHT_CMD[@]}" -p pyrightconfig.json \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  ${ARGS[@]+"${ARGS[@]}"}
