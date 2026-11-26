#!/bin/bash
# One command to check a GFQL change before you push: lint, types, and the GFQL tests on pandas and Polars (CPU).
#
#   ./bin/check-gfql.sh                 # lint + types + GFQL tests (pandas) + the Polars lane
#   ./bin/check-gfql.sh --changed       # only the test files you added or changed (fast loop; lint and types still run)
#   ./bin/check-gfql.sh --cudf          # also run the GFQL tests on cuDF (needs a GPU and RAPIDS)
#   ./bin/check-gfql.sh --polars-gpu    # also run them on polars-gpu (needs a GPU, RAPIDS and cudf-polars)
#   ./bin/check-gfql.sh --no-lint       # skip lint and types (keep the tests)
#   ./bin/check-gfql.sh --dry-run       # print what would run, run nothing
#   ./bin/check-gfql.sh -- -k name      # everything after -- goes to the pytest steps
#
# Setup once: pip install -e ".[test,polars]"   (GPU engines: see docs/contributing/gfql-gpu-colab.md)
# Non-zero exit code if any step fails; the summary names the first failing step.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

WITH_LINT=1
CHANGED=0
CUDF=0
POLARS_GPU=0
DRY=0
PYTEST_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --changed) CHANGED=1 ;;
    --cudf) CUDF=1 ;;
    --polars-gpu) POLARS_GPU=1 ;;
    --no-lint) WITH_LINT=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; PYTEST_ARGS=("$@"); break ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python; else PYTHON_BIN=python3; fi
fi

missing=()
for mod in pytest pandas; do
  "$PYTHON_BIN" -c "import $mod" >/dev/null 2>&1 || missing+=("$mod")
done
"$PYTHON_BIN" -c "import polars" >/dev/null 2>&1 || missing+=("polars")
if [ ${#missing[@]} -gt 0 ] && [ "$DRY" = 0 ]; then
  echo "missing in $PYTHON_BIN: ${missing[*]}. Install with: pip install -e \".[test,polars]\"" >&2
  exit 2
fi

# Test files to run on the pandas step: the changed ones, or the GFQL test tree.
GFQL_TESTS=(graphistry/tests/compute)
if [ "$CHANGED" = 1 ]; then
  base="$(git merge-base HEAD origin/develop 2>/dev/null || git merge-base HEAD develop 2>/dev/null || true)"
  if [ -n "$base" ]; then
    changed_tests=()
    while IFS= read -r f; do
      [ -f "$f" ] && changed_tests+=("$f")
    done < <(git diff --name-only --diff-filter=AM "$base" -- 'graphistry/tests/**/test_*.py'; git ls-files --others --exclude-standard -- 'graphistry/tests/**/test_*.py')
    if [ ${#changed_tests[@]} -gt 0 ]; then GFQL_TESTS=("${changed_tests[@]}"); else echo "no changed test files found; running the full GFQL tests"; fi
  else
    echo "no develop branch to compare with; running the full GFQL tests"
  fi
fi

FAILED=()
SUMMARY=()
run_step() {  # name, command...
  local name="$1"; shift
  echo
  echo "=== $name: $*"
  if [ "$DRY" = 1 ]; then SUMMARY+=("dry   $name"); return 0; fi
  "$@"
  local rc=$?
  if [ "$rc" = 0 ]; then
    SUMMARY+=("ok    $name")
  elif [ "$rc" = 5 ] && [ "${NONE_OK:-0}" = 1 ]; then  # pytest steps only: no tests collected (e.g. a -k filter that matches nothing in this step)
    SUMMARY+=("none  $name (no tests matched)")
  else
    SUMMARY+=("FAIL  $name"); FAILED+=("$name")
  fi
}

if [ "$WITH_LINT" = 1 ]; then
  run_step "lint" ./bin/lint.sh
  run_step "types" ./bin/typecheck.sh
fi
NONE_OK=1 run_step "pandas" "$PYTHON_BIN" -B -m pytest -q -n auto "${GFQL_TESTS[@]}" "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
NONE_OK=1 run_step "polars" ./bin/test-polars.sh "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
if [ "$CUDF" = 1 ]; then
  NONE_OK=1 run_step "cudf" env TEST_CUDF=1 "$PYTHON_BIN" -B -m pytest -q -n 2 "${GFQL_TESTS[@]}" "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
fi
if [ "$POLARS_GPU" = 1 ]; then
  NONE_OK=1 run_step "polars-gpu" env TEST_POLARS_GPU=1 "$PYTHON_BIN" -B -m pytest -q -n 2 "${GFQL_TESTS[@]}" "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
fi

echo
echo "=== summary"
printf '%s\n' "${SUMMARY[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "first failing step: ${FAILED[0]}"
  exit 1
fi
