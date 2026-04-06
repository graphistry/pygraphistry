# DGX-Spark GPU Testing Guide

How to run GFQL/cuDF tests on the shared `dgx-spark` GPU machine.

**Load when**: validating a fix or feature that touches GFQL, cuDF, or any
code path exercised by the RAPIDS container before merging to master.

---

## Prerequisites

- SSH access: `ssh dgx-spark` (config in `~/.ssh/config`)
- DGX repo clone at `~/repos/pygraphistry` (RAPIDS image already built)
- Local changes not yet pushed: sync via `cat >` or tar pipe (see below)

---

## The canonical test script

`docker/test-rapids-official-local.sh` builds and runs an official RAPIDS
image with `--gpus all`.  Key env-var defaults:

| Variable | Default | Meaning |
|---|---|---|
| `RAPIDS_VERSION` | `26.02` | RAPIDS release tag |
| `PROFILE` | `basic` | Test profile (`basic`, `gfql`, `ai`) |
| `WITH_GPU` | `1` | Pass `--gpus all` to docker run |
| `WITH_IMAGE_BUILD` | `1` | Rebuild image before run |
| `TEST_FILES` | profile default | Space-separated pytest targets |
| `WITH_LINT` | `0` | Run ruff |
| `WITH_TYPECHECK` | `0` | Run mypy |

The script **volume-mounts `graphistry/` read-only** at runtime, so you only
need to sync changed source files — no full image rebuild required for most
iterations.

---

## Sync changed files to DGX (no git push needed)

```bash
# Single file
ssh dgx-spark 'cat > ~/repos/pygraphistry/graphistry/compute/gfql_unified.py' \
    < graphistry/compute/gfql_unified.py

# Multiple files at once (tar pipe — fast)
tar czf - graphistry/compute/gfql_unified.py \
         graphistry/compute/gfql/cypher/lowering.py \
         graphistry/tests/compute/gfql/cypher/test_lowering.py \
    | ssh dgx-spark 'bash -lc "cd ~/repos/pygraphistry && tar xzf -"'

# All source files changed vs a given commit (e.g. master)
git diff <base-sha>..HEAD --name-only -- 'graphistry/*.py' 'graphistry/**/*.py' \
    | grep '^graphistry/' | grep -v '^graphistry/tests/' \
    | xargs tar czf - \
    | ssh dgx-spark 'bash -lc "cd ~/repos/pygraphistry && tar xzf -"'
```

> **Note**: sync ALL files your changed module imports from, not just the
> file you edited. If the DGX repo is on an old branch (check with
> `ssh dgx-spark 'bash -lc "cd ~/repos/pygraphistry && git log --oneline -1"'`),
> you may need to sync more files than just your diff.

---

## Running tests

### Fast iteration — skip image rebuild

```bash
ssh dgx-spark 'bash -lc "
  cd ~/repos/pygraphistry/docker &&
  WITH_IMAGE_BUILD=0 WITH_GPU=1 WITH_LINT=0 WITH_TYPECHECK=0 WITH_TEST=1 \
  PROFILE=gfql \
  TEST_FILES=\"graphistry/tests/compute/gfql/cypher/test_lowering.py::my_test\" \
  ./test-rapids-official-local.sh
"'
```

### Full GFQL profile (default gfql test set)

```bash
ssh dgx-spark 'bash -lc "
  cd ~/repos/pygraphistry/docker &&
  WITH_IMAGE_BUILD=0 WITH_GPU=1 WITH_LINT=0 WITH_TYPECHECK=0 WITH_TEST=1 \
  PROFILE=gfql \
  ./test-rapids-official-local.sh
"'
```

Default `gfql` profile tests (with `WITH_GPU=1`):
- `test_parser.py`
- `test_row_pipeline_ops.py`
- `test_lowering.py::test_graph_constructor_cudf_support`
- `test_lowering.py::test_string_cypher_formats_filtered_edge_entity_projection_on_cudf`
- `test_lowering.py::test_string_cypher_executes_real_cugraph_node_row_call_on_cudf`

### Add issue-specific tests alongside the profile defaults

```bash
ssh dgx-spark 'bash -lc "
  cd ~/repos/pygraphistry/docker &&
  WITH_IMAGE_BUILD=0 WITH_GPU=1 WITH_LINT=0 WITH_TYPECHECK=0 WITH_TEST=1 \
  PROFILE=gfql \
  TEST_FILES=\"\
    graphistry/tests/compute/gfql/cypher/test_lowering.py::test_graph_constructor_cudf_support \
    graphistry/tests/compute/gfql/cypher/test_lowering.py::test_my_new_test \
  \" \
  ./test-rapids-official-local.sh
"'
```

### Rebuild image (needed after dependency changes)

```bash
ssh dgx-spark 'bash -lc "
  cd ~/repos/pygraphistry/docker &&
  WITH_IMAGE_BUILD=1 WITH_GPU=1 WITH_LINT=0 WITH_TYPECHECK=0 WITH_TEST=1 \
  PROFILE=gfql \
  ./test-rapids-official-local.sh
"'
```

---

## Available RAPIDS image tags on DGX

```bash
ssh dgx-spark 'bash -lc "docker images | grep test-rapids"'
```

Current canonical tag: `graphistry/test-rapids-official:26.02-gfql`

---

## Checking DGX repo state

```bash
# Current branch and last commit
ssh dgx-spark 'bash -lc "cd ~/repos/pygraphistry && git log --oneline -3 && git branch --show-current"'

# Diff vs a file you just synced
ssh dgx-spark 'bash -lc "cd ~/repos/pygraphistry && git diff -- graphistry/compute/gfql_unified.py"'
```

---

## Related files

- `docker/test-rapids-official-local.sh` — main entry point
- `docker/test-rapids-official.Dockerfile` — image definition
- `docker/test-rapids-official-matrix.sh` — matrix runner (multiple RAPIDS versions)
- `plans/dgx-spark-bootstrap/plan.md` — full environment bootstrap history
