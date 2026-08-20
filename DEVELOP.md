# Development Setup

See also [CONTRIBUTING.md](CONTRIBUTING.md) and [ARCHITECTURE.md](ARCHITECTURE.md)

Development is setup for local native and containerized Python coding & testing, and with automatic GitHub Actions for CI + CD. The server tests are like the local ones, except against a wider test matrix of environments.

## LFS

We are starting to use git lfs for data:

```bash
# install git lfs: os-specific commands below
git lfs install
git lfs checkout
```

### git lfs: ubuntu

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

## Docker

### Install

```bash
cd docker && docker compose build && docker compose up -d
```

For just CPU tests, you can focus on `test-cpu` and use the run instructions below:

```bash
cd docker && docker compose build test-cpu
```


### Run local tests without rebuild

Containerized call to `pytest` for CPU + GPU modes:

```bash
cd docker

# cpu - pandas
./test-cpu-local.sh

# cpu - fast & targeted
WITH_LINT=0 WITH_TYPECHECK=0 WITH_BUILD=0 ./test-cpu-local.sh graphistry/tests/test_hyper_dask.py::TestHypergraphPandas::test_hyper_to_pa_mixed2

# gpu - pandas, cudf, dask, dask_cudf; test only one file
./test-gpu-local.sh graphistry/tests/test_hyper_dask.py
```

Connector tests (currently neo4j-only): `cd docker && WITH_NEO4J=1 ./test-cpu-local.sh` (optional `WITH_SUDO=" "`)

* Will start a local neo4j (docker) then enable+run tests against it

Remote Graphistry integration tests are opt-in because they require a live server and credentials:

```bash
TEST_REMOTE_INTEGRATION=1 \
GRAPHISTRY_API_TOKEN=<jwt> \
python -m pytest graphistry/tests/compute/test_chain_let_remote_integration.py
```

Use `GRAPHISTRY_USERNAME`/`GRAPHISTRY_PASSWORD` instead of `GRAPHISTRY_API_TOKEN` when token auth is not available. For service-account style authentication in application code, prefer `personal_key_id` + `personal_key_secret`. Optional env vars: `GRAPHISTRY_SERVER` and `GRAPHISTRY_TEST_DATASET_ID`.


## Docs

Automatically build via ReadTheDocs from inline definitions.

To manually build, see `docs/`.

## Ignore files

You may need to add ignore rules:

* ruff: pyproject.toml (or bin/lint.sh)
* mypi: mypi.ini
* sphinx: docs/source/conf.py

## Remote

Some databases like Neptune can be easier via cloud editing, especially within Jupyter:

```bash
git clone https://github.com/graphistry/pygraphistry.git
git checkout origin/my_branch
pip install --user -e .
git diff
```

and

```python
import logging
logging.basicConfig(level=logging.DEBUG)

import graphistry
graphistry.__version__
```

## CI

GitHub Actions: See `.github/workflows`

CI runs on every PR and updates them

### Cypher Surface Growth Guard

CI includes `cypher-frontend-surface-guard`, which enforces bounded growth for:

- `graphistry/compute/gfql/cypher/lowering.py` total line count
- `CompiledCypherQuery`, `CompiledGraphBinding`, `CompiledCypherGraphQuery` dataclass field/property counts

Guard implementation + baseline:

- Script: `bin/ci_cypher_surface_guard.py`
- Baseline: `bin/ci_cypher_surface_guard_baseline.json`

If growth is intentional, regenerate baseline in your branch and include explicit PR rationale:

```bash
python bin/ci_cypher_surface_guard.py --write-baseline
```

Then commit both code changes and baseline update together.

### Type Hygiene Guard

`bin/lint.sh` (run by the `python-lint-types` matrix on py3.8-3.14) runs
`bin/ci_type_hygiene_guard.py`, a stdlib-only AST check over `graphistry/`
(tests excluded, matching `mypy.ini`). It exists to catch the defect classes
that keep coming back in code review, so a reviewer does not have to.

| Check | What it flags |
|---|---|
| `missing-annotations` | a parameter or return without a type annotation (same ground as ruff `ANN001/002/003/201/202/204/205/206`) |
| `explicit-any` | `Any` anywhere inside an annotation |
| `explicit-cast` | a `cast(...)` call |
| `bare-generic` | unsubscripted `list` / `dict` / `List` / `Dict` / ... in an annotation |
| `plottable-setattr` | `setattr()` onto a parameter annotated as a `Plottable` |
| `plottable-attr-write` | `param.attr = ...` onto a parameter annotated as a `Plottable` |
| `vocab-str-param` | a closed-vocabulary parameter (`table`, `kind`, `direction`, `how`, `mode`, `engine`) annotated as plain `str` |

Enforcement is a **per-file count ratchet** against
`bin/ci_type_hygiene_baseline.json`: a file may not gain findings, and a file
absent from the baseline must have zero. Existing debt is grandfathered, new and
moved code is not.

```bash
./bin/ci_type_hygiene_guard.py             # what CI runs
./bin/ci_type_hygiene_guard.py --report    # totals per check
./bin/ci_type_hygiene_guard.py --list plottable-setattr
./bin/ci_type_hygiene_guard.py --strict    # show files that improved; time to retighten
```

When a finding is genuinely correct, annotate that line and say why:

```python
setattr(res, f"_{kind}_dbscan", dbscan)  # hygiene-ok: plottable-setattr -- res is a fresh copy
```

Do **not** raise a cap with `--update-baseline` to make a new finding go away.
Lowering caps after fixing debt is the intended use; commit the code change and
the baseline update together.

### Comment Density Guard

`bin/lint.sh` (the same `python-lint-types` matrix lane as the type-hygiene
guard) runs `bin/ci_comment_density_guard.py`, a stdlib-only `tokenize` + `ast`
check over `graphistry/`. It enforces the "Encoding: names, tests, and
structure — not prose" rules in `agents/skills/review/SKILL.md`, which were the
last rule class on that stack still enforced only by human review.

| Check | What it flags |
|---|---|
| `comment-block` | a run of 2+ adjacent full-line `#` comments (3+ for a Sphinx `#:` run) |
| `perf-claim` | performance / complexity / benchmark vocabulary in a comment or docstring |
| `issue-rationale` | a standalone comment or a docstring citing `#<issue>` as the explanation |

`comment-block` is a form rule and reads `#` comments only. `perf-claim` and
`issue-rationale` are content rules — the claim does not become admissible by
moving into a docstring — so they read docstrings too. Tests are exempt from
`comment-block` and `issue-rationale` (a test may explain its oracle) but not
from `perf-claim`: measurement belongs in pyg-bench wherever it is written.

The line limit is 127, so a constraint that genuinely cannot be expressed by a
name or a test fits on one line. `regress` and `A/B` count only next to
performance vocabulary (they also name correctness concepts), and a comment that
points at `pyg-bench` is a pointer to the measurement rather than a claim.

Enforcement is a **per-file count ratchet** against
`bin/ci_comment_density_baseline.json`, exactly like the type-hygiene guard.

```bash
./bin/ci_comment_density_guard.py             # what CI runs
./bin/ci_comment_density_guard.py --report    # totals per check
./bin/ci_comment_density_guard.py --list comment-block
./bin/ci_comment_density_guard.py --strict    # show files that improved; time to retighten
```

The fix is almost never a suppression: extract a helper whose NAME states the
rule, or write the test whose NAME states it. When a comment genuinely earns its
place and still cannot fit, annotate it:

```python
# guard-ok: comment-block -- openCypher 9.1 §4.2 wording, quoted verbatim
```

### GFQL Cache Registry

Every process-lifetime cache in `graphistry/compute/gfql/**` registers itself in
`graphistry/compute/gfql/cache_registry.py`, at its own definition site, as either
clearable (keyed by caller input; `gfql_clear_caches()` empties it) or an exempt
process singleton with a written reason. The module docstring is the spec;
`graphistry/tests/compute/gfql/test_clear_caches_covers_every_cache.py` fails CI on
any unregistered memo. Never clear a cache by name lookup -- registration hands
over the bound clear handle precisely because a name-based clear once shipped a
silent no-op and a wrong published benchmark number.

#### Conventions behind the checks

- **Never write to a caller's `Plottable`.** Caching by `setattr` keyed on
  `id()` leaked results across `gfql()` calls and returned stale answers after
  an in-place frame mutation (issue #1825). Return a new object, or thread a
  per-execution cache.
- **Prefer engine-agnostic `SeriesT` / `DataFrameT` plus a localized
  `# type: ignore`** over `Any` plus call-site `cast()`.
- **Parameterize generics.** `list[str]` needs `from __future__ import
  annotations` on py3.8 lanes; `List[str]` works without it. Both satisfy the
  check -- only the unsubscripted form is flagged.
- **A fixed vocabulary is a `Literal`, not a `str`.** A column name is legitimately
  `str`; an engine name, a `table=`/`kind=` of `'nodes'`/`'edges'`, a `how=`, or a
  `direction=` is not. Reuse an existing alias where one exists (e.g.
  `GraphEntityKind = Literal['nodes', 'edges']` in
  `graphistry/models/compute/features.py`) rather than declaring a new one.
  `vocab-str-param` only knows the six parameter names above -- it is a floor, not
  a full check; reviewers still own the general case.

Ruff additionally rejects `getattr(x, "const")` / `setattr(x, "const", v)`
(`B009`/`B010`) outright; today's offenders are grandfathered in
`pyproject.toml`'s `per-file-ignores` and are retired with
`ruff check --fix --select B009,B010 <file>`.

### GPU CI

**Today, no CI lane executes cuDF.** `ci.yml` never sets `TEST_CUDF` and no lane
installs `cudf`, and `ci-gpu.yml` is disabled: its jobs are gated on the
`GRAPHISTRY_ENABLE_GPU_PUBLIC` repository variable (unset), it needs the
`gpu_public` self-hosted runner, and a `gpu-disabled-guard` job hard-fails any
manual trigger. So a `TEST_CUDF=1` receipt is **developer-local evidence only** --
a cuDF-gated test can contradict the CPU contract, or rot outright, and stay green
on master indefinitely. Treat a GPU claim in a PR as unprotected until a GPU lane
exists: re-run it yourself rather than trusting the last receipt.

`bin/ci_gpu_gate_audit.py` (lane `gpu-gate-audit`) keeps the size of that gap
visible: it counts the cuDF gates, requires each to be attributable (a `reason=`
naming `TEST_CUDF`, so `pytest -rs` on a CPU lane names what was not run) and to
actually read the flag from the environment, and cross-checks this note against
whether any workflow sets `TEST_CUDF`. Wiring a real GPU lane retires the note;
deleting the note without wiring a lane fails the audit. The audit is static -- it
proves the gates are well formed, never that the gated assertions hold.

GPU CI can be manually triggered by core dev team members, once the lane is
re-enabled:

1. Push intended changes to protected branches `gpu-public` or `master`
2. Manually trigger action [ci-gpu](https://github.com/graphistry/pygraphistry/actions/workflows/ci-gpu.yml) on one of the above branches

GPU tests can also be run locally via `./docker/test-gpu-local.sh` , or directly
with `TEST_CUDF=1 pytest ...` on a RAPIDS-equipped box.

## Debugging Tips

* Use the unit tests
* use the `logging` module per-file


## Publish: Merge, Tag, & Upload

1. Update CHANGELOG.md in your PR branch
	- Convert `## [Development]` section to `## [X.Y.Z - YYYY-MM-DD]`
	- Document all changes following [Keep a Changelog](https://keepachangelog.com/) format
	- Commit and push to PR branch

1. Merge the PR to master (via GitHub UI or `gh pr merge`)

1. Switch to master and pull the merged changes
	```sh
	git checkout master
	git pull --ff-only origin master
	git status --short  # should be empty before tagging
	```

1. Tag the repository with the new version number (semantic versioning *X.Y.Z*)

	```sh
	git tag X.Y.Z
	git push origin refs/tags/X.Y.Z
	```

1. Confirm the [publish](https://github.com/graphistry/pygraphistry/actions?query=workflow%3A%22Publish+Python+%F0%9F%90%8D+distributions+%F0%9F%93%A6+to+PyPI+and+TestPyPI%22) Github Action published to [pypi](https://pypi.org/project/graphistry/)
	- Auto-triggers on tag push
	- **Expected gate**: on tag-triggered releases, the final `Publish distribution to PyPI` job can pause in `waiting` until a maintainer approves `Review deployments` for environment `pypi-release`.
	- If the run is waiting, open the run page and approve `Review deployments`, then wait for the PyPI job to complete.
	- If manually triggering (`workflow_dispatch`), choose `release_mode`:
	  - `evidence`: build + SBOM + provenance + evidence artifacts only (no publish)
	  - `test`: includes TestPyPI publish, skips PyPI (uses synthetic runner-local version `0.0.dev<run_id>` to avoid local-version upload rejection)
	  - `release`: TestPyPI + PyPI publish (restricted to `master`, with `pypi-release` approval)
	- Do not rerun publish for a version that is already on PyPI (duplicate-file uploads are rejected)
	- Verify version appears on PyPI: `curl -s https://pypi.org/pypi/graphistry/json | jq -r '.info.version'`
	- Verify release evidence artifacts from the workflow run:
	  - built distributions (`dist/*.whl`, `dist/*.tar.gz`)
	  - SBOM (`evidence/sbom-cyclonedx.json`)
	  - GitHub build provenance attestation for built distributions (`dist/*.whl`, `dist/*.tar.gz`)
	- Keep the PyPI Trusted Publisher binding aligned with this workflow:
	  - repository: `graphistry/pygraphistry`
	  - workflow file: `.github/workflows/publish-pypi.yml`
	  - environment: `pypi-release`
	  - refs: tag pushes and `workflow_dispatch` on `master` only
	- This workflow publishes with attestations enabled for both TestPyPI and PyPI.

1. Toggle version as active at [ReadTheDocs](https://readthedocs.org/projects/pygraphistry/versions/)

1. Create GitHub Release with detailed release notes

	```sh
	gh release create X.Y.Z --title "vX.Y.Z - Brief Title" --notes "Release notes in markdown..."
	```

	Or create via GitHub UI: https://github.com/graphistry/pygraphistry/releases/new?tag=X.Y.Z

	**Release notes should include:**
	- Critical fixes and breaking changes (if any)
	- Major features from current and recent versions
	- Links to full CHANGELOG and installation instructions
	- Highlight important API changes, new capabilities, and use cases

## CI Dependency Lockfiles

CI uses per-Python-version hashed lockfiles for supply chain security:

- **Generation**: A `generate-lockfiles` CI job runs `bin/generate-lockfiles.sh` to produce lockfiles for all profile × Python version combos. Most are uploaded as artifacts, not committed.
- **ReadTheDocs lockfile**: `requirements/rtd-py3.12.lock` is committed because `.readthedocs.yml` consumes it directly. Update it when changing RTD's Python version, docs/pygraphviz extras, `setup.py` dependency constraints that affect docs, or RTD install steps:
  ```bash
  PROFILES=rtd VERSIONS=3.12 ./bin/generate-lockfiles.sh
  ```
  CI's `check-rtd-lockfile` job regenerates only the RTD profile using the committed lockfile's `--exclude-newer` timestamp and fails if `requirements/rtd-py3.12.lock` is out of sync. To fix a red `check-rtd-lockfile`, rerun the command above and commit the resulting lockfile.
- **Spark lockfile**: `requirements/spark-py3.14.lock` is committed because the `test-spark` job installs a small Spark-specific smoke-test environment without the broader test extras. Update `requirements/spark-py3.14.in` when changing the direct Spark smoke dependencies, then regenerate and commit the lockfile:
  ```bash
  PROFILES=spark VERSIONS=3.14 ./bin/generate-lockfiles.sh
  ```
  CI's `check-spark-lockfile` job uses the committed lockfile's `--exclude-newer` timestamp and fails if `requirements/spark-py3.14.lock` is out of sync.
- **6-day cooldown**: `--exclude-newer` ensures no package published in the last 6 days is included, mitigating 0-day supply chain attacks. `UV_EXCLUDE_NEWER` is also set globally as belt-and-suspenders.
- **Hash verification**: `--require-hashes` on install ensures tamper-proof installs (except AI/umap profiles where torch conflicts prevent it).
- **Adding a dependency**: After modifying most `setup.py` extras, CI automatically regenerates artifact lockfiles. If the change affects ReadTheDocs docs dependencies, also update and commit `requirements/rtd-py3.12.lock`.
- **Emergency override**: Set `COOLDOWN_DAYS=0` in `bin/generate-lockfiles.sh` to disable the 6-day cooldown for urgent patches.
