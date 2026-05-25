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

### GPU CI

GPU CI can be manually triggered by core dev team members:

1. Push intended changes to protected branches `gpu-public` or `master`
2. Manually trigger action [ci-gpu](https://github.com/graphistry/pygraphistry/actions/workflows/ci-gpu.yml) on one of the above branches

GPU tests can also be run locally via `./docker/test-gpu-local.sh` .

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

## Optional NetworkX / SciPy Policy

The `networkx` extra supports `networkx>=2.5,<4`. Keep this range aligned with `graphistry.compute.networkx_policy.NETWORKX_VERSION_SPEC` and the local Cypher `graphistry.nx.*` CALL tests.

SciPy is optional for NetworkX-backed calls. The `networkx-scipy` extra declares the tested SciPy range, `scipy>=1.5,<2`, for environments that want NetworkX algorithms to use SciPy-backed implementations when NetworkX chooses them. The GFQL CALL surface must still keep no-SciPy fallbacks for algorithms that have local fallbacks, such as `graphistry.nx.pagerank` and `graphistry.nx.hits`.

When adding NetworkX-backed procedures:

- Use `graphistry.compute.networkx_policy` for supported-version checks.
- Update the `test-networkx-scipy-policy` CI matrix if the new procedure needs a narrower NetworkX/SciPy range.
- Add coverage for both the lower supported NetworkX bound and the current upper-bound resolver path.
