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
	- If manually triggering, run only from `master` and use it only for maintainer-led recovery scenarios
	- Do not rerun publish for a version that is already on PyPI (duplicate-file uploads are rejected)
	- Verify version appears on PyPI: `curl -s https://pypi.org/pypi/graphistry/json | jq -r '.info.version'`
	- Verify release evidence artifacts from the workflow run:
	  - built distributions (`dist/*.whl`, `dist/*.tar.gz`)
	  - SBOM (`dist/sbom-cyclonedx.json`)
	  - GitHub build provenance attestation for `dist/*`
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

- **Generation**: A `generate-lockfiles` CI job runs `bin/generate-lockfiles.sh` to produce lockfiles for all profile × Python version combos. These are uploaded as artifacts, not committed.
- **6-day cooldown**: `--exclude-newer` ensures no package published in the last 6 days is included, mitigating 0-day supply chain attacks. `UV_EXCLUDE_NEWER` is also set globally as belt-and-suspenders.
- **Hash verification**: `--require-hashes` on install ensures tamper-proof installs (except AI/umap profiles where torch conflicts prevent it).
- **Adding a dependency**: After modifying `setup.py` extras, CI automatically regenerates lockfiles. No manual lockfile updates needed.
- **Emergency override**: Set `COOLDOWN_DAYS=0` in `bin/generate-lockfiles.sh` to disable the 6-day cooldown for urgent patches.
