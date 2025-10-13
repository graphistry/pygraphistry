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

* flake8: bin/lint.sh
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

1. Manually update CHANGELOG.md

1. Merge the desired PR to master and switch to master head (`git checkout master && git pull`)

1. Tag the repository with a new version number. We use semantic version numbers of the form *X.Y.Z*.

	```sh
	git tag X.Y.Z
	git push --tags
	```

1. Confirm the [publish](https://github.com/graphistry/pygraphistry/actions?query=workflow%3A%22Publish+Python+%F0%9F%90%8D+distributions+%F0%9F%93%A6+to+PyPI+and+TestPyPI%22) Github Action published to [pypi](https://pypi.org/project/graphistry/), or manually run it for the master branch

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
