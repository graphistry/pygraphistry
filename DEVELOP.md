# Development Setup

See also [CONTRIBUTE.md](contribute.md) and [ARCHITECTURE.md](architecture.md)

Development is setup for local native and containerized Python coding & testing, and with automatic GitHub Actions for CI + CD. The server tests are like the local ones, except against a wider test matrix of environments.

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


## Debugging Tips

* Use the unit tests
* use the `logging` module per-file


## Publish: Merge, Tag, & Upload

1. Merge the desired PR to master and switch to master head (`git checkout master && git pull`)

1. Manually update CHANGELOG.md

1. Tag the repository with a new version number. We use semantic version numbers of the form *X.Y.Z*.

	```sh
	git tag X.Y.Z
	git push --tags
	```

1. Confirm the [publish](https://github.com/graphistry/pygraphistry/actions?query=workflow%3A%22Publish+Python+%F0%9F%90%8D+distributions+%F0%9F%93%A6+to+PyPI+and+TestPyPI%22) Github Action published to [pypi](https://pypi.org/project/graphistry/), or manually run it for the master branch

1. Toggle version as active at [ReadTheDocs](https://readthedocs.org/projects/pygraphistry/versions/)