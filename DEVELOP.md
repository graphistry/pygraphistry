# Development Setup

See also [CONTRIBUTE.md](contribute.md) and [ARCHITECTURE.md](architecture.md)

Development is setup for local native and containerized Python coding & testing, and with automatic GitHub Actions for CI + CD. The server tests are like the local ones, except against a wider test matrix of environments.

## Docker

### Install

```bash
cd docker && docker-compose build && docker-compose up -d
```
### Run local tests without rebuild

```bash
cd docker && ./test-cpu-local.sh
```

Connector tests (currently neo4j-only): `cd docker && WITH_NEO4J=1 ./test-cpu-local.sh` (optional `WITH_SUDO=" "`)

* Will start a local neo4j (docker) then enable+run tests against it


## Native - DEPRECATED
### Install Git Checkout - DEPRECATED

1. Remove any version installed with pip
    `pip uninstall graphistry`
2. Install local git checkout
	`./setup.py develop`

### Running Tests Locally - DEPRECATED

1. Install our test dependencies:`nose` and `mock`.
2. Run `nosetests` in the root pygraphistry folder (or `nose` or `nose2`).
3. `python setup.py test`
4. To duplicate CI tests, in python2 and 3, run ` time flake8 . --count --select=E901,E999,F821,F822,F823 --show-source --statistics`


## Docs

Automatically build via ReadTheDocs from inline definitions.

To manually build, see `docs/`.

## CI

GitHub Actions: See `.github/workflows`


## Debugging Tips

* Use the unit tests
* use the `logging` module per-file


### Native - Uninstall Git Checkout - DEPRECATED

Uninstall the local checkout (useful to rollback to packaged version) with `./setup.py develop --uninstall`

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