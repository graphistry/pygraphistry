# Development Setup

Dev is moving towards docker for easier tasks like CI automation integrations and reliable local dev

## Docker

### Install

```bash
cd docker && docker-compose build && docker-compose up -d
```
### Run local tests without rebuild

```bash
cd docker && ./test-cpu-local.sh
```


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


## CI

We intend to move to Github Actions / DockerHub Automated Builds for CPU and TBD for GPU

### Travis - DEPRECATED

Travis CI automatically runs on every branch (with a Travis CI file). To configure, go to the [Travis CI account](https://travis-ci.org/graphistry/pygraphistry) .

### Native - Uninstall Git Checkout - DEPRECATED

Uninstall the local checkout (useful to rollback to packaged version) with `./setup.py develop --uninstall`

## Publish: Merge, Tag, & Upload

1. Merge the desired PR to master and switch to master head (`git checkout master && git pull`)

2. Tag the repository with a new version number. We use semantic version numbers of the form *X.Y.Z*.

	```sh
	git tag X.Y.Z
	git push --tags
	```

3. Toggle version as active at [ReadTheDocs](https://readthedocs.org/projects/pygraphistry/versions/)

4. Confirm PyPI picked up the [release](https://pypi.org/project/graphistry/)