# Development Setup
### Install Git Checkout:

1. Remove any version installed with pip
    `pip uninstall graphistry`
2. Install local git checkout
	`./setup.py develop`

### Running Tests Locally

1. Install our test dependencies:`nose` and `mock`.
2. Run `nosetests` in the root pygraphistry folder (or `nose` or `nose2`).
3. `python setup.py test`
4. To duplicate CI tests, in python2 and 3, run ` time flake8 . --count --select=E901,E999,F821,F822,F823 --show-source --statistics`

### CI

Travis CI automatically runs on every branch (with a Travis CI file). To configure, go to the [Travis CI account](https://travis-ci.org/graphistry/pygraphistry) .

### Uninstall Git Checkout

Uninstall the local checkout (useful to rollback to packaged version) with `./setup.py develop --uninstall`

# Release Procedure: Tag, Package, & Upload
1. Tag the repository with a new version number. We use semantic version numbers of the form *X.Y.Z*.

	```sh
	git tag X.Y.Z
	git push --tags
	```

2. Get `pypirc` from your friendly colleagues

3. Run `./pipupload.sh`