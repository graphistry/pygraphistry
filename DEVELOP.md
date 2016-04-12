# Development Setup
### Install Git Checkout:

1. Remove any version installed with pip
    `pip uninstall graphistry`
2. Install local git checkout
	`./setup.py develop`

### Running Tests Locally

1. Install our test dependencies:`nose` and `mock`.
2. Run `nosetests` in the root pygraphistry folder.

### Uninstall Git Checkout

Uninstall the local checkout (useful to rollback to packaged version) with `./setup.py develop --uninstall`

# Release Procedure
1. Bump version number to X.X.X in setup.py
2. `git commit -a -m "Version X.X.X"`
3. `git tag X.X.X`
4. `git push`
5. `git push --tags`
6. In the *graphistry/config* repo, in *index.js*, update `PYGRAPHISTRY.latestVersion`.

### Package & Upload:
1. Get `~/.pypirc` file from the powers that be.
2. Run `./setup.py bdist_wheel upload -r pypi`
