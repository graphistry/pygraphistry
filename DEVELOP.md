# Development Setup
### Install Git Checkout:

1. Remove any version installed with pip
    `pip uninstall graphistry`
2. Install local git checkout
	`./setup.py develop`

### Running Tests Locally

1. Install our test dependencies:`nose` and `mock`.
2. Run `nosetests` in the root pygraphistry folder (or `nose` or `nose2`).

### Uninstall Git Checkout

Uninstall the local checkout (useful to rollback to packaged version) with `./setup.py develop --uninstall`

# Release Procedure
1. Tag the repository with a new version number. We use semantic version numbers of the form *X.Y.Z*.

	```sh
	git tag X.Y.Z`
	git push --tags
	```

2. In the [graphistry/config](https://github.com/graphistry/config) repository, update `PYGRAPHISTRY.latestVersion` in *index.js*
3. Bump the config's package version
4. Publish config to npm using `npm publish`
5. Update the config dependency version in both common (dep + package versions) and viz-app

### Package & Upload
1. Login to [Graphistry's Jenkins](http://deploy.graphistry.com/view/Package/job/Package%20PyGraphistry%20to%20PIP/build).
2. Fill the `tag` parameter with version number you have just used to tag the repository, then click *Build*.
