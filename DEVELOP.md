# CICD

[Travis CI](https://travis-ci.org/graphistry/pygraphistry) runs automatically against every branch containing a `.travis.yml` file.


Releases are based on git tags. To create a release for a given version, a tag must be created for that version.
- Use semver (`0.0.0-alpha` / `major.minor.patch-label`).
- Create a tag for the new version.
	```
	git tag 0.0.0-alpha
	git push --tags
	```

2. Login to [Graphistry's Jenkins](http://deploy.graphistry.com/view/Package/job/Package%20PyGraphistry%20to%20PIP/build).
3. Fill the `tag` parameter with version number you have just used to tag the repository, then click *Build*.
