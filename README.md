# viz-app

Mono-repo for viz-app and graphistry-client 

# Install

Requires node `v6.6.0` and npm `v3.10.3`.

1. Ensure you have `pigz` installed on your system. Mac: `brew install pigz`, Linux: `sudo apt-get install pigz`.
2. Clone this repo.
3. `npm install` to install Lerna
4. `npm run build` to install subpackages `packages/viz-app` and `packages/api-client`

# Dev Build

* In `packages/viz-app` or `packages/api-client`, run `npm run build`
* For viz-app, a common option is `npm run watch | bunyan -o short`
* For additional options for viz-app, see `packages/viz-app/README.md`

# Committing to Master

Our build server automatically builds on commits to master, publishes to npm, and manages the `packages/*/package.json` version numbers.

The development process is:

1. Prepare whatever commits, e.g., in a branch
2. Manually increment `/package.json` with a semantic version number update (NOT `/packages/viz-app/package.json`, NOR `/packages/api-client/package.json`)
3. Commit as part of the branch
4. Merge the PR

The server will then build and publish, and do a minor semantic version increase. Any packages with changes will have new version numbers corresponding to the current global version.

Committing straight to master will still work, though not advised. Even in this case, version number changes should be on the global package, and the build server will still autopublish.
