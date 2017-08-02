# viz-app

Mono-repo for viz-app and graphistry-client

# Install

Requires node `v6.6.0` and npm `v3.10.3`.

1. Ensure you have `pigz` installed on your system. Mac: `brew install pigz`, Linux: `sudo apt-get install pigz`.
2. Clone this repo.
3. `npm install` to install Lerna
4. `npm run build` to install subpackages `packages/viz-app` and `packages/api-client`

# Local Dev

* In `packages/viz-app` or `packages/api-client`, run `npm run build`
* For viz-app, a common option within that package is `npm run start:dev`
* For additional options for viz-app, see `packages/viz-app/README.md`

# Landing Code

You should manually kick off the build when commiting to master, but that's it.

## What The Build Server Does

* Our build server will automatically set package version numbers. It looks at what is in lerna.json, e.g., "2.0.0000", and turn it into "2.0.45", where "45" is the jenkins build. You don't need to manually change version numbers, but are always free to do major/minor semvar (but not patch, which the build server controls).
* It will build, publish to npm, and deploy to staging.


## Gitflow Development: Branches with PRs

1. Push whatever commits to a branch
2. Optional: Manually increment `lerna.json` with a semantic version number update (NOT `packages/viz-app/package.json` NOR `packages/api-client/package.json`)
3. Stage, review, fix, and repeat until you and your reviewers are happy
4. `Squash and Merge` the PR
5. `Delete` the branch

The server will then build and publish each package under that version.

## Straight to Master

Committing straight to master will still work, though not advised. Even in this case, version number changes should be on the global package, and the build server will still autopublish.

# Deploying Code

1. Push code to master or a branch
2. In Jenkins (deploy.graphistry.com), do `Build and push viz-app` with `Build with parameters`, specifying your branch `dev/MyBranch` or leave as `origin/master`
3. There is no step 3

Caution: If you deploy a branch, staging will be on that branch for everyone else too.
