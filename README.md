# viz-app

Mono-repo for vizapp, pivotapp, and supporting microservices

Open source clients are:
* https://github.com/graphistry/graphistry-js
* https://github.com/graphistry/pygraphistry

# Install

Requires node > `v6.6.0` and npm  > `v3.10.3`
* Known good: node 8.6.0, npm 5.5.1

1. Install `pigz`. Mac: `brew install pigz`, Linux: `sudo apt-get install pigz`.
2. `npm install`

# Local Dev

* `packages/pivot-app`: `npm run watch`
* `packages/viz-app`: `npm run start:dev`

# Gitflow

## What The Build Server Does

* Our build server automaticallys sets package version numbers. It looks at what is in lerna.json, e.g., "2.0.0000", and turn it into "2.0.45", where "45" is the jenkins build. You don't need to manually change version numbers, but are always free to do major/minor semvar (but not patch, which the build server controls).
* It will build, publish to npm, and deploy to staging.

## Branches with PRs

1. Push whatever commits to a branch
2. Optional: Manually increment `lerna.json` with a semantic version number update (NOT `packages/viz-app/package.json` NOR `packages/api-client/package.json`)
3. Stage, review, fix, and repeat until you and your reviewers are happy
4. `Squash and Merge` the PR
5. `Delete` the branch

The server will then build and publish each package under that version.

## Straight to Master

Not advised but works. The build server will still autoincrement and autopublish.


# Deploying Code

For any change in this repo,

1. Push code to master or a branch
2. In Jenkins (deploy.graphistry.com), do `Build and push viz-app` with `Build with parameters`, specifying your branch `dev/MyBranch` or leave as `origin/master`
3. There is no step 3

Caution: If you deploy a branch, staging will be on that branch for everyone else too.


# Docker

## Setup & Install Dependencies

1. Install [Docker](https://www.docker.com/docker-mac)
1. Install [`jq`](https://stedolan.github.io/jq/):

    ```sh
    brew install jq
    ```

1. Clone the [wholly inoccuous files](https://github.com/graphistry/wholly-innocuous) repo and create a `WHOLLY_INOCCUOUS` environment variable:

    ```sh
    SHELL_PROFILE="$HOME/.bash_profile" &&
    INOCCUOUS_DIR="\$HOME/graphistry/wholly-innocuous" && \
        git clone \
            git@github.com:graphistry/wholly-innocuous.git \
            $(echo $(eval "echo $INOCCUOUS_DIR")) && \
        echo "export WHOLLY_INOCCUOUS=\"$INOCCUOUS_DIR\"" \
            >> $SHELL_PROFILE && source $SHELL_PROFILE
    ```

1. Log into Graphistry Docker account: see wholly inoccuous

## Build

```npm run build```

## Run

Mysteries!

## Run the tests (uses Docker)

From the project root:

```sh
npm test
# alternatively
./build/test.sh
```


# Import existing git repositories into this repo
We have a script to fetch, rewrite, and merge the commit history of existing repositories.
The script rewrites the git history to look as if the commits were always made to this repo.
1. **Prereq: install `gnu-sed`**

    ```sh
    which sed # should print `/usr/local/bin/sed`
    sed --version # first line should say `sed (GNU sed)`
    ```

    If either of the above commands don't work, install `gnu-sed` via homebrew:

    ```sh
    brew install gnu-sed --with-default-names
    ```

1. Running `git-import-repo.sh`

    With a git URL (replace contents between the angle brackets):

    ```sh
    REPO_NAME=a-cool-node-module && \ # the repository name to import
    REPO_USER=a-nice-github-user && \ # the github user that owns the repo
    git checkout -b "import-$REPO_NAME" && \ # check out a new branch to stage the import
    sh ./git-import-repo.sh "git@github.com:$REPO_USER/$REPO_NAME.git" "packages/$REPO_NAME"
    ```

    With a local git repository:

    ```sh
    REPO_NAME=a-cool-node-module && \ # the repository name to import
    REPO_PATH=/a/cool/node/module && \ # the local path to the git repository
    git checkout -b "import-$REPO_NAME" && \ # check out a new branch to stage the import
    sh ./git-import-repo.sh "$REPO_PATH" "packages/$REPO_NAME"
    ```
    If everything looks alright after the import, congratulations 😎!

