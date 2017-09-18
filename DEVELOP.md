# Developing in the Graphistry Platform mono-repo

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

## How to import existing git repositories into this repo
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