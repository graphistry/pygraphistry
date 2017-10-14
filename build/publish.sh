#!/bin/bash -ex

# silently cd into this shell script's directory
cd $(dirname "$0") > /dev/null

source ./env.sh
export NODE_ENV=production

MAJOR_MINOR=$(jq -r .version ../lerna.json | cut -d '.' -f 1,2)
REPO_VERSION=${MAJOR_MINOR}.${BUILD_NUMBER}

sh ./lerna.sh --build=true

sh ./lerna.sh --run-cmd=\
"lerna publish \
    --yes \
    --exact \
    --skip-git \
    --skip-npm \
    --repo-version=$REPO_VERSION"

sh ./lerna.sh --script=publish.sh

echo "publish finished"
