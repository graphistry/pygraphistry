#!/bin/bash -ex

# silently cd into this shell script's directory
cd $(dirname "$0") > /dev/null

# Relevant Jenkins environment variables:
# BUILD_NUMBER - The current build number, such as "153"
# CHANGE_TARGET - The target or base branch to which the change should be merged

if [ -z $GRAPHISTRY_NAMESPACE ]; then export GRAPHISTRY_NAMESPACE=graphistry; fi
if [ -z $GRAPHISTRY_NETWORK   ]; then export GRAPHISTRY_NETWORK=${GRAPHISTRY_NAMESPACE}-network; fi
if [ -z $BUILD_NUMBER         ]; then export BUILD_NUMBER=$(jq -r .version ../lerna.json | cut -d '.' -f 3); fi

if [ -z $COMMIT_ID   ]; then export COMMIT_ID=$(git rev-parse --short HEAD); fi
if [ -z $BRANCH_NAME ]; then export BRANCH_NAME=$(git name-rev --name-only HEAD); fi
if [ -z $BUILD_TAG   ]; then export BUILD_TAG=${BUILD_TAG:-test}-${BUILD_NUMBER}; fi

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
