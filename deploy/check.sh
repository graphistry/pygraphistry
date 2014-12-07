#!/bin/bash

## Run from deploy folder to see git pull/push status of graphistry repos
## (Assumes each repo is in parent folder)

REPOS="central config datasets deploy graph-viz horizon-viz StreamGL uber-viz viz-server"
BRANCH="master"
ROOT=`pwd`/../

function check() {
  git fetch origin > /dev/null
  LOCAL=$(git rev-parse ${BRANCH})
  REMOTE=$(git rev-parse ${BRANCH}@{u})
  BASE=$(git merge-base ${BRANCH} ${BRANCH}@{u})

  if [ $LOCAL = $REMOTE ]; then
      echo "$1: Up-to-date"
  elif [ $LOCAL = $BASE ]; then
      echo "$1: Need to pull"
  elif [ $REMOTE = $BASE ]; then
      echo "$1: Need to push"
  else
      echo "$1: Diverged"
  fi
}

for REPO in $REPOS ; do
  pushd $ROOT > /dev/null
  cd $REPO
  check $REPO
  popd > /dev/null
done
