#!/bin/bash

## Run from deploy folder to see git pull/push status of graphistry repos
## (Assumes each repo is in parent folder)

REPOS="central config datasets deploy graph-viz horizon-viz node-pigz node-webcl StreamGL superconductor-proxy uber-viz viz-server"
BRANCH="master"
ROOT=`pwd`/../

function check() {
  git fetch origin &> /dev/null
  LOCAL=$(git rev-parse ${BRANCH})
  REMOTE=$(git rev-parse ${BRANCH}@{u})
  BASE=$(git merge-base ${BRANCH} ${BRANCH}@{u})
  git diff --quiet
  UNSTAGED=$?
  git diff --quiet --cached
  STAGED=$?

  if [ $STAGED = 1 ]; then
      printf "%20s: %s\n" "$1" "Staged local changes"
  elif [ $UNSTAGED = 1 ]; then
      printf "%20s: %s\n" "$1" "Unstaged local changes"
  elif [ $LOCAL = $REMOTE ]; then
      printf "%20s: %s\n" "$1" "Up-to-date ($LOCAL)"
  elif [ $LOCAL = $BASE ]; then
      printf "%20s: %s\n" "$1" "Need to pull"
  elif [ $REMOTE = $BASE ]; then
      printf "%20s: %s\n" "$1" "Need to push"
  else
      printf "%20s: %s\n" "$1" "Diverged"
  fi
}

for REPO in $REPOS ; do
  pushd $ROOT > /dev/null
  if [ -d $REPO ] ; then
    cd $REPO
    check $REPO &
  else
      printf "%20s: %s\n" $REPO "No local copy"
  fi
  popd > /dev/null
done

wait
