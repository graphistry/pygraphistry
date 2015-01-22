#!/bin/bash

## Run from deploy folder to see git pull/push status of graphistry repos
## (Assumes each repo is in parent folder)

REPOS="central config datasets deploy graph-viz horizon-viz node-pigz node-webcl splunkistry splunk-viz StreamGL superconductor-proxy uber-viz viz-server"
BRANCH="master"
ROOT=`pwd`/../
RUNTESTS=1

HELPTEXT="This is a short script that checks the status of graphistry repositories and runs tests on each repo. The -n option will prevent tests from running."

if [ -e "tests.log" ]; then
  rm tests.log
fi

while getopts ":nh" opt; do
  case $opt in
    h)
      echo "$HELPTEXT"
      exit 
      ;;
    n)
      RUNTESTS=0
      ;;
    \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
  esac
done

function check() {
  git fetch origin &> /dev/null
  LOCAL=$(git rev-parse ${BRANCH})
  REMOTE=$(git rev-parse ${BRANCH}@{u})
  BASE=$(git merge-base ${BRANCH} ${BRANCH}@{u})
  git diff --quiet
  UNSTAGED=$?
  git diff --quiet --cached
  STAGED=$?
  MESSAGE=""

  if [ $STAGED = 1 ]; then
      MESSAGE=$(printf "%20s: %s\n" "$1" "Staged local changes")
  elif [ $UNSTAGED = 1 ]; then
      MESSAGE=$(printf "%20s: %s\n" "$1" "Unstaged local changes")
  elif [ $LOCAL = $REMOTE ]; then
      MESSAGE=$(printf "%20s: %s\n" "$1" "Up-to-date ($LOCAL)")
  elif [ $LOCAL = $BASE ]; then
      MESSAGE=$(printf "%20s: %s\n" "$1" "Need to pull")
  elif [ $REMOTE = $BASE ]; then
      MESSAGE=$(printf "%20s: %s\n" "$1" "Need to push")
  else
      MESSAGE=$(printf "%20s: %s\n" "$1" "Diverged")
  fi

  if [ -e "package.json" ] && [ $2 -gt 0 ]; then
    TEST_RESULTS=$(npm test 2>&1)
    TEST_STATUS=$?
    if [ $TEST_STATUS -ne 0 ]; then
      echo "$TEST_RESULTS" >> tests.log
      MESSAGE+=$(printf "\n%22s%s\n" "" "Tests FAILED.")
    else
      MESSAGE+=$(printf "\n%22s%s\n" "" "Tests passed.")
    fi
  fi
  echo "$MESSAGE"
}

for REPO in $REPOS ; do
  pushd $ROOT > /dev/null
  if [ -d $REPO ] ; then
    cd $REPO
    check $REPO $RUNTESTS &
  else
      printf "%20s: %s\n" $REPO "No local copy"
  fi
  popd > /dev/null
done

wait

