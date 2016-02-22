#!/bin/bash

## Run from deploy folder to see git pull/push status of graphistry repos
## (Assumes each repo is in parent folder)

REPOS="central config config-public datasets deploy graph-viz horizon-viz node-pigz node-webcl splunkistry splunk-viz StreamGL superconductor-proxy uber-viz viz-server common etl-worker"
BRANCH="master"
ROOT="$(cd "$(dirname $0)"; pwd)/../.."
RUNTESTS=1

HELPTEXT="This is a short script that checks the status of graphistry repositories and runs tests on each repo. The -n option will prevent tests from running."

#Colors for output
RED=`tput setaf 1`
GREEN=`tput setaf 2`
YELLOW=`tput setaf 3`
BLUE=`tput setaf 4`
RESET=`tput sgr0`

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
  LOCALSHORT=$(git rev-parse --short ${BRANCH})
  REMOTE=$(git rev-parse ${BRANCH}@{u})
  BASE=$(git merge-base ${BRANCH} ${BRANCH}@{u})
  git diff --quiet
  UNSTAGED=$?
  git diff --quiet --cached
  STAGED=$?
  MESSAGE=""

  if [ $STAGED = 1 ]; then
      MESSAGE=$(printf "%20s: ${YELLOW}%s${RESET}" "$1" "Staged local changes")
  elif [ $UNSTAGED = 1 ]; then
      MESSAGE=$(printf "%20s: ${YELLOW}%s${RESET}" "$1" "Unstaged local changes")
  elif [ $LOCAL = $REMOTE ]; then
      MESSAGE=$(printf "%20s: ${GREEN}Up-to-date${RESET} (%s)" "$1" "$LOCALSHORT")
  elif [ $LOCAL = $BASE ]; then
      MESSAGE=$(printf "%20s: ${BLUE}%s${RESET}\t" "$1" "Need to pull")
  elif [ $REMOTE = $BASE ]; then
      MESSAGE=$(printf "%20s: ${BLUE}%s${RESET}\t" "$1" "Need to push")
  else
      MESSAGE=$(printf "%20s: ${RED}%s${RESET}\t\t" "$1" "Diverged")
  fi

  if [ -e "package.json" ] && [ $2 -gt 0 ]; then
    TEST_RESULTS=$(npm test 2>&1)
    TEST_STATUS=$?
    if [ $TEST_STATUS -ne 0 ]; then
      echo "$TEST_RESULTS" > tests.log
      MESSAGE+=$(printf "\tTests: %s\n" "${RED}Failed${RESET}")
    else
      MESSAGE+=$(printf "\tTests: %s\n" "${GREEN}Passed${RESET}")
    fi
  fi
  echo "$MESSAGE"
}

function checkDeps() {
    MISMATCH=$($ROOT/deploy/tools/setup/setup.js --versions 2>&1)
    if [[ -n $MISMATCH ]]; then
        echo "${RED}There are version mismatch in dependencies${RESET}"
        echo "$MISMATCH"
        # exit -1
    fi
}

checkDeps

# Clear out tmp
TMP_FILES=$(find /tmp/ -type f -name '*.metadata' 2>/dev/null | rev | cut -d/ -f1 | rev | sed 's/\.metadata//g')
if [ $RUNTESTS -gt 0 ]; then
  echo "Deleting cached datasets in /tmp"
  for TMP_FILE in $TMP_FILES; do
    rm "/tmp/$TMP_FILE"
    rm "/tmp/$TMP_FILE.metadata"
  done
fi

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

