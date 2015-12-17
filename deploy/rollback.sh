#!/bin/bash

## Run from deploy folder to rollback repos in parent folder
## to that of a provided date.
## Meant for use with duct tape prod
## Example $1: 5.days.ago

BRANCH="master"
REPOS=()
BASEDIRECTORY=$PWD/

for repo in ../*
do
    if [ -d $repo ]; then
        if [ -d ${repo}/.git ]; then
            REPOS+=($repo)
        fi
    fi
done

for repo in "${REPOS[@]}"
do
    cd "$BASEDIRECTORY$repo"
    if [ -z "$1" ]; then
        git checkout $BRANCH
    else
        git checkout `git rev-list -n 1 --before=$1 $BRANCH`
    fi
done

echo Rolled back repos to $1
