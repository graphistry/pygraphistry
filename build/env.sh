#!/bin/bash -ex

# Relevant Jenkins environment variables:
# BUILD_NUMBER - The current build number, such as "153"
# CHANGE_TARGET - The target or base branch to which the change should be merged

if [ -z $GRAPHISTRY_NAMESPACE ]; then export GRAPHISTRY_NAMESPACE=graphistry; fi
if [ -z $GRAPHISTRY_NETWORK   ]; then export GRAPHISTRY_NETWORK=${GRAPHISTRY_NAMESPACE}-network; fi

if [ -z $COMMIT_ID   ]; then export COMMIT_ID=$(git rev-parse --short HEAD); fi
if [ -z $BRANCH_NAME ]; then export BRANCH_NAME=$(git name-rev --name-only HEAD); fi
if [ -z $BUILD_TAG   ]; then export BUILD_TAG=${BUILD_TAG:-test}-${BUILD_NUMBER:-dev}; fi

if [ -z $PG_PORT ]; then export PG_PORT=5432; fi
if [ -z $PG_PASS ]; then export PG_PASS=pg-test-password; fi
if [ -z $PG_NAME ]; then export PG_NAME=${GRAPHISTRY_NETWORK}-pg; fi
if [ -z $DB_NAME ]; then export DB_NAME=${GRAPHISTRY_NAMESPACE}-test; fi
if [ -z $PG_USER ]; then export PG_USER=${GRAPHISTRY_NAMESPACE}-test; fi

