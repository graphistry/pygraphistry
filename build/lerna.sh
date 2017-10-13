#!/bin/bash -ex

# silently cd into the project's root directory
cd $(dirname "$0")/../ > /dev/null

if [ -z $GRAPHISTRY_NAMESPACE ]; then export GRAPHISTRY_NAMESPACE=graphistry; fi

LERNA_LS_COMMAND="lerna exec --loglevel=error"

for ARG in "$@"; do
    case $ARG in
    --build=*)
        SHOULD_BUILD_LERNA=1
        shift
        ;;
    --run-cmd=*)
        SHOULD_RUN_LERNA_CMD=1
        LERNA_CMD_TO_RUN="${ARG#*=}"
        shift
        ;;
    --script=*)
        SHOULD_RUN_SCRIPT=1
        SCRIPT_TO_RUN="${ARG#*=}"
        shift
        ;;
    --since=*)
        LERNA_LS_COMMAND="$LERNA_LS_COMMAND --since ${ARG#*=}"
        shift
        ;;
    *)
        echo "Unknown argument $ARG"
        exit 1
        ;;
    esac
done

# Path of the lerna "packages" folder, relative to the project root
PROJECTS=packages
# The lerna container name
LERNA_CONTAINER="$GRAPHISTRY_NAMESPACE/lerna"
# The lerna command to list project folders under the $PROJECTS dir, e.g. viz-app, legacy/config, etc.
LERNA_LS_COMMAND="$LERNA_LS_COMMAND -- echo \${PWD##*/$PROJECTS/}"

if [ $SHOULD_BUILD_LERNA ]; then
    docker build \
        -f build/Dockerfile-lerna \
        --build-arg NAMESPACE=${GRAPHISTRY_NAMESPACE} \
        -t ${LERNA_CONTAINER} .
fi

if [ $SHOULD_RUN_LERNA_CMD ]; then
    docker run --rm \
        -v "${PWD}":/${GRAPHISTRY_NAMESPACE} \
        ${LERNA_CONTAINER} ${LERNA_CMD_TO_RUN}
fi

if [ ! $SHOULD_RUN_SCRIPT ]; then exit 0; fi

PROJECT_DIRS=$(docker run --rm \
    -v "${PWD}":/${GRAPHISTRY_NAMESPACE} \
    ${LERNA_CONTAINER} ${LERNA_LS_COMMAND})

echo Attempting to execute $SCRIPT_TO_RUN in [ $PROJECT_DIRS ]

for PROJECT_DIR in $PROJECT_DIRS
do
    PROJECT_BUILD_DIR="$PROJECTS/$PROJECT_DIR/build"
    PROJECT_SCRIPT="$PROJECT_BUILD_DIR/$SCRIPT_TO_RUN"
    if [ ! -f "$PROJECT_SCRIPT" ]; then
        echo "no $PROJECT_SCRIPT found, skipping..."
        continue
    fi
    CONTAINER_NAME="$GRAPHISTRY_NAMESPACE/$PROJECT_DIR" ${PROJECT_SCRIPT}
done