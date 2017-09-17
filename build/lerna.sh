#!/bin/bash -ex

# silently cd into the project's root directory
cd $(dirname "$0")/../ > /dev/null

LERNA_LS_COMMAND="lerna exec --loglevel=error"

while [[ ${1} ]]; do
    case "${1}" in
    --build)
        SHOULD_BUILD_LERNA=1
        shift
        ;;
    --run-cmd)
        SHOULD_RUN_LERNA_CMD=1
        LERNA_CMD_TO_RUN="${2}"
        shift; shift;
        ;;
    --script)
        SHOULD_RUN_SCRIPT=1
        SCRIPT_TO_RUN="${2}"
        shift; shift;
        ;;
    --since)
        LERNA_LS_COMMAND="$LERNA_LS_COMMAND --since ${2}"
        shift; shift;
        ;;
    *)
        echo "Unknown argument ${1}"
        exit 1
        ;;
    esac
done

# Path of the lerna "packages" folder, relative to the project root
PROJECTS=packages
# The lerna container name
LERNA_CONTAINER="$GRAPHISTRY_NAMESPACE/lerna"
# The lerna command to list project names
LERNA_LS_COMMAND="$LERNA_LS_COMMAND -- echo \${PWD##*/}"

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

for PROJECT in $(docker run --rm \
    -v "${PWD}":/${GRAPHISTRY_NAMESPACE} \
    ${LERNA_CONTAINER} ${LERNA_LS_COMMAND})
do
    PROJECT_BUILD_DIR="$PROJECTS/$PROJECT/build"
    PROJECT_SCRIPT="$PROJECT_BUILD_DIR/$SCRIPT_TO_RUN"
    if [ ! -f "$PROJECT_SCRIPT" ]; then
        echo "no $PROJECT_SCRIPT found, skipping..."
        continue
    fi
    CONTAINER_NAME="$GRAPHISTRY_NAMESPACE/$PROJECT" sh ${PROJECT_SCRIPT}
done