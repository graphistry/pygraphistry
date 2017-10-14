#!/bin/bash -ex

# silently cd into the project's root directory
cd $(dirname "$0")/../ > /dev/null

# init build environment variables or defaults
source ./build/env.sh

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
    --run-script=*)
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

# graphistry dockerhub account/container namespace ("graphistry")
G_NS="$GRAPHISTRY_NAMESPACE"
# The lerna container name
LERNA_CONTAINER="$G_NS/lerna"

docker build \
    -f build/Dockerfile-lerna \
    --build-arg NAMESPACE=${G_NS} \
    -t ${LERNA_CONTAINER} .

if [[ ! $SHOULD_RUN_LERNA_CMD && ! $SHOULD_RUN_SCRIPT ]]; then
    exit 0
fi

trap "rv=\$?; docker rm -f lerna || true; exit \$rv" ERR EXIT

docker rm -f lerna > /dev/null || true
docker run -i -d --name lerna -v "${PWD}":/${G_NS} ${LERNA_CONTAINER} sh

if [ $SHOULD_RUN_LERNA_CMD ]; then
    docker exec -t lerna sh -c "$LERNA_CMD_TO_RUN"
fi

if [ $SHOULD_RUN_SCRIPT ]; then
    # get module package paths e.g. packages/viz-app, packages/legacy/config, etc.
    PACKAGE_PATHS=($(docker exec -t lerna $LERNA_LS_COMMAND -- echo \${PWD##*/$G_NS/} | tr -d '\r'))
    # get module package names e.g. @graphistry/viz-app, @graphistry/config, etc.
    PACKAGE_NAMES=($(docker exec -t lerna $LERNA_LS_COMMAND -- echo \$LERNA_PACKAGE_NAME | tr -d '\r'))
    for ((i=0; $i<${#PACKAGE_PATHS[*]}; i++)); do
        PACKAGE_PATH="${PACKAGE_PATHS[i]}"
        PACKAGE_NAME="${PACKAGE_NAMES[i]}"
        if [ ! -f "$PACKAGE_PATH/build/$SCRIPT_TO_RUN" ]; then
            continue
        fi
        # strip the package scope prefix and trailing slash from the
        # package name (e.g. "@graphistry/") to construct CONTAINER_NAME
        ROOT_PATH="$PWD" \
        PACKAGE_PATH="$PACKAGE_PATH" \
        PACKAGE_NAME="$PACKAGE_NAME" \
        CONTAINER_NAME="$G_NS/${PACKAGE_NAME##*@*/}" \
            ./"$PACKAGE_PATH/build/$SCRIPT_TO_RUN"
    done
fi
