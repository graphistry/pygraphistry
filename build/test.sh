#!/bin/bash -ex

# silently cd into this shell script's directory
cd $(dirname "$0") > /dev/null

# init build environment variables or defaults
source ./env.sh
export NODE_ENV=test

docker network inspect ${GRAPHISTRY_NETWORK} ||
    docker network create ${GRAPHISTRY_NETWORK}

./lerna.sh --build=true --script=test.sh

docker network rm ${GRAPHISTRY_NETWORK}

echo "test finished"
