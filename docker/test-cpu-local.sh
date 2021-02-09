#!/bin/bash
set -ex

echo "CONFIG"

WITH_NEO4J=${WITH_NEO4J:-0}
TEST_CPU_VERSION=${TEST_CPU_VERSION:-latest}

NETWORK=""
if [ "$WITH_NEO4J" == "1" ]
then
    NETWORK="--net grph_net"
fi

echo "PREP"

if [ "$WITH_NEO4J" == "1" ]
then
    ( cd ../test/db/neo4j && ./launch.sh )
fi

docker-compose build

echo "RUN"

docker run \
    -e PYTEST_CURRENT_TEST=TRUE \
    -e WITH_NEO4J=$WITH_NEO4J \
    --rm \
    ${NETWORK} \
    graphistry/test-cpu:${TEST_CPU_VERSION} \
        --maxfail=5 $@
