#!/bin/bash
set -ex

echo "CONFIG"

PIP_DEPS=${PIP_DEPS:--e .[bolt,gremlin,igraph,test,ai] neo4j==4.4.3}
WITH_NEO4J=${WITH_NEO4J:-0}
WITH_LINT=${WITH_LINT:-1}
WITH_TYPECHECK=${WITH_TYPECHECK:-1}
WITH_TEST=${WITH_TEST:-1}
WITH_BUILD=${WITH_BUILD:-1}
TEST_CPU_VERSION=${TEST_CPU_VERSION:-latest}
LOG_LEVEL=${LOG_LEVEL:-DEBUG}

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

COMPOSE_DOCKER_CLI_BUILD=1 \
DOCKER_BUILDKIT=1 \
docker-compose build \
    --build-arg PIP_DEPS="${PIP_DEPS}" \
    test-gpu

echo "RUN"

docker run \
    -e PYTEST_CURRENT_TEST=TRUE \
    -e WITH_NEO4J=$WITH_NEO4J \
    -e WITH_LINT=$WITH_LINT \
    -e WITH_TYPECHECK=$WITH_TYPECHECK \
    -e WITH_TEST=$WITH_TEST \
    -e WITH_BUILD=$WITH_BUILD \
    -e LOG_LEVEL=$LOG_LEVEL \
    -v "`pwd`/../graphistry:/opt/pygraphistry/graphistry:ro" \
    --security-opt seccomp=unconfined \
    --rm \
    ${NETWORK} \
    graphistry/test-gpu:${TEST_CPU_VERSION} \
        --maxfail=1 \
        $@
