#!/bin/bash

# Run tests using local mounts

WITH_NEO4J=${WITH_NEO4J:-1}

if [ "$WITH_NEO4J" == "1" ]
then
    ( cd ../test/db/neo4j && ./launch.sh )
fi

docker-compose build

TEST_CPU_VERSION=${TEST_CPU_VERSION:-latest}

docker run \
    -e PYTEST_CURRENT_TEST=TRUE \
    -e WITH_NEO4J=$WITH_NEO4J \
    --rm -it \
    -v ${PWD}/..:/opt/pygraphistry-mounted:ro \
    -w /opt/pygraphistry-mounted \
    --net grph_net \
    graphistry/test-cpu:${TEST_CPU_VERSION} \
    --maxfail=5 --timeout=10 $@