#!/bin/bash
set -ex

WITH_NEO4J=1
WITH_SUDO=${WITH_SUDO:-sudo}

( cd ../test/db/neo4j && WITH_SUDO="$WITH_SUDO" ./launch.sh )

docker-compose build

TEST_CPU_VERSION=${TEST_CPU_VERSION:-latest}

${WITH_SUDO} docker run \
    -e PYTEST_CURRENT_TEST=TRUE \
    -e WITH_NEO4J=1 \
    --rm \
    -v ${PWD}/..:/opt/pygraphistry-mounted:ro \
    -w /opt/pygraphistry-mounted \
    --net grph_net \
    graphistry/test-cpu:${TEST_CPU_VERSION} \
        --maxfail=5 \
        graphistry/tests/test_bolt_util.py \
        $@
