#!/bin/bash
set -ex

PYTHON_VERSION=${PYTHON_VERSION:-3.6}
WITH_NEO4J=1
WITH_SUDO=${WITH_SUDO:-sudo}

( cd ../test/db/neo4j && WITH_SUDO="$WITH_SUDO" ./launch.sh )

docker-compose build --build-arg PYTHON_VERSION=${PYTHON_VERSION} test-cpu

TEST_CPU_VERSION=${TEST_CPU_VERSION:-latest}

${WITH_SUDO} docker run \
    --security-opt seccomp=unconfined \
    -e PYTEST_CURRENT_TEST=TRUE \
    -e WITH_NEO4J=1 \
    --rm \
    -v ${PWD}/..:/opt/pygraphistry-mounted:ro \
    -w /opt/pygraphistry-mounted \
    --entrypoint=/opt/pygraphistry-mounted/docker/test-cpu-local-neo4j-only-entrypoint.sh \
    --net grph_net \
    graphistry/test-cpu:${TEST_CPU_VERSION} \
        graphistry/tests/test_bolt_util.py \
        $@
