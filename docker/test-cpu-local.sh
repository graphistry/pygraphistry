#!/bin/bash

# Run tests using local mounts

#docker-compose build

TEST_CPU_VERSION=${TEST_CPU_VERSION:-latest}

docker run \
    -e PYTEST_CURRENT_TEST=TRUE \
    --rm -it \
    -v ${PWD}/..:/opt/pygraphistry-mounted:ro \
    -w /opt/pygraphistry-mounted \
    graphistry/test-cpu:${TEST_CPU_VERSION} \
    --maxfail=5 --timeout=10 $@