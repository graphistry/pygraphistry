#!/bin/bash
set -ex

PYTHON_VERSION=${PYTHON_VERSION:-3.6}
SENTENCE_TRANSFORMER=${SENTENCE_TRANSFORMER-average_word_embeddings_komninos}
WITH_SUDO=${WITH_SUDO:-}

COMPOSE_DOCKER_CLI_BUILD=1 \
DOCKER_BUILDKIT=1 \
docker-compose build \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    --build-arg PIP_DEPS="-e .[umap-learn,test]" \
    --build-arg SENTENCE_TRANSFORMER="${SENTENCE_TRANSFOMER}" \
    test-cpu

TEST_CPU_VERSION=${TEST_CPU_VERSION:-latest}

${WITH_SUDO} docker run \
    --security-opt seccomp=unconfined \
    -e PYTEST_CURRENT_TEST=TRUE \
    --rm \
    -v ${PWD}/..:/opt/pygraphistry-mounted:ro \
    -w /opt/pygraphistry-mounted \
    --entrypoint=/opt/pygraphistry-mounted/docker/test-cpu-local-neo4j-only-entrypoint.sh \
    --net grph_net \
    graphistry/test-cpu:${TEST_CPU_VERSION} \
        graphistry/tests/test_umap_utils.py \
        $@
