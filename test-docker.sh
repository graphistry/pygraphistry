#!/bin/bash

# docker pull graphistry/graphistry-blazing

sudo docker run \
    -e GRAPHISTRY_LOG_LEVEL=TRACE \
    -e PYTEST_CURRENT_TEST=TRUE \
    --rm -it \
    --gpus all \
    -v ${PWD}:/opt/pygraphistry:ro \
    -w /opt/pygraphistry \
    graphistry/graphistry-forge-base:latest \
    /opt/pygraphistry/test.sh --maxfail=5 --timeout=10 $@