#!/bin/bash
set -ex

WITH_LINT=1 \
WITH_TYPECHECK=0 \
WITH_BUILD=0 \
PIP_DEPS=${PIP_DEPS:--e .[test]} \
    ./test-cpu-local.sh \
        --ignore=graphistry/tests/test_bolt_util.py \
        --ignore=graphistry/tests/test_gremlin.py \
        --ignore=graphistry/tests/test_ipython.py \
        --ignore=graphistry/tests/test_nodexl.py \
        --ignore=graphistry/tests/test_tigergraph \

