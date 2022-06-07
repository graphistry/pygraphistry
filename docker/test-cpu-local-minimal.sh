#!/bin/bash
set -ex

WITH_LINT=${WITH_LINT:-1} \
WITH_TYPECHECK=${WITH_TYPECHECK:-1} \
WITH_BUILD=${WITH_BUILD:-1} \
WITH_TEST=${WITH_TEST:-1} \
SENTENCE_TRANSFORMER="" \
PIP_DEPS=${PIP_DEPS:--e .[test,build]} \
    ./test-cpu-local.sh \
        --ignore=graphistry/tests/test_bolt_util.py \
        --ignore=graphistry/tests/test_gremlin.py \
        --ignore=graphistry/tests/test_ipython.py \
        --ignore=graphistry/tests/test_nodexl.py \
        --ignore=graphistry/tests/test_tigergraph \
        --ignore=graphistry/tests/test_feature_utils \
        --ignore=graphistry/tests/test_umap_utils \
        $@
