#!/bin/bash
set -ex

WITH_LINT=${WITH_LINT:-1} \
WITH_TYPECHECK=${WITH_TYPECHECK:-1} \
WITH_BUILD=${WITH_BUILD:-0} \
PIP_DEPS=${PIP_DEPS:--e .[test]} \
    ./test-cpu-local.sh \
        --ignore=graphistry/tests/test_bolt_util.py \
        --ignore=graphistry/tests/test_gremlin.py \
        --ignore=graphistry/tests/test_ipython.py \
        --ignore=graphistry/tests/test_nodexl.py \
        --ignore=graphistry/tests/test_tigergraph \
        $@
