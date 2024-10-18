#!/bin/bash
set -ex


PYTHON_VERSION=${PYTHON_VERSION:-3.8} \
APTGET_INSTALL="gcc graphviz graphviz-dev" \
PIP_DEPS=${PIP_DEPS:--e .[pygraphviz,test,build]} \
WITH_LINT=${WITH_LINT:-1} \
WITH_TYPECHECK=${WITH_TYPECHECK:-1} \
WITH_BUILD=${WITH_BUILD:-1} \
WITH_TEST=${WITH_TEST:-1} \
    ./test-cpu-local.sh \
        graphistry/tests/plugins/test_graphviz.py \
        $@
