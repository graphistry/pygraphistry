#!/bin/bash
set -ex


PYTHON_VERSION=${PYTHON_VERSION:-3.7} \
PIP_DEPS=${PIP_DEPS:--e .[ai,test]} \
WITH_LINT=${WITH_LINT:-1} \
WITH_TYPECHECK=${WITH_TYPECHECK:-1} \
WITH_BUILD=${WITH_BUILD:-1} \
WITH_TEST=${WITH_TEST:-1} \
SENTENCE_TRANSFORMER=${SENTENCE_TRANSFORMER-average_word_embeddings_komninos} \
SENTENCE_TRANSFORMER=${SENTENCE_TRANSFORMER} \
    ./test-cpu-local.sh \
        graphistry/tests/test_feature_utils.py \
        graphistry/tests/test_umap_utils.py \
        $@
