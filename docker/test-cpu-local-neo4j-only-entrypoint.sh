#!/bin/bash

set -ex

MAYBE_RAPIDS="echo 'no rapids'"
if [[ "$RAPIDS" == "1" ]]; then
    source activate rapids
    MAYBE_RAPIDS="source activate rapids"
else
    source /opt/pygraphistry/pygraphistry/bin/activate
fi

./bin/test.sh $@