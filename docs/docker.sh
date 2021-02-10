#!/bin/bash
set -ex

mkdir -p build

RUN_INSTALLS="(cp -r /doc /pygraphistry && cd /pygraphistry && python -m pip install -e .[dev] )"
RUN_SPHINX="cd docs && ./build.sh || ( echo 'Printing /tmp/sphinx*' && ( cat /tmp/sphinx* || echo no err_file ) && exit 1 )"

#TODO make a docker layer so we can cache RUN_INSTALLS
docker run \
    --entrypoint=/bin/bash \
    --rm \
    -e USER_ID=$UID \
    -v $(pwd)/..:/doc \
    ddidier/sphinx-doc:3.2.1-1 \
    -c "${RUN_INSTALLS} && ${RUN_SPHINX}"