#!/bin/bash

mkdir -p build

docker run \
    --entrypoint=/bin/bash \
    --rm -it \
    -e USER_ID=$UID \
    -v $(pwd)/..:/doc \
    ddidier/sphinx-doc:3.2.1-1 \
    -c "pip install pandas numpy pyarrow && cd docs && ./build.sh"

