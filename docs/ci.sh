#!/bin/bash
set -ex

DOCS_FORMAT=${DOCS_FORMAT:-all}  # Default to building all formats if not specified

(
    cd docker \
    && docker compose build \
    && docker compose run --rm \
        -e DOCS_FORMAT=$DOCS_FORMAT \
        -e VALIDATE_NOTEBOOK_EXECUTION=${VALIDATE_NOTEBOOK_EXECUTION:-0} \
        sphinx
)
