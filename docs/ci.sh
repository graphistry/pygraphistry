#!/bin/bash
set -ex

DOCS_FORMAT=${DOCS_FORMAT:-all}  # Default to building all formats if not specified
VALIDATE_NOTEBOOK_EXECUTION=${VALIDATE_NOTEBOOK_EXECUTION:-1}  # Default to validating notebooks

(
    cd docs/docker \
    && docker compose build sphinx \
    && docker compose run --rm \
        -e DOCS_FORMAT=$DOCS_FORMAT \
        -e VALIDATE_NOTEBOOK_EXECUTION=$VALIDATE_NOTEBOOK_EXECUTION \
        sphinx
)
