#!/bin/bash
set -ex

(
    cd docker \
    && docker compose build \
    && docker compose run --rm sphinx
)
