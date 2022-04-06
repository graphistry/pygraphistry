#!/bin/bash
set -ex

# Alias for docker-compose

COMPOSE_DOCKER_CLI_BUILD=1 \
DOCKER_BUILDKIT=1 \
    docker-compose $@
