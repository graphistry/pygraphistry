#!/usr/bin/env bash

# Graphistry-custom Bash configuration for Node.js global module directory
# WARNING: Do not edit! This file is managed by Ansible. Changes will be lost.

# Only execute this file if we haven't already done so
if [[ $_GRAPHISTRY_NODEJS_INIT != 1 ]]; then
    export npm_config_prefix='/opt/npm_global'
    export NODE_PATH="${NODE_PATH+:}${npm_config_prefix}/lib/node_modules"
    export PATH="${npm_config_prefix}/bin/:$PATH"

    umask 0002

    export _GRAPHISTRY_NODEJS_INIT=1
fi