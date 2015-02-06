#!/usr/bin/env bash

# Graphistry-custom Bash configuration for CUDA
# WARNING: Do not edit! This file is managed by Ansible. Changes will be lost.

# Only execute this file if we haven't already done so
if [[ $_GRAPHISTRY_CUDA_INIT != 1 ]]; then
    # Setup environment variables for using CUDA/OpenCL
    export DISPLAY=:0
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH+:}/usr/local/cuda/lib64"
    export LIBRARY_PATH="${LIBRARY_PATH+:}/usr/local/cuda/lib64"
    export C_INCLUDE_PATH="${C_INCLUDE_PATH+:}/usr/local/cuda/include"
    export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH+:}/usr/local/cuda/include"

    export _GRAPHISTRY_CUDA_INIT=1
fi