#!/bin/bash
# Convenience script to run notebook tests in Docker
# Usage: ./docker/test-notebooks.sh [options]
#
# Options:
#   --type TYPE        Test type: nbval (default), nbmake, or papermill
#   --path PATH        Path to notebooks (default: demos)
#   --timeout SECONDS  Timeout per notebook (default: 300)
#   --parallel         Enable parallel testing (default for nbval/nbmake)
#   --no-parallel      Disable parallel testing
#   --build            Force rebuild of Docker image
#   --gpu              Use GPU image instead of CPU
#
# Examples:
#   ./docker/test-notebooks.sh
#   ./docker/test-notebooks.sh --type nbmake --path demos/more_examples
#   ./docker/test-notebooks.sh --timeout 600 --no-parallel
#   ./docker/test-notebooks.sh --gpu --build

set -e

# Default values
TEST_TYPE="nbval"
NOTEBOOK_PATH="demos"
TIMEOUT="300"
PARALLEL="auto"
BUILD_FLAG=""
USE_GPU=0
DOCKER_TAG="${DOCKER_TAG:-latest}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --path)
            NOTEBOOK_PATH="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="auto"
            shift
            ;;
        --no-parallel)
            PARALLEL="none"
            shift
            ;;
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --gpu)
            USE_GPU=1
            shift
            ;;
        *)
            # Pass unknown arguments to pytest
            break
            ;;
    esac
done

# Set image name based on GPU flag
if [ "$USE_GPU" -eq 1 ]; then
    IMAGE_NAME="graphistry/test-notebooks-gpu:${DOCKER_TAG}"
    DOCKERFILE="test-notebooks-gpu.Dockerfile"
    RUNTIME_FLAGS="--gpus all"
else
    IMAGE_NAME="graphistry/test-notebooks:${DOCKER_TAG}"
    DOCKERFILE="test-notebooks.Dockerfile"
    RUNTIME_FLAGS=""
fi

echo "=== Notebook Testing Configuration ==="
echo "Docker Image: $IMAGE_NAME"
echo "Test Type: $TEST_TYPE"
echo "Notebook Path: $NOTEBOOK_PATH"
echo "Timeout: $TIMEOUT seconds"
echo "Parallel: $PARALLEL"
echo "Additional pytest args: $@"

# Build image if requested or if it doesn't exist
if [ -n "$BUILD_FLAG" ] || ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "=== Building Docker image ==="
    docker build \
        -f "docker/$DOCKERFILE" \
        -t "$IMAGE_NAME" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION:-3.9}" \
        .
fi

# Run tests
echo "=== Running notebook tests ==="
docker run \
    --rm \
    -v "$(pwd):/opt/pygraphistry" \
    -e TEST_TYPE="$TEST_TYPE" \
    -e NOTEBOOK_PATH="$NOTEBOOK_PATH" \
    -e TIMEOUT="$TIMEOUT" \
    -e PARALLEL="$PARALLEL" \
    $RUNTIME_FLAGS \
    "$IMAGE_NAME" \
    $@