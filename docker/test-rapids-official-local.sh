#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

resolve_default_image() {
    local rapids_version="$1"
    case "$rapids_version" in
        25.02)
            echo "nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12"
            ;;
        26.02)
            echo "nvcr.io/nvidia/rapidsai/base:26.02-cuda12-py3.13"
            ;;
        *)
            echo "Unsupported RAPIDS_VERSION: ${rapids_version}" >&2
            return 1
            ;;
    esac
}

resolve_profile_pip_deps() {
    local profile="$1"
    case "$profile" in
        basic|gfql)
            echo "-e .[test]"
            ;;
        ai)
            echo "-e .[test,testai,ai]"
            ;;
        *)
            echo "Unsupported PROFILE: ${profile}" >&2
            return 1
            ;;
    esac
}

resolve_profile_pip_pre_deps() {
    local profile="$1"
    case "$profile" in
        basic|gfql)
            echo ""
            ;;
        ai)
            echo "--index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu"
            ;;
        *)
            echo "Unsupported PROFILE: ${profile}" >&2
            return 1
            ;;
    esac
}

resolve_profile_test_files() {
    local profile="$1"
    local with_gpu="${2:-1}"
    case "$profile" in
        basic)
            cat <<'EOF'
graphistry/tests/test_hyper_dask.py::test_HyperBindings_mt
graphistry/tests/test_hyper_dask.py::test_HyperBindings_override
EOF
            ;;
        gfql)
            if [[ "$with_gpu" == "1" ]]; then
                cat <<'EOF'
graphistry/tests/compute/gfql/cypher/test_parser.py
graphistry/tests/compute/gfql/test_row_pipeline_ops.py
graphistry/tests/compute/gfql/cypher/test_lowering.py::test_graph_constructor_cudf_support
graphistry/tests/compute/gfql/cypher/test_lowering.py::test_string_cypher_formats_filtered_edge_entity_projection_on_cudf
graphistry/tests/compute/gfql/cypher/test_lowering.py::test_string_cypher_executes_real_cugraph_node_row_call_on_cudf
EOF
            else
                cat <<'EOF'
graphistry/tests/compute/gfql/cypher/test_parser.py
graphistry/tests/compute/gfql/test_row_pipeline_ops.py::test_row_pipeline_select_supports_range_scalar_function
graphistry/tests/compute/gfql/test_row_pipeline_ops.py::test_row_pipeline_select_supports_range_with_constant_series_bounds
EOF
            fi
            ;;
        ai)
            if [[ "$with_gpu" == "1" ]]; then
                cat <<'EOF'
graphistry/tests/test_umap_utils.py
graphistry/tests/test_embed_utils.py::TestEmbedCUDF::test_embed_out_basic
EOF
            else
                cat <<'EOF'
graphistry/tests/test_umap_utils.py
EOF
            fi
            ;;
        *)
            echo "Unsupported PROFILE: ${profile}" >&2
            return 1
            ;;
    esac
}

echo "CONFIG"

RAPIDS_VERSION=${RAPIDS_VERSION:-26.02}
RAPIDS_IMAGE=${RAPIDS_IMAGE:-}
PROFILE=${PROFILE:-basic}
WITH_GPU=${WITH_GPU:-1}
WITH_IMAGE_BUILD=${WITH_IMAGE_BUILD:-1}
WITH_LINT=${WITH_LINT:-0}
WITH_TYPECHECK=${WITH_TYPECHECK:-0}
WITH_TEST=${WITH_TEST:-1}
WITH_BUILD=${WITH_BUILD:-0}
LOG_LEVEL=${LOG_LEVEL:-DEBUG}
SENTENCE_TRANSFORMER=${SENTENCE_TRANSFORMER:-}
IMAGE_TAG=${IMAGE_TAG:-graphistry/test-rapids-official:${RAPIDS_VERSION}-${PROFILE}}
PIP_DEPS=${PIP_DEPS:-}
PIP_PRE_DEPS=${PIP_PRE_DEPS:-}
TEST_FILES=${TEST_FILES:-}

if [[ -z "${IMPORT_SMOKE:-}" ]]; then
    IMPORT_SMOKE='python -c "import cudf, cugraph, cuml, dask_cudf, graphistry; print({\"cudf\": cudf.__version__, \"cuml\": cuml.__version__, \"graphistry\": graphistry.__version__, \"cugraph_module\": cugraph.__name__, \"dask_cudf_module\": dask_cudf.__name__})"'
fi

if [[ -z "$RAPIDS_IMAGE" ]]; then
    RAPIDS_IMAGE="$(resolve_default_image "$RAPIDS_VERSION")"
fi

if [[ -z "$PIP_DEPS" ]]; then
    PIP_DEPS="$(resolve_profile_pip_deps "$PROFILE")"
fi

if [[ -z "$PIP_PRE_DEPS" ]]; then
    PIP_PRE_DEPS="$(resolve_profile_pip_pre_deps "$PROFILE")"
fi

if [[ -z "$TEST_FILES" ]]; then
    TEST_FILES="$(resolve_profile_test_files "$PROFILE" "$WITH_GPU" | tr '\n' ' ')"
fi

echo "RAPIDS_VERSION=${RAPIDS_VERSION}"
echo "RAPIDS_IMAGE=${RAPIDS_IMAGE}"
echo "PROFILE=${PROFILE}"
echo "WITH_GPU=${WITH_GPU}"
echo "IMAGE_TAG=${IMAGE_TAG}"
echo "PIP_PRE_DEPS=${PIP_PRE_DEPS}"
echo "PIP_DEPS=${PIP_DEPS}"
echo "TEST_FILES=${TEST_FILES}"

if [[ "$WITH_IMAGE_BUILD" == "1" ]]; then
    DOCKER_BUILDKIT=1 docker build \
        -f "${SCRIPT_DIR}/test-rapids-official.Dockerfile" \
        -t "${IMAGE_TAG}" \
        --build-arg RAPIDS_IMAGE="${RAPIDS_IMAGE}" \
        --build-arg PIP_PRE_DEPS="${PIP_PRE_DEPS}" \
        --build-arg PIP_DEPS="${PIP_DEPS}" \
        --build-arg SENTENCE_TRANSFORMER="${SENTENCE_TRANSFORMER}" \
        "${REPO_ROOT}"
fi

DOCKER_GPU_ARGS=()
if [[ "$WITH_GPU" == "1" ]]; then
    DOCKER_GPU_ARGS+=(--gpus all)
fi

docker run \
    "${DOCKER_GPU_ARGS[@]}" \
    --security-opt seccomp=unconfined \
    -e PYTEST_CURRENT_TEST=TRUE \
    -e WITH_LINT="${WITH_LINT}" \
    -e WITH_TYPECHECK="${WITH_TYPECHECK}" \
    -e WITH_TEST="${WITH_TEST}" \
    -e WITH_BUILD="${WITH_BUILD}" \
    -e LOG_LEVEL="${LOG_LEVEL}" \
    -v "${REPO_ROOT}/graphistry:/opt/pygraphistry/graphistry:ro" \
    -v "/tmp:/tmp" \
    --rm \
    --entrypoint /bin/bash \
    "${IMAGE_TAG}" \
    -lc "${IMPORT_SMOKE} && /entrypoint/test-cpu-entrypoint.sh --maxfail=1 ${TEST_FILES}"
