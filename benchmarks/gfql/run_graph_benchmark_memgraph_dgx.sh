#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RAPIDS_VERSION="${RAPIDS_VERSION:-26.02}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-1}"
GRAPH_BENCHMARK_ROOT="${GRAPH_BENCHMARK_ROOT:-${HOME}/graph-benchmark}"
RESULTS_DIR="${RESULTS_DIR:-plans/gfql-memgraph-benchmarks/results}"
MEMGRAPH_CONTAINER="${MEMGRAPH_CONTAINER:-gfql-bench-memgraph}"
MEMGRAPH_IMAGE="${MEMGRAPH_IMAGE:-memgraph/memgraph-mage}"
MEMGRAPH_PORT="${MEMGRAPH_PORT:-7687}"
MEMGRAPH_URI="${MEMGRAPH_URI:-bolt://127.0.0.1:${MEMGRAPH_PORT}}"
MEMGRAPH_BATCH_SIZE="${MEMGRAPH_BATCH_SIZE:-5000}"
MEMGRAPH_LOAD_METHOD="${MEMGRAPH_LOAD_METHOD:-csv}"
MEMGRAPH_CSV_DIR="${MEMGRAPH_CSV_DIR:-/tmp/gfql_memgraph_import}"
GFQL_QUERY_VARIANT="${GFQL_QUERY_VARIANT:-standard}"
START_MEMGRAPH="${START_MEMGRAPH:-1}"
KEEP_MEMGRAPH="${KEEP_MEMGRAPH:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

case "${RAPIDS_VERSION}" in
    25.02)
        RAPIDS_IMAGE="${RAPIDS_IMAGE:-nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12}"
        PIP_PRE_DEPS_DEFAULT="numba-cuda==0.22.2 cuda-bindings==12.9.5 cuda-core==0.3.2 cuda-python==12.9.5"
        ;;
    26.02)
        RAPIDS_IMAGE="${RAPIDS_IMAGE:-nvcr.io/nvidia/rapidsai/base:26.02-cuda12-py3.13}"
        PIP_PRE_DEPS_DEFAULT=""
        ;;
    *)
        echo "Unsupported RAPIDS_VERSION=${RAPIDS_VERSION}; expected 25.02 or 26.02" >&2
        exit 2
        ;;
esac

PIP_PRE_DEPS="${PIP_PRE_DEPS:-${PIP_PRE_DEPS_DEFAULT}}"
PIP_DEPS="${PIP_DEPS:--e .[test] neo4j}"

if [[ ! -d "${GRAPH_BENCHMARK_ROOT}/data/output/nodes" || ! -d "${GRAPH_BENCHMARK_ROOT}/data/output/edges" ]]; then
    cat >&2 <<EOF
Missing generated graph-benchmark data under ${GRAPH_BENCHMARK_ROOT}/data/output.
Create it on dgx-spark first, for example:
  git clone https://github.com/prrao87/graph-benchmark.git ${GRAPH_BENCHMARK_ROOT}
  cd ${GRAPH_BENCHMARK_ROOT} && bash generate_data.sh 100000
EOF
    exit 2
fi

if [[ "${RESULTS_DIR}" = /* ]]; then
    HOST_RESULTS_DIR="${RESULTS_DIR}"
else
    HOST_RESULTS_DIR="${REPO_ROOT}/${RESULTS_DIR}"
fi
mkdir -p "${HOST_RESULTS_DIR}"

if [[ "${START_MEMGRAPH}" == "1" ]]; then
    docker rm -f "${MEMGRAPH_CONTAINER}" >/dev/null 2>&1 || true
    docker run -d \
        --name "${MEMGRAPH_CONTAINER}" \
        -p "${MEMGRAPH_PORT}:7687" \
        -v /tmp:/tmp \
        "${MEMGRAPH_IMAGE}" >/dev/null
fi

cleanup() {
    if [[ "${START_MEMGRAPH}" == "1" && "${KEEP_MEMGRAPH}" != "1" ]]; then
        docker rm -f "${MEMGRAPH_CONTAINER}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

DOCKER_SCRIPT=$(cat <<'EOF'
set -euo pipefail
if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate base || true
    fi
fi
python --version
if [[ "${INSTALL_DEPS}" == "1" ]]; then
    python -m pip install --upgrade pip
    if [[ -n "${PIP_PRE_DEPS}" ]]; then
        python -m pip install ${PIP_PRE_DEPS}
    fi
    python -m pip install ${PIP_DEPS}
fi
python - <<'PY'
import cudf, graphistry
print({"cudf": cudf.__version__, "graphistry": graphistry.__version__})
PY
mkdir -p "${RESULTS_DIR}"
python benchmarks/gfql/graph_benchmark_q1_q9.py \
    --graph-benchmark-root "${GRAPH_BENCHMARK_ROOT}" \
    --engine pandas \
    --mode preindexed \
    --include-preindex \
    --query-variant "${GFQL_QUERY_VARIANT}" \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --output-json "${RESULTS_DIR}/graph_benchmark_gfql_cpu.json"
python benchmarks/gfql/graph_benchmark_q1_q9.py \
    --graph-benchmark-root "${GRAPH_BENCHMARK_ROOT}" \
    --engine cudf \
    --mode preindexed \
    --include-preindex \
    --query-variant "${GFQL_QUERY_VARIANT}" \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --output-json "${RESULTS_DIR}/graph_benchmark_gfql_gpu.json"
python benchmarks/gfql/graph_benchmark_memgraph_q1_q9.py \
    --graph-benchmark-root "${GRAPH_BENCHMARK_ROOT}" \
    --uri "${MEMGRAPH_URI}" \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --batch-size "${MEMGRAPH_BATCH_SIZE}" \
    --load-method "${MEMGRAPH_LOAD_METHOD}" \
    --csv-dir "${MEMGRAPH_CSV_DIR}" \
    --output-json "${RESULTS_DIR}/graph_benchmark_memgraph.json"
python benchmarks/gfql/graph_benchmark_compare.py \
    --gfql-cpu "${RESULTS_DIR}/graph_benchmark_gfql_cpu.json" \
    --gfql-gpu "${RESULTS_DIR}/graph_benchmark_gfql_gpu.json" \
    --memgraph "${RESULTS_DIR}/graph_benchmark_memgraph.json" \
    --output-md "${RESULTS_DIR}/graph_benchmark_gfql_memgraph.md"
EOF
)

docker run --rm \
    --gpus all \
    --network host \
    --security-opt seccomp=unconfined \
    --entrypoint /bin/bash \
    -e INSTALL_DEPS="${INSTALL_DEPS}" \
    -e PIP_PRE_DEPS="${PIP_PRE_DEPS}" \
    -e PIP_DEPS="${PIP_DEPS}" \
    -e GRAPH_BENCHMARK_ROOT="${GRAPH_BENCHMARK_ROOT}" \
    -e RESULTS_DIR="${RESULTS_DIR}" \
    -e RUNS="${RUNS}" \
    -e WARMUP="${WARMUP}" \
    -e MEMGRAPH_URI="${MEMGRAPH_URI}" \
    -e MEMGRAPH_BATCH_SIZE="${MEMGRAPH_BATCH_SIZE}" \
    -e MEMGRAPH_LOAD_METHOD="${MEMGRAPH_LOAD_METHOD}" \
    -e MEMGRAPH_CSV_DIR="${MEMGRAPH_CSV_DIR}" \
    -e GFQL_QUERY_VARIANT="${GFQL_QUERY_VARIANT}" \
    -v "${REPO_ROOT}:/workspace" \
    -v "${GRAPH_BENCHMARK_ROOT}:${GRAPH_BENCHMARK_ROOT}:ro" \
    -v /tmp:/tmp \
    -w /workspace \
    "${RAPIDS_IMAGE}" \
    -lc "${DOCKER_SCRIPT}"

echo "Wrote ${HOST_RESULTS_DIR}/graph_benchmark_gfql_memgraph.md"
