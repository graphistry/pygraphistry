# NOTE: context is ..

ARG RAPIDS_IMAGE=nvcr.io/nvidia/rapidsai/base:26.02-cuda12-py3.13
FROM ${RAPIDS_IMAGE}
ARG PIP_PRE_DEPS=""
ARG PIP_DEPS="-e .[test]"
ARG SENTENCE_TRANSFORMER=""
# Supply-chain: reject packages published in the last N days (pip ≥26: --uploaded-prior-to)
ARG PIP_EXCLUDE_NEWER=6d
ENV PIP_EXCLUDE_NEWER=${PIP_EXCLUDE_NEWER}

SHELL ["/bin/bash", "-lc"]
USER root

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && apt-get install -y \
        unzip \
        wget \
        zip

RUN mkdir /opt/pygraphistry
WORKDIR /opt/pygraphistry

# Install the package and selected non-RAPIDS extras on top of an official RAPIDS image.
COPY README.md setup.py setup.cfg versioneer.py MANIFEST.in ./
COPY graphistry/_version.py ./graphistry/_version.py

RUN --mount=type=cache,target=/root/.cache \
    if command -v conda >/dev/null 2>&1; then \
        conda_base="$(conda info --base 2>/dev/null || true)"; \
        if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then source "$conda_base/etc/profile.d/conda.sh" && conda activate base || true; fi; \
    fi \
    && python --version \
    && pip --version \
    && touch graphistry/__init__.py \
    && echo "PIP_PRE_DEPS: $PIP_PRE_DEPS" \
    && echo "PIP_DEPS: $PIP_DEPS" \
    && pip install --upgrade pip build \
    && if [[ -n "$PIP_PRE_DEPS" ]]; then pip install $PIP_PRE_DEPS; fi \
    && pip install $PIP_DEPS \
    && pip list

RUN --mount=type=cache,target=/root/.cache \
    mkdir -p /models \
    && cd /models \
    && if [[ "${SENTENCE_TRANSFORMER}" == "" ]]; then \
        echo "No sentence transformer specified, skipping"; \
    else \
        ( \
            wget --no-check-certificate \
            "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/${SENTENCE_TRANSFORMER}.zip" \
            && unzip "${SENTENCE_TRANSFORMER}.zip" -d "${SENTENCE_TRANSFORMER}" \
        ); \
    fi

COPY docker/test-cpu-entrypoint.sh /entrypoint/test-cpu-entrypoint.sh
COPY bin ./bin
COPY mypy.ini .
COPY pytest.ini .
COPY graphistry ./graphistry
COPY demos/data ./demos/data

ENV \
    RAPIDS=1 \
    TEST_CUDF=1 \
    TEST_DASK=1 \
    TEST_DASK_CUDF=1 \
    TEST_PANDAS=1 \
    TEST_CUGRAPH=1

ENTRYPOINT ["/entrypoint/test-cpu-entrypoint.sh"]
