#NOTE: context is ..

ARG GPU_IMAGE=graphistry/graphistry-nvidia:v2.41.0-11.8
FROM ${GPU_IMAGE}
ARG PIP_DEPS="-e .[umap-learn,test,ai]"

SHELL ["/bin/bash", "-lc"]
USER root

RUN mkdir /opt/pygraphistry
WORKDIR /opt/pygraphistry

#install tests with stubbed package
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
    && echo "PIP_DEPS: $PIP_DEPS" \
    && pip install build \
    && pip install $PIP_DEPS \
    && pip list

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
