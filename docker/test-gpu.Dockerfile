#NOTE: context is ..

ARG RAPIDS_IMAGE=rapidsai/base:25.12-cuda13-py3.12
FROM ${RAPIDS_IMAGE}
ARG PIP_DEPS="-e .[umap-learn,test,ai]"

SHELL ["/bin/bash", "-lc"]
USER root

RUN mkdir /opt/pygraphistry
WORKDIR /opt/pygraphistry

#install tests with stubbed package
COPY README.md setup.py setup.cfg versioneer.py MANIFEST.in ./
COPY graphistry/_version.py ./graphistry/_version.py

RUN --mount=type=cache,target=/root/.cache \
    python --version \
    && pip --version \
    && python -c "import cudf, cugraph, cuml; print('cudf', cudf.__version__); print('cugraph', cugraph.__version__); print('cuml', cuml.__version__)" \
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
