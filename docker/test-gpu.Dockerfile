#NOTE: context is ..

ARG BASE_VERSION=v2.39.17
ARG CUDA_SHORT_VERSION=11.4
FROM graphistry/graphistry-nvidia:${BASE_VERSION}-${CUDA_SHORT_VERSION}
ARG PIP_DEPS="-e .[umap-learn,test]"

RUN mkdir /opt/pygraphistry
WORKDIR /opt/pygraphistry

#install tests with stubbed package
COPY README.md setup.py setup.cfg versioneer.py MANIFEST.in ./
COPY graphistry/_version.py ./graphistry/_version.py

RUN --mount=type=cache,target=/root/.cache \
    source activate rapids \
    && pip list \
    && touch graphistry/__init__.py \
    && echo "PIP_DEPS: $PIP_DEPS" \
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