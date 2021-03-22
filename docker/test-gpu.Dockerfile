#NOTE: context is ..

ARG BASE_VERSION=v2.36.10
ARG CUDA_SHORT_VERSION=10.2
FROM graphistry/graphistry-nvidia:${BASE_VERSION}-${CUDA_SHORT_VERSION}

RUN mkdir /opt/pygraphistry
WORKDIR /opt/pygraphistry

#install tests with stubbed package
COPY README.md setup.py setup.cfg versioneer.py MANIFEST.in ./
COPY graphistry/_version.py ./graphistry/_version.py
RUN \
    source activate rapids \
    && pip list \
    && touch graphistry/__init__.py \
    && pip install -e .[dev] \
    && pip list

COPY docker/test-cpu-entrypoint.sh /entrypoint/test-cpu-entrypoint.sh
COPY bin ./bin
COPY mypy.ini .
COPY pytest.ini .
COPY graphistry ./graphistry

ENV RAPIDS=1
ENV TEST_CUDF=1
ENV TEST_DASK=1
ENV TEST_DASK_CUDF=1
ENV TEST_PANDAS=1

ENTRYPOINT ["/entrypoint/test-cpu-entrypoint.sh"]