#NOTE: context is ..

ARG PYTHON_VERSION=3.7
FROM python:${PYTHON_VERSION}-slim

RUN mkdir /opt/pygraphistry
WORKDIR /opt/pygraphistry

#install tests with stubbed package
COPY setup.py setup.cfg versioneer.py MANIFEST.in ./
COPY graphistry/_version.py ./graphistry/_version.py
RUN \
    pip list \
    && touch graphistry/__init__.py \
    && pip install -e .[dev] \
    && pip list

COPY docker/test-cpu-entrypoint.sh /entrypoint/test-cpu-entrypoint.sh
COPY graphistry ./graphistry
COPY pytest.ini .

ENTRYPOINT ["/entrypoint/test-cpu-entrypoint.sh"]