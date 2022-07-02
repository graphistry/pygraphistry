#NOTE: context is ..

ARG PYTHON_VERSION=3.6
FROM python:${PYTHON_VERSION}-slim
ARG PIP_DEPS="-e .[dev]"
SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && apt-get install -y \
        unzip \
        wget \
        zip

RUN mkdir /opt/pygraphistry
WORKDIR /opt/pygraphistry

RUN --mount=type=cache,target=/root/.cache \
    python -m venv pygraphistry \
    && source pygraphistry/bin/activate \
    && pip install --upgrade pip

#install tests with stubbed package
COPY README.md setup.py setup.cfg versioneer.py MANIFEST.in ./
COPY graphistry/_version.py ./graphistry/_version.py
RUN --mount=type=cache,target=/root/.cache \
    source pygraphistry/bin/activate \
    && pip list \
    && touch graphistry/__init__.py \
    && echo "PIP_DEPS: $PIP_DEPS" \
    && pip install $PIP_DEPS \
    && pip list

ARG SENTENCE_TRANSFORMER=""
RUN --mount=type=cache,target=/root/.cache \
    source pygraphistry/bin/activate \
    && mkdir -p /models \
    && cd /models \
    && if [[ "${SENTENCE_TRANSFORMER}" == "" ]]; then \
        echo "No sentence transformer specified, skipping"; \
    else \
        ( \
            wget --no-check-certificate \
            "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/${SENTENCE_TRANSFORMER}.zip" \
            && unzip "${SENTENCE_TRANSFORMER}.zip" -d "${SENTENCE_TRANSFORMER}" \
        ) ; \
    fi
# paraphrase-albert-small-v2  : 40mb
# paraphrase-MiniLM-L3-v2 (default): 60mb
# average_word_embeddings_komninos: 300mb

COPY docker/test-cpu-entrypoint.sh /entrypoint/test-cpu-entrypoint.sh
COPY bin ./bin
COPY mypy.ini .
COPY pytest.ini .
COPY graphistry ./graphistry
COPY demos/data ./demos/data

ENV RAPIDS=0
ENV TEST_CUDF=0
ENV TEST_DASK=0
ENV TEST_DASK_CUDF=0
ENV TEST_PANDAS=1

ENTRYPOINT ["/entrypoint/test-cpu-entrypoint.sh"]