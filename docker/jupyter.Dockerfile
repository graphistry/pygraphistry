ARG JUPYTER_IMAGE_TAG=latest
FROM jupyter/minimal-notebook:${JUPYTER_IMAGE_TAG}
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"
USER jovyan

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY --chown=jovyan README.md setup.py setup.cfg versioneer.py MANIFEST.in /var/graphistry/
COPY --chown=jovyan graphistry/_version.py /var/graphistry/graphistry/_version.py
RUN \
    cd /var/graphistry \
    && touch graphistry/__init__.py \
    && pip install -e .
RUN \
    cd /var/graphistry \
    && pip install -e .[gremlin,bolt]

COPY --chown=jovyan graphistry /var/graphistry/graphistry
RUN  \
    cd /var/graphistry \
    && find . \
    && python setup.py install

RUN python -c "import graphistry; print(graphistry.__version__)"