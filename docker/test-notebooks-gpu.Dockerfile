# GPU-enabled Dockerfile for testing Jupyter notebooks
# Context is ..

ARG BASE_VERSION=v2.41.0
ARG CUDA_SHORT_VERSION=11.8
FROM graphistry/graphistry-minimal:${BASE_VERSION}-${CUDA_SHORT_VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/pygraphistry

# Activate the conda environment
RUN echo "source activate rapids" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install pygraphistry and dependencies
COPY README.md setup.py setup.cfg versioneer.py MANIFEST.in ./
COPY graphistry/_version.py ./graphistry/_version.py
RUN source activate rapids && \
    touch graphistry/__init__.py && \
    pip install -e .[dev]

# Install notebook testing tools
RUN source activate rapids && \
    pip install \
    nbval==0.10.0 \
    nbmake==1.4.6 \
    papermill==2.4.0 \
    jupyter==1.0.0 \
    ipykernel==6.25.2

# Copy the full source
COPY graphistry ./graphistry
COPY demos ./demos
COPY pytest.ini .

# Install the package
RUN source activate rapids && pip install -e .

# Set environment variables for testing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RAPIDS=1

# Entry point script
COPY docker/test-notebooks-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Modify entrypoint to activate conda
RUN sed -i '2a source activate rapids' /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]