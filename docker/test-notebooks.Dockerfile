# Dockerfile for testing Jupyter notebooks
# Context is ..

ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/pygraphistry

# Create virtual environment
RUN python -m venv venv
ENV PATH="/opt/pygraphistry/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install pygraphistry and dependencies
COPY README.md setup.py setup.cfg versioneer.py MANIFEST.in ./
COPY graphistry/_version.py ./graphistry/_version.py
RUN touch graphistry/__init__.py && \
    pip install -e .[dev]

# Install notebook testing tools
RUN pip install \
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
RUN pip install -e .

# Set environment variables for testing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Entry point script
COPY docker/test-notebooks-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]