# Use the official Sphinx base image
FROM sphinxdoc/sphinx:8.0.2

# Set working directory for docs
WORKDIR /docs


# Step 1: Install necessary system dependencies including Pandoc, with caching for apt-get
COPY docs/requirements-system.txt /requirements-system.txt
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && apt-get install -y $(cat /requirements-system.txt) \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Copy project files needed for pip install (setup.py, setup.cfg, versioneer.py, README.md)
COPY ../../setup.py ../../setup.cfg ../../versioneer.py ../../README.md ./

# Step 3: Install the project dependencies with pip, including docs extras
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -e .[docs]

# Step 4: Copy the graphistry source files into the container
COPY ../../graphistry /graphistry
ENV PYTHONPATH="/graphistry:${PYTHONPATH}"

# Step 5: Copy the build script into the container
COPY docs/docker/build-docs.sh /build-docs.sh
COPY docs/source /docs/source
COPY demos /docs/source/demos
COPY README.md /docs/source/README.md
COPY ARCHITECTURE.md /docs/source/ARCHITECTURE.md
COPY CONTRIBUTING.md /docs/source/CONTRIBUTING.md
COPY DEVELOP.md /docs/source/DEVELOP.md

# Step 6: Set the working directory for Sphinx to the `source/` folder
WORKDIR /docs/source

# Step 7: Run the script to build the documentation
CMD ["/build-docs.sh"]
