# Use the official Sphinx base image
FROM sphinxdoc/sphinx:8.0.2

# Set working directory for docs
WORKDIR /docs


# Step 1: Install necessary system dependencies including Pandoc and Graphviz, with caching for apt-get
COPY docs/requirements-system.txt /requirements-system.txt
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && apt-get install -y $(cat /requirements-system.txt) \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Copy project files needed for pip install (setup.py, setup.cfg, versioneer.py, README.md)
COPY ../../setup.py ../../setup.cfg ../../versioneer.py ../../README.md ./

# Step 3: Copy the graphistry source files into the container BEFORE pip install
COPY ../../graphistry /docs/graphistry

# Step 4: Install the project dependencies with pip, including docs extras and register kernel
# Install graphistry in editable mode so it's available for notebook execution
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -e .[docs] && \
    # Pin doc-test-only deps (not in .[docs]/the RTD lock) to cooldown-safe versions —
    # uv-resolved at the repo's supply-chain anchor (exclude-newer 2026-05-14, matches
    # requirements/rtd-py3.12.lock), so a plain pip build never pulls an unpinned latest
    # (0-day) release. Regenerate: uv pip compile <(printf 'pygraphviz\nigraph\nlark\npytest\npolars')
    #   --python-version 3.12 --exclude-newer <rtd-lock-anchor> --no-deps
    # Follow-up (#1669): fold into a single hashed lock like RTD for transitive+hash pinning.
    python3 -m pip install pygraphviz==1.14 igraph==1.0.0 lark==1.3.1 pytest==9.0.3 polars==1.40.1 && \
    python3 -m ipykernel install --sys-prefix --name python3

# Step 5: Copy the build script and config files into the container
COPY docs/docker/build-docs.sh /build-docs.sh
COPY docs/.rstcheck.cfg /docs/.rstcheck.cfg
COPY docs/validate-docs.sh /docs/validate-docs.sh
COPY docs/test_doc_examples.py /docs/test_doc_examples.py
COPY docs/source /docs/source
COPY demos /docs/source/demos
COPY docs/test_notebooks /docs/test_notebooks
COPY README.md /docs/source/README.md
COPY ARCHITECTURE.md /docs/source/ARCHITECTURE.md
COPY CONTRIBUTING.md /docs/source/CONTRIBUTING.md
COPY DEVELOP.md /docs/source/DEVELOP.md

# Step 6: Set the working directory for Sphinx to the `source/` folder
WORKDIR /docs/source

# Step 7: Run the script to build the documentation
CMD ["/build-docs.sh"]
