Installation Guide - Extended
=============================

This extended guide provides detailed instructions for installing PyGraphistry, including optional configurations for enhanced performance and functionality.

GPU Mode System Requirements (Optional)
---------------------------------------

* **Nvidia RAPIDS**: PyGraphistry primarily aligns with Nvidia RAPIDS, so check their requirements for your system:

  * **Volta generation GPUs or newer** are the current Nvidia RAPIDS minimum requirement.

  * **cuDF**: Required.

  * **cuML**, **cuGraph**: Recommended.

* **PyTorch**: PyGraphistry[AI] further aligns with PyTorch for some of its more advanced methods.

Core Dependencies (Installed by Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyGraphistry depends on a small set of standard CPU-based Python data science libraries such as pandas, pyarrow, and numpy. If your system is missing these dependencies, they will get installed automatically.

Optional Dependencies
---------------------

PyGraphistry supports a variety of optional dependencies to extend its functionality.

GPU Acceleration with RAPIDS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable GPU acceleration for DataFrames and graph analytics, install **cuDF**, **cuML**, and **cuGraph** from the NVIDIA RAPIDS suite.

Follow the instructions at the `NVIDIA RAPIDS Installation Guide <https://rapids.ai/start.html>`_.

Additional Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many of the following can be used in both CPU mode and GPU mode.

- **AI Libraries**:

  - `torch` (1GB+): PyTorch and related libraries for advanced AI methods in the PyGraphistry AI packages.

    Install with:

    .. code-block:: bash

      pip install graphistry[ai]

- **Graph Libraries**:

  - `networkx`: Integration with NetworkX graphs.

    Install with:

    .. code-block:: bash

      pip install graphistry[networkx]

  - `igraph`: Support for igraph graphs.

    Install with:

    .. code-block:: bash

      pip install graphistry[igraph]

  - `pygraphviz`: Rendering graphs with Graphviz layouts.

    Install with:

    .. code-block:: bash

      pip install graphistry[pygraphviz]

- **Graph Databases and Protocols**:

  - `gremlinpython`: Working with Gremlin graph databases.

    Install with:

    .. code-block:: bash

      pip install graphistry[gremlin]

  - `neo4j`, `neotime`: Connecting to Neo4j via the Bolt protocol.

    Install with:

    .. code-block:: bash

      pip install graphistry[bolt]

- **Data Formats**:

  - `openpyxl`, `xlrd`: Reading NodeXL files.

    Install with:

    .. code-block:: bash

      pip install graphistry[nodexl]

- **Machine Learning and AI**:

  - `umap-learn`, `dirty-cat`, `scikit-learn`: For dimensionality reduction and clustering.

    Install with:

    .. code-block:: bash

      pip install graphistry[umap-learn]

  - `scipy`, `dgl`, `torch<2`, `sentence-transformers`, `faiss-cpu`, `joblib`: Advanced AI functionalities.

    Install with:

    .. code-block:: bash

      pip install graphistry[ai]

- **Jupyter Support**:

  - `ipython`: Enhanced Jupyter notebook integration.

    Install with:

    .. code-block:: bash

      pip install graphistry[jupyter]

Installing Multiple Extras
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install multiple extras by listing them separated by commas:

.. code-block:: bash

  pip install graphistry[networkx,umap-learn]

Installing All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install all optional dependencies (not generally recommended due to size and potential conflicts):

.. code-block:: bash

  pip install graphistry[all]

Common Questions
----------------

Do I Need a Server?
~~~~~~~~~~~~~~~~~~~

- **No**, you can run GFQL and other PyGraphistry CPU and GPU components locally. To use the full visualization capabilities, you do need access to a Graphistry server.

- **Options**:

  - **Graphistry Hub**: Use the public Graphistry Hub at `hub.graphistry.com <https://hub.graphistry.com/>`_.

  - **Self-Hosted Server**: Set up your own Graphistry server by following the deployment instructions in the `Graphistry CLI Admin Guide <https://github.com/graphistry/graphistry-cli>`_.

Can I Use PyGraphistry Without GPU Support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Yes**, PyGraphistry can be used without GPU support.

- **GPU Acceleration**: To leverage GPU acceleration, install optional GPU libraries like cuDF and have compatible hardware.

What Are the Benefits of Installing Optional Dependencies?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Enhanced Functionality**: Support for different graph formats, advanced analytics, machine learning, and integration with various tools and databases. For example, for visualization users needing careful layout of small trees, we recommend `pygraphviz`, while for users of big GFQL workloads, we recommend RAPIDS.

- **Customization**: Install only what you need for your specific use case.

How Do I Install Development Dependencies?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers who wish to work on PyGraphistry itself, we recommend using Docker, or for native development:

- **Install with**:

  .. code-block:: bash

    pip install graphistry[dev]

- **Includes**: Testing tools, documentation tools, and other development dependencies like `flake8`, `pytest`, `sphinx`, etc.

References
----------

- **PyGraphistry GitHub Repository**: `https://github.com/graphistry/pygraphistry <https://github.com/graphistry/pygraphistry>`_
- **Graphistry Get Started**: `https://www.graphistry.com/get-started <https://www.graphistry.com/get-started>`_
- **Graphistry CLI Admin Guide**: `https://github.com/graphistry/graphistry-cli <https://github.com/graphistry/graphistry-cli>`_
- **NVIDIA RAPIDS Installation Guide**: `https://rapids.ai/start.html <https://rapids.ai/start.html>`_
- **Graphistry Documentation**: `https://hub.graphistry.com/docs/ <https://hub.graphistry.com/docs/>`_

Happy graphing!
