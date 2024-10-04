Installation Guide
==================

This guide will help you install PyGraphistry and its dependencies, as well as walk you through optional configurations for enhanced performance and functionality.

If you are interested in capacity planning for running your own backend server, see the `Graphistry Admin Guide <hhttps://github.com/graphistry/graphistry-cli>`_.

Minimum System Requirements
----------------------------

Before installing PyGraphistry, ensure your system meets the following minimum requirements:

- **Operating System**: Windows, macOS, Linux, or any Python-capable environment.
- **Python Version**: Python 3.8 or higher.
- **Hardware**:
  - **CPU**: 1 core
  - **Memory**: 1 GB (in addition to regular OS requirements)
  - **GPU**: While optional, we recommend using a browser with WebGL enabled and a GPU, which is most phones and laptops

GPU Mode System Requirements (Optional)
---------------------------------------

* Nvidia RAPIDS: PyGraphistry primarily aligns with Nvidia RAPIDS, so check their requirements for your system
  * Volta generation GPUs or newer are the current Nvidia RAPIDS minimum requirement
  * cuDF: Required
  * cuML, cuGraph: Recommended
* PyTorch: PyGraphistry[AI] further aligns with PyTorch for some of its more advanced methods

Installing PyGraphistry
-----------------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install PyGraphistry using `pip`:

```bash
pip install graphistry
```

Importing and Version Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify the installation by importing PyGraphistry and checking its version:

.. code-block:: python

    import graphistry
    print(graphistry.__version__)


Log in to a Graphistry GPU Server
---------------------------------

While almost all of PyGraphistry can run locally, to use the PyGraphistry's visualization server, PyGraphistry needs to connecto to a Graphistry GPU server:

- **Get an Account**: Head to the server `Getting Started <https://www.graphistry.com/get-started>`_  page and pick:
  - **Graphistry Hub**: For immediate access with no install, use the public Graphistry Hub, which includes free GPU accounts
  - **Self-Host**: Quick launch on AWS/Azure, or contact staff for on-prem options

- **Log in**: Once you have an account, register in your Python environment:

  .. code-block:: python

      import graphistry

      graphistry.register(api=3, server='hub.graphistry.com', username='YOUR_USERNAME', password='YOUR_PASSWORD')

  Replace `'YOUR_USERNAME'` and `'YOUR_PASSWORD'` with your actual credentials.

  When the command finishes without an exception, you have successfully connected to the server.

For additional authentication options, see the Login and Sharing guide.

Core Dependencies (Installed by Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyGraphistry depends on a small set of standard CPU-based Python data science libraries such as pandas, pyarrow, and numpy. If your system is missing these dependencies, they will get installed.

Optional Dependencies
---------------------

PyGraphistry supports a variety of optional dependencies.

GPU Acceleration with RAPIDS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable GPU acceleration for DataFrames and graph analytics, install **cuDF**, **cuML**, and **cuGraph** from the NVIDIA RAPIDS suite.

Follow the instructions at `NVIDIA RAPIDS Installation Guide <https://rapids.ai/start.html>`_

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Note that many of the below can be used in both CPU mode and GPU mode.

- **AI Libraries**:
  - `torch` (1GB+): PyTorch and related libraries for advanced AI methods in the PyGraphistry AI packages.

    Install with:

    ```
    pip install graphistry[ai]
    ```

- **Graph Libraries**:
  - `networkx`: Integration with NetworkX graphs.

    Install with:

    ```
    pip install graphistry[networkx]
    ```

  - `igraph`: Support for igraph graphs.

    Install with:

    ```
    pip install graphistry[igraph]
    ```

  - `pygraphviz`: Rendering graphs with Graphviz layouts.

    Install with:

    ```
    pip install graphistry[pygraphviz]
    ```

- **Graph Databases and Protocols**:
  - `gremlinpython`: Working with Gremlin graph databases.

    Install with:

    ```
    pip install graphistry[gremlin]
    ```

  - `neo4j`, `neotime`: Connecting to Neo4j via the Bolt protocol.

    Install with:

    ```
    pip install graphistry[bolt]
    ```

- **Data Formats**:
  - `openpyxl`, `xlrd`: Reading NodeXL files.

    Install with:

    ```
    pip install graphistry[nodexl]
    ```

- **Machine Learning and AI**:

  - `umap-learn`, `dirty-cat`, `scikit-learn`: For dimensionality reduction and clustering.

    Install with:

    ```
    pip install graphistry[umap-learn]
    ```

  - `scipy`, `dgl`, `torch<2`, `sentence-transformers`, `faiss-cpu`, `joblib`: Advanced AI functionalities.

    Install with:

    ```
    pip install graphistry[ai]
    ```

- **Jupyter Support**:
  - `ipython`: Enhanced Jupyter notebook integration.

    Install with:

    ```
    pip install graphistry[jupyter]
    ```

Installing Multiple Extras
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install multiple extras by listing them separated by commas:

```bash
pip install graphistry[networkx,umap-learn]
```


Installing All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install all optional dependencies (not generally recommended due to size and potential conflicts):

```bash
pip install graphistry[all]
```


Common Questions
----------------

Do I Need a Server?
~~~~~~~~~~~~~~~~~~~~

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

```bash
pip install graphistry[dev]
```

- **Includes**: Testing tools, documentation tools, and other development dependencies like `flake8`, `pytest`, `sphinx`, etc.

References
----------

- **PyGraphistry GitHub Repository**: `https://github.com/graphistry/pygraphistry <https://github.com/graphistry/pygraphistry>`_
- **Graphistry Get Started**: `https://www.graphistry.com/get-started <https://www.graphistry.com/get-started>`_
- **Graphistry CLI Admin Guide**: `https://github.com/graphistry/graphistry-cli <https://github.com/graphistry/graphistry-cli>`_
- **NVIDIA RAPIDS Installation Guide**: `https://rapids.ai/start.html <https://rapids.ai/start.html>`_
- **Graphistry Documentation**: `https://hub.graphistry.com/docs/ <https://hub.graphistry.com/docs/>`_

Happy graphing!

