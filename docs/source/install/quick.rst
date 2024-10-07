Installation Guide - Quick Start
=================================

This quick start guide will help you install PyGraphistry and its essential dependencies to get you up and running quickly.

Minimum System Requirements
----------------------------

Before installing PyGraphistry, ensure your system meets the following minimum requirements:

- **Operating System**: Windows, macOS, Linux, or any Python-capable environment

- **Python Version**: Python 3.8 or higher

- **Hardware**:

  - **CPU**: 1 core

  - **Memory**: 1 GB - in addition to regular OS requirements

  - **GPU**: While optional, we recommend using a browser with WebGL enabled and a GPU, which is most phones and laptops

Installing PyGraphistry
-----------------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install PyGraphistry using `pip`:

.. code-block:: bash

  pip install graphistry

Importing and Version Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify the installation by importing PyGraphistry and checking its version:

.. code-block:: python

    import graphistry
    print(graphistry.__version__)

Log in to a Graphistry GPU Server
---------------------------------

To use PyGraphistry's visualization server, you need to connect to a Graphistry GPU server:

- **Get an Account**: Visit the `Graphistry Get Started <https://www.graphistry.com/get-started>`_ page and choose:

  - **Graphistry Hub**: For immediate access with no installation, use the public Graphistry Hub, which includes free GPU accounts.

  - **Self-Host**: Quick launch on AWS/Azure, or contact staff for on-premises options.

- **Log in**: Once you have an account, register in your Python environment:

  .. code-block:: python

      import graphistry

      graphistry.register(api=3, server='hub.graphistry.com', username='YOUR_USERNAME', password='YOUR_PASSWORD')

  Replace `'YOUR_USERNAME'` and `'YOUR_PASSWORD'` with your actual credentials.

  When the command finishes without an exception, you have successfully connected to the server.

  See the authentication guide for additional options such as logging into an organization, SSO, and using API keys.

For additional authentication options, see the Login and Sharing guide.

Happy graphing!
