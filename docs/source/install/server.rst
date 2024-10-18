Using a Server with PyGraphistry
=================================

While PyGraphistry offers robust functionalities out of the box, leveraging a server enhances its capabilities, especially for GPU-accelerated visualizations and remote operations. This guide helps you decide whether to use PyGraphistry without a server or to set up a server using various available options.

Using PyGraphistry Without a Server
-----------------------------------

For most use cases, PyGraphistry can operate seamlessly without the need for a dedicated server. This setup is ideal for:

- **Local Data Visualization**: Create and interact with visualizations directly within your local environment.
- **Basic Graph Analytics**: Perform standard graph operations and analyses without the overhead of server management.
- **Development and Testing**: Ideal for developers building and testing applications that utilize PyGraphistry.

**Note**: Without a server, advanced features like GPU-accelerated visualizations and certain remote capabilities will not be available.

Using a Graphistry Server
-------------------------

To unlock the full potential of PyGraphistry, especially for GPU-accelerated visualizations and scalable remote operations, consider setting up a Graphistry server. Below are the available options to get started:

Graphistry Hub
~~~~~~~~~~~~~~

**Graphistry Hub** offers a managed solution with the following benefits:

- **Ease of Use**: No installation required; get started immediately.
- **Free Cloud GPU Tier**: Access free GPU resources for accelerated visualizations.
- **Scalability**: Automatically scales with your project needs.

**Getting Started with Graphistry Hub**:

- Visit the `Graphistry Get Started <https://www.graphistry.com/get-started>`_ page.
- Choose **Graphistry Hub** to create an account and start using the service without any infrastructure setup.

Cloud Marketplace Deployments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deploying Graphistry on cloud platforms like **AWS** and **Azure** provides flexibility and control over your server environment.

AWS Marketplace
^^^^^^^^^^^^^^^

- **Quick Deployment**: Launch Graphistry with pre-configured settings optimized for AWS.
- **Integration**: Seamlessly integrate with other AWS services for enhanced functionality.

**Deploy on AWS**:

- Navigate to the `AWS Marketplace <https://aws.amazon.com/marketplace/>`_ and search for "Graphistry."
- Follow the deployment instructions to set up your Graphistry server on AWS.

Azure Marketplace
^^^^^^^^^^^^^^^^^^

- **Azure Integration**: Leverage Azure's robust infrastructure and services.
- **Scalable Resources**: Adjust resources based on your project's demands.

**Deploy on Azure**:

- Visit the `Azure Marketplace <https://azuremarketplace.microsoft.com/>`_ and search for "Graphistry."
- Follow the provided steps to deploy Graphistry on Azure.

Kubernetes and Docker-Compose Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For organizations preferring containerized deployments, Graphistry offers support for **Kubernetes** and **Docker-Compose**.

Kubernetes
^^^^^^^^^^

- **Orchestration**: Manage containerized applications with Kubernetes for scalability and reliability.
- **Customization**: Tailor the deployment to fit your infrastructure and scaling requirements.

**Deploy with Kubernetes**:

- Access the Kubernetes deployment guides at the `Graphistry CLI Admin Guide <https://github.com/graphistry/graphistry-cli>`_.
- Follow the instructions to deploy and manage your Graphistry server on a Kubernetes cluster.

Docker-Compose
~~~~~~~~~~~~~~~

- **Simplicity**: Ideal for smaller deployments or development environments.
- **Quick Setup**: Deploy Graphistry using Docker-Compose with minimal configuration.

**Deploy with Docker-Compose**:

- Refer to the `Graphistry CLI Admin Guide <https://github.com/graphistry/graphistry-cli>`_ for Docker-Compose setup instructions.
- Execute the provided Docker-Compose files to launch your Graphistry server locally or on a server.

Choosing the Right Option
-------------------------

- **For Beginners or Quick Setup**: Use **Graphistry Hub** for a hassle-free experience.
- **For Enterprise or Scalable Needs**: Deploy via **AWS** or **Azure Marketplace** to leverage cloud infrastructure.
- **For Containerized Environments**: Opt for **Kubernetes** or **Docker-Compose** to integrate with your existing container orchestration workflows.

Happy graphing!
