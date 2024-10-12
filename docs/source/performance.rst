.. _performance:

CPU & GPU Acceleration in PyGraphistry
==============================================

Why PyGraphistry is Fast
------------------------

PyGraphistry is designed for speed. By focusing on **vectorized processing**, it outperforms most graph libraries on standard CPUs. When you leverage GPUs and AI models, PyGraphistry can become **100X+ faster**, enabling real-time analytics and machine learning at  scale. We regularly use it on datasets with millions and billions of rows.

Just as Apache Spark used in-memory processing to replace racks of Hadoop servers with faster and smaller multicore ones, the PyGraphistry ecosystem uses GPU acceleration to increase speeds and decrease costs even further.

Flexible GPU Use: Client and Server
-----------------------------------

Strictly optional, PyGraphistry lets you harness GPUs where they make the most sense for your workflow. For smaller datasets, you can run PyGraphistry on your local GPU. Graph loading, shaping, computing, querying, ML, AI, and visualization tasks all become much more interactive and immediate, making PyGraphistry great for exploration in Jupyter notebooks and dashboards.

For larger datasets and team projects, you can offload PyGraphistry tasks like **GFQL queries** and **visualization ETL**, and even full GPU Python scripts, to shared Graphistry GPU servers. This setup handles enterprise-grade workloads, helping deliver consistent performance across web apps, dashboards, and AI pipelines.

Where PyGraphistry Accelerates with Vector Processing and GPUs
----------------------------------------------------------------

PyGraphistry uses vector processing and GPU acceleration throughout your data workflow.

In data processing, it integrates with **Apache Arrow** to seamlessly transition between **pandas** for algorithmic and hardware acceleration of datasets even on CPUs, and **cuDF** (via `NVIDIA RAPIDS <https://rapids.ai/>`_) for large, GPU-accelerated workloads, keeping your data pipelines efficient at any size. Graphistry is typically used on GPUs with 12-80 GB single-GPU RAM, and we increasingly work with teams experimenting with multi-GPU nodes (128-640 GB GPU RAM) and clusters of them.

For graph querying, **GFQL** leverages GPUs to speed up queries on massive graph datasets, delivering results in seconds on a single GPU even when traversal steps touch hundreds of millions of rows.

In visualization, GPUs enable PyGraphistry to render large, complex graphs in real time. Whether you're investigating cybersecurity threats, monitoring supply chains, or analyzing clickstreams, you get responsive visuals at any scale, locally or via shared servers.

For AI and machine learning, **PyGraphistry[AI]** uses GPUs to accelerate model training and inference for tasks like **UMAP** and **GNNs**, unlocking rapid insights from large graph datasets in areas like security and commercial analytics. When running on real-time data and billions of rows, the combination of GPU training and GPU inferencing unlocks significant velocity.

Easy Deployment Anywhere
------------------------

The Graphistry ecosystem fits into your existing infrastructure.

You can `deploy Graphistry GPU servers <https://www.graphistry.com/get-started>`_ on any modern cloud platform (`AWS <https://aws.amazon.com/>`_, `GCP <https://cloud.google.com/>`_, `Azure <https://azure.microsoft.com/en-us/>`_), and on-premises using **Docker Compose** or **Kubernetes**. PyGraphistry works with any NVIDIA GPU that are `RAPIDS-compatible <https://rapids.ai/>`_.

If you don't have a GPU, no problem. PyGraphistry is a quick `pip install graphistry` away, giving performance optimized for CPU hardware through vectorized columnar processing concepts similar to `ClickHouse <https://clickhouse.com/>`_ and `Apache Spark <https://spark.apache.org/>`_. You can also offload heavy tasks to remote Graphistry shared GPUs, including Graphistry Hub visualization servers.


Trusted Security & Compliance
-----------------------------

Many top organizations with sensitive environments — including global banks and air-gapped government systems — trust PyGraphistry. Regular security practices such as periodic penetration testing ensure systems meets strict security requirements, making it safe for some of the most stringent teams.

Next Steps
----------

Get started with PyGraphistry:

- **Installation Guide**: `Set up PyGraphistry <install/index>`_ .
- **Visualization**: Explore :ref:`10min`.
- **GFQL Documentation**: Start with :ref:`10min-gfql`.
