.. _gfql-performance:

GFQL Performance: Unleashing Vectorization and GPU Power for Scalable Graph Analytics
======================================================================================

GFQL, developed by Graphistry, rethinks graph analytics by harnessing vectorization and GPU acceleration. As datasets grow from thousands to billions of rows, traditional tools struggle to keep up without significant infrastructure investment. GFQL is rewriting the story. Start small with a quick `pip install graphistry` on your CPU system, and scale more smoothly by leveraging the power of vectorization and GPUs to handle historically tricky datasets.

Built from Real-World Necessity
-------------------------------

GFQL was born out of the challenges our team faced across many graph customer projects over the last 10 years. Projects often start with manageable datasets, and as they scale up, require tools that can grow without imposing prohibitive costs or complexities. Likewise, traditional graph solutions often require adding additional storage tier infrastructure and systems of record that duplicate a team's existing standard databases and warehouses: Too many projects died from premature distractions and complexities here.

We have `long recognizing the untapped potential of CPUs and GPUs in the compute tier <https://gradientflow.com/what-is-graph-intelligence/>`_ and the lack of effective libraries to leverage them for graph analytics. GFQL fills this gap. We designed GFQL to integrate seamlessly with the graph and dataframe ecosystem, providing a much easier, unified, and scalable solution while eliminating the need for hazardous storage tier detours.

A New Era of Graph Analytics
----------------------------

Graphistry has a history of award-winning open source data visualization and GPU acceleration engines. With GFQL, we bring our lessons learned to graph querying and analysis for real-time insights on datasets both big and small. Unlike traditional graph databases that process one path at a time, GFQL traverses entire collections simultaneously. Similar to best-of-class analytical CPU databases like Clickhouse and Google BigQuery, our vectorized approach maximizes throughput to drastically reduce query time.

When coupled with GPU acceleration, GFQL's performance reaches Graph 500 levels with even the cheapest cloud GPUs. Modern GPUs execute tens of thousands of threads in parallel, and GFQL is designed to fully saturate this capability. Whether you're traversing graphs with billions of edges or running complex algorithms, GFQL transforms previously impractical tasks into manageable ones.


Three Simple Ideas Behind GFQL's Performance
---------------------------------------------

At the core of GFQL's performance are three pioneering techniques:

**Collection-Oriented Algorithms**

GFQL operates on entire collections of nodes and edges simultaneously, different from older commercial Cypher and Gremlin graph query engines that process one path at a time. The collection-oriented approach, inspired by our research at UC Berkeley and our experience with GPUs, maximizes data throughput and minimizes computational overhead. Small queries stay interactive, and large-scale graph analytics is now more efficient than ever before.

**Vectorized Columnar Processing**

GFQL processes data in large, parallel batches using columnar data structures. This method optimizes memory usage and computational efficiency, significantly speeding up data handling compared to traditional row-based systems. Natively integrating with cutting-edge technologies like `Apache Arrow <https://arrow.apache.org/>`_, this approach ensures high performance even on CPUs, and unusually fast speeds for moving large data across systems.

**Massive Parallelism with GPUs**

Designed to saturate the tens of thousands of threads in modern GPUs, GFQL enables rapid processing of complex graph queries. This massive parallelism allows GFQL to handle tasks that are impractical on typical CPU systems, such as real-time traversals that touch hundreds of millions of edges and compute on them.


Seamless Scalability from CPUs to GPUs
--------------------------------------

GFQL allows you to start analyzing graphs on standard CPUs without specialized hardware. As your data grows, you can transition to GPU acceleration without changing your code. GFQL intelligently utilizes available hardware to optimize performance, ensuring efficient resource use whether you're on a single machine or across a cluster.

By eliminating the need for additional infrastructure, GFQL reduces time and expense, allowing you to focus on extracting insights from your data. This seamless scalability ensures that as your projects evolve, GFQL adapts to meet your needs.

Optimized for Analytical Workloads
----------------------------------

GFQL excels in scenarios requiring deep analytical capabilities. It is designed for:

- **Graph ETL and Analytics**: Efficiently process and transform large volumes of graph data.
- **Machine Learning and AI**: Accelerate graph-based ML and AI tasks, leveraging GPUs for training and inference.
- **Visualization**: Power high-performance graph visualizations, enabling real-time interaction with complex datasets.

By focusing on these areas, GFQL meets the demands of modern data projects, from initial exploration to advanced analysis, without the overhead typically associated with large-scale analytics.

Built on Graphistry's Expertise
-------------------------------

Graphistry's reputation for leveraging GPUs and vectorization in data analytics is well-established. GFQL embodies this expertise, filling gaps in the graph and dataframe ecosystem by providing tools that maximize GPU utilization and integrate with open-source technologies like Apache Arrow. Our collaboration with `NVIDIA <https://www.nvidia.com//>`_, including their investment into our team, ensures that GFQL benefits from optimized kernel methods for top-tier performance.

Empower Your Data Journey
-------------------------

With GFQL, you can start quickly, scale more smoothly, and leverage cutting-edge performance. It empowers you to:

- Begin analyzing graphs immediately on your existing hardware
- Grow from CPU to GPU processing without code changes
- Handle datasets ranging from thousands to billions of edges efficiently

Whether you're analyzing social networks, investigating cybersecurity threats, or exploring intricate datasets, GFQL transforms how you work with graph data, making complex analytics accessible and efficient.

Join the Graphistry Community
-----------------------------

We invite you to  become part of our community dedicated to advancing graph analytics through innovation in vectorization and GPU computing. Let's keep pushing the boundaries of what's possible!

---

Next Steps
----------

- **Explore GFQL**: Dive deeper into GFQL's capabilities in :ref:`10min-gfql`.
- **Get Started with PyGraphistry**: Follow the :ref:`10min` to setup and experience the performance firsthand.
- **Learn About Vectorization and GPUs**: Understand the partner ecosystem technologies behind GFQL by exploring `Apache Arrow <https://arrow.apache.org/>`_ and `NVIDIA RAPIDS <https://rapids.ai/>`_.
- **Connect with Us**: Join our :ref:`community` to share insights and collaborate with others pushing the boundaries of graph analytics.
