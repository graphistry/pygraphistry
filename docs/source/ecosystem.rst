Graphistry Ecosystem and Louie.AI
==================================

The Graphistry community of projects, open source, and partners has grown over the years:

.. graphviz::

   digraph graphistry_ecosystem_toy {
       rankdir=LR;
       node [shape=box, style=filled, fillcolor=lightgray];

       core [label="pygraphistry core"];
       gfql [label="GFQL"];
       ai [label="pygraphistry[ai]"];
       cucat [label="cu-cat"];
       louie [label="Louie.AI"];

       pandas [label="pandas"];
       arrow [label="Apache Arrow"];
       server [label="Graphistry server (optional)"];
       rapids [label="NVIDIA RAPIDS (optional)"];
       pytorch [label="PyTorch (optional)"];

       {rank=same; pandas; arrow; server; rapids; pytorch;}

       pandas -> core [label="dataframes"];
       arrow -> core [label="columnar IO"];
       rapids -> core [label="optional GPU"];
       pytorch -> ai [label="optional ML"];

       core -> gfql [label="module"];
       core -> ai [label="module"];
       core -> cucat [label="module"];
       core -> louie [label="viz + APIs"];

       core -> server [label="upload + viz store", style=dashed];
       gfql -> server [label="remote compute", style=dashed];
       louie -> server [label="uses remote server", style=dashed];
       louie -> gfql [label="generates"];
       louie -> core [label="creates viz"];
   }

Graphistry Core
---------------

* `REST API <https://hub.graphistry.com/docs/api/>`_
* `JS APIs (github) <https://github.com/graphistry/graphistry-js>`_: Node, React, and vanilla JS
* `graph-app-kit (github) <https://github.com/graphistry/graph-app-kit>`_: Python dashboarding with Graphistry and Streamlit

GFQL: Dataframe-native Graph Query Language
---------------------------------------------

Our :ref:`open-source graph query language GFQL <10min-gfql>` with optional GPU support

The Graphistry team created GFQL to fill the gap between pandas/cudf and cypher. This project has been years in the making, and is built out of need from our experiences in working with graphs of all sizes in the compute and visualization tiers.

Graphistry Louie.AI
-------------------

`Louie.AI <https://www.louie.ai/>`_  is the new genAI-native experience for Graphistry and your favorite databases

Louie.AI features:

* genAI-native notebooks: Talk to your data & databases and get back answers, visualizations, and more
* genAI-native dashboards: Build and share dashboards with your data, AI, and Graphistry
* API: Use Louie.AI's API to integrate genAI-native experiences into your own apps, both visual and headless
* Real-time AI Knowledge Graph Database: Target data, transform into preintegrated genAI-friendly indexes, then talk to it or trigger workflows

Check out the `Louie.AI homepage <https://www.louie.ai/>`_ for more information and early access.


Graphistry cu_cat
------------------

Automatic feature engineering is an important way pygraphistry[ai] streamlines ML and AI workflows. To make that fast, we have been adding GPU acceleration through a GPU-first port of dirty_cat (skrub).

Head over to the `cu-cat homepage (github) <https://github.com/graphistry/cu-cat>`_


Community
---------

Graphistry works with a variety of partners and projects, some of which include:

**GPU dataframes:**

* `Nvidia RAPIDS <https://rapids.ai/>`_: cuDF, cuGraph, cuML
* `Apache Arrow <https://arrow.apache.org/>`_: Python, JS, and more

**Graph**

* `Neo4j <https://neo4j.com/>`_
* `Trovares <https://www.trovares.com/>`_
* `Tigergraph <https://www.tigergraph.com/>`_
* `ArangoDB <https://www.arangodb.com/>`_
* `JanusGraph <https://janusgraph.org/>`_
* `Memgraph <https://memgraph.com/>`_

**Log databases and SIEMs:**

* `Elasticsearch <https://www.elastic.co/>`_
* `Microsoft Kusto <https://docs.microsoft.com/en-us/azure/data-explorer/>`_
* `Microsoft Msticpy <https://github.com/microsoft/msticpy>`_
* `OpenSearch <https://opensearch.org/>`_
* `Splunk <https://www.splunk.com/>`_

**Graph analytics:**

* `igraph <https://igraph.org/python/>`_

**Python data science ecosystem:**

* `Streamlit <https://streamlit.io/>`_
* `Jupyter <https://jupyter.org/>`_
* `Pandas <https://pandas.pydata.org/>`_
* `Dask <https://www.dask.org/>`_
