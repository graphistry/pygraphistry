.. _plugins:

Plugins
=======

PyGraphistry is frequently used with a variety of external tools such as data providers, compute engines, layout engines, and more.

Users typically prefer to go through PyGraphistry's native dataframe support (Apache Arrow, Pandas, cuDF, ...). That is often an efficient, safe, and easy starting point.

Occasionally, native PyGraphistry plugins streamline common operations, such as with graph databases. We link to the native API integrations below as appropriate.

For more examples, see also the :ref:`notebook catalog <notebooks>`.


Databases
---------------

Graph
~~~~~~~~~~~

* `Amazon Neptune <https://aws.amazon.com/neptune>`_ (:class:`graphistry.gremlin.NeptuneMixin`)
* `ArangoDB <https://www.arangodb.com>`_
* `Gremlin <https://tinkerpop.apache.org>`_ (:class:`graphistry.gremlin.GremlinMixin`)
* `Memgraph <https://memgraph.com>`_ (:meth:`graphistry.PlotterBase.PlotterBase.cypher`)
* `Neo4j <https://neo4j.com>`_ (:meth:`graphistry.PlotterBase.PlotterBase.cypher`)
* `TigerGraph <https://www.tigergraph.com>`_ (:meth:`graphistry.PlotterBase.PlotterBase.gsql`)
* `Trovares <https://trovares.com>`_


Document, Key-Value, Log, Text, and SIEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Amazon DynamoDB <https://aws.amazon.com/dynamodb>`_
* `Azure Cosmos DB <https://azure.microsoft.com/en-us/services/cosmos-db>`_ (:class:`graphistry.gremlin.CosmosMixin`)

* `Azure Data Explorer (Kusto) <https://azure.microsoft.com/en-us/services/data-explorer>`_
* `Cassandra <https://cassandra.apache.org>`_
* `Elasticsearch <https://www.elastic.co>`_
* `OpenSearch <https://opensearch.org>`_
* `Redis <https://redis.io>`_
* `Splunk <https://www.splunk.com>`_

SQL
~~~~~~~~~~~

Typically accessed via dataframe bindings

When available, we recommend exploring for accelerated bindings via `ADBC <https://arrow.apache.org/docs/format/ADBC.html>`_


* `Amazon Athena <https://aws.amazon.com/athena>`_
* `Databricks <https://databricks.com>`_
* `OpenSearch <https://opensearch.org>`_
* `PostgreSQL <https://www.postgresql.org>`_
* `Amazon Redshift <https://aws.amazon.com/redshift>`_
* `BigQuery <https://cloud.google.com/bigquery>`_
* `Snowflake <https://www.snowflake.com>`_
* `SQL Server <https://www.microsoft.com/en-us/sql-server>`_


Compute engines
----------------

Natively supported in methods such as :meth:`.nodes() <graphistry.PlotterBase.PlotterBase.nodes>` and :meth:`.edges() <graphistry.PlotterBase.PlotterBase.edges>`:

* `Apache Spark <https://spark.apache.org>`_
* `Pandas <https://pandas.pydata.org/>`_
* `cuDF <https://docs.rapids.ai/api/cudf/stable/>`_

Partial native support:

* `cuML <https://docs.rapids.ai/api/cuml/stable/>`_
* `Dask <https://www.dask.org/>`_
* `Dask-cuDF <https://docs.rapids.ai/api/cudf/stable/dask-cudf.html>`__

Accelerated interop via `Apache Arrow <https://arrow.apache.org/>`_ or Parquet:

* `DuckDB <https://duckdb.org/>`_
* `Polars <https://www.pola.rs/>`_
* `Spark <https://spark.apache.org/>`_

Graph layout and analytics
---------------------------

* :ref:`cugraph <cugraph>`: GPU-accelerated graph analytics
* :ref:`graphviz <graphviz>`: CPU graph analytics and layouts
* :ref:`igraph <igraph>`: CPU graph analytics and layouts
* :ref:`networkx <networkx>`: CPU graph analytics and layouts


Tools
---------

We are constantly experimenting, feel free to add:

* OWASP Amass

Storage engines and file formats
---------------------------------

GPU-accelerated readers via `cuDF <https://docs.rapids.ai/api/cudf/stable/>`_  (in-memory single-GPU) and `Dask-cuDF <https://docs.rapids.ai/api/cudf/stable/dask-cudf.html>`__ (bigger-than-memory, multi-GPU):

* Arrow
* CSV
* JSON
* JSONL
* LOG
* ORC
* Parquet
* TXT

Others, often via `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_:

* Azure blobstore
* GML
* S3
* XLS(X)
