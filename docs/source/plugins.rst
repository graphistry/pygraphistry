.. _plugins:

Plugins
=======

PyGraphistry is frequently used with a variety of external tools such as data providers, compute engines, layout engines, and more.

Users typically prefer to go through PyGraphistry's native dataframe support (Apache Arrow, Pandas, cuDF, ...). That is often an efficient, safe, and easy starting point.

Occasionally, such as with graph databases, graph layouts, and graph analytics, native PyGraphistry plugins streamline common operations. We link to the native API integrations below as appropriate.

See also the :ref:`layout-catalog` for:

* :ref:`cugraph <cugraph>`: GPU-accelerated graph analytics
* :ref:`graphviz <graphviz>`: CPU graph analytics and layouts
* :ref:`igraph <igraph>`: CPU graph analytics and layouts


Databases
---------------

See :ref:`demo notebooks <nb-connectors>` for data providers commonly used with Graphistry:

* ArangoDB
* AWS Neptune (:ref:`API <api-neptune>`)
* Cassandra
* Cosmos (:ref:`API <api-cosmos>`)
* Databricks
* DynamoDB
* Elasticsearch
* Gremlin (:ref:`API <api-gremlin>`)
* Kusto
* Memgraph (:meth:`API <graphistry.PlotterBase.PlotterBase.cypher>`)
* Neo4j (:meth:`API <graphistry.PlotterBase.PlotterBase.cypher>`)
* OpenSearch
* Redis
* Splunk
* Spark
* SQL (ODBC): Postgres, BigQuery, Redshift, Snowflake, SQL Server, Athena, etc.
* Tigergraph (:meth:`API <graphistry.PlotterBase.PlotterBase.gsql>`)
* Trovares

Compute engines
----------------

    * cuDF
    * Dask
    * Dask-cuDF
    * Pandas
    * Polars
    * Spark

Tools
---------

    * OWASP Amass

Storage engines and file formats
---------------------------------

    * Arrow
    * Azure blobstore
    * CSV
    * GML
    * JSON
    * JSONL
    * LOG
    * ORC
    * Parquet
    * S3
    * TXT
    * XLS(X)
