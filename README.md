# PyGraphistry: Leverage the power of graphs & GPUs to visualize, analyze, and scale your data

![Build Status](https://github.com/graphistry/pygraphistry/workflows/CI%20Tests/badge.svg)
[![CodeQL](https://github.com/graphistry/pygraphistry/workflows/CodeQL/badge.svg)](https://github.com/graphistry/pygraphistry/actions?query=workflow%3ACodeQL)
[![Documentation Status](https://readthedocs.org/projects/pygraphistry/badge/?version=latest)](https://pygraphistry.readthedocs.io/en/latest/)
[![Latest Version](https://img.shields.io/pypi/v/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
[![Latest Version](https://img.shields.io/pypi/pyversions/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
[![License](https://img.shields.io/pypi/l/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
![PyPI - Downloads](https://img.shields.io/pypi/dm/graphistry)

[![Uptime Robot status](https://img.shields.io/uptimerobot/status/m787548531-e9c7b7508fc76fea927e2313?label=hub.graphistry.com)](https://status.graphistry.com/) [<img src="https://img.shields.io/badge/slack-Graphistry%20chat-orange.svg?logo=slack">](https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g)
[![Twitter Follow](https://img.shields.io/twitter/follow/graphistry)](https://twitter.com/graphistry)


<table style="width:100%;">
  <tr valign="top">
    <td align="center"><a href="https://hub.graphistry.com/graph/graph.html?dataset=Facebook&splashAfter=true" target="_blank"><img src="https://i.imgur.com/z8SIh2E.png" title="Click to open."></a>
    <a href="https://hub.graphistry.com/graph/graph.html?dataset=Facebook&splashAfter=true" target="_blank">Demo: Interactive visualization of 80,000+ Facebook friendships</a> (<a href="http://snap.stanford.edu" target="_blank">source data</a></em>)
    </td>
  </tr>
</table>

PyGraphistry is an open source Python library for data scientists and developers to leverage the power of graph visualization, analytics, AI, including with native GPU acceleration:

* [**Python dataframe-native graph processing:**](https://pygraphistry.readthedocs.io/en/latest/10min.html) Quickly ingest & prepare data in many formats, shapes, and scales as graphs. Use tools like Pandas, Spark, [RAPIDS (GPU)](https://www.rapids.ai), and [Apache Arrow](https://arrow.apache.org/).

* [**Integrations:**](https://pygraphistry.readthedocs.io/en/latest/plugins.html) Connect to graph databases, data platforms, Python tools, and more.

  | Category | Connector Tutorials |
  |----------|---------------------|
  | **Data Platforms, SQL & Logs** | [![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.html) [![Splunk](https://img.shields.io/badge/Splunk-000000?style=flat&logo=splunk&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/splunk/splunk_demo_public.html) [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/sql/postgres.html) [![Azure Data Explorer (Kusto)](https://img.shields.io/badge/Azure_Data_Explorer_(Kusto)-0078D4?style=flat&logo=microsoftazure&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/microsoft/kusto/graphistry_ADX_kusto_demo.html) [![Google Cloud Spanner](https://img.shields.io/badge/Google_Cloud_Spanner-4285F4?style=flat&logo=googlecloud&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/spanner/google_spanner_finance_graph.html) |
  | **Graph Databases** | [![Neo4j](https://img.shields.io/badge/Neo4j-4581C3?style=flat&logo=neo4j&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/neo4j/official/graphistry_bolt_tutorial_public.html) [![Amazon Neptune](https://img.shields.io/badge/Amazon_Neptune-FF9900?style=flat&logo=amazonaws&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/neptune/neptune_cypher_viz_using_bolt.html) [![TigerGraph](https://img.shields.io/badge/TigerGraph-FF6600?style=flat)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/tigergraph/tigergraph_pygraphistry_bindings.html) [![ArangoDB](https://img.shields.io/badge/ArangoDB-DDE072?style=flat&logo=arangodb&logoColor=black)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/arango/arango_tutorial.html) [![Memgraph](https://img.shields.io/badge/Memgraph-DD2222?style=flat)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/memgraph/visualizing_iam_dataset.html) |
  | **Python Tools & Libraries** | [![CSV](https://img.shields.io/badge/CSV-217346?style=flat&logo=microsoftexcel&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/upload_csv_miniapp.html) [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/gpu_rapids/part_i_cpu_pandas.html) [![Apache Arrow](https://img.shields.io/badge/Apache_Arrow-000000?style=flat&logo=apachearrow&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/performance.html) [![NVIDIA RAPIDS](https://img.shields.io/badge/NVIDIA_RAPIDS-76B900?style=flat&logo=nvidia&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/gpu_rapids/cugraph.html) [![NetworkX](https://img.shields.io/badge/NetworkX-013243?style=flat)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/networkx/networkx.html) [![Graphviz](https://img.shields.io/badge/Graphviz-2A2A2A?style=flat&logo=graphviz&logoColor=white)](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/graphviz/graphviz.html) |

  *[View all connectors →](https://pygraphistry.readthedocs.io/en/latest/notebooks/plugins.connectors.html)*


* [**Prototype locally and deploy remotely:**](https://www.graphistry.com/get-started) Prototype from notebooks like Jupyter and Databricks using local CPUs & GPUs, and then power production dashboards & pipelines with Graphistry Hub and your own self-hosted servers.

* [**Query graphs with GFQL:**](https://pygraphistry.readthedocs.io/en/latest/gfql/index.html) Use GFQL, the first dataframe-native graph query language, to ask relationship questions that are difficult for tabular tools and without requiring a database.

* [**graphistry[ai]:**](https://pygraphistry.readthedocs.io/en/latest/gfql/combo.html#) Call streamlined graph ML & AI methods to benefit from clustering, UMAP embeddings, graph neural networks, automatic feature engineering, and more.

* [**Visualize & explore large graphs:**](https://pygraphistry.readthedocs.io/en/latest/visualization/10min.html#) In just a few minutes, create stunning interactive visualizations with millions of edges and many point-and-click built-ins like drilldowns, timebars, and filtering. When ready, customize with Python, JavaScript, and REST APIs.

* [**Columnar & GPU acceleration:**](https://pygraphistry.readthedocs.io/en/latest/performance.html) CPU-mode ingestion and wrangling is fast due to native use of Apache Arrow and columnar analytics, and the optional RAPIDS-based GPU mode delivers 100X+ speedups.


From global 10 banks, manufacturers, news agencies, and government agencies, to startups, game companies, scientists, biotechs, and NGOs, many teams are tackling their graph workloads with Graphistry.



## Gallery


The [notebook demo gallery](https://pygraphistry.readthedocs.io/en/latest/demos/for_analysis.html) shares many more live visualizations, demos, and integration examples

<table>
    <tr valign="top">
        <td width="33%" align="center"><a href="https://hub.graphistry.com/graph/graph.html?dataset=Twitter&splashAfter=true" target="_blank">Twitter Botnet<br><img width="266" src="https://i.imgur.com/qm5MCqS.jpg"></a></td>
        <td width="33%" align="center">Edit Wars on Wikipedia<br><a href="https://i.imgur.com/074zFve.png" target="_blank"><img width="266" src="https://i.imgur.com/074zFve.png"></a><em>(<a href="https://snap.stanford.edu" target="_blank">data</a></em>)</td>
        <td width="33%" align="center"><a href="https://hub.graphistry.com/graph/graph.html?dataset=bitC&splashAfter=true" target="_blank">100,000 Bitcoin Transactions<br><img width="266" height="266" src="https://i.imgur.com/axIkjfd.png"></a></td>
    </tr>
    <tr valign="top">
        <td width="33%" align="center">Port Scan Attack<br><a href="http://i.imgur.com/vKUDySw.png" target="_blank"><img width="266" src="http://i.imgur.com/vKUDySw.png"></a></td>
        <td width="33%" align="center"><a href="http://hub.graphistry.com/graph/graph.html?dataset=PyGraphistry/M9RL4PQFSF&usertag=github&info=true&static=true&contentKey=Biogrid_Github_Demo&play=3000&center=false&menu=true&goLive=false&left=-2.58e+4&right=4.35e+4&top=-1.72e+4&bottom=2.16e+4&legend={%22title%22:%22%3Ch3%3EBioGRID%20Repository%20of%20Protein%20Interactions%3C/h3%3E%22,%22subtitle%22:%22%3Cp%3EEach%20color%20represents%20an%20organism.%20Humans%20are%20in%20light%20blue.%3C/p%3E%22,%22nodes%22:%22Proteins/Genes%22,%22edges%22:%22Interactions%20reported%20in%20scientific%20publications%22}" target="_blank">Protein Interactions <br><img width="266" src="http://i.imgur.com/nrUHLFz.png" target="_blank"></a><em>(<a href="http://thebiogrid.org" target="_blank">data</a>)</em></td>
        <td width="33%" align="center"><a href="http://hub.graphistry.com/graph/graph.html?&dataset=PyGraphistry/PC7D53HHS5&info=true&static=true&contentKey=SocioPlt_Github_Demo&play=3000&center=false&menu=true&goLive=false&left=-236&right=265&top=-145&bottom=134&usertag=github&legend=%7B%22nodes%22%3A%20%22%3Cspan%20style%3D%5C%22color%3A%23a6cee3%3B%5C%22%3ELanguages%3C/span%3E%20/%20%3Cspan%20style%3D%5C%22color%3Argb%28106%2C%2061%2C%20154%29%3B%5C%22%3EStatements%3C/span%3E%22%2C%20%22edges%22%3A%20%22Strong%20Correlations%22%2C%20%22subtitle%22%3A%20%22%3Cp%3EFor%20more%20information%2C%20check%20out%20the%20%3Ca%20target%3D%5C%22_blank%5C%22%20href%3D%5C%22https%3A//lmeyerov.github.io/projects/socioplt/viz/index.html%5C%22%3ESocio-PLT%3C/a%3E%20project.%20Make%20your%20own%20visualizations%20with%20%3Ca%20target%3D%5C%22_blank%5C%22%20href%3D%5C%22https%3A//github.com/graphistry/pygraphistry%5C%22%3EPyGraphistry%3C/a%3E.%3C/p%3E%22%2C%20%22title%22%3A%20%22%3Ch3%3ECorrelation%20Between%20Statements%20about%20Programming%20Languages%3C/h3%3E%22%7D" target="_blank">Programming Languages<br><img width="266" src="http://i.imgur.com/0T0EKmD.png"></a><em>(<a href="http://lmeyerov.github.io/projects/socioplt/viz/index.html" target="_blank">data</a>)</em></td>
    </tr>
</table>



## Install

Common configurations:

* **Minimal core**

  Includes: The GFQL dataframe-native graph query language, built-in layouts, Graphistry visualization server client

  ```python
  pip install graphistry
  ```

  Does not include `graphistry[ai]`, plugins

* **No dependencies and user-level**

  ```python
  pip install --no-deps --user graphistry
  ```

* **GPU acceleration** - Optional

  Local GPU: Install [RAPIDS](https://www.rapids.ai) and/or deploy a GPU-ready [Graphistry server](https://www.graphistry.com/get-started)
  
  Remote GPU: Use the [remote endpoints](https://www.graphistry.com/blog/graphistry-2-41-3).

For further options, see the [installation guides](https://pygraphistry.readthedocs.io/en/latest/install/index.html)


## Visualization quickstart

Quickly go from raw data to a styled and interactive Graphistry graph visualization:

```python
import graphistry
import pandas as pd

# Raw data as Pandas CPU dataframes, cuDF GPU dataframes, Spark, ...
df = pd.DataFrame({
    'src': ['Alice', 'Bob', 'Carol'],
    'dst': ['Bob', 'Carol', 'Alice'],
    'friendship': [0.3, 0.95, 0.8]
})

# Bind
g1 = graphistry.edges(df, 'src', 'dst')

# Override styling defaults
g1_styled = g1.encode_edge_color('friendship', ['blue', 'red'], as_continuous=True)

# Connect: Free GPU accounts and self-hosting @ graphistry.com/get-started
graphistry.register(api=3, username='your_username', password='your_password')

# Upload for GPU server visualization session
g1_styled.plot()
```

Explore [10 Minutes to Graphistry Visualization](https://pygraphistry.readthedocs.io/en/latest/visualization/10min.html) for more visualization examples and options


## PyGraphistry[AI] & GFQL quickstart - CPU & GPU

**CPU graph pipeline** combining graph ML, AI, mining, and visualization:

```python
from graphistry import n, e, e_forward, e_reverse

# Graph analytics
g2 = g1.compute_igraph('pagerank')
assert 'pagerank' in g2._nodes.columns

# Graph ML/AI
g3 = g2.umap()
assert ('x' in g3._nodes.columns) and ('y' in g3._nodes.columns)

# Graph querying with GFQL
g4 = g3.chain([
    n(query='pagerank > 0.1'), e_forward(), n(query='pagerank > 0.1')
])
assert (g4._nodes.pagerank > 0.1).all()

# Upload for GPU server visualization session
g4.plot()
```

The **automatic GPU modes** require almost no code changes:

```python
import cudf
from graphistry import n, e, e_forward, e_reverse

# Modified -- Rebind data as a GPU dataframe and swap in a GPU plugin call
g1_gpu = g1.edges(cudf.from_pandas(df))
g2 = g1_gpu.compute_cugraph('pagerank')

# Unmodified -- Automatic GPU mode for all ML, AI, GFQL queries, & visualization APIs
g3 = g2.umap()
g4 = g3.chain([
    n(query='pagerank > 0.1'), e_forward(), n(query='pagerank > 0.1')
])
g4.plot()
```

Explore [10 Minutes to PyGraphistry](https://pygraphistry.readthedocs.io/en/latest/10min.html) for a wider variety of graph processing.


## PyGraphistry documentation

* [Main PyGraphistry documentation](https://pygraphistry.readthedocs.io/en/latest/)
* 10 Minutes to: [PyGraphistry](https://pygraphistry.readthedocs.io/en/latest/10min.html), [Visualization](https://pygraphistry.readthedocs.io/en/latest/visualization/10min.html), [GFQL](https://pygraphistry.readthedocs.io/en/latest/gfql/about.html)
* Get started: [Install](https://pygraphistry.readthedocs.io/en/latest/install/index.html), [UI Guide](https://hub.graphistry.com/docs/ui/index/), [Notebooks](https://pygraphistry.readthedocs.io/en/latest/demos/for_analysis.html)
* Performance: [PyGraphistry CPU+GPU](https://pygraphistry.readthedocs.io/en/latest/performance.html) & [GFQL CPU+GPU](https://pygraphistry.readthedocs.io/en/latest/gfql/performance.html)
* API References
    - [PyGraphistry API Reference](https://pygraphistry.readthedocs.io/en/latest/api/index.html): [Visualization & Compute](https://pygraphistry.readthedocs.io/en/latest/visualization/index.html), [PyGraphistry Cheatsheet](https://pygraphistry.readthedocs.io/en/latest/cheatsheet.html)
    - [GFQL Documentation](https://pygraphistry.readthedocs.io/en/latest/gfql/index.html):  [GFQL Cheatsheet](https://pygraphistry.readthedocs.io/en/latest/gfql/quick.html) and [GFQL Operator Cheatsheet](https://pygraphistry.readthedocs.io/en/latest/gfql/predicates/quick.html)
    - [Plugins](https://pygraphistry.readthedocs.io/en/latest/plugins.html): Databricks, Splunk, Neptune, Neo4j, RAPIDS, and more
    - Web: [iframe](https://hub.graphistry.com/docs/api/1/rest/url/#urloptions),  [JavaScript](https://hub.graphistry.com/static/js-docs/index.html?path=/docs/introduction--docs), [REST](https://hub.graphistry.com/docs/api/1/rest/auth/)

## Graphistry ecosystem

- **Graphistry server:**
  - Launch - [Graphistry Hub, Graphistry cloud marketplaces, and self-hosting](https://www.graphistry.com/get-started)
  - Self-hosting: [Administration (including Docker)](https://github.com/graphistry/graphistry-cli) & [Kubernetes](https://github.com/graphistry/graphistry-helm)

- **Graphistry client APIs:**
  - Web: [iframe](https://hub.graphistry.com/docs/api/1/rest/url/#urloptions), [JavaScript](https://hub.graphistry.com/static/js-docs/index.html?path=/docs/introduction--docs), [REST](https://hub.graphistry.com/docs/api/1/rest/auth/)
  - [PyGraphistry](https://pygraphistry.readthedocs.io/en/latest/index.html)
  - [Graphistry for Microsoft PowerBI](https://hub.graphistry.com/docs/powerbi/pbi/)

- **Additional projects**:
  - [Louie.ai](https://louie.ai/): GenAI-native notebooks & dashboards to talk to your databases & Graphistry
  - [graph-app-kit](https://github.com/graphistry/graph-app-kit): Streamlit Python dashboards with batteries-include graph packages
  - [cu-cat](https://chat.openai.com/chat): Automatic GPU feature engineering


## Community and support

- [Blog](https://www.graphistry.com/blog) for tutorials, case studies, and updates
- [Slack](https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g): Join the Graphistry Community Slack for discussions and support
- [Twitter](https://twitter.com/graphistry) & [LinkedIn](https://www.linkedin.com/company/graphistry): Follow for updates
- [GitHub Issues](https://github.com/graphistry/pygraphistry/issues) open source support
- [Graphistry ZenDesk](https://graphistry.zendesk.com/) dedicated enterprise support

## Contribute

See [CONTRIBUTING](https://pygraphistry.readthedocs.io/en/latest/CONTRIBUTING.html) and [DEVELOP](https://pygraphistry.readthedocs.io/en/latest/DEVELOP.html) for participating in PyGraphistry development, or reach out to our team

