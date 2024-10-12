# PyGraphistry: Visualize, analyze, and scale your graphs with Graphistry & GFQL

![Build Status](https://github.com/graphistry/pygraphistry/workflows/CI%20Tests/badge.svg)
[![CodeQL](https://github.com/graphistry/pygraphistry/workflows/CodeQL/badge.svg)](https://github.com/graphistry/pygraphistry/actions?query=workflow%3ACodeQL)
[![Documentation Status](https://readthedocs.org/projects/pygraphistry/badge/?version=latest)](https://pygraphistry.readthedocs.io/en/latest/)
[![Latest Version](https://img.shields.io/pypi/v/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
[![Latest Version](https://img.shields.io/pypi/pyversions/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
[![License](https://img.shields.io/pypi/l/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
![PyPI - Downloads](https://img.shields.io/pypi/dm/graphistry)

[![Uptime Robot status](https://img.shields.io/uptimerobot/status/m787548531-e9c7b7508fc76fea927e2313?label=hub.graphistry.com)](https://status.graphistry.com/) [<img src="https://img.shields.io/badge/slack-Graphistry%20chat-orange.svg?logo=slack">](https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g)
[![Twitter Follow](https://img.shields.io/twitter/follow/graphistry)](https://twitter.com/graphistry)


## Happy Graphing!

**PyGraphistry** is a leading open-source Python library that makes it easy to visualize, analyze, and scale your graph data. It unifies the Graphistry visualization engine, the GFQL dataframe-native graph query language, and the open-source PyData ecosystem. As an early core contributor to Apache Arrow and having helped start the open source GPU dataframe ecosystem, PyGraphistry makes it easy to quickly load in regular in-memory datasets of many shapes and sizes, and you can use the optional AI & RAPIDS GPU modes for billion-scale data. Combined, PyGraphistry brings seamless data handling with best-in-class performance to graph tasks. Transform any data into interactive graph visualizations and analytics that run smoothly on any system.

PyGraphistry runs in your Python application with just `pip install graphistry` . You can optionally connect it to remote Graphistry servers for enhanced performance and additional features.

## Key features

* **Interactive Visualization:** Quickly create impressive interactive visualizations with multiple layouts, data-driven styling, and built-in components like timebars, histograms, search, and filters

* **Data Integration:** Quickly load and transform data of many sources, shapes, and scales through native connectors, Arrow acceleration, and dataframe support including Pandas, Spark, and cuDF

* **GPU Acceleration:** Visualize large graphs using the built-in WebGL frontend and GPU-accelerated Graphistry servers. Speed up your graph data pipelines over 100× by enabling GPU engine modes and AI inferencing


* **Graph Querying with GFQL:** Utilize GFQL, the dataframe-native graph query language with optional GPU acceleration, directly from Python. Fast and easy compute-tier graph querying without needing outside infrastructure.

* **Graph AI Made Easy:** Simple and automated interfaces for graph machine learning and AI tasks, including automated feature engineering, UMAP clustering, and heterogeneuos graph neural networks.

* **Integrate & Deploy:** Embed PyGraphistry into notebooks, dashboards, web applications, and data pipelines. Offload intensive tasks to shareable Graphistry GPU servers for enhanced performance and scalability.


## Gallery

<table>
    <tr valign="top">
        <td width="33%" align="center"><a href="http://hub.graphistry.com/graph/graph.html?dataset=Twitter&splashAfter=true" target="_blank">Twitter Botnet<br><img width="266" src="http://i.imgur.com/qm5MCqS.jpg"></a></td>
        <td width="33%" align="center">Edit Wars on Wikipedia<br><a href="http://i.imgur.com/074zFve.png" target="_blank"><img width="266" src="http://i.imgur.com/074zFve.png"></a><em>Source: <a href="http://snap.stanford.edu" target="_blank">SNAP</a></em></td>
        <td width="33%" align="center"><a href="https://hub.graphistry.com/graph/graph.html?dataset=bitC&splashAfter=true" target="_blank">100,000 Bitcoin Transactions<br><img width="266" height="266" src="http://imgur.com/download/axIkjfd"></a></td>
    </tr>
    <tr valign="top">
        <td width="33%" align="center">Port Scan Attack<br><a href="http://i.imgur.com/vKUDySw.png" target="_blank"><img width="266" src="http://i.imgur.com/vKUDySw.png"></a></td>
        <td width="33%" align="center"><a href="http://hub.graphistry.com/graph/graph.html?dataset=PyGraphistry/M9RL4PQFSF&usertag=github&info=true&static=true&contentKey=Biogrid_Github_Demo&play=3000&center=false&menu=true&goLive=false&left=-2.58e+4&right=4.35e+4&top=-1.72e+4&bottom=2.16e+4&legend={%22title%22:%22%3Ch3%3EBioGRID%20Repository%20of%20Protein%20Interactions%3C/h3%3E%22,%22subtitle%22:%22%3Cp%3EEach%20color%20represents%20an%20organism.%20Humans%20are%20in%20light%20blue.%3C/p%3E%22,%22nodes%22:%22Proteins/Genes%22,%22edges%22:%22Interactions%20reported%20in%20scientific%20publications%22}" target="_blank">Protein Interactions <br><img width="266" src="http://i.imgur.com/nrUHLFz.png" target="_blank"></a><em>Source: <a href="http://thebiogrid.org" target="_blank">BioGRID</a></em></td>
        <td width="33%" align="center"><a href="http://hub.graphistry.com/graph/graph.html?&dataset=PyGraphistry/PC7D53HHS5&info=true&static=true&contentKey=SocioPlt_Github_Demo&play=3000&center=false&menu=true&goLive=false&left=-236&right=265&top=-145&bottom=134&usertag=github&legend=%7B%22nodes%22%3A%20%22%3Cspan%20style%3D%5C%22color%3A%23a6cee3%3B%5C%22%3ELanguages%3C/span%3E%20/%20%3Cspan%20style%3D%5C%22color%3Argb%28106%2C%2061%2C%20154%29%3B%5C%22%3EStatements%3C/span%3E%22%2C%20%22edges%22%3A%20%22Strong%20Correlations%22%2C%20%22subtitle%22%3A%20%22%3Cp%3EFor%20more%20information%2C%20check%20out%20the%20%3Ca%20target%3D%5C%22_blank%5C%22%20href%3D%5C%22https%3A//lmeyerov.github.io/projects/socioplt/viz/index.html%5C%22%3ESocio-PLT%3C/a%3E%20project.%20Make%20your%20own%20visualizations%20with%20%3Ca%20target%3D%5C%22_blank%5C%22%20href%3D%5C%22https%3A//github.com/graphistry/pygraphistry%5C%22%3EPyGraphistry%3C/a%3E.%3C/p%3E%22%2C%20%22title%22%3A%20%22%3Ch3%3ECorrelation%20Between%20Statements%20about%20Programming%20Languages%3C/h3%3E%22%7D" target="_blank">Programming Languages<br><img width="266" src="http://i.imgur.com/0T0EKmD.png"></a><em>Source: <a href="http://lmeyerov.github.io/projects/socioplt/viz/index.html" target="_blank">Socio-PLT project</a></em></td>
    </tr>
</table>

The [notebook demo gallery](https://pygraphistry.readthedocs.io/en/latest/demos/for_analysis.html) shares many more live visualizations and demos


## Install

Minimal core: Includes the GFQL dataframe-native graph query language and Graphistry visualization server client

```python
pip install graphistry
```

For more complicated environments:

```python
pip install --no-deps --user graphistry
```

See the [installation guides](https://pygraphistry.readthedocs.io/en/latest/install/index.html) for more options, GPU environments, and plugins



## Main documentation

See the [main PyGraphistry documentation](https://pygraphistry.readthedocs.io/en/latest/) for more details on installation, usage, and examples.


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
g1_styled = g1.encode_edge_color('friendship', as_continuous=True, ['blue', 'red'])

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

 *  [Main PyGraphistry documentation](https://pygraphistry.readthedocs.io/en/latest/)
* Get started
  - [Installation Guides](https://pygraphistry.readthedocs.io/en/latest/install/index.html)
  - [10 Minutes to PyGraphistry](https://pygraphistry.readthedocs.io/en/latest/10min.html)
  - [10 Minutes to Graphistry Visualization](https://pygraphistry.readthedocs.io/en/latest/visualization/10min.html) and [UI Guide](https://hub.graphistry.com/docs/ui/index/)

  - [10 Minutes to GFQL](https://pygraphistry.readthedocs.io/en/latest/gfql/about.html)
  - [Notebook demos](https://pygraphistry.readthedocs.io/en/latest/demos/for_analysis.html)
* Peformance - CPU & GPU
  - [Core](https://pygraphistry.readthedocs.io/en/latest/performance.html)
  - [GFQL](https://pygraphistry.readthedocs.io/en/latest/gfql/performance.html)
* API References
    - [PyGraphistry API Reference](https://pygraphistry.readthedocs.io/en/latest/api/index.html)
      - [Visualization & Compute](https://pygraphistry.readthedocs.io/en/latest/visualization/index.html)
      - [GFQL](https://pygraphistry.readthedocs.io/en/latest/gfql/index.html)
      - [Plugins](https://pygraphistry.readthedocs.io/en/latest/plugins.html)
    - [iframe Reference API](https://hub.graphistry.com/docs/api/1/rest/url/#urloptions)
    - [JavaScript](https://hub.graphistry.com/static/js-docs/index.html?path=/docs/introduction--docs): Browser (vanilla, React), Node.js, and storybooks
    - [REST API](https://hub.graphistry.com/docs/api/1/rest/auth/): Language-neutral

## Graphistry ecosystem

  - [Graphistry Hub and Self-Hosting](https://www.graphistry.com/get-started)
  - [Louie.AI](https://louie.ai/) (new!): GenAI-native Python notebooks, dashboards, & pipelines with native Graphistry integration
  - [graph-app-kit](https://github.com/graphistry/graph-app-kit): Streamlit Python dashboards with graph ecosystem integrations
  - [cu-cat](https://chat.openai.com/chat): End-to-end GPU automated feature engineering
  - Graphistry core API clients: [iframe](https://hub.graphistry.com/docs/api/1/rest/url/#urloptions), [JavaScript](https://hub.graphistry.com/static/js-docs/index.html?path=/docs/introduction--docs), [REST](https://hub.graphistry.com/docs/api/1/rest/auth/), [Python](https://pygraphistry.readthedocs.io/en/latest/api/index.html)

See also our partner technologies [RAPIDS](https://rapids.ai/) and [Apache Arrow](https://arrow.apache.org/), and database partners such as [Neo4j](https://neo4j.com/users/graphistry-inc/), [TigerGraph](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/tigergraph/tigergraph_pygraphistry_bindings.html), [Amazon Neptune](https://docs.aws.amazon.com/neptune/latest/userguide/visualization-graphistry.html), [Splunk](https://www.splunk.com/en_us/blog/security/supercharge-cybersecurity-investigations-with-splunk-and-graphistry-a-powerful-combination-for-interactive-graph-exploration.html), and [Databricks](https://www.databricks.com/solutions/accelerators/incident-investigation-using-graphistry). Get started with these in the [notebook data provider demo gallery](https://pygraphistry.readthedocs.io/en/latest/notebooks/plugins.connectors.html).


## Community and support

- [Blog](https://www.graphistry.com/blog) for tutorials, case studies, and updates
- [Slack](https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g): Join the Graphistry Community Slack for discussions and support
- [Twitter](https://twitter.com/graphistry) & [LinkedIn](https://www.linkedin.com/company/graphistry): Follow for updates
- [GitHub Issues](https://github.com/graphistry/pygraphistry/issues) open source support
- [Graphistry ZenDesk](https://graphistry.zendesk.com/) dedicated enterprise support

## License

PyGraphistry is open-sourced under the [BSD 3-Clause License](LICENSE.txt)

Graphistry, GFQL, and Louie.AI are trademarks of Graphistry, Inc.

## Contributing

See the [CONTRIBUTE](CONTRIBUTE.md) and [DEVELOP](DEVELOP.md) pages for details on how to participate in PyGraphistry development

