# PyGraphistry: Explore Relationships

![Build Status](https://github.com/graphistry/pygraphistry/workflows/CI%20Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pygraphistry/badge/?version=latest)](https://pygraphistry.readthedocs.io/en/latest/)
[![Latest Version](https://img.shields.io/pypi/v/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
[![Latest Version](https://img.shields.io/pypi/pyversions/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
[![License](https://img.shields.io/pypi/l/graphistry.svg)](https://pypi.python.org/pypi/graphistry)
[![Downloads](https://pepy.tech/badge/graphistry/month)](https://pepy.tech/project/graphistry/month)

[![Uptime Robot status](https://img.shields.io/uptimerobot/status/m787548531-e9c7b7508fc76fea927e2313?label=hub.graphistry.com)](https://status.graphistry.com/) [<img src="https://img.shields.io/badge/slack-Graphistry%20chat-yellow.svg?logo=slack">](https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g)
[![Twitter Follow](https://img.shields.io/twitter/follow/graphistry)](https://twitter.com/graphistry)

PyGraphistry is a Python visual graph analytics library to extract, transform, and load big graphs into [Graphistry](https://www.graphistry.com) end-to-end GPU  visual graph analytics sessions.

Graphistry gets used on problems like visually mapping the behavior of devices and users and for analyzing machine learning results. It provides point-and-click features like timebars, search, filtering, clustering, coloring, sharing, and more. Graphistry is the only tool built ground-up for large graphs. The client's custom WebGL rendering engine renders up to 8MM nodes + edges at a time, and most older client GPUs smoothly support somewhere between 100K and 1MM elements. The serverside GPU analytics engine supports even bigger graphs.

The PyGraphistry Python client helps several kinds of usage modes:

* **Data scientists**: Go from data to accelerated visual explorations in a couple lines, share live results, build up more advanced views over time, and do it all from notebook environments like Jupyter and Google Colab
* **Developers**: Quickly prototype stunning Python solutions with PyGraphistry, embed in a language-neutral way with the [REST APIs](https://hub.graphistry.com/docs/api/), and go deep on customizations like colors, icons, layouts, JavaScript, and more
* **Analysts**: Every Graphistry session is a point-and-click environment with interactive search, filters, timebars, histograms, and more
* **Dashboarding**: Embed into your favorite framework. Additionally, see our sister project [Graph-App-Kit](https://github.com/graphistry/graph-app-kit) for quickly building interactive graph dashboards by launching a stack built on PyGraphistry, StreamLit, Docker, and ready recipes for integrating with common graph libraries

PyGraphistry is a friendly and optimized PyData-native interface to the language-neutral [Graphistry REST APIs](https://hub.graphistry.com/docs/api/).
You can use PyGraphistry with traditional Python data sources like CSVs, SQL, Neo4j, Splunk, and more (see below). Wrangle data however you want, and with especially good support for Pandas dataframes, Apache Arrow tables, and Nvidia RAPIDS cuDF dataframes.

1. [Interactive Demo](#demo-of-friendship-communities-on-facebook)
2. [Graph Gallery](#gallery)
3. [Install](#install)
4. [Tutorial](#tutorial-les-misérables)
5. [Next Steps](#next-steps)
6. [Resources](#resources)

## Demo of Friendship Communities on Facebook

<table style="width:100%;">
  <tr valign="top">
    <td align="center">Click to open interactive version! <em>(For server-backed interactive analytics, use an API key)</em><a href="http://hub.graphistry.com/graph/graph.html?dataset=Facebook&splashAfter=true" target="_blank"><img src="http://i.imgur.com/Ows4rK4.png" title="Click to open."></a>
    <em>Source data: <a href="http://snap.stanford.edu" target="_blank">SNAP</a></em>
    </td>
  </tr>
</table>

## PyGraphistry is...

* **Fast & gorgeous:** Interactively cluster, filter, inspect large amounts of data, and zip through timebars. It clusters large graphs with a descendant of the gorgeous ForceAtlas2 layout algorithm introduced in Gephi. Our data explorer connects to Graphistry's GPU cluster to layout and render hundreds of thousand of nodes+edges in your browser at unparalleled speeds.

* **Easy to install:** `pip install` the client in your notebook or web app, and then connect to a [free Graphistry Hub account](https://www.graphistry.com/get-started) or [launch your own private GPU server](https://www.graphistry.com/get-started)

   ```python
  # pip install --user graphistry
  # pip install --user graphistry[bolt,gremlin,nodexl,igraph,networkx]  # optional
  import graphistry
  graphistry.register(api=3, username='abc', password='xyz')  # Free: hub.graphistry.com
  #graphistry.register(..., protocol='http', host='my.site.ngo')  # Private
  ```

* **Notebook-friendly:** PyGraphistry plays well with interactive notebooks like [Jupyter](http://ipython.org), [Zeppelin](https://zeppelin.incubator.apache.org/), and [Databricks](http://databricks.com). Process, visualize, and drill into with graphs directly within your notebooks:

    ```python
    graphistry.edges(pd.read_csv('rows.csv'), 'col_a', 'col_b').plot()
    ```

* **Great for events, CSVs, and more:** Not sure if your data is graph-friendly? PyGraphistry's `hypergraph` transform helps turn any sample data like CSVs, SQL results, and event data into a graph for pattern analysis:

     ```python
     rows = pandas.read_csv('transactions.csv')[:1000]
     graphistry.hypergraph(rows)['graph'].plot()
     ```

* **Embeddable:** Drop live views into your web dashboards and apps (and go further with [JS/React](https://hub.graphistry.com/docs)):

    ```python
    iframe_url = g.plot(render=False)
    print(f'<iframe src="{ iframe_url }"></iframe>')
    ```

* **Configurable:** In-tool or via the declarative APIs, use the powerful encodings systems for tasks like coloring by time, sizing by score, clustering by weight, show icons by type, and more.

* **Shareable:** Share live links, configure who has access, and more! [(Notebook tutorial)](https://github.com/graphistry/pygraphistry/blob/master/demos/more_examples/graphistry_features/sharing_tutorial.ipynb)  

### Explore any data as a graph

It is easy to turn arbitrary data into insightful graphs. PyGraphistry comes with many built-in connectors, and by supporting Python dataframes (Pandas, Arrow, RAPIDS), it's easy to bring standard Python data libraries. If the data comes as a table instead of a graph, PyGraphistry will help you extract and explore the relationships.

* [Pandas](http://pandas.pydata.org)

     ```python
     edges = pd.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst'])
     graphistry.edges(edges, 'src', 'dst').plot()
     ```

     ```python
     table_rows = pd.read_csv('honeypot.csv')
     graphistry.hypergraph(table_rows, ['attackerIP', 'victimIP', 'victimPort', 'vulnName'])['graph'].plot()
     ```

     ```python
     graphistry.hypergraph(table_rows, ['attackerIP', 'victimIP', 'victimPort', 'vulnName'],
         direct=True, 
         opts={'EDGES': {
           'attackerIP': ['victimIP', 'victimPort', 'vulnName'], 
           'victimIP': ['victimPort', 'vulnName'],
           'victimPort': ['vulnName']
   }})['graph'].plot()
     ```

     ```python
     ### Override smart defaults with custom settings
     g1 = graphistry.bind(source='src', destination='dst').edges(edges)
     g2 = g1.nodes(nodes).bind(node='col2')
     g3 = g2.bind(point_color='col3')
     g4 = g3.settings(url_params={'edgeInfluence': 1.0, play: 2000})
     url = g4.plot(render=False)
     ```

    ```python
    ### Read back data and create modified variants
    enriched_edges = my_function1(g1._edges)
    enriched_nodes = my_function2(g1._nodes)
    g2 = g1.edges(enriched_edges).nodes(enriched_nodes)
    g2.plot()
    ```

* [Spark](https://spark.apache.org/)/[Databricks](https://databricks.com/) ([ipynb demo](demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.ipynb), [dbc demo](demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.dbc))

    ```python
    #optional but recommended
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")

    edges_df = (
        spark.read.format('json').
          load('/databricks-datasets/iot/iot_devices.json')
          .sample(fraction=0.1)
    )
    g = graphistry.edges(edges_df, 'device_name', 'cn')

    #notebook
    displayHTML(g.plot())

    #dashboard: pick size of choice
    displayHTML(
      g.settings(url_params={'splashAfter': 'false'})
        .plot(override_html_style="""
          width: 50em;
          height: 50em;
        """)
    )
    ```

* GPU [RAPIDS.ai](https://www.rapids.ai)
  
    ```python
    edges = cudf.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst'])
    graphistry.edges(edges, 'src', 'dst').plot()
    ```

* [Apache Arrow](https://arrow.apache.org/)
  
    ```python
     edges = pa.Table.from_pandas(pd.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst']))
     graphistry.edges(edges, 'src', 'dst').plot()
    ```

* [Neo4j](http://neo4j.com) ([notebook demo](demos/demos_databases_apis/neo4j/official/graphistry_bolt_tutorial_public.ipynb))
  
    ```python
    NEO4J_CREDS = {'uri': 'bolt://my.site.ngo:7687', 'auth': ('neo4j', 'mypwd')}
    graphistry.register(bolt=NEO4J_CREDS)
    graphistry.cypher("MATCH (n1)-[r1]->(n2) RETURN n1, r1, n2 LIMIT 1000").plot()
    ```

    ```python
    graphistry.cypher("CALL db.schema()").plot()
    ```

    ```python
    from neo4j import GraphDatabase, Driver
    graphistry.register(bolt=GraphDatabase.driver(**NEO4J_CREDS))
    g = graphistry.cypher("""
      MATCH (a)-[p:PAYMENT]->(b)
      WHERE p.USD > 7000 AND p.USD < 10000
      RETURN a, p, b
      LIMIT 100000""")
    print(g._edges.columns)
    g.plot()
    ```

* [Azure Cosmos DB (Gremlin)](https://azure.microsoft.com/en-us/services/cosmos-db/)

    ```python
    # pip install --user gremlinpython
    # Options in help(graphistry.cosmos)
    g = graphistry.cosmos(
        COSMOS_ACCOUNT='',
        COSMOS_DB='',
        COSMOS_CONTAINER='',
        COSMOS_PRIMARY_KEY=''
    )
    g2 = g.gremlin('g.E().sample(10000)').fetch_nodes()
    g2.plot()
    ```

* [Amazon Neptune (Gremlin)](https://aws.amazon.com/neptune/) ([notebook demo](demos/demos_databases_apis/neptune/neptune_tutorial.ipynb), [dashboarding demo](https://aws.amazon.com/blogs/database/enabling-low-code-graph-data-apps-with-amazon-neptune-and-graphistry/))

    ```python
    # pip install --user gremlinpython==3.4.10
    #   - Deploy tips: https://github.com/graphistry/graph-app-kit/blob/master/docs/neptune.md
    #   - Versioning tips: https://gist.github.com/lmeyerov/459f6f0360abea787909c7c8c8f04cee
    #   - Login options in help(graphistry.neptune)
    g = graphistry.neptune(endpoint='wss://zzz:8182/gremlin')
    g2 = g.gremlin('g.E().limit(100)').fetch_nodes()
    g2.plot()
    ```

* [TigerGaph](https://tigergraph.com) ([notebook demo](demos/demos_databases_apis/tigergraph/tigergraph_pygraphistry_bindings.ipynb))

    ```python
    g = graphistry.tigergraph(protocol='https', ...)
    g2 = g.gsql("...", {'edges': '@@eList'})
    g2.plot()
    print('# edges', len(g2._edges))
    ```

    ```python
    g.endpoint('my_fn', {'arg': 'val'}, {'edges': '@@eList'}).plot()
    ```

* [IGraph](http://igraph.org)

    ```python
    graph = igraph.read('facebook_combined.txt', format='edgelist', directed=False)
    graphistry.bind(source='src', destination='dst').plot(graph)
    ```

* [NetworkX](https://networkx.github.io) ([notebook demo](demos/demos_databases_apis/networkx/networkx.ipynb))

    ```python
    graph = networkx.read_edgelist('facebook_combined.txt')
    graphistry.bind(source='src', destination='dst', node='nodeid').plot(graph)
    ```

* [HyperNetX](https://github.com/pnnl/HyperNetX) ([notebook demo](demos/demos_databases_apis/hypernetx/hypernetx.ipynb))

    ```python
    hg.hypernetx_to_graphistry_nodes(H).plot()
    ```

    ```python
    hg.hypernetx_to_graphistry_bipartite(H.dual()).plot()
    ```

* [Splunk](https://www.splunk.com) ([notebook demo](demos/demos_databases_apis/splunk/splunk_demo_public.ipynb))

    ```python
    df = splunkToPandas("index=netflow bytes > 100000 | head 100000", {})
    graphistry.edges(df, 'src_ip', 'dest_ip').plot()
    ```

* [NodeXL](https://www.nodexl.com) ([notebook demo](demos/demos_databases_apis/nodexl/official/nodexl_graphistry.ipynb))

    ```python
    graphistry.nodexl('/my/file.xls').plot()
    ```

    ```python
    graphistry.nodexl('https://file.xls').plot()
    ```

    ```python
    graphistry.nodexl('https://file.xls', 'twitter').plot()
    graphistry.nodexl('https://file.xls', verbose=True).plot()
    graphistry.nodexl('https://file.xls', engine='xlsxwriter').plot()
    graphistry.nodexl('https://file.xls')._nodes
    ```  

### Quickly configurable

Set visual attributes through [quick data bindings](https://hub.graphistry.com/docs/api/2/rest/upload/#createdataset2) and set [all sorts of URL options](https://hub.graphistry.com/docs/api/1/rest/url/). Check out the tutorials on [colors](demos/more_examples/graphistry_features/encodings-colors.ipynb), [sizes](demos/more_examples/graphistry_features/encodings-sizes.ipynb), [icons](demos/more_examples/graphistry_features/encodings-icons.ipynb), [badges](demos/more_examples/graphistry_features/encodings-badges.ipynb), [weighted clustering](demos/more_examples/graphistry_features/edge-weights.ipynb) and [sharing controls](https://github.com/graphistry/pygraphistry/blob/master/demos/more_examples/graphistry_features/sharing_tutorial.ipynb):

  ```python
    g
      .privacy(mode='private', invited_users=[{'email': 'friend1@site.ngo', 'action': '10'}], notify=False)
      .edges(df, 'col_a', 'col_b')
      .edges(my_transform1(g._edges))
      .nodes(df, 'col_c')
      .nodes(my_transform2(g._nodes))
      .bind(source='col_a', destination='col_b', node='col_c')
      .bind(
        point_color='col_a',
      point_size='col_b',
      point_title='col_c',
      point_x='col_d',
      point_y='col_e')
      .bind(
        edge_color='col_m',
      edge_weight='col_n',
      edge_title='col_o')
      .encode_edge_color('timestamp', ["blue", "yellow", "red"], as_continuous=True)
      .encode_point_icon('device_type', categorical_mapping={'macbook': 'laptop', ...})
      .encode_point_badge('passport', 'TopRight', categorical_mapping={'Canada': 'flag-icon-ca', ...})
      .addStyle(bg={'color': 'red'}, fg={}, page={'title': 'My Graph'}, logo={})
      .settings(url_params={
        'play': 2000,
      'menu': True, 'info': True,
      'showArrows': True,
      'pointSize': 2.0, 'edgeCurvature': 0.5,
      'edgeOpacity': 1.0, 'pointOpacity': 1.0,
      'lockedX': False, 'lockedY': False, 'lockedR': False,
      'linLog': False, 'strongGravity': False, 'dissuadeHubs': False,
      'edgeInfluence': 1.0, 'precisionVsSpeed': 1.0, 'gravity': 1.0, 'scalingRatio': 1.0,
      'showLabels': True, 'showLabelOnHover': True,
      'showPointsOfInterest': True, 'showPointsOfInterestLabel': True, 'showLabelPropertiesOnHover': True,
      'pointsOfInterestMax': 5
      })
      .plot()
  ```

### Gallery

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

## Install

### Get

You need to install the PyGraphistry Python client and connect it to a Graphistry GPU server of your choice:

1. Graphistry server account:
    * Create a free [Graphistry Hub account](https://www.graphistry.com/get-started) for open data, or [one-click launch your own private AWS/Azure instance](https://www.graphistry.com/get-started)
    * Later, [setup and manage](https://github.com/graphistry/graphistry-cli) your own private Docker instance ([contact](https://www.graphistry.com/demo-request))

2. PyGraphistry Python client:
    * `pip install --user graphistry` (Python 3.6+) or [directly call the HTTP API](https://hub.graphistry.com/docs/api/)
        * Use `pip install --user graphistry[all]` for optional dependencies such as Neo4j drivers
    * To use from a notebook environment, run your own [Jupyter](https://jupyter.org/) server ([one-click launch your own private AWS/Azure GPU instance](https://www.graphistry.com/get-started)) or another such as [Google Colab](https://colab.research.google.com)
    * See immediately following `configure` section for how to connect

### Configure

Most users connect to a Graphistry GPU server account via `graphistry.register(api=3, username='abc', password='xyz')` (hub.graphistry.com) or  `graphistry.register(api=3, username='abc', password='xyz', protocol='http', server='my.private_server.org')`

For more advanced configuration, read on for:

* Version: Use protocol `api=3`, which will soon become the default, or a legacy version

* Tokens: Connect to a GPU server by providing a `username='abc'`/`password='xyz'`, or for advanced long-running service account software, a refresh loop using 1-hour-only JWT tokens

* Private servers: PyGraphistry defaults to using the free [Graphistry Hub](https://hub.graphistry.com) public API

  * Connect to a [private Graphistry server](https://www.graphistry.com/get-started) and provide optional settings specific to it via `protocol`, `server`, and in some cases, `client_protocol_hostname`

Non-Python users may want to explore the underlying language-neutral [authentication REST API docs](https://hub.graphistry.com/docs/api/1/rest/auth/).

#### Advanced Login

* **Recommended for people:** Provide your account username/password:

```python
import graphistry
graphistry.register(api=3, username='username', password='your password')
```

* **For code**: Long-running services may prefer to use 1-hour JWT tokens:

```python
import graphistry
graphistry.register(api=3, username='username', password='your password')
initial_one_hour_token = graphistry.api_token()
graphistry.register(api=3, token=initial_one_hour_token)

# must run every 59min
graphistry.refresh() 
fresh_token = graphistry.api_token()
assert initial_one_hour_token != fresh_token
```

Alternatively, you can rerun `graphistry.register(api=3, username='username', password='your password')`, which will also fetch a fresh token.

#### Advanced: Private servers

Specify which Graphistry server to reach:

```python
graphistry.register(protocol='https', server='hub.graphistry.com')
```

Private Graphistry notebook environments are preconfigured to fill in this data for you:

```python
graphistry.register(protocol='http', server='nginx', client_protocol_hostname='')
```

Using `'http'`/`'nginx'` ensures uploads stay within the Docker network (vs. going more slowly through an outside network), and client protocol `''` ensures the browser URLs do not show `http://nginx/`, and instead use the server's name. (See immediately following **Switch client URL** section.)

#### Advanced: Switch client URL

In cases such as when the notebook server is the same as the Graphistry server, you may want your Python code to  *upload* to a known local Graphistry address without going outside the network (e.g., `http://nginx` or `http://localhost`), but for web viewing, generate and embed URLs to a different public address (e.g., `https://graphistry.acme.ngo/`). In this case, explicitly set a  client (browser) location different from `protocol` / `server`:

```python
graphistry.register(
    ### fast local notebook<>graphistry upload
    protocol='http', server='nginx', 
  
    ### shareable public URL for browsers
    client_protocol_hostname='https://graphistry.acme.ngo'
)
```

Prebuilt Graphistry servers are already setup to do this out-of-the-box.

#### Advanced: Sharing controls

Graphistry supports flexible sharing permissions that are similar to Google documents and Dropbox links

By default, visualizations are publicly viewable by anyone with the URL (that is unguessable & unlisted), and only editable by their owner.

* Private-only: You can globally default uploads to private:

```python
graphistry.privacy()
```

* Invitees: You can share access to specify users, and optionally, even email them invites

```python
VIEW = "10"
EDIT = "20"
graphistry.privacy(
  mode='private',
  invited_users=[ 
    {"email": "friend1@site1.com", "action": VIEW},
    {"email": "friend2@site2.com", "action": EDIT}
  ],
  notify=True)
```

* Per-visualization: You can choose different rules for global defaults vs. for specific visualizations

```python
graphistry.privacy(invited_users=[...])
g = graphistry.hypergraph(pd.read_csv('...'))['graph']
g.privacy(notify=True).plot()
```

See additional examples in the [sharing tutorial](https://github.com/graphistry/pygraphistry/blob/master/demos/more_examples/graphistry_features/sharing_tutorial.ipynb)

## Tutorial: Les Misérables

Let's visualize relationships between the characters in [Les Misérables](http://en.wikipedia.org/wiki/Les_Misérables).
For this example, we'll choose [Pandas](http://pandas.pydata.org) to wrangle data and [IGraph](http://igraph.org) to run a community detection algorithm. You can [view](demos/more_examples/simple/MarvelTutorial.ipynb) the Jupyter notebook containing this example.

Our [dataset is a CSV file](https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/lesmiserables.csv) that looks like this:

| source        | target        | value  |
| ------------- |:-------------:| ------:|
| Cravatte | Myriel | 1
| Valjean | Mme.Magloire | 3
| Valjean | Mlle.Baptistine | 3

*Source* and *target* are character names, and the *value* column counts the number of time they meet. Parsing is a one-liner with Pandas:

```python
import pandas
links = pandas.read_csv('./lesmiserables.csv')
```

### Quick Visualization

If you already have graph-like data, use this step. Otherwise, try the [Hypergraph Transform](demos/demos_by_use_case/logs/malware-hypergraph/Malware%20Hypergraph.ipynb) for creating graphs from rows of data (logs, samples, records, ...).

PyGraphistry can plot graphs directly from Pandas data frames, Arrow tables, cuGraph GPU data frames, IGraph graphs, or NetworkX graphs. Calling *plot* uploads the data to our visualization servers and return an URL to an embeddable webpage containing the visualization.

To define the graph, we `bind` *source* and *destination* to the columns indicating the start and end nodes of each edges:

```python
import graphistry
graphistry.register(api=3, username='YOUR_ACCOUNT_HERE', password='YOUR_PASSWORD_HERE')

g = graphistry.bind(source="source", destination="target")
g.edges(links).plot()
```

You should see a beautiful graph like this one:
![Graph of Miserables](http://i.imgur.com/dRHHTyK.png)

### Adding Labels

Let's add labels to edges in order to show how many times each pair of characters met. We create a new column called *label* in edge table *links* that contains the text of the label and we bind *edge_label* to it.

```python
links["label"] = links.value.map(lambda v: "#Meetings: %d" % v)
g = g.bind(edge_title="label")
g.edges(links).plot()
```

### Controlling Node Title, Size, Color, and Position

Let's size nodes based on their [PageRank](http://en.wikipedia.org/wiki/PageRank) score and color them using their [community](https://en.wikipedia.org/wiki/Community_structure).

#### Warmup: IGraph for computing statistics

[IGraph](http://igraph.org/python/) already has these algorithms implemented for us for small graphs. (See our cuGraph examples for big graphs.) If IGraph is not already installed, fetch it with `pip install python-igraph`. Warning: `pip install igraph` will install the wrong package!

We start by converting our edge dateframe into an IGraph. The plotter can do the conversion for us using the *source* and *destination* bindings. Then we compute two new node attributes (*pagerank* & *community*).

```python
ig = graphistry.pandas2igraph(links)
ig.vs['pagerank'] = ig.pagerank()
ig.vs['community'] = ig.community_infomap().membership
```

#### Bind node data to visual node attributes

We can then bind the node `community` and `pagerank` columns to visualization attributes:

```python
g.bind(point_color='community', point_size='pagerank').plot(ig)
```

See the [color palette documentation](https://hub.graphistry.com/docs/api/2/rest/upload/colors/#extendedpalette2) for specifying color values by using built-in ColorBrewer palettes (`int32`) or custom RGB values (`int64`).

To control the position, we can add `.bind(point_x='colA', point_y='colB').settings(url_params={'play': 0})` ([see demos](demos/more_examples/graphistry_features/external_layout) and [additional url parameters](https://hub.graphistry.com/docs/api/1/rest/url/#urloptions)]). In `api=1`, you created columns named `x` and `y`.

You may also want to bind `point_title`: `.bind(point_title='colA')`.

For more in-depth examples, check out the tutorials on [colors](demos/more_examples/graphistry_features/encodings-colors.ipynb) and [sizes](demos/more_examples/graphistry_features/encodings-sizes.ipynb).

![Second Graph of Miserables](http://i.imgur.com/P7fm5sn.png)

### Add edge colors and weights

By default, edges get colored as a gradient between their source/destination node colors. You can override this by setting `.bind(edge_color='colA')`, similar to how node colors function. ([See color documentation](https://hub.graphistry.com/docs/api/2/rest/upload/colors/#extendedpalette2).)

Similarly, you can bind the edge weight, where higher weights cause nodes to cluster closer together: `.bind(edge_weight='colA')`. [See tutorial](demos/more_examples/graphistry_features/edge-weights.ipynb).

For more in-depth examples, check out the tutorials on [colors](demos/more_examples/graphistry_features/encodings-colors.ipynb) and [weighted clustering](demos/more_examples/graphistry_features/edge-weights.ipynb).

### More advanced color and size controls

You may want more controls like using gradients or maping specific values:

```python
g.encode_edge_color('time_col', ["blue", "red"], as_continuous=True)
g.encode_edge_color('type_col', ["#000", "#F00", "#F0F", "#0FF"], as_categorical=True)
g.encode_edge_color('brand',
  categorical_mapping={'toyota': 'red', 'ford': 'blue'},
  default_mapping='#CCC')
g.encode_point_size('criticality',
  categorical_mapping={'critical': 200, 'ok': 100},
  default_mapping=50)
```

For more in-depth examples, check out the tutorials on [colors](demos/more_examples/graphistry_features/encodings-colors.ipynb).

### Custom icons and badges

You can add a main icon and multiple peripherary badges to provide more visual information. Use column `type` for the icon type to appear visually in the legend. The glyph system supports text, icons, flags, and images, as well as multiple mapping and style controls.

#### Main icon

```python
g.encode_point_icon(
  'some_column',
  shape="circle", #clip excess
  categorical_mapping={
      'macbook': 'laptop', #https://fontawesome.com/v4.7.0/icons/
      'Canada': 'flag-icon-ca', #ISO3611-Alpha-2: https://github.com/datasets/country-codes/blob/master/data/country-codes.csv
      'embedded_smile': 'data:svg...',
      'external_logo': 'http://..../img.png'
  },
  default_mapping="question")
g.encode_point_icon(
  'another_column',
  continuous_binning=[
    [20, 'info'],
    [80, 'exclamation-circle'],
    [None, 'exclamation-triangle']
  ]
)
g.encode_point_icon(
  'another_column',
  as_text=True,
  categorical_mapping={
    'Canada': 'CA',
    'United States': 'US'
    }
)
```

For more in-depth examples, check out the tutorials on [icons](demos/more_examples/graphistry_features/encodings-icons.ipynb).

#### Badges

```python
# see icons examples for mappings and glyphs
g.encode_point_badge('another_column', 'TopRight', categorical_mapping=...)

g.encode_point_badge('another_column', 'TopRight', categorical_mapping=...,
  shape="circle",
  border={'width': 2, 'color': 'white', 'stroke': 'solid'},
  color={'mapping': {'categorical': {'fixed': {}, 'other': 'white'}}},
  bg={'color': {'mapping': {'continuous': {'bins': [], 'other': 'black'}}}})
```

For more in-depth examples, check out the tutorials on [badges](demos/more_examples/graphistry_features/encodings-badges.ipynb).

### Theming

You can customize several style options to match your theme:

```python
g.addStyle(bg={'color': 'red'})
g.addStyle(bg={
  'color': '#333',
  'gradient': {
    'kind': 'radial',
    'stops': [ ["rgba(255,255,255, 0.1)", "10%", "rgba(0,0,0,0)", "20%"] ]}})
g.addStyle(bg={'image': {'url': 'http://site.com/cool.png', 'blendMode': 'multiply'}})
g.addStyle(fg={'blendMode': 'color-burn'})
g.addStyle(page={'title': 'My site'})
g.addStyle(page={'favicon': 'http://site.com/favicon.ico'})
g.addStyle(logo={'url': 'http://www.site.com/transparent_logo.png'})
g.addStyle(logo={
  'url': 'http://www.site.com/transparent_logo.png',
  'dimensions': {'maxHeight': 200, 'maxWidth': 200},
  'style': {'opacity': 0.5}
})
```

### Transforms

You can quickly manipulate graphs as well:

**Generate node table**:
```python
g = graphisty.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
g2 = g.materialize_nodes()
g2._nodes  # pd.DataFrame({'id': ['a', 'b', 'c']})
```

***Compute degrees**
```python
g = graphisty.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
g2 = g.get_degree()
g2._nodes  # pd.DataFrame({
           #  'id': ['a', 'b', 'c'],
           #  'degree_in': [0, 1, 1],
           #  'degree_out': [1, 1, 0],
           #  'degree': [1, 1, 1]
           #})
```

**Pipelining**:

```python
def capitalize(df, col):
  df2 = df.copy()
  df2[col] df[col].str.capitalize()
  return df2

g
  .cypher('MATCH (a)-[e]->(b) RETURN a, e, b')
  .nodes(lambda g: capitalize(g._nodes, 'nTitle'))
  .edges(capitalize, None, None, 'eTitle'), 
  .pipe(lambda g: g.nodes(g._nodes.pipe(capitalize, 'nTitle')))
```

**Table to graph**:

```python
df = pd.read_csv('events.csv')
hg = graphistry.hypergraph(df, ['user', 'email', 'org'], direct=True)
g = hg['graph']  # g._edges: | src, dst, user, email, org, time, ... |
g.plot()
```

**Removing nodes**

```python
g = graphisty.edges(pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'a']})
g2 = g.drop_nodes(['c'])  # drops node c, edge c->a, edge b->c, 
```

### Control layouts

```python
g = graphisty.edges(pd.DataFrame({'s': ['a', 'b', 'b'], 'd': ['b', 'c', 'd']})

g2a = g.tree_layout()
g2b = g2.tree_layout(allow_cycles=False, remove_self_loops=False, vertical=False)
g2c = g2.tree_layout(ascending=False, level_align='center')
g2d = g2.tree_layout(level_sort_values_by=['type', 'degree'], level_sort_values_by_ascending=False)

g3a = g2a.layout_settings(locked_r=True, play=1000)
g3b = g2a.layout_settings(locked_y=True, play=0)
g3c = g2a.layout_settings(locked_x=True)
```
## Next Steps

1. Create a free public data [Graphistry Hub](https://www.graphistry.com/get-started) account or [one-click launch a private Graphistry instance in AWS](https://www.graphistry.com/get-started)
2. Check out the [analyst](demos/for_analysis.ipynb) and [developer](demos/for_developers.ipynb) introductions, or [try your own CSV](demos/upload_csv_miniapp.ipynb)
3. Explore the [demos folder](demos) for your favorite [file format, database, API](demos/demos_databases_apis), use case domain, kind of analysis, and [visual analytics feature](demos/more_examples/graphistry_features)

## Resources

* Graphistry [In-Tool UI Guide](https://hub.graphistry.com/docs/ui/index/)
* [General and REST API docs](https://hub.graphistry.com/docs/api/):
  * [URL settings](https://hub.graphistry.com/docs/api/1/rest/url/#urloptions)
  * [Authentication](https://hub.graphistry.com/docs/api/1/rest/auth/)
  * [Uploading](https://hub.graphistry.com/docs/api/2/rest/upload/#createdataset2), including multiple file formats and settings
  * [Color bindings](https://hub.graphistry.com/docs/api/2/rest/upload/colors/#extendedpalette2) and [color palettes](https://hub.graphistry.com/docs/api/api-color-palettes/) (ColorBrewer)
  * Bindings and colors, REST API, embedding URLs and URL parameters, dynamic JS API, and more
  * JavaScript and more!
* Python-specific
  * [Python API ReadTheDocs](http://pygraphistry.readthedocs.org/en/latest/)
  * Within a notebook, you can always run `help(graphistry)`, `help(graphistry.hypergraph)`, etc.
* [Administration docs](https://github.com/graphistry/graphistry-cli) for sizing, installing, configuring, managing, and updating Graphistry servers
* [Graph-App-Kit Dashboarding](https://github.com/graphistry/graph-app-kit) dashboarding
