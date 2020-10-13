[![Build Status](https://travis-ci.org/graphistry/pygraphistry.svg?branch=master)](https://travis-ci.org/graphistry/pygraphistry) 
[![Documentation Status](https://readthedocs.org/projects/pygraphistry/badge/?version=latest)](https://pygraphistry.readthedocs.io/en/latest/)
[![Latest Version](https://img.shields.io/pypi/v/graphistry.svg)](https://pypi.python.org/pypi/graphistry) 
[![Latest Version](https://img.shields.io/pypi/pyversions/graphistry.svg)](https://pypi.python.org/pypi/graphistry) 
[![License](https://img.shields.io/pypi/l/graphistry.svg)](https://pypi.python.org/pypi/graphistry)

[![Downloads](https://pepy.tech/badge/graphistry/month)](https://pepy.tech/project/graphistry/month) ![Twitter Follow](https://img.shields.io/twitter/follow/graphistry)

# PyGraphistry: Explore Relationships

PyGraphistry is a Python visual graph analytics library to extract, transform, and load big graphs into [Graphistry's](http://www.graphistry.com) visual graph analytics platform. It is typically used by data scientists, developers, and operational analysts on problems like visually mapping the behavior of devices and users. 

The Python client makes it easy to go from your existing data to a Graphistry server. Through strong notebook support, data scientists can quickly go from data to accelerated visual explorations, and developers can quickly prototype stunning solutions with their users.

Graphistry supports unusually large graphs for interactive visualization. The client's custom WebGL rendering engine renders up to 8MM nodes and edges at a time, and most older client GPUs smoothly support somewhere between 100K and 1MM elements. The serverside GPU analytics engine supports even bigger graphs.

1. [Interactive Demo](#demo-of-friendship-communities-on-facebook)
2. [Graph Gallery](#gallery)
3. [Install](#install)
4. [Tutorial](#tutorial-les-misérables)
5. [Next Steps](#next-steps)
5. [Resources](#resources)

## Demo of Friendship Communities on Facebook

<table style="width:100%;">
  <tr valign="top">
    <td align="center">Click to open interactive version! <em>(For server-backed interactive analytics, use an API key)</em><a href="http://labs.graphistry.com/graph/graph.html?dataset=Facebook&usertag=github&info=true&static=true&contentKey=Facebook_Github_Demo&play=7000&center=false&menu=true&goLive=false&left=-2.02e+4&right=1.51e+4&top=-1.07e+4&bottom=9.15e+3&legend={%22nodes%22:%20%22People%20(Names%20are%20fake).%3Cbr/%3E%20Color%20indicates%20community%20and%20size%20shows%20popularity.%22,%20%22edges%22:%20%22Friendships%22,%20%22subtitle%22:%20%22%3Cp%3ECreate%20your%20own%20visualizations%20with%20%3Ca%20href=\%22https://github.com/graphistry/pygraphistry/\%22%3EPyGraphistry%3C/a%3E.%3C/p%3E%22,%20%22title%22:%20%22%3Ch3%3EFacebook%20Friendships%20(Read-only%20Mode)%3C/h3%3E%22}" target="_blank"><img src="http://i.imgur.com/Ows4rK4.png" title="Click to open."></a>
    <em>Source data: <a href="http://snap.stanford.edu" target="_blank">SNAP</a></em>
	</td>
  </tr>
</table>

<!-- IFRAME VERSION
### The Demo:

<table style="width:100%;">
  <tr valign="top">
    <td align="center"><b>Friendship Communities on Facebook.</b> (Read-only interactive version.)<br><iframe width="100%" height="500" src="http://labs.graphistry.com/graph/graph.html?dataset=PyGraphistry/VAXME5LTUN&usertag=github&info=true&static=true&contentKey=Facebook_Github_Demo&play=0&center=false&menu=true&goLive=false&left=-2.02e+4&right=1.51e+4&top=-1.07e+4&bottom=9.15e+3&legend={%22nodes%22:%20%22People%20(Names%20are%20fake).%3Cbr/%3E%20Color%20indicates%20community%20and%20size%20shows%20popularity.%22,%20%22edges%22:%20%22Friendships%22,%20%22subtitle%22:%20%22%3Cp%3ECreate%20your%20own%20visualizations%20with%20%3Ca%20href=\%22https://github.com/graphistry/pygraphistry/\%22%3EPyGraphistry%3C/a%3E.%3C/p%3E%22,%20%22title%22:%20%22%3Ch3%3EFacebook%20Friendships%20(Read-only%20Mode)%3C/h3%3E%22}" target="_blank"></iframe><br></a>
    <em>Source data: <a href="http://snap.stanford.edu" target="_blank">SNAP</a></em>
	</td>
  </tr>
</table>
-->

## PyGraphistry is...

- **Fast & gorgeous:** Cluster, filter, and inspect large amounts of data at interactive speed. We layout graphs with a descendant of the gorgeous ForceAtlas2 layout algorithm introduced in Gephi. Our data explorer connects to Graphistry's GPU cluster to layout and render hundreds of thousand of nodes+edges in your browser at unparalleled speeds.

- **Easy to install:** `pip install` the client in your notebook or web app, and then connect to a [free Graphistry Hub account](https://www.graphistry.com/get-started) or [launch your own private GPU server](https://www.graphistry.com/get-started)
   ```python
  # pip install --user graphistry
  import graphistry
  graphistry.register(api=3, username='abc', password='xyz')
  #graphistry.register(..., protocol='http', host='my.site.ngo')
  ```

-  **Notebook friendly:** PyGraphistry plays well with interactive notebooks like [Jupyter](http://ipython.org), [Zeppelin](https://zeppelin.incubator.apache.org/), and [Databricks](http://databricks.com). Process, visualize, and drill into with graphs directly within your notebooks:

    ```python
    graphistry.edges(pd.read_csv('rows.csv'), 'col_a', 'col_b').plot()
    ```

-  **Embeddable:** Drop live views into your web dashboards and apps:
    ```python
    iframe_url = g.plot(render=False)
    print(f'<iframe src="{ iframe_url }"></iframe>')
    ```

- **Great for events, CSVs, and more:** Not sure if your data is graph-friendly? PyGraphistry's `hypergraph` transform helps turn any sample data like CSVs, SQL results, and event data into a graph for pattern analysis:

     ```python
     rows = pandas.read_csv('transactions.csv')[:1000]
     graphistry.hypergraph(rows)['graph'].plot()
     ```

- **Batteries included:** PyGraphistry works out-of-the-box with popular data science and graph analytics libraries. It is also very easy to turn arbitrary data into insightful graphs:

  - [Pandas](http://pandas.pydata.org)

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
    
    
  - GPU [RAPIDS.ai](https://www.rapids.ai)
  
    ```python
    edges = cudf.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst'])
    graphistry.edges(edges, 'src', 'dst').plot()
    ```
  - [Apache Arrow](https://arrow.apache.org/)
  
    ```python
     edges = pa.Table.from_pandas(pd.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst']))
     graphistry.edges(edges, 'src', 'dst').plot()
    ```
    
  - [Neo4j](http://neo4j.com) ([notebook demo](demos/demos_databases_apis/neo4j/official/graphistry_bolt_tutorial_public.ipynb))
  
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

  - [TigerGaph](https://tigergraph.com) ([notebook demo](demos/demos_databases_apis/tigergraph/tigergraph_pygraphistry_bindings.ipynb))

      ```python
      g = graphistry.tigergraph(protocol='https', ...)
      g2 = g.gsql("...", {'edges': '@@eList'})
      g2.plot()
      print('# edges', len(g2._edges))
      ```
      ```python
      g.endpoint('my_fn', {'arg': 'val'}, {'edges': '@@eList'}).plot()      
      ```

  - [IGraph](http://igraph.org)

     ```python
     graph = igraph.read('facebook_combined.txt', format='edgelist', directed=False)
     graphistry.bind(source='src', destination='dst').plot(graph)
     ```

  - [NetworkX](https://networkx.github.io) ([notebook demo](demos/demos_databases_apis/networkx/networkx.ipynb))

     ```python
     graph = networkx.read_edgelist('facebook_combined.txt')
     graphistry.bind(source='src', destination='dst', node='nodeid').plot(graph)
     ```
  - [HyperNetX](https://github.com/pnnl/HyperNetX) ([notebook demo](demos/demos_databases_apis/hypernetx/hypernetx.ipynb))

     ```python
     hg.hypernetx_to_graphistry_nodes(H).plot()
     ```
     ```python
     hg.hypernetx_to_graphistry_bipartite(H.dual()).plot()     
     ```
     
  - [Splunk](https://www.splunk.com) ([notebook demo](demos/demos_databases_apis/splunk/splunk_demo_public.ipynb))
    
     ```python
     df = splunkToPandas("index=netflow bytes > 100000 | head 100000", {})    
     graphistry.edges(df, 'src_ip', 'dest_ip').plot()
     ```

  - [NodeXL](https://www.nodexl.com) ([notebook demo](demos/demos_databases_apis/nodexl/official/nodexl_graphistry.ipynb))

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

  - **Quickly configurable:** Set visual attributes through [quick data bindings](https://hub.graphistry.com/docs/api/2/rest/upload/#createdataset2) and set [all sorts of URL options](https://hub.graphistry.com/docs/api/1/rest/url/):
  
 
  ```python
    g
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
      .settings(url_params={
        'play': 2000,
	    'menu': True, 'info': True,
	    'bg': '%23FFFFFF',
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
  ````
	
### Gallery

<table>
    <tr valign="top">
        <td width="33%" align="center"><a href="http://labs.graphistry.com/graph/graph.html?dataset=Twitter&info=true&play=3000&static=true&contentKey=Twitter_Github_Demo&center=false&menu=true&goLive=false&left=-1.92e+3&right=1.68e+3&top=-1.03e+3&bottom=985&usertag=github&legend={%22title%22:%22%3Ch3%3ECriminal%20Twitter%20Botnet%3C/h3%3E%22,%22subtitle%22:%22%3Cp%3EThe%20botnet%20(right%20layer),%20%20launders%20Twitter%20retweets%20through%20an%20indirection%20layer%20(middle)%20%20in%20order%20to%20build%20social%20rank%20for%20fraudsters%20(left).%20%20Twitter%27s%20core%20targeting%20algorithm%20then%20routes%20the%20artificially%20trending%20tweets%20%20to%20potential%20victims%20in%20the%20precise%20demographic%20of%20FIFA/Madden%20gamers.%3C/p%3E%20%3Cp%3EMany%20victims%20have%20been%20tricked%20into%20revealing%20their%20credit%20cards%20and%20passports.%3C/p%3E%22,%22nodes%22:%22Twitter%20accounts%22,%22edges%22:%22Follow%20relationship%22}" target="_blank">Twitter Botnet<br><img width="266" src="http://i.imgur.com/qm5MCqS.jpg"></a></td>
        <td width="33%" align="center">Edit Wars on Wikipedia<br><a href="http://i.imgur.com/074zFve.png" target="_blank"><img width="266" src="http://i.imgur.com/074zFve.png"></a><em>Source: <a href="http://snap.stanford.edu" target="_blank">SNAP</a></em></td>
        <td width="33%" align="center"><a href="https://labs.graphistry.com/graph/graph.html?dataset=bitC&static=true&play=20000" target="_blank">100,000 Bitcoin Transactions<br><img width="266" height="266" src="http://imgur.com/download/axIkjfd"></a></td>
    </tr>
    <tr valign="top">
        <td width="33%" align="center">Port Scan Attack<br><a href="http://i.imgur.com/vKUDySw.png" target="_blank"><img width="266" src="http://i.imgur.com/vKUDySw.png"></a></td>
        <td width="33%" align="center"><a href="http://labs.graphistry.com/graph/graph.html?dataset=PyGraphistry/M9RL4PQFSF&usertag=github&info=true&static=true&contentKey=Biogrid_Github_Demo&play=3000&center=false&menu=true&goLive=false&left=-2.58e+4&right=4.35e+4&top=-1.72e+4&bottom=2.16e+4&legend={%22title%22:%22%3Ch3%3EBioGRID%20Repository%20of%20Protein%20Interactions%3C/h3%3E%22,%22subtitle%22:%22%3Cp%3EEach%20color%20represents%20an%20organism.%20Humans%20are%20in%20light%20blue.%3C/p%3E%22,%22nodes%22:%22Proteins/Genes%22,%22edges%22:%22Interactions%20reported%20in%20scientific%20publications%22}" target="_blank">Protein Interactions <br><img width="266" src="http://i.imgur.com/nrUHLFz.png" target="_blank"></a><em>Source: <a href="http://thebiogrid.org" target="_blank">BioGRID</a></em></td>
        <td width="33%" align="center"><a href="http://labs.graphistry.com/graph/graph.html?&dataset=PyGraphistry/PC7D53HHS5&info=true&static=true&contentKey=SocioPlt_Github_Demo&play=3000&center=false&menu=true&goLive=false&left=-236&right=265&top=-145&bottom=134&usertag=github&legend=%7B%22nodes%22%3A%20%22%3Cspan%20style%3D%5C%22color%3A%23a6cee3%3B%5C%22%3ELanguages%3C/span%3E%20/%20%3Cspan%20style%3D%5C%22color%3Argb%28106%2C%2061%2C%20154%29%3B%5C%22%3EStatements%3C/span%3E%22%2C%20%22edges%22%3A%20%22Strong%20Correlations%22%2C%20%22subtitle%22%3A%20%22%3Cp%3EFor%20more%20information%2C%20check%20out%20the%20%3Ca%20target%3D%5C%22_blank%5C%22%20href%3D%5C%22https%3A//lmeyerov.github.io/projects/socioplt/viz/index.html%5C%22%3ESocio-PLT%3C/a%3E%20project.%20Make%20your%20own%20visualizations%20with%20%3Ca%20target%3D%5C%22_blank%5C%22%20href%3D%5C%22https%3A//github.com/graphistry/pygraphistry%5C%22%3EPyGraphistry%3C/a%3E.%3C/p%3E%22%2C%20%22title%22%3A%20%22%3Ch3%3ECorrelation%20Between%20Statements%20about%20Programming%20Languages%3C/h3%3E%22%7D" target="_blank">Programming Languages<br><img width="266" src="http://i.imgur.com/0T0EKmD.png"></a><em>Source: <a href="http://lmeyerov.github.io/projects/socioplt/viz/index.html" target="_blank">Socio-PLT project</a></em></td>
    </tr>
</table>

## Install

### Get
You need to install the PyGraphistry client somewhere and connect it to a Graphistry server. 

1. Graphistry server & account:
    * Create a free [Graphistry Hub account](https://www.graphistry.com/get-started) for open data, or [one-click launch your own private AWS/Azure instance](https://www.graphistry.com/get-started)
    * Later, [setup and manage](https://github.com/graphistry/graphistry-cli) your own private enterprise Docker instance ((contact))[https://www.graphistry.com/demo-request]

2. PyGraphistry:
    * `pip install --user graphistry` (or go direct via [RESTful HTTP calls](https://hub.graphistry.com/docs/api/))
        * Use `pip install --user graphistry[all]` for optional dependencies such as Neo4j drivers
    * To use from a notebook environment, run your own [Jupyter](https://jupyter.org/) server ([one-click launch your own private AWS/Azure instance](https://www.graphistry.com/get-started)) or another such as [Google Colab](https://colab.research.google.com)
  

### Configure

#### API Credentials
Provide your API credentials to upload data to your Graphistry GPU server:

```python
import graphistry
#graphistry.register(api=3, username='username', password='your password') # 2.0 API
#graphistry.register(api=3, token='recent_2-0_token') # 2.0 API, warning: must refresh every 1hr

### Deprecated
#graphistry.register(api=1, key='Your key') # 1.0 API; note that 'key' is different from token
```

For the 2.0 API, your username/password are the same as your Graphistry account, and your session expires after 1hr. The temporary JWT token (1hr) can be generated via the REST API using your login credentials, or by visiting your landing page.

Optionally, for convenience, you may set your API key in your system environment and thereby skip the register step in all your notebooks. In your `.profile` or `.bash_profile`, add the following and reload your environment:

```export GRAPHISTRY_API_KEY="Your key"```

#### Server

Specify which Graphistry to reach:

```python
graphistry.register(protocol='https', server='hub.graphistry.com')
```

Private Graphistry notebook environments are preconfigured to fill in this data for you:

```python
graphistry.register(protocol='http', server='nginx', client_protocol_hostname='')
```

#### Client

In cases such as when the notebook server is the same as the Graphistry server, you may want your Python code to  *upload* to a known local Graphistry address (e.g., `nginx` or `localhost`), and generate and embed URLs to a different public address (e.g., `https://graphistry.acme.ngo`). In this case, explicitly set a different client (browser) location:

```
graphistry.register(
    ### fast local notebook<>graphistry upload
    protocol='http', server='nginx', 
  
    ### shareable public URL for browsers
    client_protocol_hostname='https://graphistry.acme.ngo'
)
```

Prebuilt Graphistry servers are already setup to do this out-of-the-box.


## Tutorial: Les Misérables

Let's visualize relationships between the characters in [Les Misérables](http://en.wikipedia.org/wiki/Les_Misérables).
For this example, we'll choose [Pandas](http://pandas.pydata.org) to wrangle data and [IGraph](http://igraph.org) to run a community detection algorithm. You can [view](demos/more_examples/simple/MarvelTutorial.ipynb) the Jupyter notebook containing this example.

Our [dataset is a CSV file](https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/lesmiserables.csv) that looks like this:

| source        | target        | value  |
| ------------- |:-------------:| ------:|
| Cravatte |	Myriel | 1
| Valjean	| Mme.Magloire | 3
| Valjean	| Mlle.Baptistine | 3

*Source* and *target* are character names, and the *value* column counts the number of time they meet. Parsing is a one-liner with Pandas:

```python
import pandas
links = pandas.read_csv('./lesmiserables.csv')
```

### Quick Visualization

If you already have graph-like data, use this step. Otherwise, try the [Hypergraph Transform](demos/demos_by_use_case/logs/malware-hypergraph/Malware%20Hypergraph.ipynb) for creating graphs from rows of data (logs, samples, records, ...).

PyGraphistry can plot graphs directly from Pandas data frames, Arrow tables, cuGraph GPU data frames, IGraph graphs, or NetworkX graphs. Calling *plot* uploads the data to our visualization servers and return an URL to an embeddable webpage containing the visualization.

To define the graph, we <code>bind</code> *source* and *destination* to the columns indicating the start and end nodes of each edges:

```python
import graphistry
graphistry.register(protocol='https', server='hub.graphistry.com', username='YOUR_ACCOUNT_HERE', password='YOUR_PASSWORD_HERE')

g = graphistry.bind(source="source", destination="target")
g.edges(links).plot()
```

You should see a beautiful graph like this one:
![Graph of Miserables](http://i.imgur.com/dRHHTyK.png)

### Adding Labels

Let's add labels to edges in order to show how many times each pair of characters met. We create a new column called *label* in edge table *links* that contains the text of the label and we bind *edge_label* to it.

```python
links["label"] = links.value.map(lambda v: "#Meetings: %d" % v)
g = g.bind(edge_label="label")
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

![Second Graph of Miserables](http://i.imgur.com/P7fm5sn.png)

### Add edge colors and weights

By default, edges get colored as a gradient between their source/destination node colors. You can override this by setting `.bind(edge_color='colA')`, similar to how node colors function. ([See color documentation](https://hub.graphistry.com/docs/api/2/rest/upload/colors/#extendedpalette2).)

Similarly, you can bind the edge weight, where higher weights cause nodes to cluster closer together: `.bind(edge_weight='colA')`. [See tutorial](demos/more_examples/graphistry_features/edge-weights.ipynb).

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

## Next Steps

1. Create a free public data [Graphistry Hub](https://www.graphistry.com/get-started) account or [one-click launch a private Graphistry instance in AWS](https://www.graphistry.com/get-started)
2. Check out the [analyst](demos/for_analysis.ipynb) and [developer](demos/for_developers.ipynb) introductions, or [try your own CSV](demos/upload_csv_miniapp.ipynb)
3. Explore the [demos folder](demos) for your favorite file format, database, API, or kind of analysis

## Resources

* Graphistry [UI Guide](https://hub.graphistry.com/docs/ui/index/)
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
