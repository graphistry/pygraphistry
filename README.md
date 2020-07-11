[![Build Status](https://travis-ci.org/graphistry/pygraphistry.svg?branch=master)](https://travis-ci.org/graphistry/pygraphistry) 
[![Documentation Status](https://readthedocs.org/projects/pygraphistry/badge/?version=latest)](https://pygraphistry.readthedocs.io/en/latest/)
[![Latest Version](https://img.shields.io/pypi/v/graphistry.svg)](https://pypi.python.org/pypi/graphistry) 
[![Latest Version](https://img.shields.io/pypi/pyversions/graphistry.svg)](https://pypi.python.org/pypi/graphistry) 
[![License](https://img.shields.io/pypi/l/graphistry.svg)](https://pypi.python.org/pypi/graphistry)

# PyGraphistry: Explore Relationships

PyGraphistry is a Python visual graph analytics library to extract, transform, and load big graphs into [Graphistry's](http://www.graphistry.com) visual graph analytics platform. It is typically used by data scientists, developers, and operational analysts on problems like visually mapping the behavior of devices and users. 

The Python client makes it easy to go from your existing data to a Graphistry server. Through strong notebook support, data scientists can quickly go from data to accelerated visual explorations, and developers can quickly prototype stunning solutions with their users.

Graphistry supports unusually large graphs for interactive visualization. The client's custom WebGL rendering engine renders up to 8MM nodes and edges at a time, and most older client GPUs smoothly support somewhere between 100K and 1MM elements. The serverside GPU analytics engine supports even bigger graphs.

1. [Interactive Demo](#demo-of-friendship-communities-on-facebook)
2. [Graph Gallery](#gallery)
3. [Installation](#installation)
4. [Tutorial](#tutorial-les-misérables)
5. [API Reference](#api-reference)

### Demo of Friendship Communities on Facebook

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

### PyGraphistry is...

- **Fast & Gorgeous:** Cluster, filter, and inspect large amounts of data at interactive speed. We layout graphs with a descendant of the gorgeous ForceAtlas2 layout algorithm introduced in Gephi. Our data explorer connects to Graphistry's GPU cluster to layout and render hundreds of thousand of nodes+edges in your browser at unparalleled speeds.

-  **Notebook Friendly:** PyGraphistry plays well with interactive notebooks like [Juypter](http://ipython.org), [Zeppelin](https://zeppelin.incubator.apache.org/), and [Databricks](http://databricks.com): Process, visualize, and drill into with graphs directly within your notebooks.

- **Great for Events, CSVs, and more:** Not sure if your data is graph-friendly? PyGraphistry's `hypergraph` transform helps turn any sample data like CSVs, SQL results, and event data into a graph for pattern analysis:

     ```python
     rows = pandas.read_csv('transactions.csv')[:1000]
     graphistry.hypergraph(rows)['graph'].plot()
     ```

- **Batteries Included:** PyGraphistry works out-of-the-box with popular data science and graph analytics libraries. It is also very easy to turn arbitrary data into insightful graphs:

  - [Pandas](http://pandas.pydata.org)

     ```python
     edges = pd.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst'])
     graphistry.bind(source='src', destination='dst').plot(edges)
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

  - [Neo4j](http://neo4j.com) ([notebook demo](demos/demos_databases_apis/neo4j/official/graphistry_bolt_tutorial_public.ipynb))
  
     ```python
     NEO4J_CREDS = {'uri': 'bolt://my.site.ngo:7687', 'auth': ('neo4j', 'mypwd')}
     graphistry.register(bolt=NEO4J_CREDS)
     graphistry.cypher("MATCH (a)-[p:PAYMENT]->(b) WHERE p.USD > 7000 AND p.USD < 10000 RETURN a, p, b").plot()
     ```
     ```python
     graphistry.cypher("CALL db.schema()").plot()
     ```          
     ```python
     from neo4j import GraphDatabase, Driver
     graphistry.register(bolt=GraphDatabase.driver(**NEO4J_CREDS))
     graphistry.cypher("MATCH (a)-[p:PAYMENT]->(b) WHERE p.USD > 7000 AND p.USD < 10000 RETURN a, p, b").plot()
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
     graphistry.bind(source='src_ip', destination='dest_ip').plot(df)
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

## Installation

You need to install the PyGraphistry client somewhere and connect it to a Graphistry server. We recommend the following options:

1. [Private Graphistry Server](https://www.graphistry.com/get-started): One-click launch with Graphistry, PyGraphistry, and Jupyter Notebooks preinstalled and ready to go out-of-the-box (AWS and Azure)
2. `pip install graphistry`: If you already have Jupyter Notebook installed or are using a system like Google Colab, install the PyGraphistry pip package. Connect it to a free [Graphistry Hub](https://www.graphistry.com/get-started) account or [a private Graphistry server](https://www.graphistry.com/get-started).

### Option 1: New users - Private Graphistry AWS/Azure Server with Preinstalled PyGraphistry client

For new users who have AWS accounts, simply [launch the self-serve Graphistry AMI](https://aws.amazon.com/marketplace/pp/B07Q88KH8C?ref=_ptnr_pygraphistry_ste_core). See [GrNaphistr.com](https://www.graphistry.com/get-started) for additional quick launch modes.

A private server provides several benefits for getting started:

* PyGraphistry is preinstalled
* Jupyter Notebook is preinstalled
* Starter examples of using with different files, databases, and Nvidia RAPIDS are provided
* Preconfigured backend server: Nvidia drivers, `nvidia-docker`, Graphistry server, RAPIDS environment, etc.
* Running in your private server means you can safely explore private data there

The server gracefully stops/starts: Control potential server utility fees by simply stopping the server when not using it.


### Option 2: PyGraphistry pip package - With Graphistry Hub or a private server

Install PyGraphistry into your own Python app or data science notebook environment such as [Jupyter](https://jupyter.org/) and [Google Colab](https://colab.research.google.com). Connect it to a Graphistry server such as the self-serve [Graphistry AMI](https://aws.amazon.com/marketplace/pp/B07Q88KH8C?ref=_ptnr_pygraphistry_ste_core) or a free [Graphistry Hub](https://www.graphistry.com/get-started) account.

Install PyGraphistry with Python's `pip` package manager:

- Pandas only (recommended): `pip install --user graphistry`
- + neo4j: `pip install --user "graphistry[bolt]"`
- + IGraph, NetworkX, Neo4j: `pip install --user "graphistry[all]"`

The latter two can be skipped if you already have the third-party Python packages at the appropriate versions installed.

##### Jupyter Notebook Integration

### API Credentials
Provide your API credentials to upload data to your Graphistry GPU server:

```python
import graphistry
#graphistry.register(key='Your key') # 1.0 API
#graphistry.register(api=3, username='your name', password='your pwd') # 2.0 API, logged out after 1hr
#graphistry.register(api=3, token='your JWT token') # 2.0 API, expires after 1hr
```

For the 2.0 API, your username/password are the same as your Graphistry account, and your session expires after 1hr. The temporary JWT token (1hr) can be generated via the REST API using your login credentials, or by visiting your landing page.

Optionally, for convenience, you may set your API key in your system environment and thereby skip the register step in all your notebooks. In your `.profile` or `.bash_profile`, add the following and reload your environment:

```export GRAPHISTRY_API_KEY="Your key"```

### Server

Specify which Graphistry to reach:

```python
graphistry.register(protocol='https', server='hub.graphistry.com')
```

Preconfigure private Graphistry servers to fill in this data for you.

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

If you already have graph-like data, use this step. Otherwise, try the [Hypergraph Transform](demos/demos_by_use_case/logs/malware-hypergraph/Malware%20Hypergraph.ipynb)

PyGraphistry can plot graphs directly from Pandas dataframes, IGraph graphs, or NetworkX graphs. Calling *plot* uploads the data to our visualization servers and return an URL to an embeddable webpage containing the visualization.

To define the graph, we <code>bind</code> *source* and *destination* to the columns indicating the start and end nodes of each edges:

```python
import graphistry
graphistry.register(protocol='https', server='hub.graphistry.com', token='YOUR_JWT_TOKEN_HERE')

g = graphistry.bind(source="source", destination="target")
g.plot(links)
```

You should see a beautiful graph like this one:
![Graph of Miserables](http://i.imgur.com/dRHHTyK.png)

### Adding Labels

Let's add labels to edges in order to show how many times each pair of characters met. We create a new column called *label* in edge table *links* that contains the text of the label and we bind *edge_label* to it.

```python
links["label"] = links.value.map(lambda v: "#Meetings: %d" % v)
g = g.bind(edge_label="label")
g.plot(links)
```

### Controlling Node Title, Size, Color, and Position
Let's size nodes based on their [PageRank](http://en.wikipedia.org/wiki/PageRank) score and color them using their [community](https://en.wikipedia.org/wiki/Community_structure). 


#### Warmup: IGraph for computing statistics
[IGraph](http://igraph.org/python/) already has these algorithms implemented for us. If IGraph is not already installed, fetch it with `pip install python-igraph`. Warning: `pip install igraph` will install the wrong package!

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

See the [color palette documentation](https://hub.graphistry.com/docs/api/2/rest/upload/colors/#extendedpalette2) for specifying color values. 


To control the position, we can add `.bind(point_x='colA', point_y='colB').settings(url_params={'play': 0})` ([see demos](demos/more_examples/graphistry_features/external_layout) and [additional url parameters](https://hub.graphistry.com/docs/api/1/rest/url/#urloptions)]). In `api=1`, you created columns named `x` and `y`. 

You may also want to bind `point_title`: `.bind(point_title='colA')`. 

![Second Graph of Miserables](http://i.imgur.com/P7fm5sn.png)

### Add edge colors and weights

By default, edges get colored as a gradient between their source/destination node colors. You can override this by setting `.bind(edge_color='colA')`, similar to how node colors function. ([See color documentation](https://hub.graphistry.com/docs/api/2/rest/upload/colors/#extendedpalette2).)

Similarly, you can bind the edge weight, where higher weights cause nodes to cluster closer together: `.bind(edge_weight='colA')`. [See tutorial](demos/more_examples/graphistry_features/edge-weights.ipynb).

## Next Steps

1. Create a free public data [Graphistry Hub](https://www.graphistry.com/get-started) account or [one-click launch a private Graphistry instance in AWS](https://www.graphistry.com/get-started)
2. Check out the [analyst](demos/for_analysis.ipynb) and [developer](demos/for_developers.ipynb) introductions, or [try your own CSV](demos/upload_csv_miniapp.ipynb)
3. Explore the [demos folder](demos) for your favorite file format, database, API, or kind of analysis

## References

* Graphistry [UI Guide](https://hub.graphistry.com/docs/ui/index/)
* [API docs](https://hub.graphistry.com/docs/api/): Bindings and colors, REST API, embedding URLs and URL parameters, dynamic JS API, and more
* [Python API ReadTheDocs](http://pygraphistry.readthedocs.org/en/latest/)
* Within a notebook, you can always run `help(graphistry)`, `help(graphistry.hypergraph)`, etc.
* Additional [Graphistry API docs](https://hub.graphistry.com/docs/), including the predefined [color palette values](https://hub.graphistry.com/docs/api/api-color-palettes/) (color brewer)

