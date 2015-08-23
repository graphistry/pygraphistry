# PyGraphistry: Explore Relationships

PyGraphistry is a visual graph analytics library to extract, transform, and load big graphs into [Graphistry's](http://www.graphistry.com) GPU-cloud-accelerated explorer.

### Launch The Demo:

<table style="width:100%;">
  <tr valign="top">
    <td align="center"><b>Friendship Communities on Facebook.</b> Click to open interactive version! <a href="http://proxy-labs.graphistry.com/graph/graph.html?dataset=Facebook&info=true&menu=false&play=0&mapper=opentsdb&legend={%22title%22:%22%3Ch3%3EFacebook%20Friendships%20(Read-only%20Mode)%3C/h3%3E%22,%22subtitle%22:%22%3Cp%3ECreate%20your%20own%20visualizations%20with%20%3Ca%20href=\%22https://github.com/graphistry/pygraphistry/\%22%3EPyGraphistry%3C/a%3E.%3C/p%3E%22,%22nodes%22:%22People.%20Color%20indicates%20community%20and%20size%20shows%20popularity.%22,%22edges%22:%22Friendships%22}&static=true&contentKey=Facebook_Static&center=false&left=-1.51e+4&right=1.41e+4&top=-8.81e+3&bottom=7.55e+3"><img src="http://i.imgur.com/CvO12an.png" title="Click to open."></a>
    <em>Source data: <a href="http://snap.stanford.edu">SNAP</a></em>
	</td>
  </tr>
</table>

<!-- IFRAME VESION
### The Demo:

<table style="width:100%;">
  <tr valign="top">
    <td align="center"><b>Friendship Communities on Facebook.</b> (Read-only interactive version.)<br><iframe width="100%" height="500" src="http://proxy-labs.graphistry.com/graph/graph.html?dataset=Facebook&info=true&menu=false&play=0&mapper=opentsdb&legend={%22title%22:%22%3Ch3%3EFacebook%20Friendships%20(Read-only%20Mode)%3C/h3%3E%22,%22subtitle%22:%22%3Cp%3ECreate%20your%20own%20visualizations%20with%20%3Ca%20href=\%22https://github.com/graphistry/pygraphistry/\%22%3EPyGraphistry%3C/a%3E.%3C/p%3E%22,%22nodes%22:%22People.%20Color%20indicates%20community%20and%20size%20shows%20popularity.%22,%22edges%22:%22Friendships%22}&static=true&contentKey=Facebook_Static&center=false&left=-1.51e+4&right=1.41e+4&top=-8.81e+3&bottom=7.55e+3"></iframe><br></a>
    <em>Source data: <a href="http://snap.stanford.edu">SNAP</a></em>
	</td>
  </tr>
</table>
-->
	
### PyGraphistry is...

- **Fast & Gorgeous:** Cluster, filter, and inspect large amounts of data at interactive speed. We layout graphs with a descendant of the gorgeous ForceAtlas2 layout algorithm introduced in Gephi. Our data explorer connects to Graphistry's GPU cluster to layout and render hundreds of thousand of nodes+edges in your browser at unparalleled speeds.

-  **Notebook Friendly:** PyGraphistry plays well with interactive notebooks like [IPython/Juypter](http://ipython.org), [Zeppelin](https://zeppelin.incubator.apache.org/), and [Databricks](http://databricks.com): Process, visualize, and drill into with graphs directly within your notebooks.

- **Batteries Included:** PyGraphistry works out-of-the-box with popular data science and graph analytics libraries. It is also very easy to use. To create the visualization shown above, download  [this dataset](https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/facebook_combined.txt) of Facebook communities from [SNAP](http://snap.stanford.edu) and load it with your favorite library:

  - [Pandas](http://pandas.pydata.org)

     ```python
     edges = pandas.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst'])
     graphistry.bind(source='src', destination='dst').plot(edges)
     ```

  - [IGraph](http://igraph.org)

     ```python
     graph = igraph.read('facebook_combined.txt', format='edgelist', directed=False)
     graphistry.bind(source='src', destination='dst').plot(graph)
     ```

  - [NetworkX](https://networkx.github.io)

     ```python
     graph = networkx.read_edgelist('facebook_combined.txt')
     graphistry.bind(source='src', destination='dst', node='nodeid').plot(graph)
     ```

### Gallery

<table>
    <tr valign="top">
        <td width="33%" align="center"><a href="http://i.imgur.com/qm5MCqS.jpg">Twitter Botnet<br><img width="266" src="http://i.imgur.com/qm5MCqS.jpg"></a></td>
        <td width="33%" align="center"><a href="http://i.imgur.com/074zFve.png">Edit Wars on Wikipedia<br><img width="266" src="http://i.imgur.com/074zFve.png"></a><em>Source: <a href="http://snap.stanford.edu">SNAP</a></em></td>
        <td width="33%" align="center"><a href="http://i.imgur.com/GdT4yV6.jpg">Uber Trips in SF<br><img width="266" src="http://i.imgur.com/GdT4yV6.jpg"></a></td>
    </tr>
    <tr valign="top">
        <td width="33%" align="center"><a href="http://i.imgur.com/vKUDySw.png">Port Scan Attack<br><img width="266" src="http://i.imgur.com/vKUDySw.png"></a></td>
        <td width="33%" align="center"><a href="http://i.imgur.com/nrUHLFz.png">Protein Interactions <br><img width="266" src="http://i.imgur.com/nrUHLFz.png"></a><em>Source: <a href="http://thebiogrid.org">BioGRID</a></em></td>
        <td width="33%" align="center"><a href="http://i.imgur.com/0T0EKmD.png">Programming Languages<br><img width="266" src="http://i.imgur.com/0T0EKmD.png"></a><em>Source: <a href="http://lmeyerov.github.io/projects/socioplt/viz/index.html">Socio-PLT project</a></em></td>
    </tr>
</table>

## Installation

### Dependencies
[Python](https://www.python.org) 2.7 or 3.4. 

The simplest way to install PyGraphistry is with Python's pip package manager:

- Pandas only: `pip install graphistry`
- Pandas, IGraph, and NetworkX: `pip install "graphistry[all]"`

##### API Key
An API key gives each visualization access to our GPU cluster. We currently ask for API keys to make sure our servers are not melting :) To get your own, email [pygraphistry@graphistry.com](mailto:pygraphistry@graphistry.com). Set your key after the `import graphistry` statement and you are good to go:

```python
import graphistry
graphistry.register(key='Your key')
```

##### IPython (Jupyter) Notebook Integration

We recommend [IPython](http://ipython.org) notebooks to interleave code and visualizations.

- Install IPython:`pip install "ipython[notebook]"`
- Launch notebook server: `ipython notebook`

## Tutorial: Les Misérables

Let's visualize relationships between the characters in [Les Misérables](http://en.wikipedia.org/wiki/Les_Misérables).
For this example, we'll choose [Pandas](http://pandas.pydata.org) to wrangle data and [IGraph](http://igraph.org) to run a community detection algorithm. You can download the [IPython notebook](https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/MiserablesDemo.ipynb) containing this example.

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
PyGraphistry can plot graphs directly from Pandas dataframes, IGraph graphs, or NetworkX graphs. Calling *plot* uploads the data to our visualization servers and return an URL to an embeddable webpage containing the visualization.

To define the graph, we <code>bind</code> *source* and *destination* to the columns indicating the start and end nodes of each edges:

```python
import graphistry
graphistry.register(key='YOUR_API_KEY_HERE')

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

### Controlling Node Size and Color
Let's size nodes based on their [PageRank](http://en.wikipedia.org/wiki/PageRank) score and color them using their [community](https://en.wikipedia.org/wiki/Community_structure). [IGraph](http://igraph.org/python/) already has these algorithms implemented for us. If IGraph is not already installed, fetch it with `pip install igraph-python`. (<span style="color:maroon">Warning: `pip install igraph` will install the wrong package!</span>)

We start by converting our edge dateframe into an IGraph. The plotter can do the conversion for us using the *source* and *destination* bindings. Then we create two new node attributes (*pagerank* & *community*).

```python
ig = g.pandas2igraph(links)
ig.vs['pagerank'] = ig.pagerank()
ig.vs['community'] = ig.community_infomap().membership

g.bind(point_color='community', point_size='pagerank').plot(ig)
```

![Second Graph of Miserables](http://i.imgur.com/P7fm5sn.png)

## Going Further: Marvel Comics

This is a more complex example: we link together Marvel characters who co-star in the same comic. The dataset is split in three files:

- [appearances.txt](https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/appearances.txt)
- [characters.txt](https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/characters.txt)
- [comics.txt](https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/comics.txt)

Find out who is the most popular Marvel hero! Run the code in [the Marvel Demo notebook](https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/MarvelTutorial.ipynb) to browse the entire Marvel universe.

![Marvel Universe](http://i.imgur.com/0rgPLg7.png)

## API Reference

Full Python [API](api.md) for more information on sizes, colors, palettes etc.

See also: [REST API and deprecrated Python docs](http://graphistry.com/api/api0.9.2.html#python).

## Next Step

Email [pygraphistry@graphistry.com](mailto:pygraphistry@graphistry.com) for an API key!

