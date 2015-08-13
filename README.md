# PyGraphistry: Explore Relationships

PyGraphistry is a visual graph analytics library to extract, transform, and load big graphs into [Graphistry's](http://www.graphistry.com) GPU-cloud-accelerated explorer.

### Launch The Demo:

<table style="width:100%;">
  <tr valign="top">
    <td align="center"><b>Friendship Communities on Facebook.</b> Click to open interactive version! <a href="http://proxy-staging.graphistry.com/graph/graph.html?dataset=Facebook&info=false&play=0&mapper=opentsdb&menu=false&static=true&contentKey=Facebook_readme&center=false&left=-28057.922443107804&right=19343.789165388305&top=-13990.35481117573&bottom=12682.885549380659#"><img src="http://i.imgur.com/CvO12an.png" title="Click to open."></a>
    <em>Source data: <a href="http://snap.stanford.edu">SNAP</a></em>
	</td>
  </tr>
</table>

### PyGraphistry is...

- **Fast & Gorgeous:** Cluster, filter, and inspect large amounts of data at interactive speed. We layout graphs with a descendant of the gorgeous ForceAtlas2 layout algorithm introduced in Gephi. Our data explorer connects to Graphistry's GPU cluster to layout and render hundreds of thousand of nodes+edges in your browser at unparalleled speeds.

-  **Notebook Friendly:** PyGraphistry plays well with interactive notebooks like [IPython/Juypter](http://ipython.org), [Zeppelin](https://zeppelin.incubator.apache.org/), and [Databricks](http://databricks.com): Process, visualize, and drill into with graphs directly within your notebooks.

- **Batteries Included:** PyGraphistry works out-of-the-box with popular data science and graph analytics libraries. It is also very easy to use. To create the visualization shown above, download  [this dataset](https://www.dropbox.com/s/csy1l8e3uv600mj/facebook_combined.txt?dl=1) of Facebook communities from [SNAP](http://snap.stanford.edu) and load it with your favorite library:

  - [Pandas](http://pandas.pydata.org)

     ```python
     g = pandas.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst'])
     graphistry.bind(source='src', destination='dst').plot(g)
     ```

  - [IGraph](http://igraph.org)

     ```python
     g = igraph.read('facebook_combined.txt', format='edgelist', directed=False)
     graphistry.bind(source='src', destination='dst').plot(g)
     ```

  - [NetworkX](https://networkx.github.io)

     ```python
     g = networkx.read_edgelist('facebook_combined.txt')
     graphistry.bind(source='src', destination='dst', node='nodeid').plot(g)
     ```

### Gallery

<table>
    <tr valign="top">
        <td width="33%" align="center"><a href="http://i.imgur.com/qm5MCqS.jpg">Twitter Botnet<br><img width="266" src="http://i.imgur.com/qm5MCqS.jpg"></a></td>
        <td width="33%" align="center"><a href="http://i.imgur.com/074zFve.png">Edit Wars on Wikipedia<br><img width="266" src="http://i.imgur.com/074zFve.png"></a></td>
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
An API key gives each visualization access to our GPU cluster. We currently ask for API keys to make sure our servers are not melting :) To get your own, email [pygraphistry@graphistry.com](mailto:pygraphistry@graphistry.com). Register you key after the `import graphistry` statement and you are good to go:

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
For this example, we'll choose [Pandas](http://pandas.pydata.org) to wrangle data and [Igraph](http://igraph.org) to run a community detection algorithm. You can download the [IPython notebook](https://www.dropbox.com/s/n35ahbhatshrau6/MiserablesDemo.ipynb?dl=1) containing this example.

Our [dataset is a CSV file](http://gist.github.com/thibaudh/3da4096c804680f549e6/) that looks like this:

| source        | target        | value  |
| ------------- |:-------------:| ------:|
| Cravatte |	Myriel | 1| Valjean	| Mme.Magloire | 3| Valjean	| Mlle.Baptistine | 3

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

- [appearances.txt](https://www.dropbox.com/s/yz78yy58m1mh8l2/appearances.txt?dl=1)
- [characters.txt](https://www.dropbox.com/s/7zodqsvqa9j29bb/characters.txt?dl=1)
- [comics.txt](https://www.dropbox.com/s/x1o30enl5abdpnm/comics.txt?dl=1)

Find out who is the most popular Marvel hero! Run the code in [the Marvel Demo notebook](https://www.dropbox.com/s/mzzq1mvpdwwmes1/MarvelTutorial.ipynb?dl=1) to browse the entire Marvel universe.

![Marvel Universe](http://i.imgur.com/0rgPLg7.png)

## API Reference

Have a look at the full [API](http://graphistry.com/api0.3.html#python) for more information on sizes, colors, palettes etc.

<!---

### Cheat Sheet
In a nutshell, `plot` *mandatory* arguments are:

- `edges` *pandas.DataFrame*: The edge dataframe.
- `source` *string*: The column of `edges` containing the start of each edge.
- `destination` *string*: The column of `edges` containing the end of each edge.

This is enough to define a graph.
##### Edges
We control the visual attributes of edges with the following *optional* arguments. Each of them refers to the name of a column of `edges`.

- `edge_color` *string*
- `edge_title` *string*
- `edge_label` *string*
- `edge_weight` *string*

##### Nodes
To control node visual attributes, we pass two more arguments:

- `nodes` *pandas.DataFrame*: The node dataframe.
- `node` *string*: The column of `nodes` that contains node identifiers (these are the same ids used in the `source` and `destination` columns of `edges`).

then we bind columns of `node` using:

- `point_title` *string*
- `point_label` *string*
- `point_size` *string*
- `point_color` *string*

-->


