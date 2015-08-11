# PyGraphistry

*Visually* understand and analyze relationships in large graphs using [Graphistry's](http://www.graphistry.com) browser-based data explorer. Load and process data in Python using familiar tools with PyGraphistry bindings. Here is an exported (read-only) visualization created with PyGraphistry.

<table style="width:100%;">
  <tr valign="top">
    <td>Friendship communities on Facebook. Read-only export. Dataset from <a href="http://snap.stanford.edu">SNAP</a>.<br/><iframe height="600" style="width:100%;" src="http://proxy-staging.graphistry.com/graph/graph.html?dataset=Facebook&debug=true&info=true&play=0&mapper=opentsdb&menu=false&static=true&contentKey=Facebook_readme&center=false&left=-28057.922443107804&right=19343.789165388305&top=-13990.35481117573&bottom=12682.885549380659#"></iframe></td>
  </tr>
</table>

- **Fast & Gorgeous** Our data explorer connects to Graphistry's GPU cluster show hundreds of thousand of nodes and edges in your browser. You can cluster, filter, and inspect large amounts of data at interactive speed.

-  **Notebook Friendly** PyGraphistry plays well with [IPython](http://ipython.org): You can process and visualize data directly within your notebooks.

- **Science Ready** PyGraphistry works out-of-the-box with popular data science libraries. It is also very easy to use. To create the visualization shown above, download  [this dataset](https://www.dropbox.com/s/csy1l8e3uv600mj/facebook_combined.txt?dl=1) of Facebook communities from [SNAP](http://snap.stanford.edu) and use your favorite library to load it.

  - **[Pandas](http://pandas.pydata.org)**

     ```python
     df = pandas.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst'])
     graphistry.bind(source='src', destination='dst').plot(df)
     ```

  - **[Igraph](http://igraph.org)**

     ```python
     ig = igraph.read('facebook_combined.txt', format='edgelist', directed=False)
     graphistry.bind(source='src', destination='dst').plot(ig)
     ```

  - **[NetworkX](https://networkx.github.io)**

     ```python
     g = networkx.read_edgelist('facebook_combined.txt')
     graphistry.bind(source='src', destination='dst', node='nodeid').plot(g)
     ```

### Gallery

<table>
    <tr valign="top">
        <td width="50%">Twitter Botnet<br><a href="http://TODO"><img width="400" src="http://i.imgur.com/qm5MCqS.jpg"></a></td>
        <td width="50%">Edit Wars on Wikipedia<br><a href="http://TODO"><img width="400" src="http://i.imgur.com/074zFve.png"></a></td>
    </tr>
    <tr valign="top">
        <td width="50%">Attackers Port Scanning a Network<br><a href="http://TODO"><img width="400" src="http://i.imgur.com/vKUDySw.png"></a></td>
        <td width="50%">Interactions between Proteins (Biogrid)<br><a href="http://TODO"><img width="400" src="http://i.imgur.com/nrUHLFz.png"></a></td>
    </tr>
</table>

## Installation

You need [Python](https://www.python.org) 2.7 or 3.4. The simplest way is using pip:

- With Pandas only: `pip install graphistry`
- With Pandas, IGraph, and NetworkX: `pip install "graphistry[all]"`

##### API Key
You need and API key to connect to our GPU cluster. We ask for API keys to make use our servers can handle the load. To get your own, email us at [pygraphistry@graphistry.com](mailto:pygraphistry@graphistry.com). Register you key after the `import graphistry` statement and you are good to go:

```python
import graphistry
graphistry.register(key='<Your key>')
```

##### Working IPython (Jupyter) Notebooks

We recommend [IPython](http://ipython.org) notebooks to interleave code and visualizations.

- Install IPython:`pip install "ipython[notebook]"`
- Launch notebook server: `ipython notebook`

## Tutorial: Les Misérables

For this example, we use [Pandas](http://pandas.pydata.org) to load/process data and [Igraph](http://igraph.org) to run a community detection algorithm. Download the [IPython notebook](https://www.dropbox.com/s/n35ahbhatshrau6/MiserablesDemo.ipynb?dl=1) containing this example.

#### Loading the Dataset
Let's load the characters from [Les Miserables](http://en.wikipedia.org/wiki/Les_Misérables). Our  [dataset is a CSV file](http://gist.github.com/thibaudh/3da4096c804680f549e6/) that looks like this:

| source        | target        | value  |
| ------------- |:-------------:| ------:|
| Cravatte |	Myriel | 1| Valjean	| Mme.Magloire | 3| Valjean	| Mlle.Baptistine | 3

*Source* and *target* are character names, and the *value* column counts the number of time they meet. Parsing the data is a one-liner with Pandas:

```python
import pandas
links = pandas.read_csv('./lesmiserables.csv')
```

#### Quick Visualization
The graphistry package can plot graphs directly from Pandas dataframes. We specify the name of the two columns indicating the start and end nodes of each edges by binding *source* and *destination*. The *plot* function takes a Pandas dataframe of edges, optionally a dataframe of nodes, and bindings between dataframe columns and visual attributes. Calling *plot* uploads the dataframes to our visualization servers and return the url to an embeddable interactive webpage containing the visualization.

```python
import graphistry
graphistry.register(key='YOUR_API_KEY_HERE')

plotter = graphistry.bind(source="source", destination="target")
plotter.plot(links)
```

You should see a beautiful graph like this one:
![Graph of Miserables](http://i.imgur.com/dRHHTyK.png) Since the visualization is performed on Graphistry's GPU cluster, you need an internet connection to see it.

### Adding Labels

Let's add label to edges showing how many time each pair of characters met. To to do, we create a new column called *label* in *links* containing the text of the label. Finally, create a new plotter binding the new column.

```python
links["label"] = links.value.map(lambda v: "#Meetings: %d" % v)
plotter = plotter.bind(edge_label="label")
plotter.plot(links)
```

### Controling Node Size and Color
We are going to use [igraph](http://igraph.org/python/) to size nodes based on their [PageRank](http://en.wikipedia.org/wiki/PageRank) score and color them using their [community](https://en.wikipedia.org/wiki/Community_structure). If Igraph is not already installed, fetch it with `pip install igraph-python`. (Warning: `pip install igraph` will install the wrong package!)

We start by converting our edge dateframe to an igraph. The plotter can do the conversion for us using the *source* and *destination* bindings. By computing PageRank and community clusters, we create two new attributes (*pagerank* & *community*). Both of them are attacked to nodes.

```python
ig = plotter.pandas2igraph(links)
ig.vs['pagerank'] = ig.pagerank()
ig.vs['community'] = ig.community_infomap().membership
```

Finally we bind our two new columns and plot the IGraph directly:

```python
plotter.bind(point_color='community', point_size='pagerank').plot(ig)
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


