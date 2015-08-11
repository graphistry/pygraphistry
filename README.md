# PyGraphistry

Visualize and explore large graphs using [Graphistry](http://www.graphistry.com)'s GPU cluster.

#### Fast & Gorgeous
View hundreds of thousand of nodes and edges at interactive speeds in your browser. Drill down, filter, and inspect in real-time.
![Facebook Graph](http://i.imgur.com/k4YWKUE.png)

### Science Ready

Graphistry works out-of-the-box with popular data science libraries. To explore the graph above, use your favorite library to plot [this dataset](https://www.dropbox.com/s/csy1l8e3uv600mj/facebook_combined.txt?dl=1) of Facebook communities from [SNAP](http://snap.stanford.edu).

##### [Pandas](http://pandas.pydata.org)

```python
df = pandas.read_csv('facebook_combined.txt', sep=' ', names=['src', 'dst'])
graphistry.bind(source='src', destination='dst').plot(df)
```
  
##### [Igraph](http://igraph.org)

```python
ig = igraph.read('facebook_combined.txt', format='edgelist', directed=False)
graphistry.bind(source='src', destination='dst').plot(ig)
```
  
##### [NetworkX](https://networkx.github.io)

```python
g = networkx.read_edgelist('facebook_combined.txt')
graphistry.bind(source='src', destination='dst', node='nodeid').plot(g)
```

#### Notebook Friendly
PyGraphistry integrates with [IPython](http://ipython.org): You can process, visualize and explore data directly within your notebooks.

## Installation

You need [Python](https://www.python.org) 2.7 or 3.4. The simplest way is using pip: 

- With Pandas only: `pip install graphistry`
- With Pandas, IGraph, and NetworkX: `pip install "graphistry[all]"`

##### API Key
You need and API key to connect to our GPU cluster. To get one for free, email us at [pygraphistry@graphistry.com](mailto:pygraphistry@graphistry.com) to get your API key. Registering your key is easy:

```python
import graphistry
graphistry.register(key='<Your key>')
```

##### Working IPython (Jupyter) Notebooks

We recommend [IPython](http://ipython.org) notebooks to interleave code and visualizations.

- Install IPython:`pip install "ipython[notebook]"`
- Launch notebook server: `ipython notebook`

You can download our demo example (below) in this [notebook](https://www.dropbox.com/s/n35ahbhatshrau6/MiserablesDemo.ipynb?dl=1).
 
## Tutorial: Les Misérables

For this example, we use [Pandas](http://pandas.pydata.org) to load and process data. Make sure Pandas is installed: `pip install pandas`.

#### Loading data
Let's load the characters from [Les Miserables](http://en.wikipedia.org/wiki/Les_Misérables). Our  [dataset is a CSV file](http://gist.github.com/thibaudh/3da4096c804680f549e6/) that looks like this:

| source        | target        | value  |
| ------------- |:-------------:| ------:|
| Cravatte |	Myriel | 1| Valjean	| Mme.Magloire | 3| Valjean	| Mlle.Baptistine | 3

*Source* and *target* are character names, and the *value* column counts the number of time they meet. Parsing the data is a one-liner with Pandas:

```python
import pandas

links = pandas.read_csv('./lesmiserables.csv')
```

#### Visualize Data
The graphistry package can plot graphs directly from Pandas dataframes. We do specify the name of the two columns indicating the start and end nodes of each edges with *source* and *destination*. 

```python
import graphistry

graphistry = graphistry.settings(key='YOUR_API_KEY_HERE')

graphistry.plot(links, source="source", destination="target")
```

You should see a beautiful graph like this one:
![Graph of Miserables](http://i.imgur.com/lt05Hik.png) Note that the visualization performed in the cloud: you need an Internet connection to see it.

The *plot* function takes a Pandas dataframe of edges, optionally a dataframe of nodes, and bindings between dataframe columns and visual attributes. Calling *plot* uploads the dataframes to our visualization servers and return the url to an embeddable interactive webpage containing the visualization.

### Adding Labels

Let's add label to edges showing how many time each pair of characters met. To to do, we create a new column called *label* in *links* with the text to be displayed.

```python
links['label'] = links.value.map(lambda v: "#Meeting: %d" % v)
```
Then we bind our new column by passing on extra argument to *plot*:

```python
graphistry.plot(links, source="source", destination="target", edge_label="label")
```

### Controling Node Size and Color
We are going to use [igraph](http://igraph.org/python/) to size nodes based on their [PageRank](http://en.wikipedia.org/wiki/PageRank) score and color them using their [community](https://en.wikipedia.org/wiki/Community_structure). Install igraph with `pip install igraph-python`.

We start by converting our edge dateframe to an igraph by indicating the names of the source/destination columns:

```python
import igraph
ig = pandas2igraph(links, 'source', 'target')
```
By computing the PageRank and community cluster, we create two new attributes (*pagerank* & *community*). Both of them are attacked to nodes.

```python
ig.vs['pagerank'] = ig.pagerank()
ig.vs['community'] = ig.community_infomap().membership
```

Finally, we convert our graph back to Pandas. Since we have not only edge attributes but also node attributes, our graph is represented with two Pandas dataframes: one for nodes and one for edges.

```python
(links2, nodes2) = igraph2pandas(ig, 'source', 'target')
graphistry.plot(links2, nodes2, source='source', destination='target', node='__nodeid__',\
                edge_label='label', point_color='community', point_size='pagerank')
```

![Second Graph of Miserables](http://i.imgur.com/sk5URzz.png)


The full code for the two conversion functions `pandas2igraph` and `igraph2pandas` is in the [Misérables demo notebook](https://www.dropbox.com/s/n35ahbhatshrau6/MiserablesDemo.ipynb?dl=1)

## Going Further: Marvel Comics

This is a more complex example: we link together Marvel characters who co-star in the same comic. The dataset is split in three files:

- [appearances.txt](https://www.dropbox.com/s/yz78yy58m1mh8l2/appearances.txt?dl=1)
- [characters.txt](https://www.dropbox.com/s/7zodqsvqa9j29bb/characters.txt?dl=1)
- [comics.txt](https://www.dropbox.com/s/x1o30enl5abdpnm/comics.txt?dl=1)

Run the code in [the Marvel Demo notebook](https://www.dropbox.com/s/mzzq1mvpdwwmes1/MarvelTutorial.ipynb?dl=1) to browse the entire Marvel universe. Find out who is the most popular Marvel hero!

![Marvel Universe](http://i.imgur.com/0rgPLg7.png)


## API Reference

Have a look at the full [API](http://graphistry.com/api0.3.html#python) for more information on sizes, colors, palettes etc.

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




