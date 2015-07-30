# PyGraphistry in a Nutshell

### Installation

The simplest way is to use pip:

```console
$ pip install graphistry
```

*Note*: we only support Python 2.7 for now.

#### API Key

Email us at [pygraphistry@graphistry.com](mailto:pygraphistry@graphistry.com) to get your API key.

#### Working IPython (Jupyter) Notebooks

We recommend coding in [IPython](http://ipython.org) notebooks. That way, you have your code and visualizations all in one place. You can download our demo example (below) in this [notebook](https://www.dropbox.com/s/n35ahbhatshrau6/MiserablesDemo.ipynb?dl=1).
 
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


## API Reference

Have a look at the full [API](http://graphistry.com/api0.3.html#python) for more information on sizes, colors, palettes etc.

#### Cheat Sheet
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




