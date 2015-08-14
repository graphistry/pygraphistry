# Graphistry Package v0.9.2

# Class PyGraphistry

Top-level import. Used to connect to the Graphistry server and then create a base plotter.

## `PyGraphistry.register`

### Signature ###
register(key : <em>string</em>, server = 'proxy-labs.graphistry.com' : <em>string</em> ) : None 

### About ###
Provide the API key for Graphistry's GPU cloud, and optionally, override which server to use.

Changing the key effects all derived `Plotter` instances.

### Examples
**Standard**

```Python
import graphistry
graphistry.register(key="my api key")
```

**Developer**

```Python
import graphistry
graphistry.register('my api key', 'staging')
```

## `PyGraphistry.bind`

### Signature ###
bind() : Plotter

### About
Returns a base plotter. Typically called immediately.

### Examples

```Python
import graphistry
g = graphistry.bind()
```

# Class Plotter

Graph plotting class. Chained calls successively add data and visual encodings, and end with a `plot` call. 

To streamline reuse and replayable notebooks, `Plotter` manipulations are immutable. Each chained call returns a new instance that derives from the previous one. The old plotter or the new one can then be used to create different graphs.

The class supports convenience methods for mixing calls across Pandas, NetworkX, and IGraph.

## `Plotter.bind`

### Signature ###

bind (...) : Plotter

Arguments:

* source = None : <em>String</em>

	Attribute containing an edge's source ID
	
* destination = None : <em>String</em>

 	Attribute containing an edge's destination ID

* node = None : <em>String</em>

	Attribute containing a node's ID
	
* point\_title = None : <em>HtmlString</em>

	Attribute overriding node's minimized label text. By default, the node ID is used. 

* point\_label = None : <em>HtmlString</em>

	Attribute overriding node's expanded label text. By default, scrollable list of attribute/value mappings.

* point_size = None : <em>Integer</em>

	Attribute overriding node's size. By default, uses the node degree. The visualization will normalize point sizes and adjust dynamically using semantic zoom.

* point_color = None : <em>Integer</em>

	Attribute overriding node's color. See [palette definitions](graphistry.com/palette.html) for values. Based on Color Brewer.

* edge_weight = None : <em>Integer</em>

	Attribute overriding edge weight. Default is 1. Advanced layout controls will relayout edges based on this value.
	
* edge\_title = None : <em>HtmlString</em>

	Attribute overriding edge's minimized label text. By default, the edge source and destination is used.

* edge\_label = None : <em>HtmlString</em>

	Attribute overriding edge's expanded label text. By default, scrollable list of attribute/value mappings.
	
* edge_color = None : <em>Integer</em>

	Attribute overriding edge's color. See [palette definitions](graphistry.com/palette.html) for values. Based on Color Brewer.
	

### About ###

Defines encodings that relate data attributes to graph structure and visual representation. 

To facilitate reuse and replayable notebooks, the binding call is chainable. Invocation does not effect the old binding: it instead returns a new `Plotter` instance with the new bindings added to the existing ones. Both the old and new bindings can then be used for different graphs.

### Examples ###

**Minimal**

```Python
import graphistry
g = graphistry.bind()
g.bind(source='src', destination='dst')
```

**Node colors**

```Python
import graphistry
g = graphistry.bind()
g.bind(source='src', destination='dst', node='id', point_color='color')
```

**Chaining**


```Python
import graphistry
g = graphistry.bind().bind(source='src', destination='dst', node='id')

g1 = g.bind(point_color='color1', point_size='size1')
g.bind(point_color='color1b')

g2a = g1.bind(point_color='color2a')
g2b = g1.bind(point_color='color2b', point_size='size2b')

g3a = g2a.bind(point_size='size3a')
g3b = g3b.bind(point_size='size3b')
```

All bindings use src/dst/id. Colors and sizes bind to:

* g: default/default
* g1: color1/size1
* g2a: color2a/size1
* g2b: color2b/size2b
* g3a: color2a/size3a
* g3b: color2b/size3b 

## `Plotter.edges`

### Signature

edges : (edges : <em>Graph</em>) : Plotter

Type <em>Graph</em> may be a pandas dataframe, networkx graph, or igraph graph.

### About

Specify edge list data and associated edge attribute values.

### Example

```Python
import graphistry
df = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
g = graphistry.bind().bind(source='src', destination='dst').edges(df)
g.plot()
```	

## `Plotter.nodes`

### Signature

nodes : (nodes : <em>pandasDataFrame</em>) : Plotter

### About

Specify the set of nodes and associated data. Must be a superset of the nodes referenced in the edge list.

### Example

```Python
import graphistry

es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
g = graphistry.bind().bind(source='src', destination='dst').edges(es)

vs = pandas.DataFrame({'v': [0,1,2], 'lbl': ['a', 'b', 'c']})
g = g.bind(node='v').nodes(vs)

g.plot()
```	

## `Plotter.graph`

### Signature

graph : (graph : <em>Graph</em>) : Plotter

### About

Specify the nodes and edges as a NetworkX graph or an IGraph graph.

### Example

TODO

## `Plotter.plot`

### Signature

plot : (graph = None, nodes = None)

Optional attributes follow same rules as the corresponding methods `nodes()`, `edges()`, and `graph()`. When provided, they override the existing binding.

### About

Upload bound data to the Graphistry server and show an iframe of the graph based around the currently bound visual encodings.

Outputs different results based on the embedding environment. E.g., in IPython, shows the iframe.


### Example

**Simple**

```Python
import graphistry
es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
g = graphistry.bind().bind(source='src', destination='dst').edges(es)
g.plot()
```

**Sugar**

```Python
import graphistry
es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
graphistry.bind().bind(source='src', destination='dst').plot(es)
```



## `Plotter.pandas2igraph`

TODO

## `Plotter.igraph2pandas`

TODO

## `Plotter.networkx2pandas`

TODO