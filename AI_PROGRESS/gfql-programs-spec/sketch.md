# RFC: GFQL Programs

With groups like JPMC and usage of GFQL from Louie, we’re getting into scenarios where a single Chain isn’t enough, and we do not want to require Python

This RFC looks at:

* Extending wire protocol & Python API with a DAG of GFQL expressions, similar to how you can have nested SQL statements  
* Extending for the most common scenarios where you would use these:  
  * Loading remote graphs  
  * Graph combinators like union/intersection/subtraction  
  * PyGraphistry’s rich library of Plottable-\>Plottable methods like UMAP
  
  # Core: DAGs

Instead of current pure GFQL programs being a single chain, enable a DAG composition of operations where we may have multiple graphs/subgraphs/…

## Wire Protocol

### QueryDAG

Introduce:

* “type”: “QueryDAG” : Nested composition, where some final item is the output  
  * Similar to “let\*” in FP , and SQL having multiple selects  
  * field “graph” defines a sequence of bindings mapping names to Chain/QueryDAG expressions  
* “output”: “b”: Pick which binding a QueryDAG returns, defaults to last  
* “ref”: “abc” : Both Chain and QueryDAG may specify which named item they operate on 
  * valid values for graph bindingnames: ^[a-zA-Z_][a-zA-Z0-9_-]*$ 
  * Defaults to whatever we’re dotted on  
* Similar to lexical scoping, where closest binding to a reference is the used one, and statically resolvable for a closed GFQL expression

{  
  "type": "QueryDAG",  
  "graph": {  
    "a": {“type”: “Chain”, …},  
    "a2": {“type”: “QueryDAG”, …},  
    "b": {  
      "type": "Chain",  
      "ref": "a",  
      "chain": \[ EdgeOp \]  
    }  
  },  
  "output": "b"  // optional  
}

### Dotted Refs

When the QueryDAG is simple, we just have some top-level names. But when we expect collisions, we can use dotted references to avoid. Imagine we like to have graphs like “start” and “hops”, but want disambiguated.

To disambiguate, refs can do “a.b.c” syntax, so if “b” or “c” are reused on different subgraph names, they’re disambiguated by root “a” . This gets used for “output” and “ref”.

{  
  "type": "QueryDAG",  
  "graph": {

    "alerts": {  
      "type": "QueryDAG",  
      "graph": {  
        "start": \[ ….

    "fraud": {  
      "type": "QueryDAG",  
      "graph": {  
        "start": \[ …

  "output": "alerts.start"  
}

## Python API

### New Chain() parameters

* Optional “ref” to re-root

Chain(ref="entry", chain=\[e\_forward()\])  // unbound

QueryDAG({  
  “entry”: …,  
   “out”: Chain(ref="entry", chain=\[e\_forward()\])  // bound  
})

### New QueryDAG (called ChainGraph)

from gfql.ast import ChainGraph

query \= ChainGraph({  
    "start": \[...\],  
    "next": Chain(ref="start", chain=\[...\])  
}, …)

### New Dotted Methods

# Use 1: Remote graphs

Part of the reason we may have query DAGs is because we may want to mash up graphs. A common case is working from another graph, like a remote one

Using the same GFQL user context, a pure GFQL expression can load and query a graph saved in Graphistry

### Before: Python

graphistry.bind(dataset\_id=”abc123”).chain(...)

### New: GFQL Python

from gfql.ast import ChainGraph, RemoteGraph, Chain, n, e\_forward

ChainGraph(  
    graph={  
        "abc123": RemoteGraph(dataset\_id="abc123"),  
        "b": Chain(ref="abc123", chain=\[e\_forward({"type": "tx"})\])  
    },  
    output="b"  
)

### After: Wire Protocol

{  
  "type": "ChainGraph",  
  "graph": {  
    "abc123": {  
          "type": "RemoteGraph",  
          "graph\_id": "abc123"  
     },  
    "tx": {  
      "type": "Chain",  
      "ref": "abc123",  
      "chain": \[  
        {"type": "Node", "filter\_dict": {"risk": {"type": "GT", "val": 0.9}}}  
      \]  
    }  
  },  
  "output": "tx"  
}


## Use 2: Graph combinators

TBD \- a common flow motivating multiple graphs is tasks like enrichment, intersection, etc

Target operators:

* Union  
  * policies: left, right, merge\_left, merge\_right  
* Subtract  
* Replace  
  * Policies: full, patch, extend  
* Intersect  
* From: New graph using node/edge table from diff graphs (or none)

Common policies:

* Node removal: drop\_edges, keep\_edges  
* Edge removal: drop\_all\_isolated, drop\_newly\_isolated, keep\_nodes  
* Drop\_dangling

### Wire Protocol

{  
  "type": "GraphCombinator",  
  "combinator": "union" | …,   
  "graphs": \["g1", "g2"\],  // dot-ref or graph names  
  "policy": { ... }        // combinator-specific  
  …  
}

### Python

1. Matching wire protocol:

ChainGraph({  
    "lhs": Chain(\[...\]),  
    "rhs": RemoteGraph("static-base"),  
    "union": GraphUnion("lhs", "rhs")  
}, output="union")

2. Auto-desugaring

GraphUnion(  
    Chain(\[n({"id": "A"}), e({"type": "friend"})\]),  
    RemoteGraph("static-base")  
)

When receives an expr instead of a ref, will make the QueryDAG

### Note: Relationship with Call

These may go away wrt Wire Protocol if we just treat them as Call.

However, they’re not in our core Plottable interface today as we use manual Python/Pandas for these, so they’d still need to be added to the Plottable interface


# Use 3: Call

Call exposes PyGraphistry’s Plottable interface’s many methods that return new Plottables:

* umap()  
* layout\_cugraph(), compute\_cugraph(), …  
* cypher()  
* etc

### Wire Protocol

{  
  "type": "call",  
  "function": "umap",  
  "ref": "previous\_graph",  
  "params": {  
    "x\_cols": \["age", "income"\],  
    "n\_neighbors": 10  
  }  
}

### Python API

from gfql.ast import call

call("umap", ref="input", x\_cols=\["age", "income"\], n\_neighbors=10)

### Safelisting

We may want to do some sort of safelisting controls based on Hub Tier, or user-defined

### Future: Louie Connectors

In future versions, we might consider allowing Louie connectors enabled for the calling user



