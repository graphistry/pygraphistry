# GFQL JSON Wire Protocol Examples
Generated from actual AST objects using .to_json() method

================================================================================
Example: Simple Node Matcher
================================================================================

Python form:
```python
n({'type': 'person'})
```

JSON wire protocol:
```json
{
  "type": "Node",
  "filter_dict": {
    "type": "person"
  }
}
```

================================================================================
Example: Node Matcher with Predicate
================================================================================

Python form:
```python
n({'age': gt(18), 'country': is_in(['USA', 'Canada'])})
```

JSON wire protocol:
```json
{
  "type": "Node",
  "filter_dict": {
    "age": {
      "type": "GT",
      "val": 18
    },
    "country": {
      "type": "IsIn",
      "options": [
        "USA",
        "Canada"
      ]
    }
  }
}
```

================================================================================
Example: Forward Edge Matcher
================================================================================

Python form:
```python
e_forward(edge_match={'weight': gt(0.5)}, hops=2)
```

JSON wire protocol:
```json
{
  "type": "Edge",
  "hops": 2,
  "to_fixed_point": false,
  "direction": "forward",
  "edge_match": {
    "weight": {
      "type": "GT",
      "val": 0.5
    }
  }
}
```

================================================================================
Example: Undirected Edge with Node Predicates
================================================================================

Python form:
```python
e_undirected(
    source_node_match={'type': 'user'},
    edge_match={'weight': ge(0.8)},
    destination_node_match={'type': 'post'},
    hops=1
)
```

JSON wire protocol:
```json
{
  "type": "Edge",
  "hops": 1,
  "to_fixed_point": false,
  "direction": "undirected",
  "source_node_match": {
    "type": "user"
  },
  "edge_match": {
    "weight": {
      "type": "GE",
      "val": 0.8
    }
  },
  "destination_node_match": {
    "type": "post"
  }
}
```

================================================================================
Example: Simple Chain - Find Friends of Friends
================================================================================

Python form:
```python
Chain([
    n({'type': 'person', 'name': 'Alice'}),
    e_forward(edge_match={'relationship': 'friend'}),
    n({'type': 'person'}),
    e_forward(edge_match={'relationship': 'friend'}),
    n({'type': 'person'})
])
```

JSON wire protocol:
```json
{
  "type": "Chain",
  "chain": [
    {
      "type": "Node",
      "filter_dict": {
        "type": "person",
        "name": "Alice"
      }
    },
    {
      "type": "Edge",
      "hops": 1,
      "to_fixed_point": false,
      "direction": "forward",
      "edge_match": {
        "relationship": "friend"
      }
    },
    {
      "type": "Node",
      "filter_dict": {
        "type": "person"
      }
    },
    {
      "type": "Edge",
      "hops": 1,
      "to_fixed_point": false,
      "direction": "forward",
      "edge_match": {
        "relationship": "friend"
      }
    },
    {
      "type": "Node",
      "filter_dict": {
        "type": "person"
      }
    }
  ]
}
```

================================================================================
Example: Call Operation - Hypergraph Transformation
================================================================================

Python form:
```python
ASTCall('hypergraph', {
    'entity_types': ['user', 'post', 'comment'],
    'direct': True,
    'opts': {'USE_FEAT_V2': True}
})
```

JSON wire protocol:
```json
{
  "type": "Call",
  "function": "hypergraph",
  "params": {
    "entity_types": [
      "user",
      "post",
      "comment"
    ],
    "direct": true,
    "opts": {
      "USE_FEAT_V2": true
    }
  }
}
```

================================================================================
Example: Call Operation - UMAP Layout
================================================================================

Python form:
```python
ASTCall('umap', {
    'kind': 'nodes',
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2
})
```

JSON wire protocol:
```json
{
  "type": "Call",
  "function": "umap",
  "params": {
    "kind": "nodes",
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2
  }
}
```

================================================================================
Example: Let Binding (DAG) - Multi-step Query
================================================================================

Python form:
```python
ASTLet({
    'users': Chain([n({'type': 'user'})]),
    'active_users': Chain([n({'type': 'user', 'last_login': gt(30)})]),
    'with_degrees': ASTCall('get_degrees', {'col': 'degree'}),
    'high_degree': Chain([n({'degree': gt(10)})])
})
```

JSON wire protocol:
```json
{
  "type": "Let",
  "bindings": {
    "users": {
      "type": "Chain",
      "chain": [
        {
          "type": "Node",
          "filter_dict": {
            "type": "user"
          }
        }
      ]
    },
    "active_users": {
      "type": "Chain",
      "chain": [
        {
          "type": "Node",
          "filter_dict": {
            "type": "user",
            "last_login": {
              "type": "GT",
              "val": 30
            }
          }
        }
      ]
    },
    "with_degrees": {
      "type": "Call",
      "function": "get_degrees",
      "params": {
        "col": "degree"
      }
    },
    "high_degree": {
      "type": "Chain",
      "chain": [
        {
          "type": "Node",
          "filter_dict": {
            "degree": {
              "type": "GT",
              "val": 10
            }
          }
        }
      ]
    }
  }
}
```

================================================================================
Example: Let Binding with References
================================================================================

Python form:
```python
ASTLet({
    'users': Chain([n({'type': 'user'})]),
    'user_friends': ASTRef('users', [
        e_forward(edge_match={'type': 'friend'}),
        n()
    ])
})
```

JSON wire protocol:
```json
{
  "type": "Let",
  "bindings": {
    "users": {
      "type": "Chain",
      "chain": [
        {
          "type": "Node",
          "filter_dict": {
            "type": "user"
          }
        }
      ]
    },
    "user_friends": {
      "type": "Ref",
      "ref": "users",
      "chain": [
        {
          "type": "Edge",
          "hops": 1,
          "to_fixed_point": false,
          "direction": "forward",
          "edge_match": {
            "type": "friend"
          }
        },
        {
          "type": "Node",
          "filter_dict": {}
        }
      ]
    }
  }
}
```

================================================================================
Example: String Predicates
================================================================================

Python form:
```python
n({
    'name': contains('smith'),
    'email': startswith('@example.com')
})
```

JSON wire protocol:
```json
{
  "type": "Node",
  "filter_dict": {
    "name": {
      "type": "Contains",
      "pat": "smith",
      "case": true,
      "flags": 0,
      "na": null,
      "regex": true
    },
    "email": {
      "type": "Startswith",
      "pat": "@example.com",
      "case": true,
      "na": null
    }
  }
}
```

================================================================================
Example: Complex Chain - E-commerce Pattern
================================================================================

Python form:
```python
Chain([
    n({'type': 'user', 'country': is_in(['USA', 'Canada'])}),
    e_forward(edge_match={'action': 'purchased'}),
    n({'type': 'product', 'price': gt(100)}),
    e_reverse(edge_match={'action': 'also_purchased'}),
    n({'type': 'product'})
])
```

JSON wire protocol:
```json
{
  "type": "Chain",
  "chain": [
    {
      "type": "Node",
      "filter_dict": {
        "type": "user",
        "country": {
          "type": "IsIn",
          "options": [
            "USA",
            "Canada"
          ]
        }
      }
    },
    {
      "type": "Edge",
      "hops": 1,
      "to_fixed_point": false,
      "direction": "forward",
      "edge_match": {
        "action": "purchased"
      }
    },
    {
      "type": "Node",
      "filter_dict": {
        "type": "product",
        "price": {
          "type": "GT",
          "val": 100
        }
      }
    },
    {
      "type": "Edge",
      "hops": 1,
      "to_fixed_point": false,
      "direction": "reverse",
      "edge_match": {
        "action": "also_purchased"
      }
    },
    {
      "type": "Node",
      "filter_dict": {
        "type": "product"
      }
    }
  ]
}
```

================================================================================
Example: Graph Algorithm - PageRank
================================================================================

Python form:
```python
ASTCall('compute_cugraph', {
    'alg': 'pagerank',
    'out_col': 'pagerank_score',
    'directed': True
})
```

JSON wire protocol:
```json
{
  "type": "Call",
  "function": "compute_cugraph",
  "params": {
    "alg": "pagerank",
    "out_col": "pagerank_score",
    "directed": true
  }
}
```

================================================================================
Example: Layout - Force Atlas 2
================================================================================

Python form:
```python
ASTCall('fa2_layout', {
    'fa2_params': {'iterations': 1000, 'scalingRatio': 2.0}
})
```

JSON wire protocol:
```json
{
  "type": "Call",
  "function": "fa2_layout",
  "params": {
    "fa2_params": {
      "iterations": 1000,
      "scalingRatio": 2.0
    }
  }
}
```

================================================================================
End of Examples
================================================================================

