#JSON Dataset Metadata

Metadata schema is not final yet!

##Design Goals
* Store default encodings and type/summary description of each columns.
* No hard assumption on our vgraph protobuf format. Should work with future replacements/alternatives vgraph.
* Leave options open for implementing k-partite graphs with multiple node/edge tables.
* Leave options open for composing data-sources together.

## Basic Skeleton

The metadata is divided in two parts
 
1. A header with general information such has the creator, creation time, etc.
2. A list of datasources; and for each datasource, encoding&types for both nodes and edges.

```
	HEADER: creator, version, etc.
	datasources:[
		{datasource0 description ...}
		{datasource1 description ...}
		...
	],
	nodes:[
		{
			count: <number of nodes in datasource0>
			encodings: <node encodings for datasource0>
			attributes: <attribute types+summaries for datasource0>
		}
		{
			count: <number of nodes in datasource1>
			encodings: <node encodings for datasource1>
			attributes: <attribute types+summaries for datasource1>
		}
		...
	]
	edges:[<same as nodes but for edges>]
```

We should always have `datasources.length == nodes.length == edges.length`

### Encodings
```json
"encodings":{
	"pointColor":{
    	"attributes":["spinglass"]
    },
    ...
}
```

Maps attribute names to their encoding descriptions. The `attributes` field lists the columns from which the encoding is computed.

#####Caveats
 * For now `attributes` must be an array of length one.
 * In the future, add a `transform` field to support more complex encodings.

### Attributes

Maps attribute names to a summary.

```json
attributes:{
	"spinglass":{
		"originalType":"int64",
       "min":0,
       "distinct":12,
       "max":11,
       "ctype":"int8",
       "stddev":3.4293402049363584,
       "mean":5.051250309482545
   },
   ...
}
```

The following summary fields are **mandatory**:

* `ctype`: The string-encoded numpy type
* `min`: The smallest value
* `max`: The largest value
* `distinct`: The number of unique values

The following summary fields are **optional/context dependent**:

* `originalType`: The numpy type before attempting to find the smallest representation capable of holding all values.
* `mean`, `stddev`: Stats for numeric types.
* `usertype`: How to pretty print type to users.

##Complete Example

```json
{
	"agent":"pygraphistry",
	"agentVersion":"0.9.19",
	"apiVersion":2,
	"created":1455665444898,
	"creator":"thibaud@graphistry.com",
	"name":"1GI0CJYKYQ",
	"nodes":[{
		"count":4039,
		"attributes":{
			"spinglass":{
				"originalType":"int64",
				"min":0,
				"distinct":12,
				"max":11,
				"ctype":"int8",
				"stddev":3.4293402049363584,
				"mean":5.051250309482545
			},
			"pagerank":{
				"originalType":"float64",
				"min":0.000041434683985727164,
				"distinct":3957,
				"max":0.0075745665246258935,
				"ctype":"float64",
				"stddev":0.00025797171365788026,
				"mean":0.0002475860361475613
			},
			"name":{
			"distinct":4039,
			"max":"Olympia Adams",
			"ctype":"utf8",
			"min":"Aaron Johnson"
			},
			"__nodeid__":{
				"originalType":"int64",
				"min":0,
				"distinct":4039,
				"max":4038,
				"ctype":"int16",
				"stddev":1166.103197262861,
				"mean":2019
			}
		},
		"encodings":{
			"pointColor":{
				"attributes":["spinglass"]
			},
			"pointTitle":{
				"attributes":["name"]
			},
			"pointSize":{
				"attributes":["pagerank"]
			},
			"nodeId":{
				"attributes":["__nodeid__"]
			}
		}
	}],
	"edges":[{
		"count":88234,
		"attributes":{
			"test":{
				"distinct":1,
				"max":"foo",
				"ctype":"utf8",
				"min":"foo"
			},
			"floatString":{
				"originalType":"float64",
				"min":0.000224256987401,
				"distinct":88234,
				"max":9.99979766077,
				"ctype":"float64",
				"stddev":2.8925467743982836,
				"mean":5.005153425609515
			},
			"dateObject":{
				"distinct":1,
				"userType":"datetime",
				"max":1455636636,
				"ctype":"datetime32[s]",
				"min":1455636636
			},
			"intString":{
				"originalType":"float64",
				"min":0,
				"distinct":10,
				"max":9,
				"ctype":"float64",
				"stddev":2.8686655874802707,
				"mean":4.504476732325408
			},
			"Boolean":{
				"distinct":2,
				"max":true,
				"ctype":"bool",
				"min":false
			}
		},
		"encodings":{
			"source":{
				"attributes":["src"]
			},
			"destination":{
				"attributes":["dst"]
			},
			"edgeWeight":{
				"attributes":["intString"]
			}
		}
	}],
	"datasources":[{
		"url":"s3://graphistry.data/pygraphistry/84365ae48160310d722ddc4b011f5798/ac1f08185b00e91dbeeb5220f4224140acecba7e.vgraph",
		"type":"vgraph",
		"size":1045477,
		"sha1":"ac1f08185b00e91dbeeb5220f4224140acecba7e"
	}]
}
```

[Load this example in graph-viz](http://proxy-labs.graphistry.com/graph/graph.html?dataset=s3://graphistry.data/pygraphistry/84365ae48160310d722ddc4b011f5798/dataset.json&type=jsonMeta&viztoken=a09fb35b43f1ad527b680e905691ed53cb074983&usertag=8e721f2c-pygraphistry-0.9.19&splashAfter=1455666187&info=true)