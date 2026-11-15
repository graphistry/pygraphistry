import graphistry
import pandas as pd

source = graphistry.nodes(pd.DataFrame({"id": pd.Series([], dtype="int64")}), "id").edges(
    pd.DataFrame({"s": pd.Series([], dtype="int64"), "d": pd.Series([], dtype="int64")}),
    "s", "d",
)
created = source.gfql(
    "CREATE (a:Person {name: 'Ada'})-[:KNOWS {since: 2020}]->(b:Person {name: 'Grace'})",
    engine="pandas",
)
assert len(created._nodes) == 2 and len(created._edges) == 1
assert created._edges["type"].tolist() == ["KNOWS"]
assert len(source._edges) == 0
print("DOCS REL EXAMPLE OK", list(created._edges.columns))
rev = source.gfql("CREATE (a:P {n: 1})<-[:R]-(b:P {n: 2})", engine="pandas")
print("reverse edge:", rev._edges.to_dict("records"), rev._nodes[[rev._node, "n"]].to_dict("records"))
