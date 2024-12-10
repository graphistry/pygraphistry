from typing import Set, Union
from typing_extensions import Literal


OutputTypeGraph = Literal["all", "nodes", "edges", "shape"]
output_types_graph = {"all", "nodes", "edges", "shape"}

FormatType = Literal["json", "csv", "parquet"]
from_type_values = {"json", "csv", "parquet"}

OutputTypeDf = Literal["table", "shape"]
output_types_df: Set[OutputTypeDf] = {"table", "shape"}

OutputTypeJson = Literal["json"]
output_types_json: Set[OutputTypeJson] = {"json"}

OutputTypeAll = Union[OutputTypeGraph, OutputTypeDf, OutputTypeJson]
output_types_all = output_types_graph.union(output_types_df).union(output_types_json)
