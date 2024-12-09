from typing_extensions import Literal


OutputType = Literal["all", "nodes", "edges", "shape"]
output_type_values = {"all", "nodes", "edges", "shape"}

FormatType = Literal["json", "csv", "parquet"]
from_type_values = {"json", "csv", "parquet"}
