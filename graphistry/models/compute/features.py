from typing import Set
from typing_extensions import Literal

GraphEntityKind = Literal['nodes', 'edges']
graph_entity_kind_values: Set[GraphEntityKind] = {'nodes', 'edges'}

FeatureEngineConcrete = Literal["none", "pandas", "skrub", "torch"]
FeatureEngine = Literal[FeatureEngineConcrete, "dirty_cat", "auto"]
feature_engine_concrete_values: Set[FeatureEngineConcrete] = {"none", "pandas", "skrub", "torch"}
