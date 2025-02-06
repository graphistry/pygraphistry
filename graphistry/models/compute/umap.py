from typing import Set
from typing_extensions import Literal

UMAPEngineConcrete = Literal['cuml', 'umap_learn']
UMAPEngine = Literal[UMAPEngineConcrete, "auto"]

umap_engine_values: Set[UMAPEngineConcrete] = {'cuml', 'umap_learn'}
