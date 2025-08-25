"""Type definitions for UMAP-related functionality."""

from typing import Literal, Set
from typing_extensions import Literal as Literal_ext

# Concrete UMAP engine options
UMAPEngineConcrete = Literal['cuml', 'umap_learn']

# UMAP engine including auto option
UMAPEngine = Literal['auto', 'cuml', 'umap_learn']

# Valid UMAP engine values (concrete only)
umap_engine_concrete_values: Set[UMAPEngineConcrete] = {'cuml', 'umap_learn'}

# All valid UMAP engine values including auto
umap_engine_values: Set[UMAPEngine] = {'auto', 'cuml', 'umap_learn'}

# Graph entity kinds
GraphEntityKind = Literal['nodes', 'edges']
