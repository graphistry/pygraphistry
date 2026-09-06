"""Chain specializations for the pandas/cuDF engines: admission predicates and their lanes."""
from .admission import NativeFastPathShape, native_fast_path_admits
from .hotpaths import _try_chain_fast_path

__all__ = ["NativeFastPathShape", "native_fast_path_admits", "_try_chain_fast_path"]
