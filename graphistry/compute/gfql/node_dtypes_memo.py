"""Per-frame memo for node dtype reads: the pushdown string-content gate is read once per node frame."""

import weakref
from collections import OrderedDict
from typing import Any, Optional, Tuple

from graphistry.compute.gfql.cache_registry import register_clearable_dict
from graphistry.compute.typing import DataFrameT, NodeDtypes

Fingerprint = Tuple[int, Tuple[str, ...]]

#: (id(frame), engine) -> (weakref, (length, columns), dtypes); valid while the weakref is that object
_NODE_DTYPES_MEMO: "OrderedDict[Tuple[int, str], Tuple[Any, Fingerprint, NodeDtypes]]" = OrderedDict()
_NODE_DTYPES_MEMO_MAX = 32
register_clearable_dict("_NODE_DTYPES_MEMO", _NODE_DTYPES_MEMO)


def clear_node_dtypes_memo() -> None:
    _NODE_DTYPES_MEMO.clear()


def frame_fingerprint(nodes: DataFrameT) -> Fingerprint:
    try:
        return (int(nodes.shape[0]), tuple(str(c) for c in nodes.columns))
    except Exception:
        return (-1, ())


def memo_get(nodes: DataFrameT, engine_name: str) -> Optional[NodeDtypes]:
    key = (id(nodes), engine_name)
    cached = _NODE_DTYPES_MEMO.get(key)
    if cached is None:
        return None
    ref, fingerprint, dtypes = cached
    if ref() is nodes and fingerprint == frame_fingerprint(nodes):
        _NODE_DTYPES_MEMO.move_to_end(key)
        return dict(dtypes)
    del _NODE_DTYPES_MEMO[key]
    return None


def memo_put(nodes: DataFrameT, engine_name: str, dtypes: NodeDtypes) -> None:
    try:
        ref = weakref.ref(nodes)
    except TypeError:
        return
    _NODE_DTYPES_MEMO[(id(nodes), engine_name)] = (ref, frame_fingerprint(nodes), dict(dtypes))
    while len(_NODE_DTYPES_MEMO) > _NODE_DTYPES_MEMO_MAX:
        _NODE_DTYPES_MEMO.popitem(last=False)
