"""Test helper for the #1650 structured-return change.

`RETURN a` (whole entity) now emits flattened `a.id, a.val, ...` columns instead
of a Cypher display string. This shim renders those flattened columns back to the
display string so existing conformance assertions (which encode the Cypher text
form) keep verifying render fidelity with minimal churn — the same role the
tck-gfql harness plays for the external suite.

Use `entity_text_records(result, {"a": "nodes"})` and compare to the pre-#1650
expected list-of-dicts. Scalar / expression output columns pass through unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from graphistry.compute.gfql.cypher.result_postprocess import render_entity_text


def _to_pandas(series: Any) -> Any:
    if hasattr(series, "to_pandas"):
        series = series.to_pandas()
    return series.reset_index(drop=True)


def entity_text_records(result: Any, entities: Dict[str, str]) -> List[Dict[str, Any]]:
    """Records with whole-entity columns rendered to Cypher text, scalars passed through.

    `entities` maps each whole-entity output name to its table ("nodes"/"edges").
    """
    nodes = result._nodes
    prefixes = tuple(f"{name}." for name in entities)
    data: Dict[str, Any] = {
        name: _to_pandas(render_entity_text(result, name, table=table))
        for name, table in entities.items()
    }
    for col in nodes.columns:
        if not str(col).startswith(prefixes):
            data[str(col)] = _to_pandas(nodes[col])
    return pd.DataFrame(data).to_dict(orient="records")
