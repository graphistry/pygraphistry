from __future__ import annotations

import graphistry.compute.gfql.temporal_text as temporal_text
from graphistry.compute.gfql.temporal import constructors
from graphistry.compute.gfql.temporal import durations
from graphistry.compute.gfql.temporal import folding
from graphistry.compute.gfql.temporal import values


def test_temporal_text_reexports_constructor_helpers_for_legacy_imports() -> None:
    assert temporal_text.normalize_temporal_constructor_text is constructors.normalize_temporal_constructor_text
    assert temporal_text.DATE_CALL_TEXT_RE is constructors.DATE_CALL_TEXT_RE
    assert temporal_text.TEMPORAL_CALL_EXPR_RE is constructors.TEMPORAL_CALL_EXPR_RE


def test_temporal_text_reexports_domain_helpers_for_legacy_imports() -> None:
    assert temporal_text._TemporalValue is values._TemporalValue
    assert temporal_text._comparable_datetime is values._comparable_datetime
    assert temporal_text.parse_temporal_sort_duration_components is durations.parse_temporal_sort_duration_components
    assert temporal_text.resolve_duration_text_property is durations.resolve_duration_text_property
    assert temporal_text.fold_temporal_constructor_ast is folding.fold_temporal_constructor_ast
