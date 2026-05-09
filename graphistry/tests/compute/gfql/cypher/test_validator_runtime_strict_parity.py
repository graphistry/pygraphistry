"""Validator/runtime strict-mode parity baselines (#1357).

``gfql_validate(..., strict=True)`` runs the FrontendBinder with
``strict_name_resolution=True``; the runtime ``compile_cypher_query``
path runs the post-normalize binder in **loose** mode (#1357 deferred).
This module pins where the two surfaces *diverge today* — a query that
the strict validator rejects but the runtime compile path admits.

When a follow-up PR closes a binder-coverage gap and the runtime flip is
made for that gap, the corresponding parametrized case here should be
moved into the "parity" partition (both reject) instead of the
"divergence" partition (validator-only).

Cross-kind alias rebind already has parity (binder-layer guard at #1357
ships with this PR — both validator and runtime reject), and is pinned
here as a positive parity case.
"""

from __future__ import annotations

import pandas as pd
import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql_validate import gfql_validate
from graphistry.tests.test_compute import CGFull


def _empty_g():
    """Minimal pandas-backed plottable for the validator (it reads
    g._nodes / g._edges schema even when strict=True with empty schema).
    """
    nodes_df = pd.DataFrame({"id": [], "label__S": []})
    edges_df = pd.DataFrame({"s": [], "d": [], "type": []})
    return CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")


# ---------------------------------------------------------------------------
# Parity: both validator (strict) and runtime (loose) reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # Cross-kind rebind — parity ships in this PR via the binder-layer
        # _bind_node_pattern / _bind_relationship_pattern / _bind_path_alias
        # entity_kind guard.
        "MATCH (a) MATCH ()-[a]->() RETURN a",
        "MATCH ()-[r]->() MATCH (r) RETURN r",
        "MATCH p = ()-->() MATCH (p) RETURN p",
        "MATCH (a) MATCH a = ()-->() RETURN a",
    ],
)
def test_validator_and_runtime_both_reject_cross_kind_rebind(query: str) -> None:
    """Both surfaces reject. Validator path goes through the strict binder;
    runtime path goes through the same binder via compile_cypher_query.
    Either order works — both must raise."""
    with pytest.raises(GFQLValidationError):
        gfql_validate(_empty_g(), query, strict=True)
    with pytest.raises(GFQLValidationError):
        compile_cypher(query)


# ---------------------------------------------------------------------------
# Divergence: validator (strict) rejects, runtime (loose) admits.
# These are the binder-coverage gaps documented in
# plans/1357-binder-strict-name-resolution/research/discovery-flip-blast-radius.md
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # Quantifier predicates — strict rejects 'x' as unresolved; runtime
        # admits because loose binder doesn't model comprehension scope.
        "MATCH (n) WHERE all(x IN n.labels WHERE x = 'A') RETURN n",
    ],
)
def test_validator_strict_rejects_runtime_loose_admits(query: str) -> None:
    """Pinned divergence — validator catches alias-scope leak that the
    runtime currently lets through. When the corresponding binder gap is
    closed (see follow-up issues), this test flips into the parity
    partition above."""
    with pytest.raises(GFQLValidationError):
        gfql_validate(_empty_g(), query, strict=True)
    # Runtime should NOT raise on binder dimension (loose admit). Compile
    # produces a CompiledCypher* result.
    compile_cypher(query)
