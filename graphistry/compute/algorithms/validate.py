"""Result validation, run at full scale inside the suite.

These checks need no reference output, which matters because for three of the
five kernels no usable reference exists: cuGraph has no label propagation,
neither backend has MIS, and both PageRank implementations converge to a
tolerance while LDBC fixes the iteration count.

What is checkable without a reference turns out to be most of what matters:

* WCC -- the LDBC label semantics are self-verifying. With monotone renumbering
  the label must equal the minimum vertex id of its own component, and every
  edge must join two vertices carrying the same label.
* PageRank -- mass conservation and positivity.
* SSSP -- the triangle inequality over every edge, plus dist[source] == 0.
* MIS -- independence and maximality, which together are the definition.
* CDLP -- labels must be drawn from the vertex id space and stable under a
  further application of the update rule only if it has converged, so the honest
  check is structural rather than a fixpoint assertion.

A `fail` here marks the cell `validation_failed` and its timing is not published.
"""
from __future__ import annotations

from typing import Any, Mapping

from ._dfops import concat_frames, df_cons, gather, to_host_int


def _ok(**extra: Any) -> dict[str, Any]:
    return {"status": "ok", **extra}


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "fail", "reason": reason, **extra}


def _both_directions(edges: Any, src: str, dst: str, vec: Any):
    return gather(vec, edges[src]), gather(vec, edges[dst])


def validate_wcc(labels: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    edges = prepared["edges"]
    ls, ld = _both_directions(edges, "src", "dst", labels)
    if to_host_int((ls != ld).sum()) != 0:
        return _fail("an edge joins two different component labels")

    # The label must be a fixed point: the representative labels itself.
    if to_host_int((gather(labels, labels) != labels).sum()) != 0:
        return _fail("a component label is not a fixed point (label[label[v]] != label[v])")

    # LDBC semantics: the label IS the minimum vertex id of its component. This
    # is only assertable because dense_renumber is monotone, so dense order and
    # original order agree -- no reference output needed.
    from ._dfops import arange

    vids = arange(edges, len(labels), "int64")
    pairs = df_cons(edges, {"lbl": labels.reset_index(drop=True), "v": vids})
    mins = pairs.groupby("lbl", sort=False)["v"].min().reset_index()
    if to_host_int((mins["lbl"] != mins["v"]).sum()) != 0:
        return _fail("a component label is not the minimum vertex id of its component")

    return _ok(components=int(len(mins)))


def validate_pagerank(pr: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    total = float(pr.sum())
    if abs(total - 1.0) > 1e-9:
        return _fail(f"mass not conserved: sum={total!r}")
    if to_host_int((pr <= 0).sum()) != 0:
        return _fail("non-positive PageRank value")
    return _ok(mass=round(total, 12), max_rank=float(pr.max()))


def validate_cdlp(labels: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    v_count = prepared["vertices"]
    if to_host_int(((labels < 0) | (labels >= v_count)).sum()) != 0:
        return _fail("label outside the vertex id range")
    return _ok(communities=int(labels.nunique()))


def validate_sssp(dist: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    edges = prepared["edges"]
    source = prepared["sssp_source"]
    if float(dist.iloc[source]) != 0.0:
        return _fail(f"dist[source]={float(dist.iloc[source])!r}, expected 0")

    # Triangle inequality: no edge may offer a cheaper route than the recorded
    # distance. One masked pass over E; finite-only so inf - inf never appears.
    ds = gather(dist, edges["src"])
    dd = gather(dist, edges["dst"])
    w = edges["w"].reset_index(drop=True)
    slack = dd - (ds + w)
    finite = (ds == ds) & (ds < float("inf")) & (dd < float("inf"))
    violations = to_host_int(((slack > 1e-6) & finite).sum())
    if violations:
        return _fail(f"{violations} edges violate the triangle inequality")

    reached = to_host_int((dist < float("inf")).sum())
    return _ok(reached=reached, unreachable=int(prepared["vertices"]) - reached)


def validate_mis(in_set: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    edges = prepared["edges"]
    a, b = _both_directions(edges, "src", "dst", in_set)
    if to_host_int((a & b).sum()) != 0:
        return _fail("not independent: an edge has both endpoints in the set")

    # Maximality: every vertex outside the set needs a neighbour inside it.
    from .kernels import _sym_any, _sorted_copies
    from ._dfops import align, full

    by_src, by_dst = _sorted_copies(edges, "src", "dst")
    v_count = prepared["vertices"]
    nbr = _sym_any(by_src, by_dst, "src", "dst", in_set, v_count, chunks=1)
    nbr = align(edges, v_count, nbr, "v", "p", full(edges, v_count, 0, "int8"))
    orphans = to_host_int(((~in_set) & (nbr == 0)).sum())
    if orphans:
        return _fail(f"not maximal: {orphans} vertices outside the set have no neighbour in it")

    size = to_host_int(in_set.sum())
    if size == 0:
        return _fail("empty independent set")
    return _ok(set_size=size, fraction=round(size / v_count, 4))


_VALIDATORS = {
    "wcc": validate_wcc,
    "pagerank": validate_pagerank,
    "cdlp": validate_cdlp,
    "sssp": validate_sssp,
    "mis": validate_mis,
}


def validate_result(kernel: str, result: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    fn = _VALIDATORS.get(kernel)
    if fn is None:
        return {"status": "skipped", "reason": f"no validator for {kernel!r}"}
    try:
        return fn(result, prepared)
    except Exception as exc:
        return _fail(f"validator raised {type(exc).__name__}: {exc}")
