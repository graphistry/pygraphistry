"""Machine-checkable grammar invariants for the GFQL Cypher LALR parser.

The parser's correctness argument is grammar-level, not implementation-level:

1. **Conflict profile is pinned** (``test_lalr_conflict_profile_pinned``): the
   unified grammar builds under LALR(1) with a fixed, reviewed set of
   shift/reduce conflicts and ZERO reduce/reduce conflicts. Reduce/reduce is
   the line that must never be crossed: it means two rules derive the same
   string (genuine ambiguity), which LALR resolves arbitrarily. A grammar edit
   that changes the profile fails this test and forces a review.

2. **Semantic ambiguity is ZERO** (``test_semantic_ambiguity_zero``):
   expanding EVERY Earley derivation of every corpus query
   (``ambiguity='explicit'`` + ``CollapseAmbiguities``), all derivations
   transform to the SAME AST, unconditionally. Two residual DERIVATION
   ambiguities exist (each exactly binary, each AST-identical, matching the
   two pinned shift/reduce conflicts) and are machine-surfaced as pinned
   witnesses in ``test_residual_derivation_ambiguity_witnesses``. The former
   WITH..WHERE attachment ambiguity was ELIMINATED at grammar level (WHERE is
   bundled into its preceding clause), as was the dotted-name redundancy
   (name-rooted dot chains derive only via ``qualified_name``).

3. **Parser-choice neutrality** (``test_lalr_ast_equals_earley_ast_differential``):
   running the production pipeline with an Earley parser over the same grammar
   yields byte-identical ASTs to the LALR parser across the corpus, and both
   reject the same invalid inputs. LALR is a pure speed change (~70x).

Together: the grammar (a declarative artifact) carries the correctness
argument, and these tests make its safety properties machine-checked so
syntactic extensions cannot silently regress them.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

import pytest

from graphistry.compute.exceptions import GFQLSyntaxError
from graphistry.compute.gfql.cypher import parse_cypher
from graphistry.compute.gfql.cypher import parser as parser_mod

lark = pytest.importorskip("lark")


# --- Corpus --------------------------------------------------------------------
# One query per grammar construct family. Every query must parse in production.

DIFFERENTIAL_CORPUS = [
    # patterns
    "MATCH (n) RETURN n",
    "MATCH (n:Person) RETURN n",
    "MATCH (n:Person {id: 1, name: 'x'}) RETURN n",
    "MATCH (a)-[r]->(b) RETURN a, r, b",
    "MATCH (a)<-[r:KNOWS]-(b) RETURN a",
    "MATCH (a)-[r:A|B|C]-(b) RETURN r",
    "MATCH (a)-[r:KNOWS*1..3]->(b) RETURN b",
    "MATCH (a)-[:R*]->(b) RETURN b",
    "MATCH p = shortestPath((a:X)-[:R*1..4]->(b:Y)) RETURN p",
    "MATCH (a)-[:R*2..]->(b) RETURN b",
    "MATCH (a)-[:R]->(b), (b)-[:S]->(c) RETURN a, c",
    "OPTIONAL MATCH (n)-[r]->(m) RETURN n, m",
    "MATCH (n) MATCH (m) RETURN n, m",
    # WHERE: structured predicate forms (flat AND chain lifts)
    "MATCH (n) WHERE n.x = 50 RETURN n",
    "MATCH (n) WHERE n.x <> 1 AND n.y >= 2 AND n.z < 3 RETURN n",
    "MATCH (n) WHERE n.name CONTAINS 'ab' RETURN n",
    "MATCH (n) WHERE n.name STARTS WITH 'a' AND n.name ENDS WITH 'z' RETURN n",
    "MATCH (n) WHERE n.x IS NULL AND n.y IS NOT NULL RETURN n",
    "MATCH (n) WHERE n:Admin AND n.active = true RETURN n",
    "MATCH (a), (b) WHERE a.id = b.id RETURN a",
    "MATCH (n) WHERE n.x = $param RETURN n",
    # WHERE: generic expression forms (stay on expr_tree)
    "MATCH (n) WHERE n.a > 3 OR n.b = 1 RETURN n",
    "MATCH (n) WHERE n.x = 1 XOR n.y = 2 RETURN n",
    "MATCH (n) WHERE NOT n.deleted = true RETURN n",
    "MATCH (n) WHERE (n.x = 1 AND n.y = 2) OR n.z = 3 RETURN n",
    "MATCH (n) WHERE n.x = 1 AND (n.y = 2 AND n.z = 3) RETURN n",
    "MATCH (n) WHERE n.x IN [1, 2, 3] RETURN n",
    "MATCH (n) WHERE n.x + 1 > n.y * 2 RETURN n",
    "MATCH (n) WHERE NOT (n)-[:BLOCKED]->() RETURN n",
    # RETURN / projection
    "MATCH (n) RETURN DISTINCT n.city AS city",
    "MATCH (n) RETURN count(n) AS c, sum(n.x) AS s",
    "MATCH (n) RETURN n.a, n.b ORDER BY n.a ASC, n.b DESC SKIP 2 LIMIT 5",
    "MATCH (n) RETURN CASE WHEN n.x > 1 THEN 'hi' ELSE 'lo' END AS bucket",
    "MATCH (n) RETURN CASE n.x WHEN 1 THEN 'a' ELSE 'b' END AS k",
    "RETURN 1 AS one",
    # WITH pipelines
    "MATCH (n) WITH n.city AS city, count(n) AS c WHERE c > 1 RETURN city, c",
    "MATCH (n) WITH n ORDER BY n.x LIMIT 3 RETURN n.id",
    "MATCH (n:Person) WITH n AS m WHERE m.age > 1 RETURN m",
    "MATCH (a)-[r]->(x) WITH a, x WHERE a.animal = x.animal RETURN a, x",
    "MATCH (a:A)-[:R]->(b) WITH collect(b) AS bs UNWIND bs AS b2 MATCH (b2)-[:S]->(c) RETURN c",
    "MATCH (n) WITH n WHERE n.x = 1 MATCH (m) WHERE m.y = n.x RETURN m",
    # UNWIND / CALL / UNION / params
    "UNWIND [1, 2, 3] AS x RETURN x",
    "MATCH (n) UNWIND n.tags AS t RETURN t",
    "CALL gds.pageRank({maxIter: 10}) YIELD nodeId, score RETURN nodeId, score",
    "MATCH (a:X) RETURN a.id AS id UNION MATCH (b:Y) RETURN b.id AS id",
    "MATCH (a:X) RETURN a.id AS id UNION ALL MATCH (b:Y) RETURN b.id AS id",
    # graph constructor / USE
    "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'x' } USE g1 MATCH (n) RETURN n",
    # comprehensions / quantifiers / subscripts / composite property access
    "MATCH (n) RETURN [x IN n.tags WHERE x > 1 | x + 1] AS t",
    "MATCH (n) RETURN [x IN n.tags WHERE x > 1] AS t",
    "MATCH (n) RETURN [x IN n.tags | x + 1] AS t",
    "MATCH (n) WHERE any(x IN n.tags WHERE x = 'a') RETURN n",
    "MATCH (n) WHERE all(x IN n.tags WHERE x > 0) RETURN n",
    "MATCH (n) WHERE none(x IN n.tags WHERE x < 0) RETURN n",
    "MATCH (n) WHERE single(x IN n.tags WHERE x = 1) RETURN n",
    "MATCH (n) RETURN n.tags[0] AS first, head(n.tags) AS h",
    "MATCH (n) RETURN (n:Admin) AS is_admin",
    "MATCH (n) RETURN [x IN n.tags] AS t",
    # remaining construct families (rule-coverage: every grammar rule must be
    # exercised -- see test_every_grammar_rule_is_exercised_by_the_corpus)
    "MATCH p = (a)-[:R]->(b) RETURN p",
    "MATCH (n) RETURN count(DISTINCT n.city) AS c",
    "MATCH (n) WHERE n.flag = false AND n.gone = null RETURN n",
    "GRAPH { MATCH (a)-[r]->(b) }",
    "MATCH (n) RETURN {name: n.x, 'k': 1} AS m",
    "MATCH (n) RETURN *",
    "MATCH ()-[r]->() RETURN count(*)",
    "MATCH (a)-->(b) RETURN a, b",
    "MATCH (a)<--(b) RETURN a, b",
    "MATCH (a)--(b) RETURN a, b",
    "MATCH (a)<-->(b) RETURN a, b",
    "MATCH (a)-[:R*2]->(b) RETURN b",
    # arithmetic / comparison operators + unary
    "MATCH (n) WHERE 1 < n.x < 10 RETURN n",
    "MATCH (n) WHERE n.x / 2 > n.y % 3 RETURN n",
    "MATCH (n) WHERE n.x - 1 > 0 RETURN n",
    # unary +/- on a NON-literal operand: a signed literal (-1) is an
    # ambiguous literal-vs-uminus form, but -n.x is an unambiguous unary node
    "MATCH (n) RETURN -n.x AS v, +n.y AS w",
    # composite-root property access + list subscript slices
    "MATCH (n) RETURN head(n.tags).name AS v",
    "MATCH (n) RETURN n.tags[1..3] AS a, n.tags[1..] AS b, n.tags[..3] AS c, n.tags[..] AS d",
]

# Queries that are GRAMMATICAL but rejected by the transformer by design
# (known-unsupported features that must still parse to give a good error).
# They contribute to grammar-rule coverage via raw parse only.
GRAMMAR_ONLY_COVERAGE = [
    "MATCH p = allShortestPaths((a)-[:R]->(b)) RETURN p",  # raises unsupported at transform
]

# Invalid inputs: both parsers must reject (rejection parity, not message parity).
REJECTED_CORPUS = [
    "MATCH (n) WHERE RETURN n",
    "MATCH (n RETURN n",
    "MATCH (n) RETURN",
    "MATCH (n)-[r->-(m) RETURN n",
    "WHERE n.x = 1 RETURN n",  # WHERE without MATCH (rejected post-parse)
    "MATCH (a)-[:R*..4]->(b) RETURN b",  # open LOWER hop bound is not in the grammar
]

# Residual DERIVATION ambiguity: strings with exactly TWO parse trees that
# transform to ONE identical AST (harmless, but machine-surfaced so a grammar
# edit can't silently grow it). Maps 1:1 to the single pinned shift/reduce
# conflict:
#   - `[x IN xs]`: list_comprehension (no body) vs list_literal of an in_op
#     element — the IN conflict. Both mean the identity comprehension text.
#     Inherent to Cypher's `[NAME IN ...` overlap; unremovable under LALR(1)
#     without duplicating the whole expr hierarchy (zero semantic gain).
RESIDUAL_DERIVATION_AMBIGUITY = [
    "MATCH (n) RETURN [x IN n.tags] AS t",
]


def _fresh_lalr_with_log() -> List[logging.LogRecord]:
    """Build a fresh (uncached) LALR parser with debug on, capturing lark's
    grammar-analysis log records (conflicts are reported at WARNING)."""
    records: List[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    lark_logger = logging.getLogger("lark")
    handler = _Capture()
    old_level = lark_logger.level
    lark_logger.addHandler(handler)
    lark_logger.setLevel(logging.DEBUG)  # lark.utils defaults it to CRITICAL
    try:
        lark.Lark(
            parser_mod._GRAMMAR,
            start="start",
            parser="lalr",
            debug=True,
            maybe_placeholders=False,
            propagate_positions=True,
        )
    finally:
        lark_logger.removeHandler(handler)
        lark_logger.setLevel(old_level)
    return records


def _earley_parser(ambiguity: str = "resolve") -> Any:
    return lark.Lark(
        parser_mod._GRAMMAR,
        start="start",
        parser="earley",
        ambiguity=ambiguity,
        maybe_placeholders=False,
        propagate_positions=True,
    )


# --- 1. Conflict profile -------------------------------------------------------

def test_lalr_conflict_profile_pinned() -> None:
    """The grammar's LALR conflict profile is pinned: 0 reduce/reduce (genuine
    ambiguity — never acceptable), and exactly these shift/reduce conflicts,
    all resolved as shift. A grammar edit that changes this set must update
    the pin AND justify each new conflict here:

    - IN x1 at ``qualified_name : NAME``: inside ``[``, a NAME followed by IN
      is a comprehension head (shift) rather than an expr atom (reduce). The
      residual-witness test pins that both readings of the overlapping string
      ``[x IN xs]`` produce the same AST. This is the ONE conflict inherent to
      Cypher's grammar (``[NAME IN ...`` is a comprehension or a list whose
      first element is an ``in_op``); removing it needs a duplicated expr
      hierarchy for zero semantic gain, so it stays — pinned and characterized.

    Historical note: DOT x5 (dotted-name redundancy) and WHERE x1 (WITH..WHERE
    attachment) were eliminated at grammar level — name-rooted dot chains
    derive only via ``qualified_name``, and WHERE is bundled into its
    preceding clause. RPAR x1 (``(n:Admin)`` label predicate) was eliminated
    by dropping the redundant ``label_predicate_expr`` rule (it parses via
    ``grouped_expr`` over ``bare_label_predicate_expr``). None must reappear.
    """
    conflict_re = re.compile(
        r"(Shift/Reduce|Reduce/Reduce) conflict for terminal (\w+): \(resolving as (\w+)\)"
    )
    profile: Dict[Tuple[str, str, str], int] = {}
    for record in _fresh_lalr_with_log():
        m = conflict_re.match(record.getMessage())
        if m:
            key = (m.group(1), m.group(2), m.group(3))
            profile[key] = profile.get(key, 0) + 1

    assert not any(kind == "Reduce/Reduce" for kind, _, _ in profile), (
        f"Reduce/reduce conflict introduced (genuine grammar ambiguity): {profile}"
    )
    assert profile == {
        ("Shift/Reduce", "IN", "shift"): 1,
    }, f"LALR conflict profile changed: {profile}"


# --- 2. Semantic ambiguity -----------------------------------------------------

def test_semantic_ambiguity_zero() -> None:
    """Every Earley derivation of every corpus query transforms to the SAME
    AST — semantic ambiguity is zero, no exceptions. (The former WITH..WHERE
    attachment ambiguity was eliminated by bundling WHERE into its preceding
    clause in the grammar.)

    Note: ``CollapseAmbiguities`` rebuilds trees without ``meta`` positions, so
    derivation ASTs have degraded spans/text and CANNOT be compared against the
    production AST — only against each other (all equally degraded, so any
    structural divergence between derivations still shows)."""
    from lark.visitors import CollapseAmbiguities

    explicit = _earley_parser(ambiguity="explicit")
    unexpected: List[str] = []
    for query in DIFFERENTIAL_CORPUS:
        trees = CollapseAmbiguities().transform(explicit.parse(query))
        asts = {
            repr(parser_mod._build_transformer(query).transform(t)) for t in trees
        }
        if len(asts) != 1:
            unexpected.append(f"{query!r}: {len(trees)} derivations, {len(asts)} ASTs")
    assert not unexpected, (
        "Semantic ambiguity introduced (derivations disagree on AST):\n"
        + "\n".join(unexpected)
    )


def test_residual_derivation_ambiguity_witnesses() -> None:
    """Machine-surfaced witnesses for the ONLY residual derivation
    ambiguities: each is exactly binary and AST-identical (see
    RESIDUAL_DERIVATION_AMBIGUITY for the 1:1 mapping to the two pinned
    shift/reduce conflicts). Every other corpus query has exactly ONE
    derivation. A grammar edit that grows either count fails here and must
    characterize the new ambiguity before landing."""
    from lark.visitors import CollapseAmbiguities

    explicit = _earley_parser(ambiguity="explicit")
    for query in RESIDUAL_DERIVATION_AMBIGUITY:
        trees = CollapseAmbiguities().transform(explicit.parse(query))
        asts = {
            repr(parser_mod._build_transformer(query).transform(t)) for t in trees
        }
        assert len(trees) == 2, f"expected binary derivation ambiguity: {query!r}"
        assert len(asts) == 1, f"derivations must agree on the AST: {query!r}"
    for query in DIFFERENTIAL_CORPUS:
        if query in RESIDUAL_DERIVATION_AMBIGUITY:
            continue
        trees = CollapseAmbiguities().transform(explicit.parse(query))
        assert len(trees) == 1, (
            f"unexpected derivation ambiguity ({len(trees)} trees): {query!r}"
        )


# --- Rule coverage: the corpus must exercise EVERY grammar rule ------------------

def test_every_grammar_rule_is_exercised_by_the_corpus() -> None:
    """Every tree-producing grammar rule must appear in some corpus parse.

    This is what makes syntactic EXTENSIONS safe-by-construction: adding a
    grammar rule without a corpus query fails HERE with the rule's name, and
    adding the query automatically subjects the new construct to every other
    invariant in this file (ambiguity probe, LALR==Earley differential) plus
    the full-repo differential. There is no way to extend the grammar and
    silently skip the safety net.

    Mechanics: 'tree-producing' = rule aliases plus non-inlined rule names
    (lark's ``?rule:`` single-child inlining and ``_``-prefixed helpers never
    appear as tree nodes, so they are not coverable and not counted).
    Coverage counts raw parses of DIFFERENTIAL_CORPUS + GRAMMAR_ONLY_COVERAGE
    (grammatical-but-unsupported constructs) on all three grammar entry
    points (whole query, WHERE-lift chain, pattern fragment)."""
    lalr = parser_mod._parser_lalr()

    expected = set()
    for rule in lalr.rules:  # type: ignore[attr-defined]  # concrete lark.Lark
        if rule.alias:
            expected.add(rule.alias)
        elif not rule.options.expand1 and not rule.origin.name.startswith("_"):
            expected.add(rule.origin.name)

    observed = set()

    def walk(tree: Any) -> None:
        if isinstance(tree, lark.Tree):
            observed.add(tree.data)
            for child in tree.children:
                walk(child)

    for query in DIFFERENTIAL_CORPUS + GRAMMAR_ONLY_COVERAGE:
        walk(lalr.parse(query))
    # the two sub-grammar entry points (their rules are unused from start=)
    walk(parser_mod._where_predicate_chain_parser().parse(
        "n.x = 1 AND n.y IS NULL AND n.z IS NOT NULL AND n.s CONTAINS 'a' "
        "AND n.t STARTS WITH 'b' AND n.u ENDS WITH 'c' AND n:Admin"
    ))
    walk(parser_mod._pattern_parser().parse("(a:X {k: 1})-[r:R*1..2]->(b)"))

    uncovered = sorted(expected - observed)
    assert not uncovered, (
        "Grammar rules with NO corpus coverage (add a DIFFERENTIAL_CORPUS "
        "query exercising each, or GRAMMAR_ONLY_COVERAGE if the construct is "
        f"grammatical-but-unsupported): {uncovered}"
    )


# --- 3. LALR == Earley differential ---------------------------------------------

def test_lalr_ast_equals_earley_ast_differential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the production pipeline with Earley instead of LALR yields the
    identical AST for every corpus query: the parser swap is AST-neutral."""
    earley = _earley_parser()

    def _with_parser(parser: Any, query: str) -> str:
        monkeypatch.setattr(parser_mod, "_parser_lalr", lambda: parser)
        parser_mod._parse_cypher_cached.cache_clear()
        try:
            return repr(parse_cypher(query))
        finally:
            monkeypatch.undo()
            parser_mod._parse_cypher_cached.cache_clear()

    for query in DIFFERENTIAL_CORPUS:
        lalr_ast = _with_parser(parser_mod._parser_lalr(), query)
        earley_ast = _with_parser(earley, query)
        assert lalr_ast == earley_ast, f"AST divergence on: {query}"


def test_lalr_and_earley_reject_the_same_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    earley = _earley_parser()
    for query in REJECTED_CORPUS:
        for parser in (parser_mod._parser_lalr(), earley):
            monkeypatch.setattr(parser_mod, "_parser_lalr", lambda p=parser: p)
            parser_mod._parse_cypher_cached.cache_clear()
            with pytest.raises(GFQLSyntaxError):
                parse_cypher(query)
            monkeypatch.undo()
            parser_mod._parse_cypher_cached.cache_clear()


# The complete set of DELIBERATE language fixes vs the old Earley parser,
# found by full repo-corpus differentials (1800+ queries; everything else is
# accept/reject- and AST-identical). Each was an accept-by-accident with
# ill-defined semantics; all are now honest syntax errors.
DELIBERATE_LANGUAGE_FIXES = [
    # Earley's dynamic lexer re-lexed DISTINCT (absent from the NAME reserved
    # lookahead) as a NAME and accepted this as returning a *variable named*
    # DISTINCT with distinct=False. The LALR contextual lexer keeps it a keyword.
    "MATCH (n) RETURN DISTINCT",
    # The old flat clause-item grammar accepted WHERE anywhere and attached it
    # positionally in Python; these shapes had ill-defined attachment (a double
    # WHERE kept BOTH predicates in different AST fields). WHERE now binds to
    # its preceding clause in the grammar, so they are syntax errors:
    "MATCH (a) WHERE a.x = 1 WHERE a.y = 2 RETURN a",                      # double WHERE
    "MATCH (n) WITH n WHERE n.x = 1 WHERE n.y = 2 RETURN n",               # double post-WITH WHERE
    "MATCH (n) UNWIND n.tags AS t WHERE t = 'a' RETURN t",                 # WHERE after UNWIND (not openCypher)
    "GRAPH g1 = GRAPH { WHERE a.x = 1 MATCH (a) } USE g1 MATCH (n) RETURN n",  # WHERE before MATCH
]


@pytest.mark.parametrize("query", DELIBERATE_LANGUAGE_FIXES)
def test_deliberate_language_fixes_are_rejected(query: str) -> None:
    with pytest.raises(GFQLSyntaxError):
        parse_cypher(query)
