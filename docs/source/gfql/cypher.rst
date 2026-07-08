.. _gfql-cypher:

Cypher Syntax In GFQL
=====================

PyGraphistry supports a read-only Cypher surface directly through GFQL on a
bound graph. This is the on-ramp for Cypher users who want familiar
declarative syntax and graph-pattern semantics, but executed by GFQL's fully
vectorized columnar engine and open-source GPU runtime instead of a
database-only runtime. Start here when you want to execute a Cypher query
through ``g.gfql("MATCH ...")`` instead of translating it into native GFQL
operators by hand.

Choose The Right Cypher Entrypoint
----------------------------------

- Use ``g.gfql("MATCH ...")`` or ``g.gfql("...", language="cypher")`` for
  Cypher syntax on the current bound ``Plottable``.
- Use ``g.gfql_remote([...])`` for remote GFQL execution when the dataset size
  or hardware profile calls for running GFQL on remote infrastructure.
- Use ``graphistry.cypher("...")`` or ``g.cypher("...")`` for remote database
  Cypher over Bolt/Neo4j-style integrations. That is a separate execution path.

Quickstart
----------

Assume ``g`` is a bound graph with nodes and edges already attached.

.. code-block:: python

    top_people = g.gfql(
        "MATCH (p:Person) "
        "RETURN p.name AS name "
        "ORDER BY name DESC "
        "LIMIT 5"
    )

    top_people._nodes

String queries default to ``language="cypher"``, so the explicit selector is
usually optional:

.. code-block:: python

    same_result = g.gfql(
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name DESC LIMIT 5",
        language="cypher",
    )

Parameters
----------

Use ``params=...`` instead of manual string interpolation:

.. code-block:: python

    limited = g.gfql(
        "MATCH (p:Person) "
        "RETURN p.name AS name "
        "ORDER BY name DESC "
        "LIMIT $top_n",
        params={"top_n": 2},
    )

    limited._nodes

Graph Constructors (``GRAPH { }``)
-----------------------------------

.. note::

   ``GRAPH { }`` is a **GFQL extension** — not part of openCypher, and GQL's
   first edition (ISO/IEC 39075:2024) deferred graph constructors to a future
   revision. Standard Cypher and GQL both force query results through
   row/path-centric serialization. ``GRAPH { }`` closes that gap by keeping
   results in **graph state** (both ``_nodes`` and ``_edges``), enabling
   composable graph-pipeline workflows.

Use ``GRAPH { MATCH ... }`` when you want the matched subgraph back in graph
state instead of a row table:

.. code-block:: python

    subgraph = g.gfql(
        "GRAPH { "
        "  MATCH (seed)-[reach]-(nbr) "
        "  WHERE seed.degree >= $degree_cutoff "
        "}",
        params={"degree_cutoff": 10},
    )

    subgraph._nodes
    subgraph._edges

Use ``WHERE`` inside ``GRAPH { }`` to reduce the graph that flows into the
next stage. The result still stays graph-shaped: matching nodes and edges remain
in ``_nodes`` and ``_edges``, so a later ``USE``, ``GRAPH { ... }``, or
``CALL graphistry.*.write()`` can continue without first turning the result into
a row table.

The current implementation supports filters that can be decided from one node
alias or one edge alias at a time, for example ``seed.degree >= 10``,
``reach.weight > 5``, ``seed.score > 0.25 OR seed.score IS NULL``, and
``searchAny(seed, 'alice')``. GFQL applies those filters while building the
subgraph, so fewer nodes and edges move forward. When a node filter removes a
node, edges attached to removed nodes are removed too.

Filters that need the joined match rows are intentionally rejected inside
``GRAPH { }`` for now, because GFQL cannot yet turn every such row condition
back into one unambiguous node-and-edge graph to pass to the next stage. That
includes pattern-existence checks such as ``WHERE (a)-[:R]->()`` or
``EXISTS { ... }``, and expressions that combine multiple aliases in one
condition. Use a row query such as ``MATCH ... RETURN ...`` for those cases, or
split the work into smaller graph stages.

Use ``GRAPH g = GRAPH { ... }`` to bind a named graph, then ``USE g`` to
query it:

.. code-block:: python

    result = g.gfql(
        "GRAPH g1 = GRAPH { "
        "  MATCH (seed)-[reach]-(nbr) "
        "  WHERE seed.degree >= $degree_cutoff "
        "} "
        "USE g1 "
        "MATCH (n) "
        "RETURN n.id AS id, n.degree AS degree "
        "ORDER BY degree DESC LIMIT 10",
        params={"degree_cutoff": 10},
    )

Multi-stage graph pipelines chain constructors:

.. code-block:: python

    g1 = g.gfql(
        "GRAPH { "
        "  MATCH (seed)-[reach]-(nbr) "
        "  WHERE seed.degree >= $degree_cutoff "
        "}",
        params={"degree_cutoff": 10},
    )

    g2 = g1.gfql("CALL graphistry.cugraph.pagerank.write()", engine="cudf")

    g3 = g2.gfql(
        "GRAPH { "
        "  MATCH (core)-[halo]-(nbr) "
        "  WHERE core.pagerank >= $pagerank_cutoff "
        "}",
        params={"pagerank_cutoff": 0.25},
    )

The graph constructor surface supports:

- ``MATCH`` and ``WHERE`` clauses inside ``GRAPH { }``
- Graph-preserving ``CALL graphistry.*.write()`` inside ``GRAPH { }``
- ``GRAPH g = GRAPH { }`` named bindings with lexical scoping
- ``USE g`` to switch the current graph (works both inside constructors and
  in the outer query)
- Variable scoping: pattern variables inside ``GRAPH { }`` do not leak

This enables single-expression pipelines that filter, enrich, and query:

.. code-block:: python

    result = g.gfql(
        "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) WHERE a.score > 10 } "
        "GRAPH g2 = GRAPH { USE g1 CALL graphistry.degree.write() } "
        "USE g2 "
        "MATCH (n) RETURN n.id AS id, n.degree AS degree "
        "ORDER BY degree DESC LIMIT 10"
    )

Inside ``GRAPH { }``, only ``MATCH``, ``WHERE``, ``USE``, and graph-preserving
``CALL graphistry.*.write()`` are allowed. Row-oriented clauses (``RETURN``,
``ORDER BY``, ``SKIP``, ``LIMIT``, ``DISTINCT``, ``UNWIND``, ``WITH``) and
row-returning ``CALL`` (without ``.write()``) are not allowed inside the
constructor — use them in the outer query after ``USE``.

Supported Cypher Surface Through ``g.gfql()``
---------------------------------------------

The current GFQL Cypher compiler intentionally supports a bounded read-only
surface. At a high level, that includes:

- ``MATCH`` and a bounded ``OPTIONAL MATCH`` subset
- ``WHERE``
- ``RETURN`` and ``WITH``
- ``ORDER BY``, ``SKIP``, ``LIMIT``, and ``DISTINCT``
- ``UNWIND``
- supported ``UNION`` / ``UNION ALL`` and ``CALL graphistry.*`` flows when executed
  directly through ``g.gfql("...")``

For exact ``RETURN`` / ``WITH`` row semantics after pattern matching, see
:doc:`return`. For same-path ``WHERE`` comparisons, see :doc:`where`.

Support Matrix
--------------

.. list-table::
   :header-rows: 1

   * - Query shape
     - Status
     - Notes
   * - ``MATCH`` / ``WHERE`` / ``RETURN`` / ``WITH`` / ``ORDER BY`` / ``SKIP`` / ``LIMIT`` / ``DISTINCT``
     - Supported
     - Core read-only Cypher-in-GFQL path.
   * - ``GRAPH { MATCH ... }`` / ``GRAPH g = ...`` / ``USE g``
     - Partial
     - **GFQL extension** (not in openCypher; GQL deferred graph constructors).
       Returns matched subgraph in graph state. Supports named bindings and
       scoped graph switching via USE.
   * - ``OPTIONAL MATCH``
     - Partial
     - Supported for a bounded Cypher-in-GFQL subset, not the full Cypher null-extension surface.
   * - ``UNWIND``
     - Partial
     - Supported at top level, after ``MATCH``, in row-only pipelines, and in
       the narrow graph-backed
       ``MATCH ... WITH collect([DISTINCT] alias) AS list UNWIND list AS alias MATCH ... RETURN``
       continuation shape, but not in arbitrary graph/row interleavings.
   * - ``UNION`` / ``UNION ALL`` and ``CALL graphistry.*``
     - Partial
     - Execute directly through ``g.gfql("...")``. Helper translation to a single ``Chain`` is stricter.
   * - Variable-length relationship patterns
     - Partial
     - Direct Cypher supports endpoint traversals such as ``[*2]``,
       ``[*1..3]``, ``[*]``, and typed forms like ``[:R*2..4]``; connected
       multi-relationship variable-length patterns; and bounded/exact/fixed-point
       variable-length ``WHERE`` pattern predicates in the current row-shaped
       subset. Path/list-carrier uses and unsupported path/row-shaping cases
       still fail fast.
   * - ``CREATE`` / ``DELETE`` / ``SET``
     - Not supported
     - GFQL's Cypher surface is read-only.
   * - Multiple disconnected ``MATCH`` patterns and arbitrary joins
     - Not supported
     - Split the work into separate GFQL / dataframe steps.
   * - Full Cypher expression and function surface in row expressions
     - Partial
     - The current row-expression subset is intentionally smaller than full Cypher; finish advanced logic in pandas/cuDF when needed.

Supported Syntax Forms
----------------------

The matrix above is clause-level. This section lists the main user-visible
syntax forms on the current Cypher-in-GFQL surface.

Pattern Matching Forms
~~~~~~~~~~~~~~~~~~~~~~

- Single-pattern ``MATCH`` queries with node aliases, relationship aliases,
  inline property maps, and top-level ``params=...`` binding.
- Node labels and multi-label node patterns such as ``(p:Person:Admin)``.
- Relationship direction forms ``->``, ``<-``, and undirected ``-[]-``.
- Relationship type alternation such as ``[r:KNOWS|HATES]``.
- Single variable-length relationship patterns, including ``[*n]``,
  ``[*m..n]``, ``[*]``, and typed forms such as ``[:R*2..4]``.
- Connected patterns that mix variable-length and fixed-length relationships,
  such as ``MATCH (a)-[:R*2]->()-[:S]->(c) RETURN c``.
- Connected comma-separated patterns such as
  ``MATCH (a)-[:A]->(b), (b)-[:B]->(c)``.
- Repeated ``MATCH`` clauses when they stay connected through shared aliases.
- Path variable binding such as ``MATCH p = (n)-[r]->(b)`` when the path
  variable itself is not the projected output.

WHERE Forms
~~~~~~~~~~~

- Literal and parameter comparisons on node and edge properties.
- Same-path alias comparisons such as ``WHERE p.team = q.team``.
- ``IS NULL`` and ``IS NOT NULL`` predicates.
- String predicates ``STARTS WITH``, ``ENDS WITH``, and ``CONTAINS``.
- Regex match ``=~`` (openCypher/neo4j-standard), e.g.
  ``WHERE n.name =~ '(?i)al.*'``. Uses a **full-string / anchored** match
  (like neo4j's Java-regex ``=~``), so ``n.name =~ 'AB'`` matches only
  ``'AB'`` — use ``.*`` / ``^..$`` for partial matches. Inline flags such as
  ``(?i)`` (case-insensitive), ``(?m)``, and ``(?s)`` are honored. Composes
  through ``AND`` / ``OR`` / ``NOT``. Engine caveat: on ``engine='cudf'``,
  ``(?m)`` / ``(?s)``, lookaround, backreferences, and ``(?i)`` combined with
  uppercase escape classes (e.g. ``(?i)\\D+``; lowercase ``\\d``/``\\.``
  work), case-crossing character ranges (``(?i)[A-z]``), hex escapes, and
  non-ASCII patterns raise ``NotImplementedError`` (libcudf
  regex limits — declined honestly rather than approximated); use
  ``engine='pandas'`` for those patterns. (``LIKE`` / ``ILIKE`` are not part of
  Cypher or GQL — use ``=~``, ``CONTAINS``, or ``STARTS WITH`` instead.)
- Label predicates such as ``WHERE b:Foo:Bar``.
- Relationship-type predicates such as ``WHERE type(r) = 'KNOWS'``.
- Positive relationship-existence pattern predicates such as
  ``WHERE (n)-[:R]->()`` and variable-length existence checks such as
  ``WHERE (n)-[*]-()`` and ``WHERE (n)-[:R*2]->()``.
- ``EXISTS { <pattern> }`` pattern-existence subqueries (openCypher-standard) in
  WHERE position, e.g. ``WHERE EXISTS { (n)-[:R]->() }`` and ``WHERE NOT EXISTS
  { (n)--() }`` — the declarative prune-isolated building blocks. Aliases
  introduced inside the braces are existentially scoped (``EXISTS { (n)--(m) }``
  is fine even when ``m`` is unbound outside), inline property maps work
  (``EXISTS { (n)-[:R {w: 1}]->() }``), and the one supported inner ``WHERE``
  form is endpoint inequality: ``EXISTS { (n)--(m) WHERE m <> n }`` — "has a
  neighbor other than itself", i.e. the drop-self-loop prune-isolated flavor.
  Runs natively on all four engines. Not yet supported (clear errors):
  ``EXISTS`` in RETURN/WITH projections, general inner ``WHERE`` clauses,
  multi-pattern bodies, full ``MATCH .. RETURN`` subquery bodies, and ``EXISTS``
  inside ``GRAPH { }`` pipelines. For prune-isolated in GRAPH STATE (nodes AND
  edges back), use edge patterns instead: ``GRAPH { MATCH (a)-[e]-(b) }`` keeps
  every edge-touching node with ALL its edges (self-loops included); add
  ``WHERE a.id <> b.id`` for the drop-self-loop variant.
- Pattern predicates can be combined with row predicates in the current
  boolean subset, including ``AND`` / ``OR`` / ``XOR`` and ``NOT`` forms.

Scalar Functions and Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard openCypher/neo4j scalar functions and operators usable in ``WHERE``
and ``RETURN`` expressions:

- Arithmetic ``+ - * / %``. Chained comparisons such as
  ``WHERE 1 < n.age < 65`` are supported. (The ``^`` exponentiation operator is
  not yet available.)
- Numeric functions ``abs``, ``sqrt``, ``sign``, ``floor``, ``ceil`` (alias
  ``ceiling``, per Cypher 25 / GQL), and ``round(x)`` / ``round(x, precision)``
  (returns a float). ``round`` follows neo4j's tie-breaking: precision 0 rounds
  ties toward positive infinity (``round(-1.5)`` → ``-1.0``, ``round(2.5)`` →
  ``3.0``); precision > 0 rounds ties away from zero (``round(-1.55, 1)`` →
  ``-1.6``). One documented deviation at precision > 0: neo4j rounds via the
  number's decimal string (Java ``BigDecimal.valueOf``), so a value like
  ``2.675`` — stored as the binary double ``2.67499…`` — gives ``2.68`` in
  neo4j but ``2.67`` here (both engines, consistently binary-double).
  Precision above 308 is the identity (a float64 has no digits there).
- String helpers ``toLower`` / ``toUpper`` and their GQL-conformance aliases
  ``lower`` / ``upper`` (the idiomatic case-insensitive compare, e.g.
  ``WHERE toLower(n.name) = 'bob'``), plus ``substring`` and
  ``size``, and conversions ``toInteger`` / ``toFloat`` / ``toString`` /
  ``toBoolean`` and ``coalesce``.
- Regex ``=~`` (see WHERE Forms above).
- ``searchAny(entity, term[, opts])`` — cross-column search predicate (WHERE
  position; GFQL extension for the viz filter pipeline): True where ANY of the
  entity's columns matches ``term``. Inspector semantics: OR across columns,
  case-insensitive substring by default, regex opt-in; dtype-gated — string
  columns always, integer columns iff the term is a numeric literal
  (``/^[0-9.-]+$/``); floats/dates/booleans only via the explicit list. Options
  map: ``{caseSensitive: true, regex: true, columns: ['name', ...]}`` (unknown
  keys error, listing the valid ones). Composes with other WHERE predicates
  through AND/OR/NOT; nodes and edges independently searchable with different
  terms. Runs natively on all four engines for node aliases; an edge-alias
  ``searchAny(r, ...)`` declines honestly on polars pending multi-entity
  binding-row support (use ``engine='pandas'``), and explicit non-string
  columns beyond ints/bools likewise decline on polars and cuDF rather than
  risk divergent stringification (float repr differs across engines). The regex path obeys the same
  per-engine decline rules as ``=~``. Python twins:
  :meth:`ComputeMixin.search_nodes` / :meth:`ComputeMixin.search_edges`.

``LIKE`` / ``ILIKE`` and ``BETWEEN`` are intentionally not provided — they are
not part of Cypher or GQL; use ``=~`` / ``CONTAINS`` / ``STARTS WITH`` and
``a <= x AND x <= b`` (or chained ``a <= x <= b``) respectively.

Variable-Length Relationship Boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct Cypher multihop support remains intentionally bounded. The supported
direct forms include endpoint traversals, connected multi-relationship
patterns, and variable-length ``WHERE`` pattern predicates where the result
stays in the current row-shaping subset, for example:

- ``MATCH (a)-[*2]->(b) RETURN b``
- ``MATCH (a)-[:R*1..3]->(b) RETURN b``
- ``MATCH (a)<-[*2]-(b) RETURN b``
- ``MATCH (a)-[:R*1..2]-(b) RETURN b``
- ``MATCH (a)-[:R*2]->(b)-[:S]->(c) RETURN c``
- ``MATCH (a)-[:R]->(b), (b)-[:S*1..2]->(c) RETURN a.id AS a_id, c.id AS c_id``
- ``MATCH (n) WHERE (n)-[:R*2]->() RETURN n``
- ``MATCH (n) WHERE NOT (n)-[:R*2]->() RETURN n.id AS id``

The current compiler explicitly rejects these remaining subfamilies with
``GFQLValidationError`` instead of attempting unsound execution:

- path/list-carrier use of a variable-length relationship alias, such as
  ``RETURN r`` or ``count(r)``
- shapes that still require unsupported path/relationship-carrier row shaping
  around a variable-length segment
- connected multi-pattern relationship-alias projection such as
  ``RETURN r`` / ``r.prop`` when it would require unsupported row shaping
- multi-alias ``RETURN *`` projections that would require unsupported
  path/multi-source row shaping

Row And Row-Pipeline Forms
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Top-level row-only queries such as ``RETURN 1 AS x``.
- ``RETURN`` / ``WITH`` projections with aliases, ``RETURN *``, ``DISTINCT``,
  ``ORDER BY``, ``SKIP``, and ``LIMIT``.
- Terminal ``WITH`` queries and multiple ``WITH`` stages.
- ``WITH ... WHERE`` row filtering.
- Aggregation/grouping via Cypher projection semantics, including ``count``,
  ``count(DISTINCT ...)``, ``collect``,
  ``collect(DISTINCT ...)``, ``sum``, ``max``, and ``size(...)``.
- Top-level ``UNWIND ... RETURN ...`` queries.
- Mixed graph/row queries such as ``MATCH ... UNWIND ... RETURN ...``.
- Connected multi-alias scalar projection such as
  ``MATCH (a)-[:R]->(b), (b)-[:S]->(c) RETURN a.id AS a_id, c.id AS c_id``.
- The bounded ``MATCH ... WITH ... MATCH ... RETURN`` re-entry shape,
  including connected suffix projections in the current supported row-binding
  subset.

Whole-Entity RETURN Output Shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A terminal ``RETURN`` of a whole node or relationship (``RETURN a`` rather than
``RETURN a.prop``) emits **structured flattened columns**, one per field, named
``<alias>.<field>``::

    g.gfql("MATCH (a:Person) RETURN a")
    # result._nodes columns: a.id, a.name, a.age, ...  (one column per field)

This is directly usable (no string to re-parse) and survives JSON / CSV / Parquet /
Arrow serialization and ``plot()``. To recover the human-readable Cypher display
string (``(:Person {name: 'Alice'})``) on demand, use the presentation helper::

    from graphistry.compute.gfql.cypher.result_postprocess import render_entity_text
    text_series = render_entity_text(result, "a")            # nodes
    text_series = render_entity_text(result, "r", table="edges")  # relationships

Notes:

- An aliased property projection of the same field (``RETURN a, a.val``) is
  de-duplicated — you get a single ``a.val`` column, not two.
- A whole entity with no fields to flatten (no id binding, no properties, no
  type/label — in practice only an edge whose graph has no edge-id binding) has
  nothing to flatten and falls back to a single Cypher-display-text column under the
  bare alias. Nodes always carry an id field and always flatten.

Procedure And Multi-Branch Forms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Direct execution of ``UNION`` row queries through ``g.gfql("...")``.
- Direct execution of ``UNION ALL`` row queries through ``g.gfql("...")``.
- Direct execution of row-returning ``CALL graphistry.*`` procedures,
  including:

  - ``CALL graphistry.degree()``
  - ``CALL graphistry.degree YIELD nodeId RETURN nodeId``
  - ``CALL graphistry.igraph.pagerank() YIELD nodeId, pagerank RETURN nodeId``
  - ``CALL graphistry.cugraph.louvain()``
  - ``CALL graphistry.cugraph.edge_betweenness_centrality()``

- Direct execution of standalone graph-preserving ``CALL graphistry.*.write()``
  procedures, including:

  - ``CALL graphistry.degree.write()``
  - ``CALL graphistry.igraph.pagerank.write()``
  - ``CALL graphistry.cugraph.edge_betweenness_centrality.write()``
  - ``CALL graphistry.cugraph.k_core.write()``
  - ``CALL graphistry.igraph.spanning_tree.write()``
  - ``CALL graphistry.nx.edge_betweenness_centrality.write()``
  - ``CALL graphistry.nx.k_core.write()``
  - ``CALL graphistry.nx.pagerank.write()``

- Bare procedures without ``.write()`` stay row-returning even when you omit
  ``YIELD ... RETURN ...``. For example, ``CALL graphistry.degree()`` projects
  its default outputs into ``_nodes`` and leaves an empty placeholder
  ``_edges`` frame (for example, ``assert result._edges.empty``); use
  ``.write()`` when you want enrich-then-``MATCH`` graph workflows with
  traversable edges (for example, ``assert not result._edges.empty``).

- For ``graphistry.igraph.<alg>()`` and node-oriented ``graphistry.cugraph.<alg>()``,
  row mode uses ``nodeId`` plus the algorithm output columns (for example,
  ``pagerank`` or ``louvain``). For edge-oriented ``graphistry.cugraph.<alg>()``,
  row mode uses ``source`` / ``destination`` plus the edge result columns.
  Topology-returning procedures such as ``graphistry.cugraph.k_core()`` or
  ``graphistry.igraph.spanning_tree()`` require ``.write()``.

- ``graphistry.nx.*`` remains a curated CPU subset rather than the full
  NetworkX API, but it includes common node, edge, multi-output, and
  graph-returning forms:

  - ``graphistry.nx.pagerank()`` / ``.write()``
  - ``graphistry.nx.betweenness_centrality()`` / ``.write()``
  - ``graphistry.nx.degree_centrality()`` / ``.write()``
  - ``graphistry.nx.closeness_centrality()`` / ``.write()``
  - ``graphistry.nx.eigenvector_centrality()`` / ``.write()``
  - ``graphistry.nx.katz_centrality()`` / ``.write()``
  - ``graphistry.nx.connected_components()`` / ``.write()``
  - ``graphistry.nx.strongly_connected_components()`` / ``.write()``
  - ``graphistry.nx.core_number()`` / ``.write()``
  - ``graphistry.nx.hits()`` / ``.write()``
  - ``graphistry.nx.edge_betweenness_centrality()`` / ``.write()``
  - ``graphistry.nx.k_core.write()``

  Node calls use ``nodeId`` + the algorithm column, edge calls use
  ``source`` / ``destination`` + the algorithm column, and topology-returning
  calls such as ``k_core`` require ``.write()``. Multi-output ``hits`` returns
  ``nodeId``, ``hubs``, and ``authorities``.

  The same curated NetworkX algorithm subset is available from regular Python
  as ``g.compute_networkx(...)`` for users who do not need the local Cypher
  ``CALL`` path.

- Local Cypher ``CALL`` options accept one optional map argument. The top-level
  keys mirror ``compute_igraph()`` / ``compute_cugraph()`` options such as
  ``out_col``, ``directed``, ``kind``, ``use_vids``, and ``params``; any extra
  keys are forwarded into the nested algorithm ``params`` dictionary.

Component-labeling examples:

.. code-block:: python

    # Graph-enrichment mode (keeps traversable _edges)
    g.gfql(
        "CALL graphistry.cugraph.connected_components.write({out_col: 'wcc_id', directed: false})",
        language="cypher",
    )
    g.gfql(
        "CALL graphistry.cugraph.strongly_connected_components.write({out_col: 'scc_id', directed: true})",
        language="cypher",
    )

    # Row mode (no .write): returns nodeId + output column rows
    g.gfql(
        "CALL graphistry.cugraph.connected_components({out_col: 'wcc_row', directed: false})",
        language="cypher",
    )
    g.gfql(
        "CALL graphistry.nx.connected_components({directed: false})",
        language="cypher",
    )

- Outside that curated ``networkx`` subset, ``graphistry.nx.*`` is not part of
  the current local Cypher ``CALL`` surface.

- ``cypher_to_gfql()`` stays stricter than direct execution and intentionally
  rejects ``UNION`` / ``UNION ALL`` and row-returning ``CALL`` flows because
  they are not representable as a single GFQL ``Chain``.

Expression Families
~~~~~~~~~~~~~~~~~~~

- Arithmetic, boolean, comparison, and null-propagation expressions.
- ``CASE`` expressions.
- Graph introspection helpers such as ``labels()``, ``type()``, ``keys()``, and
  ``properties()``.
- Dynamic graph property lookup such as ``n['name']`` and ``n[$idx]``.
- List predicates such as ``all(...)``, ``any(...)``, ``none(...)``, and
  ``single(...)``.
- Temporal constructors and operations over ``date``, ``time``, ``datetime``,
  ``localtime``, ``localdatetime``, and ``duration`` in the current vectorized
  subset.

Bounded / Partial Forms
~~~~~~~~~~~~~~~~~~~~~~~

- ``OPTIONAL MATCH`` works for a bounded subset, including top-level and
  bound optional rows, but not the full Cypher null-extension surface.
- ``UNWIND`` works at top level, after ``MATCH``, in row-only pipelines, and
  in the narrow graph-backed
  ``MATCH ... WITH collect([DISTINCT] alias) AS list UNWIND list AS alias MATCH ... RETURN``
  continuation shape, but not in arbitrary graph/row interleavings.
- ``MATCH ... WITH ... MATCH ... RETURN`` is limited to the bounded single
  re-entry shape. Connected suffix projections with whole-row and carried
  scalar bindings are supported in the current subset, but this still does not
  generalize to arbitrary re-entry plans.

Not Supported Today
~~~~~~~~~~~~~~~~~~~

- Variable-length relationship aliases used as path/list carriers, such as
  ``RETURN r`` or ``count(r)``.
- Connected multihop shapes that still require unsupported
  path/relationship-carrier row shaping.
- Multiple disconnected ``MATCH`` patterns used as arbitrary joins.
- Multi-pattern re-entry shapes beyond the bounded single
  ``MATCH ... WITH ... MATCH ... RETURN`` form.
- ``RETURN *`` after unsupported re-entry shapes or when it would require
  unsupported multi-alias/path projection shaping.
- ``CREATE``, ``DELETE``, ``SET``, and other write clauses.
- Generic database procedures outside ``CALL graphistry.*``.
- The full Cypher expression/function surface.

Validation And Unsupported Shapes
---------------------------------

- Unsupported but syntactically valid query shapes on this Cypher surface raise
  ``GFQLValidationError``, usually before execution starts.
- Invalid Cypher syntax raises ``GFQLSyntaxError``.
- Passing a string query with an unsupported ``language=...`` selector also
  raises ``GFQLValidationError``.

That fail-fast behavior is intentional: the current GFQL Cypher compiler
prefers explicit validation over silently returning wrong rows.

Static Validation / Preflight Check
-----------------------------------

If you want to know whether a query fits the current Cypher-in-GFQL subset before
execution, start with the bound-graph inline preflight APIs:

.. code-block:: python

    g.gfql_validate(
        "MATCH (p) RETURN p.name AS name ORDER BY name DESC LIMIT $top_n",
        params={"top_n": 5},
        # strict=True is the default for local bound-graph preflight
    )

    # On failure:
    # - GFQLSyntaxError for invalid syntax
    # - GFQLValidationError for unsupported/scheme-invalid shapes

- Use ``g.gfql_validate(...)`` when you want a stable validate-only entrypoint
  that never executes query operators and raises structured exceptions on invalid queries.
- Use ``g.gfql(..., validate=True)`` when you want execution guarded by a
  local preflight check. For Cypher strings, this uses schema-aware strict
  preflight by default.
- Use ``g.gfql_remote(..., validate=True)`` when you want remote execution
  guarded by local preflight before upload/network dispatch. For Cypher strings,
  remote preflight uses ``strict=False`` by default because remote schema is authoritative.
- Use ``parse_cypher()`` when you only want grammar validation and access to
  the parsed representation.
- Use ``compile_cypher()`` when you need low-level compiler/lowering output for
  tooling or whitebox inspection.
- Use ``cypher_to_gfql()`` only when you specifically need a single GFQL
  ``Chain``. It is intentionally stricter than direct execution through
  ``g.gfql("...")``.

Low-level helper example:

.. code-block:: python

    from graphistry.compute.exceptions import GFQLSyntaxError, GFQLValidationError
    from graphistry.compute.gfql.cypher import parse_cypher, compile_cypher

    query = "MATCH (p:Person) RETURN p.name AS name"

    try:
        parsed = parse_cypher(query)   # grammar + AST checks
        compiled = compile_cypher(query)  # compiler/lowering checks
    except GFQLSyntaxError as exc:
        print("Invalid Cypher syntax for g.gfql(\"MATCH ...\"):", exc)
    except GFQLValidationError as exc:
        print("Valid Cypher, but outside the current GFQL Cypher surface:", exc)

Common Rewrites
---------------

- Need remote execution on Graphistry infrastructure instead of running against
  the current bound graph? Prefer ``g.gfql_remote(...)`` for remote GFQL, and
  keep ``validate=True`` (default) for local preflight before upload.
- Need remote database Cypher against Neo4j/Bolt-style backends instead of
  remote GFQL? Use ``graphistry.cypher("...")`` or ``g.cypher("...")``.
- Need a pure GFQL chain object? Use ``cypher_to_gfql()`` when the query fits a
  single ``Chain``.
- Need fixed-length, bounded, or fixed-point endpoint traversal? Direct Cypher
  already supports ``[*2]``, ``[*1..3]``, and ``[*]`` for that endpoint-only
  slice.
- Need aliasable intermediate hops, output slicing, or mixed connected-pattern
  multihop control? Rewrite in native GFQL with explicit hop bounds such as
  ``e_forward(min_hops=1, max_hops=3)`` or ``e_forward(to_fixed_point=True)``,
  or unroll the traversal into explicit chain steps.
- Need write semantics or arbitrary joins? Keep Cypher syntax for the supported
  read-only part and finish the rest in a database or in pandas/cuDF.

Compiler Helper APIs
--------------------

Use the helper APIs when you want to inspect or reuse compiler output rather
than run the query immediately:

.. code-block:: python

    from graphistry.compute.gfql.cypher import (
        parse_cypher,
        compile_cypher,
        cypher_to_gfql,
        gfql_from_cypher,
    )

    parsed = parse_cypher("MATCH (p:Person) RETURN p.name AS name")
    compiled = compile_cypher("MATCH (p:Person) RETURN p.name AS name")
    chain = cypher_to_gfql("MATCH (p:Person) RETURN p.name AS name")

``cypher_to_gfql()`` / ``gfql_from_cypher()`` are intentionally limited to
queries that can be represented as a single GFQL ``Chain``. If a query requires
``UNION`` or a row-returning ``CALL`` flow, execute it directly through
``g.gfql("...", language="cypher")`` instead.

See :doc:`/api/gfql/cypher` for the helper reference.

Translation Vs Direct Execution
-------------------------------

This page is about **direct Cypher-syntax execution** through
``g.gfql("MATCH ...")`` on a bound graph.

- If you want to run the query now, use ``g.gfql("MATCH ...")``.
- If you want to understand how Cypher maps into GFQL operators and wire
  protocol, use :doc:`spec/cypher_mapping`.
- If you want native GFQL chain syntax instead of strings, start with
  :doc:`quick`.
