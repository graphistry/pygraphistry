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
   * - ``OPTIONAL MATCH``
     - Partial
     - Supported for a bounded Cypher-in-GFQL subset, not the full Cypher null-extension surface.
   * - ``UNWIND``
     - Partial
     - Supported in the current Cypher-in-GFQL subset, but not in every placement and combination.
   * - ``UNION`` / ``UNION ALL`` and ``CALL graphistry.*``
     - Partial
     - Execute directly through ``g.gfql("...")``. Helper translation to a single ``Chain`` is stricter.
   * - Variable-length relationship patterns
     - Partial
     - Direct Cypher supports endpoint-only single variable-length relationship
       traversals such as ``[*2]``, ``[*1..3]``, ``[*]``, and typed forms like
       ``[:R*2..4]``. Path/list-carrier uses, bounded/exact ``WHERE`` pattern
       predicates, and mixed connected patterns still fail fast.
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
- Single variable-length relationship patterns when they are the only
  relationship in the connected pattern, including ``[*n]``, ``[*m..n]``,
  ``[*]``, and typed forms such as ``[:R*2..4]``.
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
- Label predicates such as ``WHERE b:Foo:Bar``.
- Relationship-type predicates such as ``WHERE type(r) = 'KNOWS'``.
- Positive relationship-existence pattern predicates such as
  ``WHERE (n)-[:R]->()`` and bare fixed-point variable-length existence checks
  such as ``WHERE (n)-[*]-()``.
- One positive relationship-existence pattern predicate may be combined with
  ordinary row filters through top-level ``AND``, for example
  ``WHERE n.kind = 'x' AND (n)-[:R*]->() AND n.id <> 'a'``.

Variable-Length Relationship Boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct Cypher multihop support is intentionally narrow in the current landing
slice. The supported direct forms are endpoint traversals where the
variable-length relationship is the only relationship in the connected pattern,
for example:

- ``MATCH (a)-[*2]->(b) RETURN b``
- ``MATCH (a)-[:R*1..3]->(b) RETURN b``
- ``MATCH (a)<-[*2]-(b) RETURN b``
- ``MATCH (a)-[:R*1..2]-(b) RETURN b``

The current compiler explicitly rejects these remaining subfamilies with
``GFQLValidationError`` instead of attempting unsound execution:

- path/list-carrier use of a variable-length relationship alias, such as
  ``RETURN r`` or ``count(r)``
- exact or bounded variable-length ``WHERE`` pattern predicates such as
  ``WHERE (n)-[:R*2]-()``
- top-level ``OR`` / ``NOT`` around variable-length ``WHERE`` pattern
  predicates, or more than one positive pattern predicate in the same
  ``WHERE`` clause
- connected patterns containing more than one relationship when any one of
  them is variable-length
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
- The bounded ``MATCH ... WITH ... MATCH ... RETURN`` re-entry shape.

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

- ``graphistry.nx.*`` remains a deliberately smaller compatibility subset than
  ``igraph`` / ``cugraph``, but it now includes representative node, edge, and
  graph-returning forms:

  - ``graphistry.nx.pagerank()`` / ``.write()``
  - ``graphistry.nx.betweenness_centrality()`` / ``.write()``
  - ``graphistry.nx.edge_betweenness_centrality()`` / ``.write()``
  - ``graphistry.nx.k_core.write()``

  Node calls use ``nodeId`` + the algorithm column, edge calls use
  ``source`` / ``destination`` + the algorithm column, and topology-returning
  calls such as ``k_core`` require ``.write()``.

- Local Cypher ``CALL`` options accept one optional map argument. The top-level
  keys mirror ``compute_igraph()`` / ``compute_cugraph()`` options such as
  ``out_col``, ``directed``, ``kind``, ``use_vids``, and ``params``; any extra
  keys are forwarded into the nested algorithm ``params`` dictionary.

- Outside that smaller ``networkx`` subset, ``graphistry.nx.*`` is not part of
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
- ``UNWIND`` works at top level, after ``MATCH``, and in row-only pipelines,
  but not in every graph/row interleaving.
- ``MATCH ... WITH ... MATCH ... RETURN`` is limited to the bounded single
  re-entry shape and does not generalize to arbitrary re-entry plans.

Not Supported Today
~~~~~~~~~~~~~~~~~~~

- Variable-length relationship aliases used as path/list carriers, such as
  ``RETURN r`` or ``count(r)``.
- Exact or bounded variable-length ``WHERE`` pattern predicates such as
  ``WHERE (n)-[:R*2]-()``.
- Connected patterns that mix a variable-length relationship with other
  relationship segments in the same connected pattern.
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
execution, preflight it with the helper APIs:

.. code-block:: python

    from graphistry.compute.exceptions import GFQLSyntaxError, GFQLValidationError
    from graphistry.compute.gfql.cypher import parse_cypher, compile_cypher

    query = "MATCH (p:Person) RETURN p.name AS name"

    try:
        parse_cypher(query)      # grammar + AST checks
        compile_cypher(query)    # GFQL Cypher compiler / lowering checks
    except GFQLSyntaxError as exc:
        print("Invalid Cypher syntax for g.gfql(\"MATCH ...\"):", exc)
    except GFQLValidationError as exc:
        print("Valid Cypher, but outside the current GFQL Cypher surface:", exc)

- Use ``parse_cypher()`` when you only want syntax and AST validation.
- Use ``compile_cypher()`` for the strongest compiler preflight, because it also
  catches unsupported-but-valid query shapes in lowering.
- Use ``cypher_to_gfql()`` only when you specifically need a single GFQL
  ``Chain``. It is intentionally stricter than direct execution through
  ``g.gfql("...")``.

Common Rewrites
---------------

- Need remote execution on Graphistry infrastructure instead of running against
  the current bound graph? Prefer ``g.gfql_remote([...])`` for remote GFQL.
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
