.. _gfql-engines:

Choosing a GFQL Engine: pandas, Polars, cuDF, Polars-GPU
========================================================

GFQL runs the **same query** on four interchangeable execution engines. You pick
the engine with one keyword — ``engine=``, accepted uniformly by ``g.gfql()`` and
``g.hop()`` — and GFQL returns **identical results** on every one (differential parity
is a release gate). Pick the engine that fits your hardware and workload; nothing else changes.

The one-line speedup
--------------------

On real graphs, switching the default ``pandas`` engine to the columnar **Polars**
engine is a one-keyword change — no GPU, same results:

.. code-block:: python

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # up to ~38x faster on real graphs, no GPU, identical results

Your existing pandas, Polars, or cuDF graph works as-is: the input frames are accepted and
coerced once; the only change is the keyword. The catch: below ~1M edges ``pandas`` can be
faster (Polars has fixed startup overhead), and a few exotic Cypher features still require
``engine='pandas'`` — see *When not to use Polars* below.

.. warning::
   **Already a Polars user? Pass** ``engine='polars'`` **— the default does not.** With the
   default ``engine='auto'``, a graph built from ``polars.DataFrame`` is **silently coerced to
   pandas** (``auto`` resolves to ``cudf`` for cuDF input and ``pandas`` for everything else,
   *including Polars*; it never selects the Polars engine). To stay native end-to-end, pass
   ``engine='polars'`` explicitly:

   .. code-block:: python

      import polars as pl, graphistry
      g = graphistry.edges(edges_pl, 'src', 'dst').nodes(nodes_pl, 'id')  # polars frames
      out = g.gfql(query)                    # auto -> coerced to PANDAS (out._nodes is pandas!)
      out = g.gfql(query, engine='polars')   # native Polars in and out (out._nodes is polars)

.. note::
   **Result frames match the engine.** With ``engine='polars'`` or ``'polars-gpu'`` the
   output is Polars — ``result._nodes`` and ``result._edges`` are ``polars.DataFrame`` (and
   ``cudf.DataFrame`` for ``engine='cudf'``). If downstream code is pandas-specific (``.iloc``,
   ``.loc``, ``groupby().apply()``), call ``result._nodes.to_pandas()`` to convert back.

The four engines
----------------

.. list-table::
   :header-rows: 1
   :widths: 16 14 18 12 40

   * - Engine
     - Hardware
     - Frame type
     - Opt-in?
     - In one line
   * - ``pandas``
     - CPU
     - ``pandas``
     - default
     - Universal default; best on small/interactive graphs.
   * - ``polars``
     - CPU
     - ``polars``
     - explicit
     - Columnar + fused lazy plan; the CPU speed win, **no GPU needed**.
   * - ``cudf``
     - NVIDIA GPU
     - ``cudf``
     - explicit
     - RAPIDS GPU, eager op-by-op; great for one very large materialization.
   * - ``polars-gpu``
     - NVIDIA GPU
     - ``polars``
     - explicit
     - The Polars fused plan executed on GPU (cudf_polars); fastest on heavy multi-hop.

``engine='auto'`` resolves to ``cudf`` for cuDF input and ``pandas`` otherwise. **AUTO
never selects Polars or Polars-GPU** — they are explicit opt-in (see *Why opt-in?* below).

Motivating comparison (real graphs)
-----------------------------------

Same query, same answers, four engines. Warm-median latency on **Orkut** (3.1M nodes /
**117M edges**, SNAP), measured on a single machine:

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 16

   * - Workload (Orkut, 117M edges)
     - ``pandas``
     - ``polars``
     - ``cudf``
     - ``polars-gpu``
   * - 1-hop from 10K seeds
     - 2613 ms
     - **68 ms**
     - 1005 ms
     - 63 ms
   * - 2-hop from 10K seeds
     - 18161 ms
     - 2695 ms
     - 2774 ms
     - **1518 ms**
   * - Full out-degree aggregation
     - 799 ms
     - 205 ms
     - 314 ms
     - **167 ms**
   * - 2-hop from 100K seeds (~85M output rows)
     - 28822 ms
     - 8215 ms
     - **6002 ms**
     - 8559 ms

*Warm median, identical result rows across all four engines. Reproducer:*
``benchmarks/gfql/index_bulk_olap_bench.py``. *See Methodology below.*

Reading the table:

- **Polars-CPU beats pandas up to ~38x** on bulk traversal and ~4x on aggregation — **with no
  GPU**. On the 1-hop workload it is ~38x faster than pandas (68 ms vs 2613 ms).
- **Polars-CPU also beats cuDF** on these shapes (68 ms vs 1005 ms on 1-hop). cuDF runs
  GFQL *eagerly*, op by op (a kernel launch + a materialized intermediate per hop), while
  Polars builds **one fused lazy plan and collects once**. The fused plan wins until the
  work is large enough to amortize GPU launch costs.
- **Polars-GPU is fastest on heavy multi-hop** (2-hop from 10K seeds: 1518 ms) and on
  aggregation — the same fused plan, executed on the GPU.
- **cuDF wins the one extreme case** — a 2-hop from 100K seeds materializing ~85M output rows
  (6.0 s) — where raw GPU throughput on a single massive join overtakes everything and
  Polars-GPU comes under memory pressure (footnote F3).
- On a smaller graph (**LiveJournal**, 35M edges) the pattern holds: 1-hop from 10K seeds is
  pandas 1129 ms → polars **37 ms** (~30x). Filter- and lookup-heavy workloads favor Polars
  even more strongly — a separate **LDBC SNB sf1** benchmark shows order-of-magnitude gains
  (tens of × over pandas; see ``benchmarks/gfql/`` and the GFQL benchmark notes).

.. note::
   There is **no universal winner** — route by workload shape and size (next section).
   Below ~1M edges, ``pandas`` often wins because Polars/GPU pay a fixed startup overhead
   that the small amount of work cannot amortize.

Decision matrix
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 16 18 22 14

   * - Workload shape
     - Size (edges)
     - Hardware
     - Recommended engine
     - Notes
   * - Filter / ``WHERE`` / aggregation
     - > ~1M
     - CPU
     - ``polars``
     - order-of-magnitude over pandas [F1]
   * - Bulk 1-hop frontier expansion
     - > ~1M
     - CPU
     - ``polars``
     - up to ~38x pandas, ~15x cuDF (Orkut 1-hop)
   * - Heavy multi-hop (2-hop+)
     - large
     - GPU
     - ``polars-gpu``
     - fastest until extreme materialization [F3]; GPU-or-error [F4]
   * - Full-graph aggregation
     - 100M+
     - GPU
     - ``polars-gpu`` / ``cudf``
     - GPU work-bound [F2]
   * - One very large single materialization
     - 80M+ output rows
     - GPU
     - ``cudf``
     - Polars-GPU can hit memory pressure here [F3]
   * - Anything
     - < ~1M
     - CPU
     - ``pandas``
     - fixed Polars/GPU overhead dominates [F1]
   * - Selective / seeded traversal
     - any
     - CPU
     - ``pandas``/``polars`` + **CSR index**
     - O(degree), not an engine choice [F5]

**[F1] Crossover ~1M edges.** Below it pandas often wins (Polars/GPU fixed overhead).
Above it Polars/GPU pull away. Some fast-path shapes (filter/lookup) cross over below 100K.

**[F2] GPU is work-bound, not size-bound.** A GPU wins when there is enough work to amortize
its ~3 ms kernel-launch floor: big frontiers, dense joins, full-graph aggregation. Tiny or
seeded work finishes faster on CPU.

**[F3] Polars-GPU memory pressure.** On an extreme single materialization (~85M output rows,
2-hop from 100K seeds on Orkut) raw ``cudf`` leads (6.0 s) and ``polars-gpu`` slips (8.6 s)
as its in-memory GPU executor comes under memory pressure. Prefer ``cudf`` for that regime.

**[F4] Polars-GPU is GPU-or-error.** It never silently falls back to CPU and reports the
result as a GPU run (see *Honesty* below).

**[F5] Selective traversal is an indexing problem, not an engine choice.** A seeded ``hop``
from a few nodes is fastest with the opt-in **CSR adjacency index** (``g.create_index(...)``,
``index_policy=``), which turns the O(E) scan into an O(degree) gather — on CPU, independent
of ``engine=``. (A dedicated index guide is in progress; the methods live under the GFQL API.)

cuDF vs Polars-GPU
------------------

Both run on an NVIDIA GPU, so which do you use?

- **cuDF is not deprecated.** It remains a first-class, supported engine and is the right
  choice for one very large materialization (footnote F3).
- **They execute differently.** ``cudf`` runs GFQL eagerly — each hop is a separate kernel
  launch with a materialized intermediate. ``polars-gpu`` runs the **same fused lazy plan as
  the CPU Polars engine**, collected once on the GPU. Fusing the plan is why ``polars-gpu``
  leads on heavy multi-hop and why even **CPU Polars often beats eager cuDF** on bulk work.
- **Frame type.** ``cudf`` operates on ``cudf.DataFrame``; ``polars-gpu`` operates on
  ``polars.DataFrame`` (only the lazy ``.collect()`` runs on the GPU). Either way, a graph
  built from pandas frames is accepted and coerced for you — only the keyword changes.
- **Install.** ``cudf`` and ``polars-gpu`` both need the RAPIDS GPU stack; ``polars-gpu``
  additionally uses ``cudf_polars``. ``polars`` (CPU) only needs ``pip install polars``.

When **not** to use Polars
--------------------------

Honesty matters more than a bigger number:

- **Small graphs (< ~1M edges):** ``pandas`` often wins; the Polars/GPU startup overhead is
  not worth it (footnote F1).
- **A few exotic Cypher features** are not yet native on Polars (e.g. cross-entity same-path
  ``WHERE``, some temporal/entity-text forms). They raise an honest ``NotImplementedError``
  pointing at ``engine='pandas'`` — GFQL **never** silently bridges Polars to pandas, because
  that would misreport pandas performance as Polars (see *Honesty*).
- **One extreme materialization (80M+ output rows):** prefer ``cudf`` over ``polars-gpu``
  (footnote F3).
- **vs graph databases:** GFQL-Polars beats embedded kuzu on frontier expansion (up to ~87x
  on LiveJournal 1-hop in our runs — reproducer ``benchmarks/gfql/index_vs_kuzu_prepared.py``),
  and separately beats Neo4j+GDS end-to-end (:doc:`benchmark_filter_pagerank`). The honest
  boundary: kuzu's worst-case-optimal joins target **cyclic / multi-way join** patterns
  (triangles, cliques) that we have **not** yet benchmarked, and kuzu may lead there.

Parity and honesty
------------------

- **Identical results across engines.** Differential parity — every engine's output must match
  the pandas oracle — is a release gate, exercised across forward/reverse/undirected, 1-3 hop,
  filters, and aggregations.
- **No silent fallback (NO-CHEATING).** The Polars engine runs natively or raises
  ``NotImplementedError`` — it never quietly converts to pandas. ``polars-gpu`` is
  **GPU-or-error**: if any step of the plan cannot run on the GPU it raises (pointing at
  ``engine='polars'``) rather than silently running on CPU and labelling it a GPU result.
  So any latency you measure is real work on the engine you asked for.

Methodology
-----------

- Host: ``dgx-spark`` (GB10 Grace-Blackwell, unified memory — the F3 memory-pressure
  boundary is partly a property of this box), RAPIDS container
  ``graphistry/test-rapids-official:26.02-gfql-polars``.
- Datasets: `SNAP <https://snap.stanford.edu/data/>`_ **com-LiveJournal** (35M edges),
  **com-Orkut** (117M edges). The order-of-magnitude filter/lookup figure is from a separate
  **LDBC SNB sf1** benchmark, not the table above.
- Measurement: **warm median** after 2 warmups (5 timed runs on Orkut, 8 on LiveJournal);
  every reported cell is **guarded** — the result rows are verified identical across engines
  before any timing is kept.
- Reproduce: ``benchmarks/gfql/index_bulk_olap_bench.py`` (engine comparison),
  ``benchmarks/gfql/pandas_vs_polars.py``, and ``benchmarks/gfql/index_vs_kuzu_prepared.py``
  (vs kuzu). Numbers on this page are rendered from saved runs; the page does not re-run them.

Install
-------

.. code-block:: bash

   pip install graphistry          # base; pandas engine works out of the box
   pip install graphistry polars   # adds the CPU 'polars' engine
   # 'cudf' and 'polars-gpu' require the NVIDIA RAPIDS stack (GPU);
   # 'polars-gpu' additionally uses cudf_polars.

Then change one keyword — your existing graph and query are unchanged:

.. code-block:: python

   import graphistry
   g = graphistry.edges(df, 'src', 'dst')          # your existing pandas, Polars, or cuDF graph
   g.gfql("MATCH (a)-[e]->(b) RETURN b", engine='polars')      # CPU columnar
   g.gfql("MATCH (a)-[e]->(b) RETURN b", engine='polars-gpu')  # same plan on GPU

Why opt-in?
-----------

Polars and Polars-GPU are explicit (``engine='polars'`` / ``'polars-gpu'``; ``auto`` never
picks them) for two honest reasons: (1) below the ~1M-edge crossover they can be *slower* than
pandas, so a blanket default would regress small/interactive workloads; and (2) a few exotic
Cypher features still require ``engine='pandas'`` and raise rather than silently bridge. Opting
in keeps the default behavior unchanged and makes the performance trade-off explicit.

See also
--------

- :doc:`performance` — GFQL performance overview
- :doc:`benchmark_filter_pagerank` — end-to-end CPU/GPU vs Neo4j+GDS
- :doc:`/api/gfql/index` — GFQL API reference
- :doc:`remote` — run GFQL on a remote GPU
