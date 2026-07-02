GFQL in Pyodide
================

This page shows how to run a small GFQL workload in JavaScript with
`Pyodide <https://pyodide.org/>`__.

Why Pyodide 314?
----------------

Pyodide 314 aligns Pyodide versioning with Python 3.14, ships as a native ES
module runtime, and standardizes package publishing around PyEmscripten wheels.
That makes it a good baseline for a browser-side GFQL proof: the JavaScript
entrypoint can import Pyodide as an ES module, load Pyodide-native packages
such as ``pandas`` and ``pyarrow``, and install pure Python wheels at runtime.

For this GFQL demo, the important runtime pieces are:

- Pyodide packages: ``micropip``, ``pandas``, ``requests``, ``packaging``, and
  ``typing-extensions``.
- Pure Python wheels: ``lark`` for the local Cypher parser.
- A pure Python wheel for this repository.

The browser demo intentionally avoids the top-level ``graphistry.edges(...)``
constructor and uses a small in-Pyodide graph object backed by
``ComputeMixin``. That keeps the live demo on a pandas-only path. ``pyarrow``
is still useful for normal PyGraphistry upload and Arrow workflows, but it is
not loaded for this CSV/GFQL browser proof.

Build the bundle
----------------

From the repository root:

.. code-block:: bash

   node demos/gfql/pyodide/build-bundle.mjs /tmp/pygraphistry-gfql-pyodide-bundle

Choose a bundle flavor:

- ``self-hosted`` copies Pyodide, Python stdlib, and required Pyodide wheels
  into the bundle. It is the most reproducible/offline option.
- ``cdn`` publishes only the demo files plus Graphistry/``lark`` wheels and
  loads the pinned Pyodide 314 runtime/packages from jsDelivr. It keeps the
  hosted artifact small, but first cold load still downloads Pyodide and
  ``pandas`` from the CDN.

.. code-block:: bash

   node demos/gfql/pyodide/build-bundle.mjs /tmp/gfql-cdn --flavor cdn
   node demos/gfql/pyodide/build-bundle.mjs /tmp/gfql-self-hosted --flavor self-hosted

For Read the Docs, build the small CDN-backed flavor into the Sphinx static tree
before the HTML build:

.. code-block:: bash

   node demos/gfql/pyodide/build-bundle.mjs --docs-static --flavor cdn

This writes the live demo under ``docs/source/static/gfql/pyodide/``. The
directory is ignored by git because it contains generated wheels and static
assets; publish it as a generated docs artifact, not as checked-in source.
The Read the Docs build runs this command before the HTML build so the live
page is included in the published ``_static`` directory.

The builder:

1. Copies the repo to a temporary build directory.
2. Builds a pure Python ``graphistry`` wheel.
3. Downloads wheels that Pyodide does not ship directly, currently ``lark``.
4. Installs ``pyodide@314.0.0``, resolves the required Pyodide package closure
   from the Pyodide lockfile, and verifies checksums for downloaded wheels.
5. Writes a static bundle containing ``pyodide/``, ``gfql.js``, ``edges.csv``,
   ``browser.html``, ``manifest.json``, ``size-report.json``, and wheels under
   ``wheels/``. The ``cdn`` flavor omits ``pyodide/`` and points the manifest
   at the pinned CDN runtime instead.
6. Prunes non-runtime Pyodide files such as source maps, legacy console pages,
   and TypeScript declarations.

Try it live
-----------

When the generated static bundle is published with the docs, open the live
browser demo:

.. raw:: html

   <p>
     <a class="reference external" href="../_static/gfql/pyodide/browser.html">
       Try GFQL in your browser
     </a>
   </p>

The live page runs entirely in the browser: it loads Pyodide, installs the
bundled Graphistry wheel and small pure-Python dependencies, reads the sample
CSV, then executes both a native GFQL predicate query and a Cypher-style
``MATCH`` query. The page includes editable CSV, editable Cypher, rendered
tables, raw JSON, and per-step timing.

If the link returns 404 in a local docs build, generate the static bundle first
with ``node demos/gfql/pyodide/build-bundle.mjs --docs-static`` and rebuild the
HTML docs.

Run the browser tutorial
------------------------

Serve the generated directory as static files:

.. code-block:: bash

   cd /tmp/pygraphistry-gfql-pyodide-bundle
   python -m http.server 8000

Then open:

.. code-block:: text

   http://localhost:8000/browser.html

The page loads ``pyodide/pyodide.mjs``, installs the local wheels listed in
``manifest.json``, loads ``edges.csv``, and runs:

.. code-block:: javascript

   const astResult = await runtime.runEdgeWeightAtLeast({ csv, minWeight: 2 });
   const cypherResult = await runtime.runCypherCsv({
     csv,
     query: "MATCH (a)-[e]->(b) WHERE e.weight >= 2 RETURN e",
   });

The expected output is:

.. code-block:: json

   {
     "astEdges": [
       {"src": "bob", "dst": "carol", "weight": 2},
       {"src": "alice", "dst": "carol", "weight": 3}
     ],
     "cypherRows": [
       {"e": "[{weight: 2}]"},
       {"e": "[{weight: 3}]"}
     ]
   }

Run the Node smoke
------------------

The same bundle can be checked in Node:

.. code-block:: bash

   PYODIDE_MODULE=/tmp/pygraphistry-gfql-pyodide-bundle/pyodide/pyodide.mjs \
     node /tmp/pygraphistry-gfql-pyodide-bundle/run-node.mjs \
     /tmp/pygraphistry-gfql-pyodide-bundle/wheels/graphistry-0+unknown-py3-none-any.whl

Benchmark and size report
-------------------------

The builder writes ``size-report.json``. Recent local builds of this demo
reported:

.. list-table::
   :header-rows: 1

   * - Flavor
     - Bundle part
     - Approximate bytes
     - Approximate size
   * - ``cdn``
     - Total static bundle
     - 975,000
     - 0.9 MiB
   * - ``cdn``
     - Pyodide runtime and package cache
     - 0
     - CDN-backed
   * - ``cdn``
     - Graphistry and ``lark`` wheels
     - 943,000
     - 0.9 MiB
   * - ``self-hosted``
     - Total static bundle
     - 23,126,000
     - 22.0 MiB
   * - ``self-hosted``
     - Pyodide runtime and package cache
     - 22,151,080
     - 21.1 MiB
   * - ``self-hosted``
     - Graphistry and ``lark`` wheels
     - 943,000
     - 0.9 MiB

Run the benchmark:

.. code-block:: bash

   node /tmp/pygraphistry-gfql-pyodide-bundle/benchmark-node.mjs \
     /tmp/pygraphistry-gfql-pyodide-bundle

Set ``GFQL_BENCH_SIZES`` and ``GFQL_BENCH_REPEAT`` to change the workload:

.. code-block:: bash

   GFQL_BENCH_SIZES=10,1000,10000 GFQL_BENCH_REPEAT=3 \
     node /tmp/pygraphistry-gfql-pyodide-bundle/benchmark-node.mjs \
     /tmp/pygraphistry-gfql-pyodide-bundle

Example local Node timings after one warmup run:

.. list-table::
   :header-rows: 1

   * - Edges
     - Native GFQL median
     - Cypher median
     - Returned rows
   * - 10
     - 43.0 ms
     - 95.9 ms
     - 4
   * - 1,000
     - 51.2 ms
     - 105.7 ms
     - 400
   * - 10,000
     - 138.2 ms
     - 179.4 ms
     - 4,000

In the same run, creating the Pyodide GFQL runtime took about 6.82 seconds.
Browser numbers will vary with network, cache state, CPU, and whether the
server compresses static assets.

Run the browser smoke
---------------------

The browser smoke starts a local static server, opens ``browser.html`` in
Chromium, waits for the auto-run output, and checks that both native GFQL and
Cypher return the expected rows:

.. code-block:: bash

   npm install --prefix demos/gfql/pyodide --no-audit --no-fund
   npm exec --prefix demos/gfql/pyodide -- playwright install chromium
   node demos/gfql/pyodide/build-bundle.mjs /tmp/gfql-cdn --flavor cdn
   node demos/gfql/pyodide/test-browser.mjs /tmp/gfql-cdn

CI runs the same smoke against the ``cdn`` flavor. A recent cold CDN browser
run took about 46.6 seconds to create the Pyodide runtime, then 78 ms for the
native GFQL query and 487 ms for the Cypher query. Warm browser/cache behavior
should be faster; keep release numbers benchmark-driven.

Hosting and versioning
----------------------

The generated RTD demo uses the ``cdn`` flavor under
``_static/gfql/pyodide/``. That keeps the published docs artifact small while
pinning all Pyodide URLs to ``v314.0.0``.

For standalone apps, two patterns are reasonable:

- **Self-hosted bundle**: publish the generated directory with the app or docs.
  This is the most reproducible option and works offline after the first page
  load if the browser cache keeps the assets.
- **Pinned CDN runtime**: load Pyodide from
  ``https://cdn.jsdelivr.net/pyodide/v314.0.0/full/`` and host only
  ``browser.html``, ``gfql.js``, the Graphistry wheel, ``lark``, and the
  manifest. Pyodide's docs list this versioned JsDelivr URL as the cached
  release CDN path.

In either mode, keep ``manifest.json`` pinned to the Pyodide version and the
Graphistry wheel built for the release. Static hosts must serve ``.wasm`` with
the WebAssembly MIME type; Python's local ``http.server`` and many static hosts
do this correctly.

Implementation notes
--------------------

- ``gfql.js`` accepts a browser URL wheel or byte-mounted wheel data. URL
  wheels use ``micropip.install(url, deps=False)``. Byte-mounted local wheels
  are extracted into Pyodide ``purelib`` after validating wheel member paths.
- ``gfql.js`` also accepts byte-mounted dependency wheels. In Node, those are
  installed through Pyodide's ``emfs:`` wheel URI support.
- The bundle manifest points ``lark`` at a local wheel and records the resolved
  Pyodide package closure. The ``self-hosted`` flavor serves those wheels from
  ``pyodide/``; the ``cdn`` flavor serves them from the pinned Pyodide CDN.
- The demo binds a small ``id`` nodes table derived from CSV endpoints before
  running GFQL. This avoids pandas 3.0 concat edge cases in Pyodide when
  Graphistry has to synthesize nodes.
- The Cypher example intentionally uses ``RETURN e`` because broader
  multi-alias Cypher projection is outside the currently supported local GFQL
  compiler subset.
