# Pyodide GFQL proof

This is a small `gfql.js` proof for running PyGraphistry GFQL inside Pyodide.

It uses:

- Pyodide `pandas`, `requests`, `packaging`, and `typing-extensions` packages.
- `micropip` for the pure Python `lark` runtime dependency used by the Cypher parser.
- A pure Python wheel for this repo, installed into Pyodide with `deps=False` after the runtime deps are already present.

For a browser URL wheel, `gfql.js` uses `micropip.install(url, deps=False)`. For a Node byte-mounted local wheel, it writes the wheel into Pyodide FS and extracts it into `purelib`; Pyodide/Node `fetch` does not resolve Pyodide FS paths as URLs.

Build a wheel from a writable copy of the repo:

```bash
rm -rf /tmp/pygraphistry-pyodide-src /tmp/pygraphistry-pyodide-dist
rsync -a --exclude .git --exclude plans --exclude uv.lock --exclude '=2' ./ /tmp/pygraphistry-pyodide-src/
uv run --no-project --with build python -m build --wheel --outdir /tmp/pygraphistry-pyodide-dist /tmp/pygraphistry-pyodide-src
```

Run the Node smoke proof:

```bash
rm -rf /tmp/pygraphistry-pyodide-node
npm install --prefix /tmp/pygraphistry-pyodide-node pyodide@314.0.0
PYODIDE_MODULE=/tmp/pygraphistry-pyodide-node/node_modules/pyodide/pyodide.mjs node demos/gfql/pyodide/run-node.mjs /tmp/pygraphistry-pyodide-dist/graphistry-0+unknown-py3-none-any.whl
```

The smoke uses `edges.csv` and validates both:

- AST GFQL: `e(edge_match={"weight": ge(2)})`, returning two filtered edges.
- Cypher parser path: `MATCH (a)-[e]->(b) WHERE e.weight >= 2 RETURN e`, returning two projected rows.

Both paths bind a small `id` nodes table derived from the CSV endpoints before running GFQL. That avoids pandas 3.0.2 concat edge cases in the current Pyodide runtime when Graphistry has to synthesize nodes.

Build a static Pyodide 314 bundle:

```bash
node demos/gfql/pyodide/build-bundle.mjs /tmp/pygraphistry-gfql-pyodide-bundle
```

The builder supports two flavors:

- `self-hosted`: copies Pyodide, Python stdlib, and required Pyodide wheels into
  the bundle. This is the most reproducible/offline option and is about 22 MiB.
- `cdn`: keeps only the demo files plus Graphistry/`lark` wheels, and loads the
  pinned Pyodide 314 runtime/packages from jsDelivr. This is the smallest hosted
  artifact and is about 1 MiB, but first cold load still downloads Pyodide and
  pandas from the CDN.

```bash
node demos/gfql/pyodide/build-bundle.mjs /tmp/gfql-cdn --flavor cdn
node demos/gfql/pyodide/build-bundle.mjs /tmp/gfql-self-hosted --flavor self-hosted
```

To generate the Read the Docs "Try it live" payload before a Sphinx HTML build:

```bash
node demos/gfql/pyodide/build-bundle.mjs --docs-static --flavor cdn
```

That writes the bundle to `docs/source/static/gfql/pyodide/`, which is ignored
by git because it contains generated docs artifacts and local wheels.

The bundle includes `gfql.js`, `browser.html`, `edges.csv`, `manifest.json`,
`size-report.json`, and wheels under `wheels/`. The `self-hosted` flavor also
includes `pyodide/`. Serve it with:

```bash
cd /tmp/pygraphistry-gfql-pyodide-bundle
python -m http.server 8000
```

Then open `http://localhost:8000/browser.html`.

Run the browser smoke:

```bash
npm install --prefix demos/gfql/pyodide --no-audit --no-fund
npm exec --prefix demos/gfql/pyodide -- playwright install chromium
node demos/gfql/pyodide/test-browser.mjs /tmp/pygraphistry-gfql-pyodide-bundle
```

Benchmark it:

```bash
GFQL_BENCH_SIZES=10,1000,10000 GFQL_BENCH_REPEAT=3 \
  node /tmp/pygraphistry-gfql-pyodide-bundle/benchmark-node.mjs \
  /tmp/pygraphistry-gfql-pyodide-bundle
```
