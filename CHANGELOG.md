# Changelog

All notable changes to the PyGraphistry are documented in this file. The PyGraphistry client and other Graphistry components are tracked in the main [Graphistry major release history documentation](https://graphistry.zendesk.com/hc/en-us/articles/360033184174-Enterprise-Release-List-Downloads).

The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and all PyGraphistry-specific breaking changes are explictly noted here.

## [Development]

### Added

* igraph handlers: `graphistry.from_igraph`, `g.from_igraph`, `g.to_igraph`
* docs: README.md examples of using new igraph methods

### Changed

* Deprecation warnings for old igraph methods: ...
* Internal igraph handlers upgraded to new igraph methods 

### Breaking

* `network2igraph` and `igraph2pandas` renamed output node ID column to `_n_implicit` (`constants.NODE`)

## [0.24.1 - 2022-04-29]

### Fixed

* Expose symbols for `.chain()` predicates as top-level: previous `ast` export was incorrect

## [0.24.0 - 2022-04-29]

### Added

* Use buildkit with pip install caching for test dockerfiles
* Graph AI branch: Autoencoding via dirty_cat and sentence_transformers (`g.featurize()`)
* Graph AI branch: UMAP via umap_learn (`g.umap()`)
* Graph AI branch: GNNs via DGL (`g.build_dgl_graph()`)
* `g.reset_caches()` to clear upload and compute caches (last 100)
* Central `setup_logger()`
* Official Python 3.10 support

### Changed

* Logging: Refactor to `setup_logger(__name__)`

### Fixed

* hypergraph: use default logger instead of DEBUG

## [0.23.3 - 2022-04-23]

### Added

* `g.collapse(node='root_id', column='some_col', attribute='some_val')

## [0.23.2 - 2022-04-11]

### Fixed

* Avoid runtime import exn when on GPU-less systems with cudf/dask_cudf installed

## [0.23.1 - 2022-04-08]

### Added

* Docs: `readme.md` digest of compute methods

### Fixed

* Docs: `get_degree()` -> `get_degrees()` (https://github.com/graphistry/pygraphistry/issues/330)
* Upload memoization handles column renames (https://github.com/graphistry/pygraphistry/issues/326)

## [0.23.0 - 2022-04-08]

### Breaking

* `g.edges()` now takes an optional 4th named parameter `edge` ID

Code that looks like `g.edges(some_fn, None, None, some_arg)` should now be like `g.edges(some_fn, None, None, None, some_arg)`

* Similar new optional `edge` ID parameter in `g.bind()`

### Changed

* `g.hop()` now takes optional `return_as_wave_front=False`, primarily for internal use by `chain()`

### Added

* `g.chain([...])` with `graphistry.ast.{n, e_forward, e_reverse, e_undirected}`

## [0.22.0 - 2022-04-06]

### Added

* Node dictionary-based filtering: `g.filter_nodes_by_dict({"some": "value", "another": 2})`
* Edge dictionary-based filtering: `g.filter_edges_by_dict({"some": "value", "another": 2})`
* Hops support edge filtering: `g.hop(hops=2, edge_match={"type": "transaction"})`
* Hops support pre-node filtering: `g.hop(hops=2, source_node_match={"type": "account"})`
* Hops support post-node filtering: `g.hop(hops=2, destination_node_match={"type": "wallet"})`
* Hops defaults to full graph if no initial nodes specified: `g.hop(hops=2, edge_match={"type": "transaction"})`

## [0.21.4 - 2022-03-30]

### Added

* Horizontal and radial axis using `.encode_axis(rows=[...])`

### Fixed

* Docs: Work around https://github.com/sphinx-doc/sphinx/issues/10291

## [0.21.0 - 2022-03-13]

### Added

* Better implementation of `.tree_layout(...)` using Sugiyama; good for small/medium DAGs
* Layout rotation method `.rotate(degree)`
* Compute method `.hops(nodes, hops, to_fixed_point, direction)`

### Changed

* Infra: `test-cpu-local-minimum.sh` accepts params

## [0.20.6 - 2022-03-12]

### Fixed

* Docs: Point color encodings

## [0.20.5 - 2021-12-06]

### Changed

* Unpin Networkx

### Fixed

* Docs: Removed deprecated `api=1`, `api=2` registration calls (#280 by @pradkrish)
* Docs: Fixed bug in honeypot nb (#279 by @pradkrish)
* Tests: Networkx test version sniffing

## [0.20.3 - 2021-11-21]

### Added

* Databricks notebook connector ([PR](https://github.com/graphistry/pygraphistry/pull/277))
* Databricks notebook + dashboard example ([PR](https://github.com/graphistry/pygraphistry/pull/277), [ipynb](https://github.com/graphistry/pygraphistry/blob/ad31a227136430bcd578feac1c18e90920ab4f00/demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.ipynb))

## [0.20.2 - 2021-10-18]

### Added

* Docs: [umap_learn tutorial notebook](https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/umap_learn/umap_learn.ipynb)

## [0.20.1 - 2021-08-24]

### Added

* Docs: Sharing control [demos/more_examples/graphistry_features/sharing_tutorial.ipynb](https://github.com/graphistry/pygraphistry/blob/master/demos/more_examples/graphistry_features/sharing_tutorial.ipynb)

## [0.20.0 - 2021-08-24]

### Added
* Feature: global `graphistry.privacy()` and compositional `Plotter.privacy()`
* Docs: How to use `privacy()`

### Changed

* Docs: Start removing deprecated 1.0 API docs

## [0.19.6 - 2021-08-15]

### Fixed

* Fix: NetworkX 2.5+ support - accept minor version tags

## [0.19.5 - 2021-08-15]

### Fixed

* Fix: igraph `.plot()` arrow coercion syntax error (https://github.com/graphistry/pygraphistry/issues/257)
* Fix: Lint duplicate import warning

### Changed

* CI: Treat lint warnings as CI failures


## [0.19.4 - 2021-07-22]

### Added
* Infra: Add CI stage that installs and tests with minimal core deps (https://github.com/graphistry/pygraphistry/issues/254)

### Fixed
* Fix: Core tests pass with minimal install dependencies (https://github.com/graphistry/pygraphistry/issues/253, https://github.com/graphistry/pygraphistry/issues/254)

## [0.19.3 - 2021-07-16]

### Added
* Feature: Compute methods `materialize_nodes`, `get_degrees`, `drop_nodes`, `get_topological_levels`
* Feature: Layout methods `tree_layout`, `layout_settings`
* Docs: New compute and layout methods

## [0.19.2 - 2021-07-14]

### Added
* Feature: `g.fetch_edges()` for neptune/gremlin edge attributes

### Fixed
* Fix: `g.fetch_nodes()` for neptune/gremlin node attrbutes

## [0.19.1 - 2021-07-09]

### Added
* Docs: [demos/demos_databases_apis/neptune/neptune_tutorial.ipynb](https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/neptune/neptune_tutorial.ipynb)

### Changed
* Docs: Updated [demos/for_analysis.ipynb](https://github.com/graphistry/pygraphistry/blob/master/demos/for_analysis.ipynb) to `api=3`

### Fixed
* Fix: Gremlin (Neptune) connector deduplicates nodes/edges

## [0.19.0 - 2021-07-09]

### Added
* Feature: Gremlin connector (GraphSONSerializersV2d0)
* Feature: Cosmos connector
* Feature: Neptune connector
* Feature: Chained composition operators:
  * `g.pipe((lambda g, a1, ...: g2), a1, ...)`
  * `g.edges((lambda g, a1, ...: df), None, None, a1, ...)`
  * `g.nodes((lambda g, a1, ...: df), None, a1, ...)`
* Feature: plotter::infer_labels: Guess node label names when not set, instead of defaulting to node_id. Runs during plots.
* Infra: Jupyter notebook: `cd docker && docker-compose build jupyter && docker-compose up jupyter`
* Docs: Neptune, Cosmos, chained composition

### Changed
* Refactor: Split out PlotterBase, interface Plottable

### Fixed
* Fix: Plotter has `hypergraph()`

## [0.18.2 - 2021-04-29]

### Added
* Docs: security.md

### Fixed

* Hypergraphs - detect and handle mismatching types across partitions

### Changed

* Infra: Speedup testing containers via incrementalization and docker settings
* Infra: Update testing container base builds

## [0.18.1 - 2021-03-26]

### Added

* Feature: Hypergraphs in dask, dask_cudf modes. Mixed nan support. (https://github.com/graphistry/pygraphistry/pull/225)
* Feature: Dask/dask_cuda frames can be passed in, which will be .computed(), memoized, and converted to arrow (https://github.com/graphistry/pygraphistry/pull/225)
* Infra: Test env var controls - WITH_LINT=1, WITH_TYPECHECK=1, WITH_BUILD=1 (https://github.com/graphistry/pygraphistry/pull/225)
* Docs: Inline hypergraph examples  (https://github.com/graphistry/pygraphistry/pull/225)

### Changed
* CI: Disable seccomp during test (docker perf) (https://github.com/graphistry/pygraphistry/pull/225)

## [0.18.0 - 2021-03-21]

### Added

* Feature: cudf mode for hypergraph (https://github.com/graphistry/pygraphistry/pull/224)
* Feature: pandas mode for hypergraph uses all-vectorized operations (https://github.com/graphistry/pygraphistry/pull/224)
* Infra: Engine class for picking dataframe engine - pandas/cudf/dask/dask_cudf (https://github.com/graphistry/pygraphistry/pull/224)
* CI: mypy type checking (https://github.com/graphistry/pygraphistry/pull/222)
* CI: GPU test harness (https://github.com/graphistry/pygraphistry/pull/223)

### Changed

* Hypergraph: Uses new pandas/cudf implementations (https://github.com/graphistry/pygraphistry/pull/224)

### Added

* Infra: Issue templates for bugs and feature requests

## [0.17.0 - 2021-02-08]

### Added

* Docs: Overhaul Sphinx docs - Update, clean all warnings, add to CI, reject commits that fail
* Docs: Setup.py (pypi) now includes full README.md
* Docs: Added ARCHITECTURE, CONTRIBUTE, and expanded DEVELOP
* Garden: DRY for CI + local dev via shared bin/ scripts
* Docker: Downgrade local dev 3.7 -> 3.6 to more quickly catch minimum version errors
* CI: Now tests building docs (fail on warnings), pypi wheels distro, and neo4j connector

### Breaking

* Changes in setup.py extras_require: 'all' installs more

## [0.16.3 - 2021-02-08]

### Added

* Docs: ARCHITECTURE.md and CONTRIBUTE.md

### Changed
* Quieted memoization fail warning
* CI: Removed TravisCI in favor of GHA
* CD: GHA now handles PyPI publish on tag push
* Docs: Readme install clarifies Python 3.6+
* Docs: Update DEVELOP.md dev flow

## [0.16.2 - 2021-02-08]

### Added

* Friendlier error message for calling .cypher(...) without setting BOLT auth/driver (https://github.com/graphistry/pygraphistry/issues/204) 
* CI: Run containerized neo4j connector tests
* Infrastructure: Set Python 3.9 support metadata

### Fixed

* Memoization: When memoize hashes throw exceptions, emit warning and fallback to unmemoized (b7a25c74e)

## [0.16.1] - 2021-02-07

### Added

* Friendlier error message for api=1, 2 server non-json responses (https://github.com/graphistry/pygraphistry/pull/187)
* CI: Moved to GitHub Actions for CI + optional manual publish
* CI: Added Python 3.9 to test matrix
* Infrastructure: Upgraded Versioneer to 0.19
* Infrastructure: Fewer warnings and enforce flake8 CI checks

### Breaking

* None known; many small changes to fix warnings so version bump out of caution

## [0.15.0] - 2021-01-11

### Added
* File API: Enable via `.plot(as_files=True)`. By default, auto-skips file re-uploads (disable via `.plot(memoize=False)`) for tables with same hash as those already uploaded in the same session. Use with `.register(api=3)` clients on Graphistry `2.34`+ servers. More details at  (https://github.com/graphistry/pygraphistry/pull/195) .
* Dev: More docs and logging as part of https://github.com/graphistry/pygraphistry/pull/195
* Auth service account docs in README.md (12.2.2020)

## [0.14.1] - 2020-11-16

### Added
* Examples for icons, badges, and new node/edge bindings
* graph-app-kit links
* Slack link

### Fixed
* Python test matrix: Removed 3.9
* Propagate misformatted etl1/2 server errors 

## [0.14.0] - 2020-10-12

### Breaking
* Warnings: Standardizing on Python's warnings.warn

### Fixed
* Neo4j: Improve handling of empty query results (https://github.com/graphistry/pygraphistry/issues/178)

### Added
* Icons: Add new as_text, blend_mode, border, and style options (Graphistry 2.32+)
* Badges: Add new badge encodings (Graphistry 2.32.+)
* Python 3.8, 3.9 in test matrix
* New binding shortcuts `g.nodes(df, col)` and `g.nodes(df, src_col, dst_col)`

### Changed
* Python 2.7: Removed __future__ (Python 2.7 has already been EOL so not breaking)
* Redid ipython detection
* Imports: Refactoring for more expected style
* Testing: Fixed most warnings in preperation for treating them as errors
* Testing: Integration tests against self-contained neo4j instance

## [0.13.0] - 2020-09-17

### Added

- Chainable methods `.addStyle()` and `.style()` in `api=3` for controlling foreground, background, logo, and page metadata. Requires Graphistry 2.31.10+ [08eddb8](https://github.com/graphistry/pygraphistry/pull/175/commits/08eddb8fe3ce8ebe66ad2773fc8ee57dfad2dc58)
- Chainable methods `.encode_[point|edge]_[color|icon|size]()` for more powerful *complex* encodings, and underlying generic handler `__encode()`. Requires Graphistry 2.31.10+ [f370ca8](https://github.com/graphistry/pygraphistry/pull/175/commits/f370ca82931e8fb61e40d62ba397b95e0d474f7f)
- More usage examples in README.md

### Changed

- Split `ArrowLoader::*encoding*`methods to `*binding*` vs. `*encoding*` ones to more precisely reflect the protocol. Not considered breaking as an internal method.

## [0.12.0] - 2020-09-08

### Added

- Neo4j 4 temporal and spatial type support - [#172](https://github.com/graphistry/pygraphistry/pull/172)
- CHANGELOG.md

### Changed
- Removed deprecated docker test harness in favor of `docker/` - [#172](https://github.com/graphistry/pygraphistry/pull/172)
