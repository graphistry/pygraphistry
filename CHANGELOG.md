# Changelog

All notable changes to the PyGraphistry are documented in this file. The PyGraphistry client and other Graphistry components are tracked in the main [Graphistry major release history documentation](https://graphistry.zendesk.com/hc/en-us/articles/360033184174-Enterprise-Release-List-Downloads).

The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and all PyGraphistry-specific breaking changes are explictly noted here.

## Dev

### Added
* GFQL temporal predicates and type system for date/time comparisons
  * Support for datetime, date, and time comparisons with operators: `gt`, `lt`, `ge`, `le`, `eq`, `ne`, `between`, `is_in`
  * Proper timezone handling for datetime comparisons
  * Type-safe temporal value handling with TypeGuard annotations
  * Temporal value classes: `DateTimeValue`, `DateValue`, `TimeValue` for explicit temporal types
  * Wire protocol support for JSON serialization of temporal predicates
  * Comprehensive documentation: datetime filtering guide, wire protocol reference, and examples notebook

### Breaking
* Plottable is now a Protocol
* py.typed added, type checking active on PyGraphistry!
* transform() and transform_umap() now require some parameters to be keyword-only

## [0.38.0 - 2025-06-17]

### Changed
* PyPI publish workflow now uses Trusted Publishing (OIDC) instead of password authentication

## [0.38.0 - 2025-06-17]

### Feat
* Kusto/Azure Data Explorer integration. `PyGraphistry.kusto()`, `kusto_query()`, `kusto_query_graph()`
* Extra kusto install target `pip install graphistry[kusto]` installs azure-kusto-data, azure-identity

### Fixed
* Fix sentence transformer model name handling to support both legacy format and new organization-prefixed formats (e.g., `mixedbread-ai/mxbai-embed-large-v1`)

### Changed
* Legacy `Plottable.spanner_init()` & `PyGraphistry.spanner_init()` helpers no longer shipped. Use `spanner()`

### Breaking
* Kusto device authentication doesn't persist.

### Test
* Add comprehensive tests for sentence transformer model name formats including legacy, organization-prefixed, and local path formats

## [0.37.0 - 2025-06-05]

### Fixed

* Fix embed_utils.py modifying global logging.StreamHandler.terminator ([#660](https://github.com/graphistry/pygraphistry/issues/660)) ([8480cd06](https://github.com/graphistry/pygraphistry/commit/8480cd06))

### Breaking 🔥

* `FeatureMixin.transform()` now raises `ValueError` for invalid `kind` parameter instead of silently continuing ([25e4bf51](https://github.com/graphistry/pygraphistry/commit/25e4bf51))
* `FeatureMixin._transform()` now raises `ValueError` when encoder is not initialized instead of returning `None` ([25e4bf51](https://github.com/graphistry/pygraphistry/commit/25e4bf51))
* `UMAPMixin.transform_umap()` now always returns `pd.DataFrame` (possibly empty) instead of `None` for `y_` in tuple return ([d2941ec4](https://github.com/graphistry/pygraphistry/commit/d2941ec4))

### Chore

* Switch to setup_logger utility in multiple modules ([842fb904](https://github.com/graphistry/pygraphistry/commit/842fb904))
* Add AI_PROGRESS/ and PLAN.md to .gitignore ([f0c18b3b](https://github.com/graphistry/pygraphistry/commit/f0c18b3b), [ac25a356](https://github.com/graphistry/pygraphistry/commit/ac25a356))

### Docs

* Add AI assistant prompt templates and conventional commits guidance ([a52048a7](https://github.com/graphistry/pygraphistry/commit/a52048a7))
* Simplify CLAUDE.md to point to ai_code_notes README ([e5393381](https://github.com/graphistry/pygraphistry/commit/e5393381))
* Update AI assistant documentation with Docker-first testing ([db5496eb](https://github.com/graphistry/pygraphistry/commit/db5496eb))

## [0.36.2 - 2025-05-16]

### Feat

* GFQL: Hop pattern matching now supports node ID column having same name as edge source or destination column

### Perf

* GFQL: Optimize hop operations with improved memory usage and reduced redundancy 

### Test

* GFQL: Comprehensive tests for column name conflicts in chain pattern matching

### Infra

* Add CLAUDE.md with performance guidelines

## [0.36.1 - 2025-04-17]

### Feat

* Add "erase_files_on_fail" option to plot and upload functions

## [0.36.0 - 2025-02-05]

### Breaking

* `from_cugraph` returns using the src/dst bindings of `cugraph.Graph` object instead of base `Plottable`
* `pip install graphistry[umap-learn]` and `pip install graphistry[ai]` are now Python 3.9+ (was 3.8+)
* `Plottable`'s fields `_node_dbscan` / `_edge_dbscan` are now `_dbscan_nodes` / `_dbscan_edges`

### Feat

* Switch to `skrub` for feature engineering
* More AI methods support GPU path
* Support cugraph 26.10+, numpy 2.0+
* Add more umap, dbscan fields to `Plottable`

### Infra

* `[umap-learn]` + `[ai]` unpin deps - scikit, scipy, torch (now 2), etc

### Refactor

* Move more type models to models/compute/{feature,umap,cluster}
* Turn more print => logger

### Fixes

* Remove lint/type ignores and fix root causes

### Tests

* Stop ignoring warnings in featurize and umap
* python version tests use corresponding python version for mypy
* ci umap tests: py 3.8, 3.9 => 3.9..3.12
* ci ai tests: py 3.8, 3.9 => 3.9..3.12
* ci tests dgl
* plugin tests check for module imports

## [0.35.10 - 2025-01-24]

### Fixes: 

* Spanner: better handling of spanner_config issues: #634, #644

## [0.35.9 - 2025-01-22]

### Docs 

* Spanner: minor changes to html and markdown in notebook for proper rendering in readthedocs 

## [0.35.8 - 2025-01-22]

### Docs

* Spanner: fix for plots rendering in readthedocs demo notebooks


## [0.35.7 - 2025-01-22]

### Feat 

* added support for Google Spanner Graph and Google Spanner `spanner_gql_to_g` and `spanner_query_to_df` 
* added new Google Spanner Graph demo notebook 

## [0.35.6 - 2025-01-11]

### Docs

* Fix typo in new shaping tutorial
* Privacy-preserving analytics

## [0.35.5 - 2025-01-10]

### Docs

* New tutorial on graph shaping

## [0.35.4 - 2024-12-28]

### Fixes

* `Plottable._render` now typed as `RenderModesConcrete`
* Remote GFQL - Handle `output_type is None` 

## [0.35.3 - 2024-12-24]

### Docs

* Rename CONTRIBUTE.md to CONTRIBUTING.md to match OSS standards (snyk)
* setup.py: add project_urls
* Add FUNDING.md
* Add CODE_OF_CONDUCT.md

### Feat

* GFQL serialization: Edge AST node deserializes as more precise `ASTEdge` subclasses

### Fixes

* GFQL Hop: Detect #614 of node id column name colliding with edge src/dst id column name and raise `NotImplementedError`
* MyPy: Remove explicit type annotation from Engine

## [0.35.2 - 2024-12-13]

### Docs

* Python remote mode notebook: Fixed engine results
* Python remote mode: Add JSON example

## [0.35.1 - 2024-12-11]

### Fixes

* Fix broken imports in new GFQL remote mode

## [0.35.0 - 2024-12-10]

### Docs

* New tutorial on GPU memory management and capacity planning in the GPU section
* New tutorial on remote-mode GFQL
* New tutorial on remote-mode Python

### Feat

* `plot(render=)` supports literal-typed mode values: `"auto"`, `"g"`, `"url"`, `"ipython"`, `"databricks"`, where `"g"` is a new Plottable
* Remote metadata: Expose and track uploaded `._dataset_id`, `._url`, `._nodes_file_id`, and `._edges_file_id`
* Remote upload: Factor out explicit upload method `g2 = g1.upload(); assert all([g2._dataset_id, g2._nodes_file_id, g2._edges_file_id])` from plot interface
* Remote dataset: Remote dataset binding via `g1 = graphistry.bind(dataset_id='abc123')`
* Remote GFQL: Remote GFQL calls via `g2 = g1.chain_remote([...])` and `meta_df = g1.chain_remote_shape([...])`
* Remote GPU Python: Remote Python calls via `g2 = g1.python_remote_g(...)`, `python_remote_table()`, and `python_remote_json()` for different return types

### Changed

* `plot(render=)` now `Union[bool, RenderMode]`, not just `bool` 

## [0.34.17 - 2024-10-20]

### Added

* Layout: `circle_layout()` that moves points to one or more rings based on an ordinal field such as degree
* Layout: `fa2_layout()` that uses the ForceAtlas2 algorithm to lay out the graph, with some extensions.
* Layout: `group_in_a_box_layout()` is significantly faster in CPU and GPU mode
* Layout: `group_in_a_box_layout()` exposes attribute `_partition_offsets`
* Compute methods `g.to_pandas()` and `g.to_cudf()`

### Fix

* cugraph methods now handle numerically-typed node/edge index columns
* repeat calls to `fa2_layout` and `group_in_a_box_layout` now work as expected

### Docs

* Add group-in-a-box layout tutorial

### Infra

* Expose `scene_settings` in `Plottable`

## [0.34.16 - 2024-10-13]

### Docs

* Update and streamline readme.md
* Add quicksheet for overall
* More crosslinking

### Infra

* Add markdown support to docsite
* ReadTheDocs homepage reuses github README.md
* Docs pip install caches
* Drop SVGs and external images during latexpdf generation

### Changed

* Treemap import `squarify` deferred to use to allow core import without squarify installed, such as in `--no-deps`

## [0.34.15 - 2024-10-11]

### Docs

* Improve GFQL translation doc
* Add examples and API links: Shaping, Hypergraphs, AI & ML
* Add performance docs
* Add AI examples

## [0.34.14 - 2024-10-09]

### Added

* HTTP responses with error status codes log an `logging.ERROR`-level message of the status code and response body

## [0.34.13 - 2024-10-07]

### Docs

* Add more GFQL cross-references

## [0.34.12 - 2024-10-07]

### Docs

* Fix ipynb examples in ReadTheDocs distribution

## [0.34.11 - 2024-10-07]

### Fix

* Types

### Infra

* Enable more Python version checks

## [0.34.10 - 2024-10-07]

### Fix

* Docs: Notebook builds

### Docs

* More links, especially around plugins
* Update color theme to match Graphistry branding

## [0.34.9 - 2024-10-07]

### Fix

* Docs: 10 Minutes to PyGraphistry links

## [0.34.8 - 2024-10-06]

### Fix

* Docs: PDF support
* Docs: Links

### Docs

* More accessible theme

## [0.34.7 - 2024-10-06]

### Docs

* RTD: Added notebook tutorials 
* RTD: Added various guides
* RTD: Added cross-references
* RTD: Cleaner navigation

### Infra

* Python: Add Python 12 to CI and document support
* Docs: Udated dependencies - Sphinx 8, Python 12, and various related
* Docs: Added nbsphinx - hub url grounding, ...
* Docs: Redo as a docker compose flow with incremental builds (docker, sphinx)
* Docs: Updated instructions for new flow

### Fix

* Docs: 2024
* Notebooks: Compatibility with nbsphinx - exactly one title heading, no uncommented `!`, correct references, ...

## [0.34.6 - 2024-10-04]

### Added

* Plugins: graphviz bindings, such as `g.layout_graphviz("dot")`

### Docs

* Reorganized readthedocs
* Added intro tutorials: `10 Minutes to PyGraphistry`, `10 Minutes to GFQL`, `Login and Sharing`

## [0.34.5 - 2024-09-23]

### Fixed

* GFQL: Fix `chain()` regression around an incorrectly disabled check manifesting as https://github.com/graphistry/pygraphistry/issues/583
* GFQL: Fix `chain()`, `hop()` traverse filtering logic for a multi-hop edge scenarios
* GFQL: Fix `hop()` predicate handling in multihop scenarios

### Infra

* GFQL: Expand test suite around multihop edge predicates in `hop()` and `chain()`

## [0.34.4 - 2024-09-20]

### Added

* UMAP: Optional kwargs passthrough to umap library constructor, fit, and transform methods: `g.umap(..., umap_kwargs={...}, umap_fit_kwargs={...}, umap_transform_kwargs={...})`
* Additional GPU support in featurize paths

### Changed

* Replace `verbose` with `logging`

### Refactor

* Narrow `use_scaler` and `use_scaler_target` typing to `ScalerType` (`Literal[...]`) vs `str`
* Rename `featurize_or_get_nodes_dataframe_if_X_is_None` (and edges variant) as non-private due to being shared

### Fixed

* get_indegrees: Fix warning https://github.com/graphistry/pygraphistry/issues/587

## [0.34.3 - 2024-08-03]

### Added

* Layout `modularity_weighted_layout` that uses edge weights to more strongly emphasize community structure

### Docs

* Tutorial for `modularity_weighted_layout`

### Infra

* Upgrade tests to`docker compose` from `docker-compose` 
* Remove deprecated `version` to address warnings

## [0.34.2 - 2024-07-22]

### Fixed

* Graceful CPU fallbacks: When lazy GPU dependency imports throw `ImportError`, commonly seen due to broken CUDA environments or having CUDA libraries but no GPU, warn and fall back to CPU.

* Ring layouts now support filtered inputs, giving expected positions

* `encode_axis()` updates are now functional, not inplace

### Changed

* Centralize lazy imports into `graphistry.utils.lazy_import`
* Lazy imports distinguish `ModuleNotFound` (=> `False`) from `ImportError` (warn + `False`)

## [0.34.1 - 2024-07-17]

### Infra

* Upgrade pypi automation to py3.8

## [0.34.0 - 2024-07-17]

### Added

* Ring layouts: `ring_categorical_layout()`, `ring_continuous_layout()`, `time_ring_layout()`
* Plottable interface includes `encode_axis()`, `settings()`
* Minimal global config manager

### Infra

* Test GPU infra updated to Graphistry 2.41 (RAPIDS 23.10, CUDA 11.8)
* Faster test preamble
* More aggressive low-memory support in GPU UMAP unit tests 

### Fixed

* cudf materialize nodes auto inference
* workaround feature_utils typecheck fail 

### Docs

* Ring layouts

### Breaking 🔥

* Dropping support for Python 3.7 (EOL)

## [0.33.9 - 2024-07-04]

### Added

* Added `personalized_pagerank` to the list of supported `compute_igraph` algorithms.

## [0.33.8 - 2024-04-30]

### Fixed

* Fix from_json when json object contains predicates.

## [0.33.7 - 2024-04-06]

### Fixed

* Fix refresh() for SSO

## [0.33.6 - 2024-04-05]

### Added

* `featurize()`, on error, coerces `object` dtype cols to `.astype(str)` and retries

## [0.33.5 - 2024-03-11]

### Fixed

* Fix upload-time validation rejecting graphs without a nodes table

## [0.33.4 - 2024-02-29]

### Added

* Fix validations import.

## [0.33.3 - 2024-02-28]

### Added

* Validations for dataset encodings.

## [0.33.2 - 2024-02-24]

### Added

* GFQL: Export shorter alias `e` for `e_undirected`
* Featurize: More auto-dropping of non-numerics when no `dirty_cat`

### Fixed

* GFQL: `hop()` defaults to `debugging_hop=False`
* GFQL: Edge cases around shortest-path multi-hop queries failing to enrich against target nodes during backwards pass

### Infra

* Pin test env to work around test fails: `'test': ['flake8>=5.0', 'mock', 'mypy', 'pytest'] + stubs + test_workarounds,` +  `test_workarounds = ['scikit-learn<=1.3.2']`
* Skip dbscan tests that require umap when it is not available

## [0.33.0 - 2023-12-26]

### Added

* GFQL: GPU acceleration of `chain`, `hop`, `filter_by_dict`
* `AbstractEngine`  to `engine.py::Engine` enum
* `compute.typing.DataFrameT` to centralize df-lib-agnostic type checking

### Refactor

* GFQL and more of compute uses generic dataframe methods and threads through engine

### Infra

* GPU tester threads through LOG_LEVEL

## [0.32.0 - 2023-12-22]

### Added

* GFQL `Chain` AST object
* GFQL query serialization - `Chain`, `ASTObject`, and `ASTPredict` implement `ASTSerializable`
  - Ex:`Chain.from_json(Chain([n(), e(), n()]).to_json())`
* GFQL predicate `is_year_end`

### Docs

* GFQL in readme.md

### Changes

* Refactor `ASTEdge`, `ASTNode` field naming convention to match other `ASTSerializable`s

### Breaking 🔥

* GFQL `e()` now aliases `e_undirected` instead of the base class `ASTEdge`

## [0.31.1 - 2023-12-05]

### Docs

* Update readthedocs yml to work around ReadTheDocs v2 yml interpretation regressions
* Make README.md pass markdownlint
* Switch markdownlint docker channel to official and pin

## [0.30.0 - 2023-12-04]

### Added

* Neptune: Can now use PyGraphistry OpenCypher/BOLT bindings with Neptune, in addition to existing Gremlin bindings
* chain/hop: `is_in()` membership predicate, `.chain([ n({'type': is_in(['a', 'b'])}) ])`
* hop: optional df queries - `hop(..., source_node_query='...', edge_query='...', destination_node_query='...')`
* chain: optional df queries:
  - `chain([n(query='...')])`
  - `chain([e_forward(..., source_node_query='...', edge_query='...', destination_node_query='...')])`
* `ASTPredicate` base class for filter matching
* Additional predicates for hop and chain match expressions:
  - categorical: is_in (example above), duplicated
  - temporal: is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end, is_leap_year
  - numeric: gt, lt, ge, le, eq, ne, between, isna, notna
  - str: contains, startswith, endswith, match, isnumeric, isalpha, isdigit, islower, isupper, isspace, isalnum, isdecimal, istitle, isnull, notnull

### Fixed

* chain/hop: source_node_match was being mishandled when multiple node attributes exist
* chain: backwards validation pass was too permissive; add `target_wave_front` check`
* hop: multi-hops with `source_node_match` specified was not checking intermediate hops
* hop: multi-hops reverse validation was mishandling intermediate nodes
* compute logging no longer default-overrides level to DEBUG

### Infra

* Docker tests support LOG_LEVEL

### Changed

* refactor: move `is_in`, `IsIn` implementations to `graphistry.ast.predicates`; old imports preserved
* `IsIn` now implements `ASTPredicate`
* Refactor: use `setup_logger(__name__)` more consistently instead of `logging.getLogger(__name__)`
* Refactor: drop unused imports
* Redo `setup_logger()` to activate formatted stream handler iff verbose / LOG_LEVEL

### Docs

* hop/chain: new query and predicate forms
* hop/chain graph pattern mining tutorial: [ipynb demo](demos/more_examples/graphistry_features/hop_and_chain_graph_pattern_mining.ipynb)
* Neptune: Initial tutorial for using PyGraphistry with Amazon Neptune's OpenCypher/BOLT bindings

## [0.29.7 - 2023-11-02]

### Added

* igraph: support `compute_igraph('community_optimal_modularity')`
* igraph: `compute_igraph('articulation_points')` labels nodes that are articulation points

### Fixed

* Type error in arrow uploader exception handler
* igraph: default coerce Graph-type node labels to strings, enabling plotting of g.compute_igraph('k_core')
* igraph: fix coercions when using numeric IDs that were confused by igraph swizzling

### Infra

* dask: Fixed parsing error in hypergraph dask tests
* igraph: Ensure in compute_igraph tests that default mode results coerce to arrow tables
* igraph: Test chaining
* tests: mount source folders to enable dev iterations without rebuilding

## [0.29.6 - 2023-10-23]

### Docs

* Memgraph: Add tutorial (https://github.com/graphistry/pygraphistry/pull/507 by https://github.com/karmenrabar)

### Fixed

* Guard against potential `requests`` null dereference in uploader error handling

### Security

* Add control `register(..., sso_opt_into_type='browser' | 'display' | None)`
* Fix display of SSO URL

## [0.29.5 - 2023-08-23]

### Fixed

* Lint: Update flake8 in test
* AI: UMAP ignores reserved columns and fewer exceptions on low dimensionaltiy

## [0.29.4 - 2023-08-22]

### Fixed

* Lint: Dynamic type checks

### Infra

* Adding Python 3.10, 3.11 to more of test matrix
* Unpin `setuptools` and `pandas`
* Fix tests that were breaking on pandas 2+

## [0.29.3 - 2023-07-19]

### Changed

* igraph: Change dependency to new package name before old deprecates (https://github.com/graphistry/pygraphistry/pull/494)

## [0.29.2 - 2023-07-10]

### Fixed

* Security: Allow token register without org
* Security: Refresh logic

## [0.29.1 - 2023-05-03]

### Fixed 

* AI: cuml OOM fix [#482](https://github.com/graphistry/pygraphistry/pull/482)

## [0.29.0 - 2023-05-01]

### Breaking 🔥
* AI: moves public `g.g_dgl` from KG `embed` method to private method `g._kg_dgl`
* AI: moves public `g.DGL_graph` to private attribute `g._dgl_graph`
* AI: To return matrices during transform, set the flag:  `X, y = g.transform(df, return_graph=False)` default behavior is ~ `g2 = g.transform(df)` returning a `Plottable` instance. 

## [0.28.7 - 2022-12-22]

### Added
* AI: all `transform_*` methods return graphistry Plottable instances, using an infer_graph method. To return matrices, set the `return_graph=False` flag. 
* AI: adds `g.get_matrix(**kwargs)` general method to retrieve (sub)-feature/target matrices
* AI: DBSCAN -- `g.featurize().dbscan()` and `g.umap().dbscan()` with options to use UMAP embedding, feature matrix, or subset of feature matrix via `g.dbscan(cols=[...])`
* AI: Demo cleanup using ModelDict & new features, refactoring demos using `dbscan` and `transform` methods.
* Tests: dbscan tests
* AI: Easy import of featurization kwargs for `g.umap(**kwargs)` and `g.featurize(**kwargs)`
* AI: `g.get_features_by_cols` returns featurized submatrix with `col_part` in their columns
* AI: `g.conditional_graph` and `g.conditional_probs` assessing conditional probs and graph
* AI Demos folder: OSINT, CYBER demos
* AI: Full text & semantic search (`g.search(..)` and `g.search_graph(..).plot()`)
* AI: Featurization: support for dataframe columns that are list of lists -> multilabel targets
                  set using `g.featurize(y=['list_of_lists_column'], multilabel=True,...)`
* AI: `g.embed(..)` code for fast knowledge graph embedding (2-layer RGCN) and its usage for link scoring and prediction
* AI: Exposes public methods `g.predict_links(..)` and `g.predict_links_all()`
* AI: automatic naming of graphistry objects during `g.search_graph(query)` -> `g._name = query`
* AI: RGCN demos - Infosec Jupyterthon 2022, SSH anomaly detection

### Fixed

* GIB: Add missing import during group-in-a-box cudf layout of 0-degree nodes
* Tests: SSO login tests catch more unexpected exns

## [0.28.6 - 2022-11-29]


### Added

* Personal keys: `register(personal_key_id=..., personal_key_secret=...)`
* SSO: `register()` (no user/pass), `register(idp_name=...)` (org-specific IDP)

### Fixed

* Type errors

## [0.28.4 - 2022-10-22]

### Added

* AI: `umap(engine='cuml')` now supports older RAPIDS versions via knn fallback for edge creation. Also: `"umap_learn"`, defaults to `"auto"`
* `prune_self_edges()` to drop any edges where the source and destination are the same

### Fixed

* Infra: Updated github actions versions and Ubuntu environment for publishing 

## [0.28.3 - 2022-10-12]

### Added

* AI: full text & semantic search (`g.search(..)` and `g.search_graph(..).plot()`)
* Featurization: support for dataframe columns that are list of lists -> multilabel targets
                  set using `g.featurize(y=['list_of_lists_column'], multilabel=True,...)`
                  Only supports single-column data targets


## [0.28.2 - 2022-10-11]

### Changed

* Infra: Updated github actions

### Fixed

* `encode_axis()` now correctly sets axis
* work around mypy mistyping operator & on pandas series 

## [0.28.1 - 2022-10-06]

### Changed

* Speed up `g.umap()` >100x by using cuML UMAP engine
* Drop official support for Python 3.6 - its LTS security support stopped 9mo ago
* neo4j: v5 support - backwards-compatible changing derefs from id to element_id

### Added

* umap: Optional `engine` parameter (default `cuml`) for `UMAP()`
* ipynb: UMAP purpose, functionality and parameter details, with general UMAP notebook planned in future (features folder)

### Fixed

* has_umap: removed as no longer necessary

## [0.28.0 - 2022-09-23]

### Added

* neo4j: v5 support (experimental)

### Changed

* Infra: suppress igraph pandas FutureWarnings

## [0.27.3 - 2022-09-07]

### Changed

* Infra: Remove heavy AI dependencies from `pip install graphistry[dev]`

### Added

* igraph: Optional `use_vids` parameter (default `False`) for `to_igraph()` and its callers (`layout_igraph`, `compute_graph`)
* igraph: add `coreness` and `harmonic_centrality` to `compute_igraph`

### Fixed

* igraph: CI errors around igraph
* igraph: Tolerate deprecation warning of `clustering`
* Docs: Typos and updates - thanks @gadde5300 + @szhorvat !

## [0.27.2 - 2022-09-02]

### Changed

* Speed up `import graphistry` 10X+ by lazily importing AI dependencies. Use of `pygraphistry[ai]` features will still trigger slow upstream dependency initialization times upon first use.

### Fixed

* Docs: Update Labs references to Hub

## [0.27.1 - 2022-07-25]

### Fixed

* `group_in_a_box_layout()`: Remove verbose output
* `group_in_a_box_layout()`: Remove synthesized edge weight

## [0.27.0 - 2022-07-25]

### Breaking 🔥

* Types: Switch `materialize_nodes` engine param to explicitly using`Engine` typing (no change to untyped user code)

### Added

* `g.keep_nodes(List or Series)`
* `g.group_in_a_box_layout(...)`: Both CPU (pandas/igraph) and (cudf/cugraph) versions, and various partitioning/layout/styling settings
* Internal clientside Brewer palettes helper for categorical point coloring

### Changed

* Infra: CI early fail on deeper lint
* Infra: Move Python 3.6 from core to minimal tests due to sklearn 1.0 incompatibility

### Fixed

* lint
* suppress known dgl bindings test type bug

## [0.26.1 - 2022-07-01]

### Breaking 🔥

* `_table_to_arrow()` for `cudf`: Updated for RAPIDS 2022.02+ to handle deprecation of `cudf.DataFrame.hash_columns()` in favor of new `cudf.DataFrame.hash_values()`

### Added

* `materialize_nodes()`: Supports `cudf`, materializing a `cudf.DataFrame` nodes table when `._edges` is an instance of `cudf.DataFrame`
* `to_cugraph()`, `from_cugraph()`, `compute_cugraph()`, `layout_cugraph()`
* docs: [cugraph demo notebook](demos/demos_databases_apis/gpu_rapids/cugraph.ipynb)

### Changed

* Infra: Update GPU test env settings
* `materialize_nodes`: Return regular index

### Fixed

* `hypergraph()` in dask handles failing metadata type inference
* tests: gpu env tweaks
* tests: umap logging was throwing warnings

## [0.26.0 - 2022-06-03]

### Added
* `g.transform()`
* `g.transform_umap()`
* `g.scale()`
* Memoization on UMAP and Featurize calls
* Adds **kwargs and propagates them through to different function calls (featurize, umap, scale, etc)

### Breaking 🔥

* Final deprecation of `register(api=2)` protobuf/vgraph mode - also works around need for protobuf test upgrades

## [0.25.3 - 2022-06-22]

### Added

* `register(..., org_name='my_org')`: Optionally upload into an organization
* `g.privacy(mode='organization')`: Optionally limit sharing to within your organization

### Changed

* docs: `org_name` in `README.md` and sharing tutorial

## [0.25.2 - 2022-05-11]

### Added

* `compute_igraph()`
* `layout_igraph()`
* `scene_settings()`

### Fixed

* `from_igraph` uses `g._node` instead of `'name'` in more cases 

## [0.25.1 - 2022-05-08]

### Fixed

* `g.from_igraph(ig)` will use IDs (ex: strings) for src/dst values instead of igraph indexes

## [0.25.0 - 2022-05-01]

Major version bump due to breaking igraph change

### Added

* igraph handlers: `graphistry.from_igraph`, `g.from_igraph`, `g.to_igraph`
* docs: README.md examples of using new igraph methods

### Changed

* Deprecation warnings in old igraph methods: `g.graph(ig)`, `igraph2pandas`, `pandas2igraph`
* Internal igraph handlers upgraded to use new igraph methods 

### Breaking 🔥

* `network2igraph` and `igraph2pandas` renamed output node ID column to `_n_implicit` (`constants.NODE`)

## [0.24.1 - 2022-04-29]

### Fixed

* Expose symbols for `.chain()` predicates as top-level: previous `ast` export was incorrect

## [0.24.0 - 2022-04-29]

Major version bump due to large dependency increases for kitchen-sink installs and overall sizeable new feature

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

### Breaking 🔥

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

### Breaking 🔥

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

### Breaking 🔥

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

### Breaking 🔥
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
