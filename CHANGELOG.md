# Changelog

All notable changes to the PyGraphistry are documented in this file. The PyGraphistry client and other Graphistry components are tracked in the main [Graphistry major release history documentation](https://graphistry.zendesk.com/hc/en-us/articles/360033184174-Enterprise-Release-List-Downloads).

The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and all PyGraphistry-specific breaking changes are explictly noted here.

## Dev

### Breaking
* Plottable is now a Protocol
* py.typed added, type checking active on PyGraphistry!
* transform() and transform_umap() now require some parameters to be keyword-only

### Added

* GFQL: Comparison predicates (`gt`, `lt`, `ge`, `le`, `eq`, `ne`, `between`) and `is_in` now support datetime, date, and time values with timezone awareness

### Changed

* GFQL: Temporal value classes (`DateTimeValue`, `DateValue`, `TimeValue`) are now exported from `graphistry.compute`

### Internal

* Remove unused imports across predicate modules

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

### Breaking

* PyGraphistry now installs lightgbm & sentence-transformers by default

### Added

* GPU-accelerated embedded-AI environment: new Docker env `cuda11.8-rapids24.12-pygraphistry-dev`
* Use Apache Arrow Files (via IPC mechanism) for on-disk caching and as PyGraphistry's native cache format
* Automatic loading of appropriate dataframe engine (pandas, cudf) when computing
* Auto conversion between pandas/cudf. `@autocast` decorator added for seamless transitions.
* Align PyGraphistry core with Graphistry's production data science stack
* Update RAPIDS from 22.x to 24.12
* Better typing support for pandas/cudf/polars edge cases
* Google Spanner integration, including `PyGraphistry.spanner()`, `g.spanner_query()`, `g.spanner_query_to_dataframe()`

### Changed

* Allow passing `config` into plot(), upload() methods when `graphistry.privacy()` is enabled. Adjusted behavior for all related methods.
* Databricks-connect now supported
* Arrow Files (Feather V2) implementation used as primary caching mechanism
* Caching speed improvements up to 10x for pandas and up to 60x for cudf
* Update lightgbm lower bound to `>=4.0.0`
* RAPIDS environments now support both CUDA 11.8 and CUDA 12 (12.5)
* Better type annotations for dataframe operations, add runtime_checkable Protocol for cudf typing
* Update minimum supported scikit-learn to `>=1.1`
* Update docker base images to use `nvidia/cuda:11.8.0-devel-ubuntu22.04` for RAPIDS compatibility

### Fixed

* Fix optional cuda dependencies installation flow
* Allow RAPIDS to run without cuml/cugraph installed
* Fix CI and testing infrastructure for large environments
* Resolve dependency conflicts between lightgbm, scikit-learn, and RAPIDS versions
* Import error handling for optional GPU packages

### Test

* Update test infrastructure to support both CPU and GPU environments
* Add tests for Arrow File caching mechanisms
* Add tests for automatic dataframe conversions
* Google Spanner comprehensive integration tests (requires credentials)

### Internal

* Large internal code cleanup and modernization
* Update pre-commit hooks and development tools
* Better separation of CPU and GPU code paths
* Refactor imports to avoid circular dependencies with heavy dependencies

### Examples

* New notebooks demonstrating Arrow File caching
* New notebooks showing automatic pandas/cudf conversions

## [0.36.0 - 2025-06-05]

### Added

* Google BigQuery integration PyGraphistry.bigquery(), g.bigquery_query(), g.bigquery_query_to_dataframe()

### Changed

* Improve environment variables loading and allow changing server credentials after setting g = graphistry.nodes(df) and later g.register(...).plot()
* Updated demos to Plotter 2.0, new folder structure

### Test

* CI for BigQuery

## [0.35.0 - 2025-06-24]

### Changed

* g.name() now returns the graph g itself, not the name string.
  * Migration: Change `name_str = g.name('my_graph')` to `g2 = g.name('my_graph'); name_str = g2._name`

### Breaking

* **Stricter `g.name()` typing: breaks `name_str = g.name('my_graph')`**

### Fixed

* Fixed duplicate setting of graph name in `g.name()` method
* Fixed missing return self in g.name() method
* `g.name()` now correctly returns the Plotter instance for method chaining

### Test

* Added tests for g.name() method to ensure it returns self and sets name correctly

## [0.34.12 - 2025-06-04]

### Added

* Python 3.12 support

### Fixed

* Fixed .gitignore
* Fixed import issues with RAPIDS cudf and dask-cudf
* Removed `setup_logger` from public APIs
* Fixed type annotation

### Test

* Type checking, coverage checking, minimal tests for Python 3.12

### Internal

* Various type fixes

### Docs

* Fix URLs in README

## [0.34.11 - 2025-06-01]

### Fixed

* Suppress warnings related to using `pyarrow` with `pandas` 2.2.x during `g.plot()`. This issue appears when using `pandas >= 2.2.2` with `pyarrow >= 15`. For users experiencing issues, we recommend using `pyarrow == 16.0.0` with `pandas >= 2.2.2`, though all recent pyarrow versions should work.

### Test

* Test various CI pandas versions with PyArrow

### Internal

* Update pre-commit hooks including Black to 24.4.2

## [0.34.10 - 2025-05-22]

### Fixed

* Fix client-side JWT token handling

## [0.34.9 - 2025-05-13]

### Added

* Add support for JWT token refresh for enterprise SSO environments. OAuth provider must support refresh tokens. More info: https://auth0.com/docs/secure/tokens/refresh-tokens

## [0.34.8 - 2025-05-05]

### Fixed

* Handle mixed `np.nan`/`pd.NaT` values correctly

### Test

* Better PyArrow handling for tests

## [0.34.7 - 2025-04-26]

### Fixed

* Silence a warning that `pd.Timedelta` is deprecated and to use `pd.Timestamp` (even though we aren't using `pd.Timedelta` in the first place)
* Fixed hypergraph `.hop()` when using column names that are reserved words in pandas query language

### Test

* Test for using edge attributes that are reserved words

## [0.34.6 - 2025-04-24]

### Added

* New Plotter methods `g.edges_df()` and `g.nodes_df()` as aliases to `g._edges` and `g._nodes` 

### Test

* Add test for `g.edges_df()` and `g.nodes_df()`

## [0.34.5 - 2025-04-12]

### Added

* Support both UUID and classic bigint IDs from Louie.AI

### Test

* Add Louie tests

## [0.34.4 - 2025-04-12]

### Fixed

* Restore broken `g.scene_settings()` used internally by GFQL

### Test

* Test `g.scene_settings()`

## [0.34.3 - 2025-04-10]

### Fixed

* Fix `register()` not passing dataset_id and other parameters to plot()

### Test

* Test register() parameters

## [0.34.2 - 2025-04-09]

### Fixed

* Fix `.hop()` mishandling edge attributes when used with `return_as_wave_front`

### Test

* Add test for `.hop()` with edge attributes and `return_as_wave_front`

## [0.34.1] - 2025-04-08

### Added

* Add env var `GRAPHISTRY_HAS_NO_REAL_IPYTHON` for Databricks Connect compatibility where IPython lib exists but not browser display 

### Docs

* README Databricks Connect example

## [0.34.0] - 2025-04-05

### Changed

* `transform_umap()` signature moved **X** out of positional, so need to use kw form `g.transform_umap(X=some_df)` - aligns all internal libs
* 3.7 wheels deprecated

### Breaking

* **`transform_umap(X)` must now be called as `transform_umap(X=X)` as X changed from positional to keyword-only argument**

### Fixed

* `hop()` and GFQL validation errors on `filter_dict` for boolean values

### Test

* `hop()` and GFQL `filter_dict` checks for literals like booleans

## [0.33.3] - 2025-03-17

### Fixed

* Increased max width setting to prevent content truncation in hop/chain outputs in api docs

## [0.33.2] - 2025-03-17

### Added

* Additional Hop and Chain functionality documentation
* Type definition exports

### Fixed

* Type exports

### Internal

* Documentation infrastructure updates
* Separated chain and hop docs for better maintainability and readability

## [0.33.1] - 2025-03-17

### Fixed

* Fixed wheel build

## [0.33.0] - 2025-03-17

### Added

* Docs including more inline examples

### Fixed

* Fixed CI builds for Python 3.12
* Better dependency resolution in dev env
* Removed install-time dependencies on heavy libs

### Changed

* Updated pyproject.toml to use dynamic versioning from graphistry/\_\_init\_\_.py
* Updated Docker build configurations for Python 3.12 support
* Removed runtime version dependency on dirty-equals
* Update dev env

### Internal

* Switched to using pyproject.toml for build configuration
* Improved development environment setup

## [0.32.0] - 2025-03-12

### Added

* **Databricks Delta Sharing Support**: Direct integration with Databricks Delta Sharing for seamless data access.
  * PyGraphistry methods: `PyGraphistry.delta_sharing()`, `g.delta_lake()`, `g.delta_lake_nodes()`, `g.delta_lake_edges()`
  * Install: `pip install graphistry[databricks]`
  * Configure via environment variables or method parameters
  * Support for both table and dataframe operations

### Changed

* Default to fast `lz4` during `.plot()`, `.upload()`, and other upload operations
* Upload format and compression are now configurable: `.plot(file_compression='lz4')`, `.plot(file_format='parquet')` where supported formats are `['parquet', 'arrow', 'csv', 'json']` and compression can be `['lz4', 'gzip', 'brotli', 'snappy', 'zstd', None]`

### Fixed

* Fix RAPIDS environment detection in newer versions
* Fixed Python 3.6 compatibility in tests
* Fix error in optional cudf calls
* CI: Update test actions

### Test

* Delta Sharing comprehensive test coverage

### Docs

* Full Delta Sharing documentation in docstrings and notebooks

## [0.31.0] - 2025-02-27

### Added

* **Privacy-preserving mode**: `graphistry.privacy()` with managed compute and controls for what leaves a notebook
  * `graphistry.privacy()` returns a managed Plotter instance
  * Granular control for data redaction: `g.plot(render=True, memoize=False)`
  * Control for privacy-preserving vs regular sessions

### Changed

* CI back to Python 3.6 for legacy compatibility testing

### Docs

* Complete documentation for privacy mode
* All public methods updated with privacy mode behavior
* Updated code of conduct and contributing guidelines

### Test

* All major methods tested for privacy mode compliance

## [0.30.4] - 2025-02-26

### Fixed

* Docs

## [0.30.3] - 2025-02-26

### Fixed

* CI release flow

## [0.30.2] - 2025-02-25

### Changed

* CI now on python 3.8 and more robust CI

### Added

* Python 3.11 and 3.12 testing

## [0.30.1] - 2025-02-23

### Fixed

* SSO/JWT refresh on initial plot when creds expired

## [0.30.0] - 2025-02-21

### Added

* Neo4j 5.x bolt driver support
* Neo4j Property Graph support

### Changed

* Neo4j bolt driver connector defaults to Neo4j 5.x (explicit `graphistry.register(bolt_driver_version='4.4')` to pick 4.x)
* Upgraded neo4j driver dependency
* Bump apache-arrow, pyarrow, and pandas versions used in CI

### Fixed

* Neo4j 5.x neotime errors by removing Neo4j `DateTime` conversion
* Neo4j test infra using env vars `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`

### Breaking

* Neo4j `DateTime` conversion removed for Neo4j 5.x driver compatibility; library defaults to Neo4j 5.x