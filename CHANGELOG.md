# Changelog

All notable changes to the PyGraphistry are documented in this file. The PyGraphistry client and other Graphistry components are tracked in the main [Graphistry major release history documentation](https://graphistry.zendesk.com/hc/en-us/articles/360033184174-Enterprise-Release-List-Downloads).

The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and all PyGraphistry-specific breaking changes are explictly noted here.

## [Development]

### Adding
* Gremlin / AWS Neptune connector <-- not for this release
* Compressed uploads: Snappy-compressed parquet for network-bound scenarios (https://github.com/graphistry/pygraphistry/issues/188)
* Improved error message: Include response body on failed arrow.post()

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
