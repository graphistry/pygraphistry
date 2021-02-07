# Changelog

All notable changes to the PyGraphistry are documented in this file. The PyGraphistry client and other Graphistry components are tracked in the main [Graphistry major release history documentation](https://graphistry.zendesk.com/hc/en-us/articles/360033184174-Enterprise-Release-List-Downloads).

The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and all PyGraphistry-specific breaking changes are explictly noted here.

## [Development]

### Adding
* Gremlin / AWS Neptune connector <-- not for this release
* Compressed uploads: Snappy-compressed parquet for network-bound scenarios (https://github.com/graphistry/pygraphistry/issues/188)
* CI: Move to GitHub Actions

### Added

* Friendlier error message for api=1, 2 server non-json responses (https://github.com/graphistry/pygraphistry/pull/187)


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
