# Architecture

This document should help get started with modifying code. See also [develop.md](DEVELOP.md) for developer commands and [CONTRIBUTE.md](CONTRIBUTE.md) for community guidelines.

## Client/Server Wrangling Tool

PyGraphistry is Python client library primarily for:
* Loading in PyData, or converting non-PyData into PyData
* Setting declarative bindings for graph shaping and visual encodings
* Turning that into a live, embeddable Graphistry visualization

It is also increasingly used for intermediate compute and deeper access to Graphistry APIs. However, these should not get in the way of the primary use case.

It is currently heavily using Pandas, and slowly porting to Arrow-based and RAPIDS.ai-based internals.

## Functional

Most user interaction is functional, where every `.bind()` will create a new Graphistry object that is a clone of the one being chained. Thus most calls have a cheap `.copy()` over shallow immutable bindings.

Account-related settings are generally global with local functional overrides.

## DetaFrames: Lazy vs. Eager

At the `plot()` call, lazily bound data is materialized into the format needed for upload processing, e.g., API=3 will require pandas dataframes to be transformed into Arrow.

Database connectors should convert to Pandas, or better, Arrow, upon load. This enables immediate analysis using `._nodes` and `._edges`.

## Plugins

New code is increasingly put into separate files:

* Graphistry APIs: Arrow conveniences classes to match Graphistry server APIs. Each entity type generally has different types. These are being written to allow standalone use.

* Per-Connector: These should be separate files and standalone. For convenience, their `connect()` and `query()` methods can be added to the global namespace. Ex: `g.bolt(...).cypher(...).plot()`.

* Methods are intended for Notebook-based inspection. For example, calling `g.cypher` should guide users with the available arguments and examples of use.