"""Per-backend lazy GFQL engines. Each backend lowers GFQL to its own deferred
plan representation (polars: ``pl.LazyFrame``; future duckdb: relations) and
executes via the shared target/collect framework in ``graphistry.compute.gfql.lazy``.
The lowering is per-backend; only the target/collect framework is shared."""
