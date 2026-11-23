# Benchmarking opt-in CPU Parquet hop

Run these commands from the repository root on Linux with the Polars extra
installed. The benchmark emits JSONL with the invocation, ingestion and layout
build measurements, per-attempt runs, and per-method medians.

```bash
python -m benchmarks.gfql.parquet_hop \
  --data /tmp/gfql-synthetic --synthetic-edges 1000000 \
  --row-group-size 16384 --cluster --seeds 0 10000 250000 --hops 4 \
  > synthetic.jsonl

curl -fL https://snap.stanford.edu/data/soc-Epinions1.txt.gz -o /tmp/soc-Epinions1.txt.gz
python -m benchmarks.gfql.parquet_hop \
  --data /tmp/gfql-epinions --snap-gzip /tmp/soc-Epinions1.txt.gz \
  --row-group-size 4096 --cluster --seeds 18 42 --hops 2 \
  > epinions.jsonl
```

Use a fresh scratch directory for each build. The small SNAP importer uses
the endpoint list in its recorded direction, assigns edge IDs by input row
and creates deterministic synthetic properties. It loads the source edge
list into memory during ingestion. For larger inputs, pass existing local
`--nodes` and `--edges` Parquet files or directories. Directory inputs scan
all nested `*.parquet` files with Hive partitioning enabled. Their byte totals
and cache advice cover those same files. The optional clustering step writes a
second edge file sorted by source. `--row-group-size` sets the Parquet write
granularity for ingestion and clustering. For existing files it leaves the
stored row groups intact.

All timed methods share the same files and seeds in each run. The eager control
calls public `.hop()` without opt-in. The bounded route calls public `.hop()`
with `limits=HopLimits(...)`. The native control builds equivalent membership and
range expressions with public Polars APIs and uses an independent traversal
loop. It omits the feature's row guards and uses the same endpoint checks,
NaN-to-null property normalization and complete output requirements. The
resident-index comparison separately measures eager input load and forward
index construction. Its query timing starts after both. To compare source and
clustered layouts, run the same query twice with the same source files and
enable `--cluster` for the second run.

Each attempt runs in a fresh subprocess. Cold-advised runs use Linux
`POSIX_FADV_DONTNEED` and warm runs prime one query in the worker. The advice is
advisory. Physical reads and writes are kernel process counters. The RSS field
is process high-water memory through the named phase, including any prior setup
and warmup in that worker. Index and input setup I/O appear in those phase
records. Query time includes property rehydration. The bounded and native
routes report separate rehydration time. Digests cover both output schemas and
the values of all rows sorted by identity, outside the timed phase. Compare
digests within the same Polars version and digest algorithm. Different digests fail the run. Failed
attempts remain in the JSONL. When both lazy routes are requested, the
benchmark requires every bounded and native attempt to complete. Eager and
resident-index allocation failures remain visible in the attempt records.

The runner defaults to four Polars threads. For an existing large graph, use
`--address-space-mib` to add a Linux virtual-address cap to the query workers.
A failure at that cap is evidence about the configured constraint. Ingestion
and optional layout builds run separately and should be sized for the host.
For regression checks, verify digests, completion counts and scan plans.
Latency and I/O medians are for local comparison.
