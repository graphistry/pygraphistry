# Graph Benchmark q1-q9 (graph-benchmark)

This benchmark replays q1-q9 from `prrao87/graph-benchmark` against Graphistry using pandas/cuDF and GFQL filters.
It expects the benchmark repo to be checked out as a sibling (default: `/home/lmeyerov/Work/graph-benchmark`) and
its dataset generated with `generate_data.sh`.

## Setup

```sh
# In the sibling repo
cd /home/lmeyerov/Work/graph-benchmark
bash generate_data.sh 100000
```

## Run

```sh
cd /home/lmeyerov/Work/pygraphistry
python benchmarks/graph_benchmark_q1_q9.py --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark
```

Optional flags:

```sh
python benchmarks/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --runs 5 \
  --warmup 1 \
  --output-json /tmp/graph_benchmark_q1_q9.json
```

Preindexed variant (relation/type split per query, still vectorized pandas):

```sh
python benchmarks/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode preindexed \
  --runs 5 --warmup 1
```

Include preindex build time in per-query medians (adds `preindex_ms` and `median_ms_with_preindex`):

```sh
python benchmarks/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode preindexed \
  --include-preindex \
  --runs 5 --warmup 1
```

Presorted variant (global sort by rel/src/dst and node_type/node_id):

```sh
python benchmarks/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode presorted \
  --runs 5 --warmup 1
```

## Notes

- q1-q7 use GFQL filters to match the graph-benchmark query intent, then pandas aggregates for counts/averages.
- q8-q9 count all length-2 paths (including multiplicity) with vectorized degree math over FOLLOWS edges.
- The dataset uses separate ID spaces per node type; the loader offsets them into a single ID space.
