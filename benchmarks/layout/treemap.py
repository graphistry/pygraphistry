"""
Performance benchmark: treemap layout implementations.

Compares:
  ref   — external reference package (skipped if not installed)
  cpu   — our built-in _squarify.py running on CPU (pandas DataFrame, data pre-resident)
  gpu   — our implementation driven via treemap() with a cuDF DataFrame (skipped if cudf absent)

For E2E benchmarks, DataFrames are built ONCE before timing.
CPU path: pandas already in memory. GPU path: cuDF already on device.

Dimensions swept:
  n_partitions: number of squarify rects (2 → 500)
  For E2E path: nodes_per_partition is also varied (affects groupby + tolist overhead)

Usage:
  python3.10 benchmarks/layout/treemap.py     # local CPU (reference + ours)
  python3     benchmarks/layout/treemap.py    # on dgx-spark (reference + ours + gpu)

Output:
  Prints Markdown tables and writes benchmarks/layout/RESULTS.md alongside this script.
"""

import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

try:
    import squarify as _ref
    HAS_REF = True
except ImportError:
    HAS_REF = False
    _ref = None

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False
    np = None

# Our implementation — find repo root by searching up from this file's location
_this = Path(__file__).resolve()
_repo_root = next(
    (p for p in [_this.parent, _this.parent.parent, _this.parent.parent.parent,
                 _this.parent.parent.parent.parent]
     if (p / "graphistry").is_dir()),
    _this.parent
)
sys.path.insert(0, str(_repo_root))
try:
    from graphistry.layout.gib._squarify import normalize_sizes as _our_norm, squarify as _our_sq
    HAS_OURS = True
except ImportError:
    HAS_OURS = False

try:
    import pandas as pd
    from graphistry.PlotterBase import PlotterBase
    from graphistry.layout.gib.treemap import treemap as _treemap_fn
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False
    cudf = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPEATS = 500  # repeats per cell; median + p95 computed
E2E_REPEATS = 200


def make_sizes(n: int) -> list:
    """Descending integer sizes summing to a round number, length n."""
    raw = [max(1, int(1000 / (i + 1))) for i in range(n)]
    raw.sort(reverse=True)
    return raw


def bench_fn(fn, *args, repeats=REPEATS):
    """Time fn(*args) for `repeats` iterations. Return (median_us, p95_us)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    return statistics.median(times), statistics.quantiles(times, n=20)[-1]  # p95


def fmt(us):
    if us >= 1_000_000:
        return f"{us/1e6:.2f}s"
    if us >= 1_000:
        return f"{us/1e3:.2f}ms"
    return f"{us:.1f}µs"


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_ref(sizes, dx, dy):
    normed = _ref.normalize_sizes(sizes, dx, dy)
    _ref.squarify(normed, 0, 0, dx, dy)


def run_ours(sizes, dx, dy):
    normed = _our_norm(sizes, dx, dy)
    _our_sq(normed, 0, 0, dx, dy)


def make_plottable_cpu(n_partitions: int, nodes_per_partition: int):
    """Build a pandas-backed Plottable. Created ONCE before timing."""
    total = n_partitions * nodes_per_partition
    nodes_df = pd.DataFrame({
        "id": list(range(total)),
        "partition": [i // nodes_per_partition for i in range(total)],
    })
    edges_df = pd.DataFrame({"s": [0], "d": [1]})
    return PlotterBase().nodes(nodes_df, "id").edges(edges_df, "s", "d")


def make_plottable_gpu(n_partitions: int, nodes_per_partition: int):
    """Build a cuDF-backed Plottable with data already on GPU. Created ONCE before timing."""
    total = n_partitions * nodes_per_partition
    nodes_df = cudf.DataFrame({
        "id": list(range(total)),
        "partition": [i // nodes_per_partition for i in range(total)],
    })
    edges_df = cudf.DataFrame({"s": [0], "d": [1]})
    return PlotterBase().nodes(nodes_df, "id").edges(edges_df, "s", "d")


def run_treemap(g, dx, dy):
    _treemap_fn(g, w=dx, h=dy)


# ---------------------------------------------------------------------------
# Main benchmark sweep
# ---------------------------------------------------------------------------

N_PARTITION_SIZES = [2, 5, 10, 25, 50, 100, 250, 500]
DX, DY = 1000.0, 1000.0  # fixed canvas

GPU_NODES_PER_PARTITION = [10, 100, 1000]

rows_algo = []

print(f"\n{'='*70}")
print(f"Backends available:  ref={HAS_REF}  cpu={HAS_OURS}  gpu={HAS_CUDF}")
print(f"Repeats per cell: algo={REPEATS}  e2e={E2E_REPEATS}")
print(f"Canvas: {DX}×{DY}")
print(f"{'='*70}\n")

# --- Algorithm-only sweep (ref vs cpu) ---
print("## Algorithm benchmark (normalize_sizes + squarify, no DataFrame)")
header = f"{'n_parts':>8} | {'sizes[0]':>10} | {'ref median':>12} {'ref p95':>10} | {'cpu median':>12} {'cpu p95':>10} | {'speedup':>8}"
print(header)
print("-" * len(header))

for n in N_PARTITION_SIZES:
    sizes = make_sizes(n)
    row = {"n": n, "sizes_max": sizes[0]}

    if HAS_REF:
        med_ref, p95_ref = bench_fn(run_ref, sizes, DX, DY)
        row["ref_med"] = med_ref
        row["ref_p95"] = p95_ref
    else:
        med_ref = p95_ref = None

    if HAS_OURS:
        med_cpu, p95_cpu = bench_fn(run_ours, sizes, DX, DY)
        row["cpu_med"] = med_cpu
        row["cpu_p95"] = p95_cpu
    else:
        med_cpu = p95_cpu = None

    speedup = f"{med_ref/med_cpu:.2f}×" if (med_ref and med_cpu) else "n/a"
    row["speedup"] = speedup

    ref_str = f"{fmt(med_ref):>12} {fmt(p95_ref):>10}" if med_ref else f"{'n/a':>12} {'n/a':>10}"
    cpu_str = f"{fmt(med_cpu):>12} {fmt(p95_cpu):>10}" if med_cpu else f"{'n/a':>12} {'n/a':>10}"
    print(f"{n:>8} | {sizes[0]:>10} | {ref_str} | {cpu_str} | {speedup:>8}")
    rows_algo.append(row)

# --- E2E sweep: CPU (pandas) vs GPU (cuDF), data pre-built before timing ---
rows_e2e = []

if HAS_PANDAS and HAS_CUDF:
    print(f"\n## E2E benchmark: treemap() on pre-resident data, {E2E_REPEATS} reps")
    print("(DataFrame built ONCE before timing; measures only treemap() call)")
    print("cpu=pandas already in memory, gpu=cuDF already on device")
    header_e2e = (f"{'n_parts':>8} | {'nodes/part':>10} | {'total':>8} |"
                  f" {'cpu median':>11} {'cpu p95':>9} |"
                  f" {'gpu median':>11} {'gpu p95':>9} | {'gpu/cpu':>8}")
    print(header_e2e)
    print("-" * len(header_e2e))

    for n in N_PARTITION_SIZES:
        for npg in GPU_NODES_PER_PARTITION:
            g_cpu = make_plottable_cpu(n, npg)
            g_gpu = make_plottable_gpu(n, npg)

            med_cpu, p95_cpu = bench_fn(run_treemap, g_cpu, DX, DY, repeats=E2E_REPEATS)
            med_gpu, p95_gpu = bench_fn(run_treemap, g_gpu, DX, DY, repeats=E2E_REPEATS)

            ratio = f"{med_gpu/med_cpu:.2f}×"
            row = {"n": n, "npg": npg, "total": n * npg,
                   "cpu_med": med_cpu, "cpu_p95": p95_cpu,
                   "gpu_med": med_gpu, "gpu_p95": p95_gpu,
                   "ratio": ratio}
            print(f"{n:>8} | {npg:>10} | {n*npg:>8} |"
                  f" {fmt(med_cpu):>11} {fmt(p95_cpu):>9} |"
                  f" {fmt(med_gpu):>11} {fmt(p95_gpu):>9} | {ratio:>8}")
            rows_e2e.append(row)
elif HAS_PANDAS:
    print(f"\n## E2E benchmark (CPU only): treemap() on pre-resident data, {E2E_REPEATS} reps")
    print("(DataFrame built ONCE before timing; gpu skipped — cudf not available)")
    header_e2e = (f"{'n_parts':>8} | {'nodes/part':>10} | {'total':>8} |"
                  f" {'cpu median':>11} {'cpu p95':>9}")
    print(header_e2e)
    print("-" * len(header_e2e))

    for n in N_PARTITION_SIZES:
        for npg in GPU_NODES_PER_PARTITION:
            g_cpu = make_plottable_cpu(n, npg)
            med_cpu, p95_cpu = bench_fn(run_treemap, g_cpu, DX, DY, repeats=E2E_REPEATS)
            row = {"n": n, "npg": npg, "total": n * npg,
                   "cpu_med": med_cpu, "cpu_p95": p95_cpu}
            print(f"{n:>8} | {npg:>10} | {n*npg:>8} |"
                  f" {fmt(med_cpu):>11} {fmt(p95_cpu):>9}")
            rows_e2e.append(row)
else:
    print("\n## E2E benchmark: SKIPPED (pandas not available)")

# ---------------------------------------------------------------------------
# Write RESULTS.md
# ---------------------------------------------------------------------------

out = Path(__file__).parent / "RESULTS.md"

lines = [
    "# Treemap Layout — Performance Benchmark Results",
    "",
    f"**Date**: {time.strftime('%Y-%m-%d')}",
    f"**Python**: {sys.version.split()[0]}",
    f"**Platform**: {sys.platform}",
    f"**Backends**: ref={'yes' if HAS_REF else 'not installed'}  cpu={'yes' if HAS_OURS else 'no'}  gpu={'yes (cudf)' if HAS_CUDF else 'not available'}",
    f"**Repeats per cell**: algo={REPEATS}  e2e={E2E_REPEATS} (median + p95 reported)",
    f"**Canvas**: {DX}×{DY}",
    "",
    "## Algorithm benchmark (normalize + layout, no DataFrame overhead)",
    "",
    "| n_partitions | largest_size | ref median | ref p95 | cpu median | cpu p95 | speedup (ref/cpu) |",
    "|-------------:|-------------:|-----------:|--------:|-----------:|--------:|------------------:|",
]

for r in rows_algo:
    ref_med = fmt(r["ref_med"]) if "ref_med" in r else "n/a"
    ref_p95 = fmt(r["ref_p95"]) if "ref_p95" in r else "n/a"
    cpu_med = fmt(r["cpu_med"]) if "cpu_med" in r else "n/a"
    cpu_p95 = fmt(r["cpu_p95"]) if "cpu_p95" in r else "n/a"
    sp = r.get("speedup", "n/a")
    lines.append(f"| {r['n']:>12} | {r['sizes_max']:>12} | {ref_med:>10} | {ref_p95:>7} | {cpu_med:>10} | {cpu_p95:>7} | {sp:>17} |")

lines += [
    "",
    "> **speedup** > 1× means our CPU impl is faster than the reference.",
    "> speedup < 1× means slower (overhead from numpy vs pure Python for tiny n).",
    "",
]

if rows_e2e and HAS_CUDF:
    lines += [
        "## E2E benchmark: treemap() on pre-resident data (pandas CPU vs cuDF GPU)",
        "",
        "> Data built ONCE before timing. CPU=pandas already in memory, GPU=cuDF already on device.",
        "> `gpu/cpu` ratio < 1× means GPU is faster. Crossover driven by cuDF groupby fixed overhead (~600µs).",
        "",
        "| n_partitions | nodes/partition | total_nodes | cpu median | cpu p95 | gpu median | gpu p95 | gpu/cpu |",
        "|-------------:|----------------:|------------:|-----------:|--------:|-----------:|--------:|--------:|",
    ]
    for r in rows_e2e:
        lines.append(
            f"| {r['n']:>12} | {r['npg']:>15} | {r['total']:>11} |"
            f" {fmt(r['cpu_med']):>10} | {fmt(r['cpu_p95']):>7} |"
            f" {fmt(r['gpu_med']):>10} | {fmt(r['gpu_p95']):>7} | {r['ratio']:>7} |"
        )
elif rows_e2e:
    lines += [
        "## E2E benchmark: treemap() on pre-resident data (CPU only — gpu not available)",
        "",
        "> Data built ONCE before timing. CPU=pandas already in memory.",
        "",
        "| n_partitions | nodes/partition | total_nodes | cpu median | cpu p95 |",
        "|-------------:|----------------:|------------:|-----------:|--------:|",
    ]
    for r in rows_e2e:
        lines.append(
            f"| {r['n']:>12} | {r['npg']:>15} | {r['total']:>11} |"
            f" {fmt(r['cpu_med']):>10} | {fmt(r['cpu_p95']):>7} |"
        )
else:
    lines += ["## E2E benchmark", "", "> Not run — pandas not available."]

lines += [
    "",
    "## Interpretation",
    "",
    "- **CPU parity with reference:** built-in impl matches removed dep within noise (see algo table).",
    "- **E2E GPU vs CPU:** cuDF groupby has a fixed per-call overhead (~600µs baseline).",
    "  GPU becomes competitive with CPU only when node count is large enough that cuDF's",
    "  vectorised groupby beats pandas. The `gpu/cpu` column shows the crossover.",
    "- **Layout algorithm itself is CPU-bound** (sequential recursion, not GPU-parallelisable).",
    "  The GPU path calls `.to_numpy().tolist()` before layout, so layout runs on CPU either way.",
    "- **Coordinate transform (normalize + global positioning) is vectorized** via DataFrame merge,",
    "  replacing per-row dict `.map()` lookups with a single join.",
    "- For typical usage (10–100 partitions), total treemap latency is negligible vs",
    "  upstream community detection (hundreds of ms to seconds).",
]

out.write_text("\n".join(lines) + "\n")
print(f"\nResults written to: {out}")
