"""Auto-applied RMM allocation-limit guard for dgx GB10 (unified memory).

Python imports `sitecustomize` at startup if it's on sys.path. When
GFQL_RMM_LIMIT_GB is set, this caps ALL RMM (cudf/cugraph/cupy) device
allocations so an over-allocation raises a clean caught MemoryError instead of
consuming the shared 119 GB unified host RAM and OOM-thrashing the box.

Proven (2026-07-01): docker --memory is TRANSPARENT to cudf/unified allocs;
RMM LimitingResourceAdaptor caps both cudf AND cugraph cleanly. This is the
containment mechanism. No-op (silent) when GFQL_RMM_LIMIT_GB unset or rmm absent
(CPU runs). Mount this dir + prepend to PYTHONPATH via benchmarks/dgx/safe_run.sh.
"""
import os as _os


def _apply_rmm_limit() -> None:
    gb = _os.environ.get("GFQL_RMM_LIMIT_GB")
    if not gb:
        return
    try:
        limit = int(float(gb) * 1024 ** 3)
    except ValueError:
        return
    try:
        import rmm
        rmm.mr.set_current_device_resource(
            rmm.mr.LimitingResourceAdaptor(rmm.mr.CudaMemoryResource(), allocation_limit=limit))
        try:
            import cupy
            from rmm.allocators.cupy import rmm_cupy_allocator
            cupy.cuda.set_allocator(rmm_cupy_allocator)
        except Exception:
            pass
        import sys
        print(f"[dgx-guard] RMM device allocation limit = {gb} GB (unified-memory safety)",
              file=sys.stderr)
    except Exception:
        # rmm not present (CPU run) or set failed — do not block the workload.
        pass


_apply_rmm_limit()
