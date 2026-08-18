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


def _apply_host_limit() -> None:
    """Cap host (pandas/numpy) address space when GFQL_HOST_LIMIT_GB is set.

    RMM only contains device/unified allocations, so a pure-CPU workload has NO
    hard cap: the safe_run.sh watchdog polls every 5 s and only fires once host
    RAM has already fallen below the floor -- i.e. after the box is swapping.
    That is the failure shape that cost 9.5 hours on a 1.8B-edge run.

    RLIMIT_AS makes an over-allocating CPU run raise a clean MemoryError at the
    allocation site instead, mirroring what the RMM cap does for GPU runs.

    Caveat: RLIMIT_AS limits *virtual* address space, so it must not be applied
    to GPU runs -- CUDA reserves large VA ranges up front and would fail
    spuriously. Set GFQL_HOST_LIMIT_GB only on CPU lanes.
    """
    gb = _os.environ.get("GFQL_HOST_LIMIT_GB")
    if not gb:
        return
    try:
        limit = int(float(gb) * 1024 ** 3)
    except ValueError:
        return
    try:
        import resource
        import sys
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        print(f"[dgx-guard] host address-space limit = {gb} GB (CPU-lane safety)",
              file=sys.stderr)
    except Exception:
        # Never block the workload on a failure to tighten a guard.
        pass


_apply_rmm_limit()
_apply_host_limit()
