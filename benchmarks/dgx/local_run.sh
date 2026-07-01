#!/bin/bash
# LOCAL desktop run guard. The local box is a ~31GB WORKSTATION (not dgx's 119GB) —
# a runaway alloc OOM-kills the desktop SESSION (logout). Caps address space so a
# runaway process dies with a clean MemoryError instead of taking the session down.
# TESTED 2026-07-01: ulimit -v 2GB + 3GB alloc -> MemoryError, desktop survived.
# Use for ANY local python; keep local data TINY (<~1M rows). Benchmarks -> dgx safe_run.sh.
# Usage: benchmarks/dgx/local_run.sh python3 script.py   (LOCAL_CAP_GB=8 default)
CAP_GB=${LOCAL_CAP_GB:-8}
( ulimit -v $((CAP_GB*1024*1024)) 2>/dev/null; exec "$@" )
