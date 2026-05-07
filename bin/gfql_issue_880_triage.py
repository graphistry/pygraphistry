#!/usr/bin/env python3

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphistry.compute.gfql.benchmark_residual_triage import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli())
