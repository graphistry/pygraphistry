"""Combine per-worker GFQL engagement records; fail closed on incomplete runs."""

import argparse
import json
from pathlib import Path

from graphistry.tests.compute.gfql.routes.switch import ROUTES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    files = sorted(args.record_dir.glob("*.json"))
    if len(files) != args.workers:
        raise ValueError(f"expected {args.workers} worker records, found {len(files)}")
    records = [json.loads(path.read_text()) for path in files]
    if any(record["exitstatus"] != 0 for record in records):
        raise ValueError("normal baseline failed")
    hashes = {record["collection_sha256"] for record in records}
    counts = {record["collected"] for record in records}
    if len(hashes) != 1 or len(counts) != 1:
        raise ValueError("worker collections differ")
    seen = set().union(*(record["seen"] for record in records))
    if len(seen) != next(iter(counts)):
        raise ValueError("not every collected test ran in the normal baseline")
    hits = {route: sorted(set().union(*(record["hits"][route] for record in records))) for route in ROUTES}
    if any(set(ids) - seen for ids in hits.values()):
        raise ValueError("route hit refers to an unknown test")
    manifest = {
        "routes": list(ROUTES),
        "collection_sha256": next(iter(hashes)),
        "collected": next(iter(counts)),
        "hits": hits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, sort_keys=True))
    print(f"normal baseline: {manifest['collected']} tests")
    for route, ids in hits.items():
        print(f"{route}: {len(ids)} relevant tests")
    print(f"all-off: {len(set().union(*(set(ids) for ids in hits.values())))} relevant tests")


if __name__ == "__main__":
    main()
