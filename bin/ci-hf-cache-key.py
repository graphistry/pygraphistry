#!/usr/bin/env python3
"""Emit a GitHub Actions cache-key fragment for HuggingFace model caches."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


HF_DEPENDENCIES = (
    "huggingface-hub",
    "sentence-transformers",
    "tokenizers",
    "transformers",
)

PACKAGE_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\\\s]+)")


def parse_versions(lockfile: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lockfile.read_text(encoding="utf-8").splitlines():
        match = PACKAGE_RE.match(line.strip())
        if not match:
            continue
        name, version = match.groups()
        normalized_name = name.lower().replace("_", "-")
        if normalized_name in HF_DEPENDENCIES:
            versions[normalized_name] = version
    return versions


def build_key(versions: dict[str, str]) -> tuple[str, str]:
    missing = [name for name in HF_DEPENDENCIES if name not in versions]
    if missing:
        raise SystemExit(
            "Missing HF dependency pins in lockfile: " + ", ".join(missing)
        )

    summary = ",".join(f"{name}={versions[name]}" for name in HF_DEPENDENCIES)
    digest = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16]
    return f"hfdeps-{digest}", summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lockfile", type=Path)
    args = parser.parse_args()

    key, summary = build_key(parse_versions(args.lockfile))
    print(f"key={key}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
