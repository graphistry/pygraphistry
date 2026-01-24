#!/usr/bin/env python3
import argparse
import os
import sys

from detect_secrets.constants import VerifiedResult
from detect_secrets.core import baseline
from detect_secrets.core.scan import get_files_to_scan
from detect_secrets.core.secrets_collection import SecretsCollection
from detect_secrets.settings import default_settings, get_settings

EXCLUDE_FILTER = "detect_secrets.filters.regex.should_exclude_file"
VERIFY_FILTER = "detect_secrets.filters.common.is_ignored_due_to_verification_policies"
BASELINE_FILTER = "detect_secrets.filters.common.is_baseline_file"


def configure_filters(exclude_files: str, only_verified: bool, baseline_path: str) -> None:
    if exclude_files:
        get_settings().filters[EXCLUDE_FILTER] = {"pattern": [exclude_files]}

    if baseline_path:
        get_settings().filters[BASELINE_FILTER] = {"filename": baseline_path}

    min_level = VerifiedResult.VERIFIED_TRUE if only_verified else VerifiedResult.UNVERIFIED
    get_settings().filters[VERIFY_FILTER] = {"min_level": min_level.value}

def ensure_default_plugins() -> None:
    if get_settings().plugins:
        return
    with default_settings() as settings:
        pass
    get_settings().set(settings)


def load_baseline(baseline_path: str):
    baseline_data = baseline.load_from_file(baseline_path)
    return baseline.load(baseline_data, filename=baseline_path)


def scan_paths(paths, root: str) -> SecretsCollection:
    secrets = SecretsCollection(root=root)
    for filename in get_files_to_scan(*paths, should_scan_all_files=False, root=root):
        secrets.scan_file(filename)
    return secrets


def format_secrets(secrets: SecretsCollection) -> str:
    lines = []
    for filename, secret in secrets:
        line_number = getattr(secret, "line_number", 0) or 0
        lines.append(f"{filename}:{line_number} {secret.type}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secret detection wrapper (serial scan)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_cmd = subparsers.add_parser("create-baseline", help="Create or update baseline")
    create_cmd.add_argument("--baseline", required=True, help="Baseline path")
    create_cmd.add_argument("--exclude-files", default="", help="Regex for files to exclude")
    create_cmd.add_argument("paths", nargs="*", help="Paths to scan (default: .)")

    scan_cmd = subparsers.add_parser("scan", help="Scan for new secrets")
    scan_cmd.add_argument("--baseline", required=True, help="Baseline path")
    scan_cmd.add_argument("--exclude-files", default="", help="Regex for files to exclude")
    scan_cmd.add_argument("--only-verified", action="store_true", help="Only report verified secrets")
    scan_cmd.add_argument("paths", nargs="*", help="Paths to scan (default: .)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = os.getcwd()
    paths = args.paths or ["."]

    if args.command == "create-baseline":
        ensure_default_plugins()
        configure_filters(args.exclude_files, False, args.baseline)
        secrets = scan_paths(paths, root=root)
        baseline.save_to_file(secrets, args.baseline)
        return 0

    if not os.path.exists(args.baseline):
        print(f"ERROR: baseline not found at {args.baseline}", file=sys.stderr)
        return 2

    baseline_secrets = load_baseline(args.baseline)
    ensure_default_plugins()
    configure_filters(args.exclude_files, args.only_verified, args.baseline)
    scanned = scan_paths(paths, root=root)
    new_secrets = scanned - baseline_secrets

    if new_secrets:
        report = format_secrets(new_secrets)
        if report:
            print(report)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
