#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import pathlib
import sys
import unicodedata

ROOT = pathlib.Path.cwd()
TARGETS = [
    ROOT / "README.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "DEVELOP.md",
    ROOT / "docs" / "source",
]
TEXT_EXTENSIONS = {".md", ".rst", ".ipynb", ".txt"}


def iter_targets():
    for target in TARGETS:
        if target.is_file():
            yield target
            continue
        if not target.is_dir():
            continue
        for path in sorted(target.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
                yield path


issues = []
for path in iter_targets():
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    for line_no, line in enumerate(text.splitlines(), start=1):
        for col_no, ch in enumerate(line, start=1):
            if ord(ch) > 0xFFFF:
                rel = path.relative_to(ROOT)
                name = unicodedata.name(ch, "UNKNOWN")
                issues.append(f"{rel}:{line_no}:{col_no}: U+{ord(ch):04X} {name}")

if issues:
    print("ERROR: Found non-BMP Unicode characters that frequently break pdflatex docs builds:")
    for issue in issues:
        print(issue)
    sys.exit(1)

print("OK: No non-BMP Unicode characters detected in docs-fed text sources.")
PY
