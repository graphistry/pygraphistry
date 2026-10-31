import re
from pathlib import Path

import pytest


BLOCKED_NAME = re.compile(r"^(report|summary|findings|analysis).*\.md$", re.IGNORECASE)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.md", True),
        ("REPORT.md", True),
        ("summary.md", True),
        ("findings-graphistry-compute-hop.md", True),
        ("analysis.md", True),
        ("security-report.md", False),
        ("issues-report.md", False),
        ("wave-report.md", False),
        ("final-report.md", False),
        ("feature-risk-analysis.md", False),
    ],
)
def test_blocked_name_fixture(filename, expected):
    assert bool(BLOCKED_NAME.match(filename)) is expected


def test_skill_files_use_subagent_safe_names():
    skills = Path(__file__).resolve().parents[1] / ".agents" / "skills"
    blocked = [str(path) for path in skills.rglob("*.md") if BLOCKED_NAME.match(path.name)]
    assert blocked == []
