import re
from pathlib import Path

import pytest


BLOCKED_NAME = re.compile(r"^(report|summary|findings|analysis).*\.md$", re.IGNORECASE)
CONTENT_TOKEN = re.compile(r"[\w<>{}*./-]+\.md")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("write <dimension>/report.md", True),
        ("write <dimension>/REPORT.md", True),
        ("write <dimension>/summary.md", True),
        ("write <dimension>/findings-graphistry-compute-hop.md", True),
        ("write <dimension>/analysis.md", True),
        ("write security-report.md", False),
        ("write issues-report.md", False),
        ("write wave-report.md", False),
        ("write final-report.md", False),
        ("write feature-risk-analysis.md", False),
    ],
)
def test_blocked_content_fixture(line, expected):
    assert bool(blocked_names(line)) is expected


def blocked_names(content):
    tokens = CONTENT_TOKEN.findall(content)
    basenames = (token.rsplit("/", 1)[-1] for token in tokens)
    return sorted({basename for basename in basenames if BLOCKED_NAME.match(basename)})


@pytest.mark.parametrize(
    "pre_s0_line",
    [
        "  <dimension>/findings-<file-slug>.md",
        "  <dimension>/report.md",
        "  adversarial/report.md",
        "- `findings.md` (bugs, repros, evidence)",
    ],
)
def test_pre_s0_skill_content_is_flagged(pre_s0_line):
    token = CONTENT_TOKEN.search(pre_s0_line).group(0)
    assert blocked_names(pre_s0_line) == [token.rsplit("/", 1)[-1]]


def test_skill_files_use_subagent_safe_names():
    skills = Path(__file__).resolve().parents[1] / ".agents" / "skills"
    blocked = {
        str(path.relative_to(skills)): blocked_names(path.read_text(encoding="utf-8"))
        for path in skills.rglob("*.md")
    }
    assert {path: names for path, names in blocked.items() if names} == {}
