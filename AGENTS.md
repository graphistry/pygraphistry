# AGENTS.md

**Who this is for: coding agents** (Claude Code, Codex, Cursor, Copilot and others). It holds the rules that apply to every agent task in this repository. People read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

| Document | For whom | What it is |
|---|---|---|
| `AGENTS.md` (this file; `CLAUDE.md` points here) | agents | rules for every task |
| `.agents/skills/contribute/SKILL.md` | agents | the first-contribution flow |
| `CONTRIBUTING.md` | people | the first-contribution flow |
| `DEVELOP.md`, `ARCHITECTURE.md` | people and agents | reference: setup, test commands, CI, release; code layout |
| `ai/` | agents | topic guides for developing this library |

## Skill Locations

Canonical skills live under:

- `.agents/skills/`

## Platform Path Conventions

- **OpenAI Codex**: uses `.agents/skills/`
- **Claude Code**: often uses `.claude/skills/`
- **Other tools (Cursor/Copilot/etc.)**: may use `.agents/skills/` or tool-specific paths

Recommended local setup:

```bash
# Claude path to shared skills
mkdir -p .claude
ln -s ../.agents/skills .claude/skills
```

## Guidance Priority

1. Follow repository docs and tests as source of truth.
2. Keep edits minimal and verifiable.
3. Run relevant lint/tests for touched areas, and run `./bin/lint.sh` before every push (the whole lint gate: ruff, type-hygiene and comment-density ratchets, relative imports, plus the quick CI guards `ci_gpu_gate_audit.py` and `ci_cypher_surface_guard.py`, enabled by default with `LINT_QUICK_GUARDS=1`; set it to `0` to skip those two). On Windows or without Bash, run `ruff check` and `python bin/ci/ci_type_hygiene_guard.py`, `python bin/ci/ci_comment_density_guard.py`, `python bin/ci/ci_gpu_gate_audit.py`, and `python bin/ci/ci_cypher_surface_guard.py` from the repository root.
4. No dynamic typing in code you add under `graphistry/`: no `hasattr`/`getattr`/`setattr`, no `__module__` or `type(x).__name__` sniffing, no `except AttributeError` probes, no bare `# type: ignore`, no `object` annotations. Use the typed helpers in `graphistry/Engine.py` (`is_series_like`, `is_cudf_series`, `is_polars_series`, `is_polars_df`), `isinstance`, or a Protocol. See `DEVELOP.md` "Type Hygiene Guard".
5. Reviewing someone else's pull request (for the reviewing agent): first follow the mandatory [screen-untrusted skill](.agents/skills/screen-untrusted/SKILL.md) before fetching, installing, running, or reading contributor text. The AI must read the natural-language text; regexes and hints only triage. Record the exact SHA, manifest, sources, diff categories, natural-language read, findings and non-findings. A new commit voids the screen. After a pass, run only sandboxed tests in a throwaway tree with no network or credentials and a memory cap.

## First Contribution

When a person asks you to work on an issue or to open a pull request, follow the `contribute` skill: [.agents/skills/contribute/SKILL.md](.agents/skills/contribute/SKILL.md). It is the whole flow, written for agents (claim, branch from `develop`, checks cheapest first, engines, benchmark, draft pull request). Do not work from `CONTRIBUTING.md`: that page is the same flow written for people.

## Where Issues Go

- File performance issues in [graphistry/pyg-bench](https://github.com/graphistry/pyg-bench), not here: slowdowns, regressions, profiling findings, benchmark results, and proposals whose motivation is speed or memory. Link the pygraphistry code path or PR from the pyg-bench issue.
- Correctness, API, docs, and engine-support issues stay in pygraphistry. When a perf investigation finds a wrong result, file that bug here and cross-link it.
- This routing applies only to people with access to the benchmark, TCK and evals repositories; outside contributors have no access, so they file in pygraphistry and a maintainer moves the issue.

## Release Assistant Reminder (maintainers only; skip when contributing)

- During publish assistance, after pushing tag `X.Y.Z`, always check `.github/workflows/publish-pypi.yml` run state.
- Treat `Publish distribution to PyPI` in `waiting` as an expected manual gate for environment `pypi-release`; explicitly instruct maintainer to click `Review deployments` before expecting PyPI completion.
- After the workflow completes, verify both TestPyPI and PyPI public metadata report `X.Y.Z`; PyPI may lag briefly after the job succeeds.
- Create or verify the matching GitHub Release for tag `X.Y.Z`, then check Read the Docs. `latest` may update automatically, while tag-specific versions can require maintainer activation in the RTD dashboard.
- Record release follow-ups discovered during readiness audits as issues in the owning repo before final handoff.

## Published Benchmarks (maintainers only; skip when contributing)

- Never remove or suppress a published benchmark number without the maintainer's explicit approval. Stale numbers remain published with their measurement date, a staleness label, and a linked re-measure issue.
