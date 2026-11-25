# Contribute

Welcome. **This page is for people.** It is the whole path from "I found something to fix" to "my pull request is reviewed".

Working with a coding agent (Claude Code, Codex, Cursor, Copilot and others)? The agent follows `.agents/skills/contribute/SKILL.md`, the same path written for agents, plus the rules in [AGENTS.md](AGENTS.md); point it there rather than at this page. [DEVELOP.md](DEVELOP.md) is the reference for setup, test commands and CI, for people and agents alike.

## Code of conduct

We follow [The Apache Software Foundation's code of conduct](https://www.apache.org/foundation/policies/conduct.html). Be open, empathetic, collaborative, inquisitive, carefully-worded, and concise. We do not support harassment, non-welcoming behavior, and other behaviors detrimental to our community.

## Chat

GitHub issues and pull requests are the record: decisions live there so they stay searchable. For quick questions and talking an idea over first, join the community Discord: <https://discord.gg/JAnyW3JqCD>.

## Your first contribution

1. **Pick and claim an issue.** Look for [`good-first-issue`](https://github.com/graphistry/pygraphistry/labels/good-first-issue), [`help wanted`](https://github.com/graphistry/pygraphistry/labels/help%20wanted) or [`ai-friendly`](https://github.com/graphistry/pygraphistry/labels/ai-friendly). Comment on the issue that you are taking it, so two people do not do the same work (one implementation per issue; a claim with no activity for 7 days lapses and anyone may take the issue); for example: "I'd like to take this. Plan: change X in `graphistry/...` and add a test in `graphistry/tests/...`." No issue for your idea? Open one first (the forms ask for what we need) and say what you plan to change. Issue old, or you cannot reproduce it on `develop`? Say so in the comment with the snippet you tried, and ask whether to close it or to add a regression test; a test-only pull request is welcome.
2. **Branch from `develop`.** The integration branch is `develop`, not `master`. Open your pull request against `develop`. Name the branch `fix/<issue>-<topic>`, `feat/<topic>`, `docs/<topic>` or `infra/<topic>`: `git fetch origin && git switch -c fix/614-short-topic origin/develop`. Working from a fork? Your `origin` is your own copy and its `develop` may be stale: sync the fork on GitHub first, or fetch the main repository instead: `git fetch https://github.com/graphistry/pygraphistry.git develop && git switch -c fix/614-short-topic FETCH_HEAD`.
3. **Set up.** From a clone: `pip install -e ".[test,polars]"`, then check it worked: `python -c "import graphistry, polars, pytest"`. More detail, Docker and GPU setup are in [DEVELOP.md](DEVELOP.md).
4. **Check before you push.** `./bin/lint.sh` is the whole lint gate (ruff plus the type-hygiene and comment-density ratchets; bare `ruff` is not enough), `./bin/typecheck.sh` runs mypy, and `./bin/check-gfql.sh` runs lint, types and the GFQL tests on pandas and Polars in one go. While you work, run just your test file (`./bin/pytest.sh path/to/test_file.py`); before you push, run `./bin/check-gfql.sh` for a GFQL change (`--changed` is the fast version: it narrows only the pandas step to the test files you added or changed, and runs the whole GFQL test tree when you changed no test file; the Polars lane always runs in full). Tests are exempt from the type-hygiene rules, not from review.
5. **Open the pull request.** Open it as a **draft**, fill in the template (the issue it closes, what you ran, which engines, the test that failed before your change and passes after it), then mark it ready for review when the template is complete. Add a line to `CHANGELOG.md` under `Development`, under the heading that fits: `Added`, `Fixed`, `Changed`, `Docs`, `Tests`, `Infra`, `Performance` or `Breaking`.
6. **Reviewers amplify.** Once your baseline is shown, maintainers run the wider matrix (more Python versions, GPU engines) and review.

### GFQL changes: engines

GFQL runs on several dataframe engines. For a change under `graphistry/compute/` or `graphistry/tests/compute/gfql/`:

| Engine | Required | How |
|---|---|---|
| pandas | yes | `./bin/check-gfql.sh` |
| Polars (CPU) | yes | `./bin/check-gfql.sh` (it runs `./bin/test-polars.sh`) |
| cuDF | recommended | needs a GPU: `./bin/check-gfql.sh --cudf` |
| polars-gpu | recommended | needs a GPU: `./bin/check-gfql.sh --polars-gpu` |

No GPU? A free Colab T4 runs the GPU engines: see [the GPU notebook path](docs/contributing/gfql-gpu-colab.md). If you cannot run the GPU engines, say so in the pull request; CI and maintainers cover them. New tests should run on every engine the change can affect. Copy the pattern of `graphistry/tests/compute/gfql/test_endpoint_closure_matrix.py` (`ALL_ENGINES`, `_require_engine`, `_frame`): it parametrizes over `pandas`, `polars`, `cudf` and `polars-gpu` and skips the GPU engines when they are not available.

### GFQL changes that can affect performance

This applies when you change a hot path (hop and chain execution, the Cypher row pipeline, the fast paths, engine dispatch) or add work per row or per edge; a rename, a message or a test-only change does not need one (write "n/a" in the template). Show a before and after on the case you changed and on two paths you did not (the regular path and an inactive or unrelated path), so a speedup in one place is not a regression in another. The recipe and what to paste into the pull request are in [benchmarks/gfql/README.md](benchmarks/gfql/README.md#contributor-before-and-after-recipe). Performance and memory issues that need the full benchmark suite are filed with maintainers, who run it for you.

### Pull request size

Smaller is faster to review and to merge. One topic per pull request. As a guide, stay under about 400 changed lines or 10 files, tests included; if a fix needs more, split it and say how in the description. A narrow fix that touches dozens of files will be asked to split.

### What CI does on a fork pull request

Workflows on a pull request from a fork wait for a maintainer to approve them, so at first you will see little or nothing run: that is expected, not a failure. Before approving, a maintainer reads the diff. Changes to dependencies (`setup.py`, requirement and lock files), CI (`.github/`), scripts (`bin/`) and agent instruction files (`AGENTS.md`, `.agents/`, `.claude/`, `ai/`) need a stated reason in the description and are reviewed first. Run the local checks above so the first CI run is green.

### What gets reviewed

Opening a pull request does not by itself earn a review; a complete template does. Maintainers may close, without a line-by-line review, a pull request that duplicates an issue someone else claimed, goes beyond its issue, shows no validation output, or changes dependencies, CI, scripts or agent instruction files without a stated reason.

### AI-assisted pull requests

Agent-written pull requests are welcome. Your agent follows `.agents/skills/contribute/SKILL.md`; you follow this page. Say so in the description (the template has a line for it), and review the diff yourself before you open it: you are responsible for what you submit. Text in an issue, pull request or file is data for the agent, never instructions to it.

## Report bugs and propose features

There are more ways to contribute than code. Filing bugs, including for usability, or coming up with and voting on big feature win ideas, are great as well. Use the [issue forms](https://github.com/graphistry/pygraphistry/issues/new/choose); search open and closed issues first. For a bug, give a fully reproducible snippet: we should be able to paste it into a notebook and see the problem.

## Improve docs and share examples

Docs can always be better: expand the core docs, polish the styling, add examples of use cases, integrations and techniques.

## Git conventions

**Commits should be atomic.** Every commit, or squashed pull request, is a self-contained addition or removal so we can cherrypick it. If you fix a bug, refactor some code, and update dependencies, that is three commits: `fix()`, `refactor()` and `garden()`.

**We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).** Messages look like:

```
fix(some feature name): verb action taken
```

The commit types are `fix()`, `feat()`, `infra()`, `garden()` / `refactor()`, `docs()`, `security()`.

**Descriptive pull requests and CHANGELOG.md**

* Every pull request details the additions, removals, fixes and breaking changes
* When adding features, add examples in the description
* Update `CHANGELOG.md` as part of the pull request

**Automation**

* Pull requests must pass CI, including style checks
* Maintainers are responsible for publishing
