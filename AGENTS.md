# AGENTS.md

This file defines AI assistant guidance for this repository.

## Skill Locations

Canonical skills live under:

- `agents/skills/`

For tools that expect `.agents/skills/`, use a local symlink:

```bash
ln -s agents .agents
```

## Platform Path Conventions

- **OpenAI Codex**: prefers `.agents/skills/`
- **Claude Code**: often uses `.claude/skills/`
- **Other tools (Cursor/Copilot/etc.)**: may use `.agents/skills/` or tool-specific paths

Recommended local setup:

```bash
# Canonical -> tool-expected path
ln -s agents .agents

# Claude path to shared skills
mkdir -p .claude
ln -s ../.agents/skills .claude/skills
```

`/.agents/` is ignored in git to keep local tool wiring out of commits.

## Guidance Priority

1. Follow repository docs and tests as source of truth.
2. Keep edits minimal and verifiable.
3. Run relevant lint/tests for touched areas.

## Contract Bundle Bump Protocol (AI-critical)

If you edit exported contract shapes for these bundles, you MUST update bundle version metadata in the matching `contract_version.py` file:

- `graphistry_frontend` bundle sources:
  - `graphistry/models/surfaces/graphistry_frontend/react_settings.py`
  - `graphistry/models/surfaces/graphistry_frontend/url_params.py`
  - `graphistry/models/surfaces/graphistry_frontend/axis.py`
  - `graphistry/models/surfaces/graphistry_frontend/contract_version.py`
- `graphistry_server_dataset` bundle sources:
  - `graphistry/io/contracts/graphistry_server/dataset.py`
  - `graphistry/io/contracts/graphistry_server/contract_version.py`

Required steps when shape/default/mapping changes:

1. Bump `*_CONTRACT_VERSION`.
2. Register the new computed signature in `*_CONTRACT_SIGNATURES_BY_VERSION`.
3. Optionally set `*_UPSTREAM_VERSIONS` pins when targeting a known upstream release.

Validation expectation:

- Run `graphistry/tests/test_validate_settings.py`.
- Import-time checks in `contract_version.py` will raise with actionable instructions if version/signature bookkeeping is missing.
