# .agents

Assistant-facing support material for this repository.

## Purpose

- Keep reusable agent docs and helper scripts in one place.
- Keep task-oriented skill definitions in `ai/skills/`.

## Layout

```text
.agents/
├── README.md
├── assets/
│   ├── find_dynamic_imports.sh
│   ├── generate_comment_inventory.sh
│   └── pysa_extract_callers.py
└── docs/
    └── gfql/
        ├── README.md
        ├── calls_checklist.md
        ├── conformance.md
        ├── oracle.md
        └── predicates_checklist.md
```

## Related Paths

- Skills: `ai/skills/`
- Skill index: `ai/skills/SKILLS.md`
- Plans (gitignored): `plans/`

## Quick Commands

```bash
# Generate dynamic import inventory for refactors
./.agents/assets/find_dynamic_imports.sh master plans/<task>/dynamic_imports.md

# Generate comment inventory for cleanup passes
./.agents/assets/generate_comment_inventory.sh master plans/<task>/comment_inventory.md

# Extract callers from pysa call graph
python3 .agents/assets/pysa_extract_callers.py pysa_results/call-graph.json PlotterBase.PlotterBase.bind
```
