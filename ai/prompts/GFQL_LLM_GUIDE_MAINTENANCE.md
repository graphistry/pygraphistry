# LLM_GUIDE.md Maintenance Process

**Purpose**: Update `LLM_GUIDE.md` when GFQL features change or user patterns emerge.
**For**: Maintainers doing updates after code changes, LLM failures, or user requests.

## Quick Start

**Scenarios**:
1. **New GFQL feature** → [Change Types](#change-types)
2. **Breaking change** → [Breaking Changes](#breaking-changes)
3. **Emergency rollback** → [Rollback](#breaking-change-rollback-emergency)
4. **Users asking "how do I...?"** → [Common Patterns](#common-patterns)
5. **LLMs generating invalid JSON** → [LLM Failures](#llm-failures)
6. **Multiple changes** → [Multi-Change Releases](#multi-change-releases)

**See also**: the end-to-end checklist in `ai/docs/gfql/calls_checklist.md`
whenever a `call()` surface changes.

## Update Triggers

| Trigger | Source | Action |
|---------|--------|--------|
| New algorithm | `call_safelist.py` changes | Add to Graph Algorithms |
| New predicate | `predicates/*.py` added | Add to Predicates |
| Breaking change | `ASTSerializable.py` modified | Update ALL examples + migration |
| Common user query | 3+ issues same pattern | Add to Common Patterns |
| LLM failure | 3+ tickets invalid JSON | Add to Common Mistakes |
| New domain demand | 3+ requests from industry | Add domain section |

## Line Limit Policy

**Target**: <500 lines · **Acceptable**: 500-520 · **Warning**: >520 · **Hard**: 550

### Approaching limit:
- **<510**: Prefer compact variations
- **510-520**: OK for critical content
- **>520**: MUST condense OR deprecate OR escalate

### Condensation strategies:

| Strategy | Savings | Impact | When |
|----------|---------|--------|------|
| Remove blank lines | ~10 | Low | First |
| Inline headings | ~8 | Low | Second |
| Compact variations | ~12 | Med | Third |
| Remove domain | ~22 | High | Last |

**Domain priority**: Tier 1 (Cyber, Healthcare, Finance) > Tier 2 (Supply Chain, Fraud) > Tier 3 (Social, E-commerce)

## Deprecation Policy

**When**: 515-520 lines OR obsolete features

### Criteria (any 2+):
- [ ] Usage <2%
- [ ] Superseded by newer algorithm
- [ ] Redundant with existing
- [ ] Domain-specific (move to domain section)
- [ ] No requests 6+ months

### Process:
1. Create issue `[DOC-DEPRECATION]` with usage data
2. Wait 2 weeks for feedback
3. If no objections: Remove + CHANGELOG note
4. If objections: Re-evaluate or compact

### Trajectory Monitoring:
```
Current: [X] lines (v[Y.Z])
Growth: ~4 lines/scenario
Remaining: [550-X] → ~[(550-X)/4] scenarios
```

**Thresholds**:
- **515-520**: Yellow - Identify candidates
- **520-540**: Orange - MUST deprecate OR justify
- **540-550**: Red - Required before release
- **>550**: Critical - Immediate action

### Long-Term Options:
**A. Deprecation + Compact** (2-3 releases): Remove 2-3 low-usage algs (~45) + compact 3-4 examples (~40) = 85 lines → 20+ scenarios
**B. Increase limit 600-650** (medium-term): Modern LLMs support 128k tokens → 20-25 scenarios
**C. Split document** (long-term): Core <400 + Extended <250 → modular but overhead

**Recommendation**: A → monitor → B if needed

## Change Types

### New Algorithm

**Trigger**: New in `call_safelist.py` (e.g., `betweenness`, `triangle_count`)

**Research**:
```bash
grep -n "new_fn" graphistry/compute/gfql/call_safelist.py
grep "new_fn" graphistry/tests/compute/ -rn
```

**Update**: `## Graph Algorithms` (add example) + `## Call Functions` (list)
- Cross-check `ai/docs/gfql/calls_checklist.md` for required code/tests/doc updates

**Format**:
```markdown
**[Name] ([Use Case]):**
```python
# Dense: call('fn', {'param': val})
```
```json
{"type": "Call", "function": "fn", "params": {"param": val}}
```
**Use case**: [One-line explanation]
```

**Placement**: END of Graph Algorithms (chronological)
**Lines**: ~15-18 · **Time**: 25-30 min

### New Predicate

**Trigger**: New in `predicates/*.py` (e.g., `Matches`, `Between`)

**Research**:
```bash
read graphistry/compute/predicates/new_pred.py
grep "to_json" graphistry/compute/predicates/new_pred.py -A10
```

**Update**: `## Predicates` (by category)

**Format**:
```markdown
**[Category]:**
```json
{"type": "Name", "param": val}
```
**Note**: [When to use vs similar, if applicable]
```

**Placement**: Within Predicates by category
**Lines**: ~3-5 · **Time**: 20-25 min

### Common Patterns

**Trigger**: Users asking "how do I [pattern]?" (e.g., degree filtering, ego networks)

**Workflow** (check duplicates FIRST):
1. `grep -i "keywords" LLM_GUIDE.md`
2. Exact exists → STOP
3. Similar with terminology gap → Enhance with aliases inline
4. None exists → Decision tree

**Decision tree**:
1. **3+ domains?** → Same semantics: full example · Different semantics: cross-reference
2. **Variation?** → Compact (~5 lines)
3. **Domain-specific?** → Add to domain section
4. **Edge case?** → Generation Rules or Common Mistakes

**Compact format** (saves space):
```markdown
**[Pattern]:** `[dense]` Pattern: [explanation]. [params].
```

**Full format**:
```markdown
**[Pattern]:**
```python
# Dense: [code]
```
```json
{[JSON]}
```
**Pattern**: [Explanation]
```

**Lines**: ~5 (compact) or ~15-20 (full) · **Time**: 25-35 min

### Breaking Changes

**Trigger**: `ASTSerializable.py` field renames, new required fields, type changes

**Research**:
```bash
git diff master graphistry/compute/ASTSerializable.py
grep '"old_field"' LLM_GUIDE.md | wc -l
```

**Update**: Core Types + ALL examples + version note + migration guide

**Automation**:
```bash
grep -c '"old"' LLM_GUIDE.md  # Count
grep -n '"old"' LLM_GUIDE.md  # Preview
# Editor find-replace: "old": → "new":
```

**Version note** (after title):
```markdown
**Version**: GFQL v[X.Y]+. Breaking v[X.Y]: [desc]. See Common Mistakes for migration.
```

**Migration** (in Common Mistakes):
```markdown
❌ v[OLD] `old` → ✅ v[NEW] `new` (breaking v[X.Y])
```

**Validate**:
```bash
grep '"old"' LLM_GUIDE.md  # Empty
python3 plans/.../generate_examples.py
```

**Lines**: +10-15 (may update 30-50 existing) · **Time**: 40-50 min
**Critical**: Use automation, validate thoroughly

### Breaking Change Rollback (Emergency)

**Scenario**: Production issue from breaking change

**Workflow**:
1. **Identify**: `git log --oneline LLM_GUIDE.md -5`
2. **Check isolation**: `git show <hash> --stat`
3. **Revert**: `git revert <hash> --no-edit`
4. **Remove version note** (line ~3)
5. **Validate**:
   ```bash
   grep '"deprecated"' LLM_GUIDE.md  # Empty
   grep '"correct"' LLM_GUIDE.md | wc -l  # Expected
   ```
6. **Check Common Mistakes** still accurate
7. **Commit**: `git commit -m "docs: Rollback vX.Y (INC-XXX)"`
8. **Deploy** (emergency bypass)
9. **Monitor 30 min**

**Checklist**:
- [ ] Old removed (grep empty)
- [ ] Correct restored (count matches)
- [ ] Version note updated
- [ ] Common Mistakes accurate
- [ ] Other features NOT reverted
- [ ] Line count OK
- [ ] Incident referenced

**Emergency bypass** (when): P0/P1 outage, LLM breakage, security, data breach
**Process**: Direct push, post-mortem PR

**Best practice - Commit isolation**:
✅ commit 1: Add betweenness · commit 2: Breaking change
❌ commit 1: Add betweenness + breaking change
**Why**: Clean rollback without losing features

**Time**: 25-30 min (15 active + 15 monitoring)

### LLM Failures

**Trigger**: 3+ tickets same error pattern

**Examples**: `"filter"` vs `"filters"`, `"node_type"` vs `"type"`, filters outside Node, deprecated fields

**Process**: Analyze → Categorize → Add to Common Mistakes

**Format** (ultra-compact):
```markdown
❌ `"wrong"` → ✅ `"correct"` ([context])
❌ Wrong structure → ✅ Correct: [inline]
```

**Placement**: Common Mistakes
**Coverage**: Top 5 errors (80/20)
**Lines**: ~7 total · **Time**: 40-50 min

### New Domain

**Trigger**: 3+ requests from industry (healthcare, logistics, energy)

**Process**:
1. Identify 3-5 entity types
2. Identify relationships
3. Identify common filters
4. Map to Font Awesome 4 icons
5. Create example: search + filter (+ optional: algorithm + viz)

**Format**:
```markdown
### [Domain]
**Use case**: [Description]
**Dense**: `[Python]`
**JSON**:
```json
{[AST]}
```
```

**Placement**: After existing domains
**Lines**: ~22-28 · **Time**: 50-60 min
**Risk**: May exceed limit - see [Escalation](#escalation-process)

## Multi-Change Releases

**Scenario**: Multiple GFQL changes (algorithm + predicate + param)

**Workflow**:
1. Research ALL changes first
2. Estimate total lines (additions - removals)
3. If exceeds: Condense BEFORE adding
4. Update order: Core Types → Predicates → Patterns → Algorithms → Domains
5. Validate together
6. Cross-reference release notes

**Example budget**:
```
Start: 500
+1 (edge param) +3 (predicate) +15 (algorithm) = +19
Target: 519 → Remove 19 OR use 500-520 range
```

**Condensation**: Blank lines (-10), inline headings (-8), compact (-5)
**Time**: 50-60 min for 3 changes

## Escalation Process

**When**: Stuck after all options, limit exceeded, conflicting priorities

**Steps**:
1. Document: Tried + failed
2. Options table: Pros/cons
3. Recommendation + rationale
4. Create issue `[DOC-DECISION]`
5. Tag maintainer
6. Await decision

**Examples**: New domain > 520 lines, critical Common Mistakes won't fit, conflicting domain priority, breaking change needs more space

## Update Workflow

1. Identify trigger (1-2 min)
2. Research: `read`, `grep`, tests/docs (3-6 min)
3. Check line budget: `wc -l`, estimate (1-2 min)
4. Create example: Python → `.to_json()` (5-10 min)
5. Decide placement: Decision tree (2-3 min)
6. Draft: Dense + JSON + annotations (6-10 min)
7. Verify: `wc -l`, validate, cross-ref (3-5 min)
8. Update this doc if new pattern (5 min)

**Total**: 25-50 min

## Maintenance Checklist

- [ ] Line count <520 (<500 ideal)
- [ ] Dual format (Dense + JSON)
- [ ] Inline annotations (`// required`, `// default`)
- [ ] Qualified lists ("Examples:", "and more")
- [ ] Real values (not "foo"/"bar")
- [ ] JSON validated (`generate_examples.py`)
- [ ] Common Mistakes updated (breaking/LLM)
- [ ] Cross-referenced release notes

## Validation

### JSON
```bash
python3 -c "import json; json.loads('{\"type\": \"Node\"}')"
python3 plans/.../generate_examples.py
```

### Line Count
```bash
wc -l plans/.../LLM_GUIDE.md
```

### Completeness
```bash
grep "new_feature" LLM_GUIDE.md
grep '"deprecated"' LLM_GUIDE.md  # Empty
```

## Common Mistakes (Meta)

**Placement uncertainty**: Algorithms → END (chrono), Predicates → by category, Patterns → decision tree, Domains → after existing

**Exceeding limit**: Check 500-520 acceptable (essential), else condense

**Breaking without migration**: ALWAYS add version note + migration

**Missing similar check**: Grep for related, add note

**Placeholder values**: Use real from tests/docs

## Template Patterns

### Full Example
```markdown
**[Name] ([Use Case]):**
```python
# Dense: [code]
```
```json
{[JSON]}
```
**Pattern**: [Explanation]
```
**Lines**: ~15-20

### Compact Variation
```markdown
**[Name]:** `[dense]` Pattern: [explanation]. [params].
```
**Lines**: ~5

### Common Mistake
```markdown
❌ `"wrong"` → ✅ `"correct"` ([context])
```
**Lines**: 1

### Cross-Reference

**Use**: Same structure, different semantics across domains

**Example**: "Trace propagation" = laundering (fraud), malware (cyber), contamination (supply)

**Template**:
```markdown
**[Pattern] ([Generic])**
```python
# Dense: [generic]
```
**Pattern**: [Structure]
**Domain examples**:
- **[Domain1]**: [Specific] - See [Link](#link)
- **[Domain2]**: [Specific] - See [Link](#link)
**Key params**: [Important]
```

**Lines**: ~10-12 (vs 3×20=60)
**Benefits**: No duplication, connects generic→specific, searchable

**Validate**:
- [ ] Examples exist
- [ ] Links work
- [ ] Truly structural not semantic

## File Locations

**Main**: `plans/gfql_json_wire_spec_generator/LLM_GUIDE.md`
**Process**: `plans/gfql_json_wire_spec_generator/PROCESS_DISTILLATION.md`
**Validation**: `plans/gfql_json_wire_spec_generator/generate_examples.py`

**Monitor**:
- `graphistry/compute/gfql/call_safelist.py` - Algorithms
- `graphistry/compute/predicates/*.py` - Predicates
- `graphistry/compute/ASTSerializable.py` - Breaking
- `graphistry/tests/compute/` - Patterns

## Success Signals

**Green**: LLM >80% success, users reference guide, no support tickets, examples work across releases

**Yellow**: Approaching 500, 2-3 new domains pending, occasional LLM failures

**Red**: Multiple "how do I" same query, 3+ tickets same invalid JSON, release breaks examples, test failures

## Background

**Goal**: LLMs generate valid GFQL JSON for common use cases
**Not**: Implementation details, wire protocol internals, exhaustive API
**Audience**: LLMs (Claude, GPT)

**Principles**:
1. Purpose over types (organize by DO not IS)
2. Examples over explanation
3. Rich details (real palettes, icons, terms)
4. Top-down (context → details)
5. Dual format (Dense + JSON)
6. Inline annotations (`// required`)

**Evolution**: 5 phases: "document wire" → "enable LLM codegen"

## When to Use

**Use**: LLM docs, code generation guides (JSON/DSL/config), teaching APIs to AI

**Don't**: Developer docs (need impl details), reference manuals (exhaustive), internal docs (org context)

## References

**Sources**: `gfql/spec/cypher_mapping.md` (analogies), `gfql/about.rst` (multi-step), `gfql/builtin_calls.rst` (rich params)

**Artifacts**: `PLAN.md` (6 phases + simulations), `PHASE4_RESEARCH_SUMMARY.md` (gap analysis), `simulations/` (8+ personas)

**Result**: `LLM_GUIDE.md` 484-520 lines, self-contained, LLM-optimized
