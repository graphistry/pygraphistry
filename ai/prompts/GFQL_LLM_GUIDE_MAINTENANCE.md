# LLM_GUIDE.md Maintenance Process

**Purpose**: Update `LLM_GUIDE.md` when GFQL features change or user patterns emerge.

**For**: Maintainers doing updates after code changes, LLM failures, or user requests.

---

## Quick Start: I need to update the guide

**Scenarios**:
1. **New GFQL feature** → See [Change Types](#change-types)
2. **Breaking change in release** → See [Breaking Changes](#breaking-changes)
3. **Emergency: Need to rollback breaking change** → See [Breaking Change Rollback](#breaking-change-rollback-emergency)
4. **Users asking "how do I...?"** → See [Common Patterns](#common-patterns)
5. **LLMs generating invalid JSON** → See [LLM Failures](#llm-failures)
6. **Multiple changes in one release** → See [Multi-Change Releases](#multi-change-releases)

---

## Update Triggers

| Trigger | Source | Action |
|---------|--------|--------|
| **New algorithm** | `call_safelist.py` changes | Add to Graph Algorithms |
| **New predicate** | `predicates/*.py` added | Add to Predicates |
| **Breaking change** | `ASTSerializable.py` modified | Update ALL examples + migration |
| **Common user query** | 3+ GitHub issues/tickets same pattern | Add to Common Patterns |
| **LLM failure pattern** | 3+ tickets with invalid JSON | Add to Common Mistakes |
| **New domain demand** | 3+ requests from industry | Add domain section |

---

## Line Limit Policy

**Target**: <500 lines (ideal for LLM context)
**Acceptable**: 500-520 lines (if essential content like breaking changes, domain examples, error prevention)
**Warning**: >520 lines (requires justification or condensation)
**Hard limit**: 550 lines (MUST condense or split document)

### When approaching limit:
- **<510 lines**: Prefer compact variations over full examples
- **510-520 lines**: Acceptable for critical content (new domains, breaking changes, Common Mistakes)
- **>520 lines**: MUST condense existing content OR deprecate old features OR escalate decision

### Condensation strategies (apply in order):

| Strategy | Savings | Readability Impact | When to Use |
|----------|---------|-------------------|-------------|
| Remove blank lines | ~10 lines | Low | First step |
| Inline domain headings | ~8 lines | Low | Second step |
| Convert full examples to compact variations | ~12 lines | Medium | Third step |
| Remove lowest-priority domain | ~22 lines | High | Last resort |

**Domain priority** (for space constraints):
- **Tier 1**: Cyber Security, Healthcare, Finance (enterprise demand)
- **Tier 2**: Supply Chain, Fraud (industry demand)
- **Tier 3**: Social Media, E-commerce (consumer demand)

---

## Deprecation Policy

**When**: Approaching line limit (515-520 lines) OR features become obsolete

### Deprecation Criteria

**Evaluate for deprecation if**:
- [ ] Usage <2% (check analytics, GitHub issue mentions, or forum discussions)
- [ ] Superseded by newer algorithm (e.g., old PageRank API → new PageRank API)
- [ ] Redundant with existing examples (e.g., betweenness + closeness both centrality → keep 1)
- [ ] Domain-specific (better suited for domain example, not core guide)
- [ ] Rarely requested (no user questions in 6+ months)

### Deprecation Process

1. **Identify candidate**: Use criteria above
2. **Create GitHub issue**: "Deprecate [algorithm/predicate] from LLM_GUIDE.md"
   - Tag: `[DOC-DEPRECATION]`
   - Include: Usage data, rationale, alternative recommendations
3. **Wait 2 weeks** for community feedback
4. **If no objections**:
   - Remove from LLM_GUIDE.md
   - Add to CHANGELOG with deprecation note
   - Update cross-references
5. **If objections**:
   - Re-evaluate criteria
   - Consider compromise (compact variation vs full example)

### Line Limit Trajectory Monitoring

**Track growth rate**:
```
Current: [X] lines (as of v[Y.Z])
Average growth: ~4 lines/scenario
Capacity remaining: [550 - X] lines
Estimated scenarios until limit: ~[(550 - X) / 4] scenarios
```

**Action Thresholds**:
- **515-520 lines**: Yellow flag - Document TODO, identify deprecation candidates
- **520-540 lines**: Orange flag - MUST deprecate OR justify increase
- **540-550 lines**: Red flag - Emergency - deprecation required before next release
- **>550 lines**: Critical - Immediate action (deprecate OR split document)

### Long-Term Sustainability Options

**Option A: Deprecation + Compact Variations** (short-term, 2-3 releases)
- Remove 2-3 low-usage algorithms (~45 lines)
- Convert 3-4 full examples to compact variations (~40 lines)
- **Savings**: ~85 lines → Buys 20+ scenarios (~5-6 releases)

**Option B: Increase Hard Limit to 600-650** (medium-term)
- **Rationale**: LLM context windows increased (GPT-4: 128k tokens vs original 8k)
- **Impact**: Buys 20-25 scenarios (~6-8 releases)
- **Trade-off**: Longer guide, but still fits modern LLM context

**Option C: Split Document** (long-term, major refactor)
- **Core guide**: <400 lines (essential patterns, common use cases)
- **Extended guide**: <250 lines (advanced algorithms, edge cases)
- **Impact**: Modular, scalable, but coordination overhead

**Recommendation**: Start with A (deprecation), monitor growth, evaluate B (limit increase) if needed

---

## Change Types

### New Algorithm

**Trigger**: New function in `call_safelist.py` (e.g., `betweenness_centrality`, `triangle_count`)

**Research**:
```bash
grep -n "new_function" graphistry/compute/gfql/call_safelist.py
grep "new_function" graphistry/tests/compute/ -r --output_mode content -n
grep "new_function" docs/source/gfql/builtin_calls.rst -A 20
```

**Update sections**:
- Primary: `## Graph Algorithms` (add example)
- Secondary: `## Call Functions` (add to list)
- Optional: Domain examples (if highly relevant)

**Format**:
```markdown
**[Algorithm Name] ([Use Case]):**
```python
# Dense: call('function_name', {'param': value})
```
```json
{
  "type": "Call",
  "function": "function_name",
  "params": {"param": value}
}
```
**Use case**: [One-line explanation]
```

**Placement**: Add at END of `## Graph Algorithms` section (chronological order)

**Lines**: ~15-18 lines per algorithm

**Time estimate**: 25-30 min

---

### New Predicate

**Trigger**: New predicate class in `predicates/*.py` (e.g., `Matches`, `Between`, `JsonContains`)

**Research**:
```bash
read graphistry/compute/predicates/new_predicate.py
grep "to_json" graphistry/compute/predicates/new_predicate.py -A 10
grep "NewPredicate" graphistry/tests/compute/predicates/ -r --output_mode content
```

**Update sections**:
- Primary: `## Predicates` (add to appropriate category)
- Check: Similar predicates exist? Add clarifying note

**Format**:
```markdown
**[Category]:**
```json
{"type": "PredicateName", "param1": value, "param2": value}
```
**Note**: [When to use vs similar predicates, if applicable]
```

**Placement**: Within `## Predicates`, grouped by category (Numeric, String, Pattern Matching, etc.)

**Lines**: ~3-5 lines per predicate

**Time estimate**: 20-25 min

---

### Common Patterns

**Trigger**: Multiple users asking "how do I [pattern]?" (e.g., degree filtering, ego networks, temporal aggregation)

**Research Workflow** (ALWAYS check for duplicates first):
1. **Check for duplicates**:
   ```bash
   grep -i "pattern_keywords" LLM_GUIDE.md
   ```
2. **If exact pattern exists**: STOP - Don't duplicate
3. **If similar pattern exists with terminology gap**:
   - Enhance existing example with aliases/synonyms inline
   - Example: "degree" exists but users search "neighbor count" → Add "(neighbor count/connections)"
4. **If no pattern exists**: Proceed with decision tree below

**Decision tree**:
1. **Used by 3+ domains?** → **Decision Point**:
   - **Same semantics**: Add full common pattern example
   - **Different semantics**: Add cross-reference pattern (generic + domain pointers)
2. **Variation of existing pattern?** → Add as compact variation (~5 lines)
3. **Domain-specific?** → Add to relevant `## Domain` section
4. **Edge case?** → Add to `## Generation Rules` or `## Common Mistakes`

**Process**:
1. Identify pattern from GitHub issues/tickets
2. Find implementation in tests: `grep "pattern" graphistry/tests/compute/ -r`
3. Create minimal example
4. Validate JSON
5. Add to appropriate section

**Compact variation format** (saves space):
```markdown
**Use case: [Pattern Name]**
```python
# Dense: [compact code]
```
Pattern: [One-line explanation]. [Key parameter notes].
```

**Full example format**:
```markdown
**[Pattern Name]:**
```python
# Dense: [code]
```
```json
{[full JSON]}
```
**Pattern**: [Explanation]
```

**Lines**: ~5 lines (compact) or ~15-20 lines (full)

**Time estimate**: 25-35 min

---

### Breaking Changes

**Trigger**: Changes to `ASTSerializable.py`, field renames, new required fields, type name changes

**Examples**:
- Field rename: `filter_dict` → `filters`
- New required field: Add `version` to all types
- Type name change: `Node` → `NodeMatch`

**Research**:
```bash
git diff master graphistry/compute/ASTSerializable.py
grep "class AST" graphistry/compute/ast.py
grep '"old_field_name"' plans/gfql_json_wire_spec_generator/LLM_GUIDE.md | wc -l
```

**Update sections**:
1. `## Core Types` (update schemas)
2. **ALL examples** (find-replace for field renames)
3. Add version note at top of guide
4. Add migration guide to `## Common Mistakes`

**Automation strategy**:
```bash
# Count instances
grep -c '"old_field"' LLM_GUIDE.md

# Dry run to see what changes
grep -n '"old_field"' LLM_GUIDE.md

# Use editor find-replace (safer than sed for complex JSON)
# Replace: "old_field": → "new_field":
```

**Version note template** (add after title):
```markdown
**Version Note**: This guide reflects GFQL v[X.Y]+. Breaking change in v[X.Y]: [description]. See Common Mistakes for migration.
```

**Migration guide template** (add to Common Mistakes):
```markdown
❌ v[OLD] `old_syntax` → ✅ v[NEW] `new_syntax` (breaking change v[X.Y])
```

**Validation**:
```bash
# Verify old field names gone
grep '"old_field"' LLM_GUIDE.md  # Should return empty

# Validate JSON examples still work
python3 plans/gfql_json_wire_spec_generator/generate_examples.py
```

**Lines**: +10-15 (version note + migration) but may update 30-50 existing lines

**Time estimate**: 40-50 min

**Critical**: Breaking changes affect EVERY example - use automation, validate thoroughly

---

### Breaking Change Rollback (Emergency)

**Scenario**: Production issue caused by breaking change, need to revert docs immediately

**Rollback Workflow**:
1. **Identify commit**:
   ```bash
   git log --oneline LLM_GUIDE.md -5
   ```
2. **Check isolation**: Ensure breaking change commit is separate from other features
   ```bash
   git show <commit-hash> --stat
   ```
3. **Revert commit**:
   ```bash
   git revert <commit-hash> --no-edit
   ```
4. **Remove version note**: Check line ~3 for breaking change warning, remove if present
5. **Validate rollback**:
   ```bash
   # Verify old field names gone
   grep '"deprecated_field"' LLM_GUIDE.md  # Should be empty

   # Verify correct field names restored
   grep '"correct_field"' LLM_GUIDE.md | wc -l  # Should match expected count
   ```
6. **Check Common Mistakes**: Ensure migration guide still accurate for current version
7. **Commit with incident reference**:
   ```bash
   git commit -m "docs: Rollback vX.Y breaking change (INC-XXXX)"
   ```
8. **Deploy immediately** (emergency bypass of PR process)
9. **Monitor for 30 min**: Check user reports, GitHub issues, support tickets

**Rollback Validation Checklist**:
- [ ] Old field names removed (grep returns empty)
- [ ] Correct field names restored (grep count matches expected)
- [ ] Version note removed or updated to previous version
- [ ] Common Mistakes section still accurate
- [ ] Other features NOT reverted (check unrelated additions)
- [ ] Line count reasonable (within 500-520)
- [ ] Commit message references incident number

**Emergency Bypass Criteria** (when to skip PR review):
- Production outage (P0/P1 incident)
- Breaking change affecting LLM JSON generation
- Security vulnerability in examples
- Data breach in sample code

**Process**: Direct push to master, create post-mortem PR with explanation

**Time estimate**: 25-30 min (15 min active + 15 min monitoring)

**Breaking Change Commit Isolation** (best practice):
When making breaking changes, commit them SEPARATELY from other features:

✅ **Good**:
```
commit 1: Add betweenness algorithm
commit 2: Add Between predicate
commit 3: Breaking change - filters rename
```

❌ **Bad**:
```
commit 1: Add betweenness + Between + filters rename
```

**Why**: Enables clean rollback of breaking change without losing other features

---

### LLM Failures

**Trigger**: 3+ support tickets with same LLM-generated error pattern

**Examples**:
- Using `"filter"` instead of `"filters"`
- Using `"node_type"` instead of `"type"`
- Putting filters outside Node object
- Using deprecated field names

**Process**:
1. Analyze tickets to identify pattern
2. Categorize error (field name, structure, missing required, deprecated)
3. Add to `## Common Mistakes` section

**Format** (ultra-compact to fit line budget):
```markdown
❌ `"wrong_field"` → ✅ `"correct_field"` ([context])
❌ Wrong structure → ✅ Correct: [inline example]
```

**Placement**: `## Common Mistakes` section (after domains, before end)

**Coverage**: Top 5 most common errors (80/20 rule)

**Lines**: ~7 lines total (strict budget)

**Time estimate**: 40-50 min (includes compression iterations)

---

### New Domain

**Trigger**: 3+ requests from new industry vertical (healthcare, logistics, energy)

**Process**:
1. Identify 3-5 key entity types (e.g., healthcare: patient, doctor, facility)
2. Identify typical relationships (e.g., referral, admission, treatment)
3. Identify common filters (age ranges, risk scores, dates)
4. Map entities to Font Awesome 4 icons
5. Create complete example: search + filter (optional: + algorithm + visualization)

**Format**:
```markdown
### [Domain Name]

**Use case**: [One-line description]

**Dense**: `[compact Python form]`

**JSON**:
```json
{[compact JSON AST]}
```
```

**Placement**: After existing domains (Cyber, Fraud, Supply Chain, Social Media)

**Lines**: ~22-28 lines per domain

**Time estimate**: 50-60 min

**Blocker risk**: May exceed line limit - see [Escalation Process](#escalation-process)

---

## Multi-Change Releases

**Scenario**: Release includes multiple GFQL changes (e.g., new algorithm + predicate + edge parameter)

**Coordination workflow**:
1. **Research ALL changes first** - Don't start editing until scope is clear
2. **Estimate total line impact** - Calculate (additions - removals)
3. **If exceeds limit**: Proactively condense BEFORE adding new content
4. **Update order**: Core Types → Predicates → Common Patterns → Algorithms → Domains (top-to-bottom to avoid line number confusion)
5. **Validate together**: Test all changes as a set (catch interactions)
6. **Cross-reference**: Ensure every release note item is documented

**Example line budget**:
```
Starting: 500 lines
Change 1 (edge param): +1 line (Core Types)
Change 2 (predicate): +3 lines (Predicates)
Change 3 (algorithm): +15 lines (Algorithms)
Total additions: +19 lines
Target: 519 lines → Need to remove 19 lines OR use 500-520 acceptable range
```

**Condensation targets**: Remove blank lines (-10), inline domain headings (-8), compact variation format (-5)

**Time estimate**: 50-60 min for 3 coordinated changes

---

## Escalation Process

**When to escalate**: Stuck after trying all options, line limit exceeded with essential content, conflicting priorities

**Steps**:
1. **Document**: What you tried, why it failed
2. **Options table**: Pros/cons of alternatives
3. **Recommendation**: Preferred option with rationale
4. **Create issue**: GitHub issue tagged `[DOC-DECISION]`
5. **Tag maintainer**: Request decision from core team
6. **Await decision**: Don't proceed until resolved

**Example escalation scenarios**:
- New domain would exceed 520 lines
- Critical Common Mistakes section doesn't fit
- Conflicting domain priorities (which to keep?)
- Breaking change requires more space than available

---

## Update Workflow (Standard)

1. **Identify trigger** - What changed? (1-2 min)
2. **Research** - `read [file]`, `grep "feature"`, find tests/docs (3-6 min)
3. **Check line budget** - `wc -l LLM_GUIDE.md`, estimate impact (1-2 min)
4. **Create example** - Test in Python → `.to_json()` (5-10 min)
5. **Decide placement** - Use decision tree (2-3 min)
6. **Draft update** - Dense + JSON + annotations (6-10 min)
7. **Verify** - `wc -l`, validate JSON, cross-reference (3-5 min)
8. **Update this doc** - If new pattern discovered (5 min)

**Total time**: 25-50 min depending on complexity

---

## Maintenance Checklist

Before committing updates:

- [ ] Line count <520 (ideally <500)
- [ ] Dual format (Dense Python + JSON) for all examples
- [ ] Inline annotations (`// required`, `// optional`, `// default: X`)
- [ ] Qualified lists ("Examples:", "and more" to avoid sounding exhaustive)
- [ ] Real values (not "foo", "bar" - use actual colors, icons, field names)
- [ ] JSON validated (run `generate_examples.py` or manual validation)
- [ ] Common Mistakes updated (if breaking change or LLM failure pattern)
- [ ] Cross-referenced release notes (all features documented)

---

## Validation

### JSON Validation
```bash
# Validate individual example
python3 -c "import json; json.loads('{\"type\": \"Node\", \"filters\": {}}')"

# Run full validation suite
python3 plans/gfql_json_wire_spec_generator/generate_examples.py
```

### Line Count Check
```bash
wc -l plans/gfql_json_wire_spec_generator/LLM_GUIDE.md
```

### Completeness Check
```bash
# Verify all release features documented
grep "new_feature" LLM_GUIDE.md

# Verify old field names removed (breaking changes)
grep '"deprecated_field"' LLM_GUIDE.md  # Should return empty
```

---

## Common Mistakes (Meta)

### Mistake 1: Placement uncertainty
**Problem**: Unsure where to add new content
**Solution**:
- Algorithms → END of `## Graph Algorithms` (chronological)
- Predicates → Within `## Predicates` by category
- Patterns → Decision tree (3+ domains → Common Patterns)
- Domains → After existing domains

### Mistake 2: Exceeding line limit
**Problem**: Update would push over 500 lines
**Solution**: Check if 500-520 acceptable (essential content), else condense using strategies table

### Mistake 3: Breaking changes without migration guide
**Problem**: Users confused by v[NEW] syntax
**Solution**: ALWAYS add version note + migration guide to Common Mistakes

### Mistake 4: Missing similar feature check
**Problem**: New predicate overlaps with existing (e.g., Matches vs Contains with regex)
**Solution**: Grep Predicates section for related features, add clarifying note

### Mistake 5: Placeholder values
**Problem**: Examples use "foo", "bar", generic names
**Solution**: Use real values from tests/docs - actual colors, icons, domain field names

---

## Template Patterns

### Full Example Pattern
```markdown
**[Feature Name] ([Use Case]):**
```python
# Dense: [compact Python code]
```
```json
{
  "type": "[Type]",
  [... full JSON ...]
}
```
**Pattern**: [Explanation]
```

**Lines**: ~15-20

---

### Compact Variation Pattern
```markdown
**Use case: [Variation Name]**
```python
# Dense: [compact code]
```
Pattern: [One-line explanation]. [Key parameters].
```

**Lines**: ~5

---

### Common Mistake Pattern
```markdown
❌ `"wrong"` → ✅ `"correct"` ([context])
```

**Lines**: 1

---

### Cross-Reference Pattern

**Use when**: Structural pattern identical across domains, but semantics differ

**Example**: "Trace propagation" = money laundering (fraud) vs malware spread (cyber) vs contamination (supply chain)

**Template**:
```markdown
**[Pattern Name] ([Generic Description])**

```python
# Dense: [generic GFQL code]
```

**Pattern**: [Generic structural explanation]

**Domain examples**:
- **[Domain 1]**: [Domain-specific name/use case] - See [Domain Section](#link)
- **[Domain 2]**: [Domain-specific name/use case] - See [Domain Section](#link)
- **[Domain 3]**: [Domain-specific name/use case] - See [Domain Section](#link)

**Key parameters**: [Important configurable params]
```

**Lines**: ~10-12 (compact)

**Benefits**:
- Avoids duplication (vs 3× 20-line examples = 60 lines)
- Connects generic concept to domain-specific implementations
- Improves searchability (users search generic term, find domain examples)

**Validation**:
- [ ] Domain examples actually exist and show this pattern
- [ ] Links work (section IDs correct)
- [ ] Generic pattern is truly structural (not semantic)

---

## File Locations

**Main guide**: `plans/gfql_json_wire_spec_generator/LLM_GUIDE.md`
**This process doc**: `plans/gfql_json_wire_spec_generator/PROCESS_DISTILLATION.md`
**Validation script**: `plans/gfql_json_wire_spec_generator/generate_examples.py`

**Monitor these files** for changes:
- `graphistry/compute/gfql/call_safelist.py` - New algorithms
- `graphistry/compute/predicates/*.py` - New predicates
- `graphistry/compute/ASTSerializable.py` - Breaking changes
- `graphistry/tests/compute/` - Usage patterns

---

## Success Signals

**Green** (guide working well):
- LLM success rate >80% (generates valid JSON first try)
- Users reference guide in GitHub issues/PRs
- No support tickets about LLM-generated errors
- Examples run unchanged across releases

**Yellow** (needs attention):
- Approaching 500 lines (condensation needed soon)
- 2-3 new domain requests building up
- Occasional LLM failures (not systematic yet)

**Red** (update immediately):
- Multiple users asking same "how do I...?"
- 3+ tickets with same LLM-generated invalid JSON
- Release breaks examples (breaking change)
- Test failures in `generate_examples.py`

---

## Background: Why This Guide Exists

**Goal**: Enable LLMs to generate valid GFQL JSON for common use cases

**Not**: Document implementation details, wire protocol internals, or exhaustive API reference

**Target audience**: LLMs (Claude, GPT, etc.) generating code for users

**Key principles**:
1. **Purpose over types** - Organize by what users want to DO (Search, Algorithms, Viz)
2. **Examples over explanation** - Show complete workflows, not just type definitions
3. **Rich details** - Actual palette values, icon names, domain terminology
4. **Top-down structure** - Context before details, examples before schemas
5. **Dual format** - Dense Python + JSON for every example
6. **Inline annotations** - `// required`, `// optional` directly in schemas

**Evolution**: 5 phases from "document wire protocol" (wrong) → "enable LLM code generation" (right)

---

## When to Use This Process

**Use for**:
- LLM-focused documentation
- Code generation guides (JSON, DSL, config)
- Teaching APIs to AI assistants

**Don't use for**:
- Developer documentation (needs implementation details)
- Reference manuals (needs exhaustive coverage)
- Internal documentation (needs organizational context)

---

## References

**Created this guide from**:
- `docs/source/gfql/spec/cypher_mapping.md` - Framing (analogies)
- `docs/source/gfql/about.rst` - Multi-step examples
- `docs/source/gfql/builtin_calls.rst` - Rich parameter values (palettes, icons)

**Research artifacts**:
- `PLAN.md` - 6-phase evolution log with simulation framework
- `PHASE4_RESEARCH_SUMMARY.md` - Gap analysis leading to Phase 5 audit
- `simulations/` - 8+ persona scenarios validating this process

**Result**:
- `LLM_GUIDE.md` - 484-520 lines (target), self-contained, LLM-optimized
