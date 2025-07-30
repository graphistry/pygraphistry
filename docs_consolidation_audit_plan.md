# GFQL Documentation Consolidation Audit Plan

## Overview
Systematic audit to ensure no documentation was lost during consolidation from multiple branches into gfql-docs-clean-v3.

## Branches to Audit
1. **gfql-docs-v2** - Had comprehensive builtin_calls.rst improvements (PR #718, now closed)
2. **gfql-docs-combined** - Had cypher_mapping improvements and group_in_a_box_layout docs
3. **gfql-docs-clean-v3** - New consolidated branch (PR #719)

## Audit Steps

### Phase 1: Branch Comparison Setup
- [ ] Compare gfql-docs-v2 vs gfql-docs-clean-v3
- [ ] Compare gfql-docs-combined vs gfql-docs-clean-v3
- [ ] Identify all changed files in each comparison

### Phase 2: File-by-File Audit - builtin_calls.rst
- [ ] Check line count (should be ~1400 lines)
- [ ] Verify all 24 safelist methods are documented
- [ ] Confirm algorithm categorization (Centrality, Community, etc.)
- [ ] Verify external links (cuGraph, igraph, RAPIDS)
- [ ] Check group_in_a_box_layout addition
- [ ] Validate GPU acceleration section
- [ ] Check See Also references

### Phase 3: File-by-File Audit - cypher_mapping.md
- [ ] Verify GPU vs CPU decision guide section
- [ ] Check algorithm mapping table (should have 12+ algorithms)
- [ ] Confirm GFQL advantages section
- [ ] Verify WITH and CALL pattern mappings
- [ ] Check all code examples updated to gfql()
- [ ] Validate wire protocol examples

### Phase 4: File-by-File Audit - translate.rst
- [ ] Check betweenness example (should use igraph)
- [ ] Verify get_degrees example added
- [ ] Confirm mixed GPU/CPU examples with comments
- [ ] Check community detection section has call() variant
- [ ] Validate all chain() → gfql() updates

### Phase 5: File-by-File Audit - Other Files
- [ ] about.rst - Check all chain() → gfql() updates
- [ ] overview.rst - Verify API updates
- [ ] quick.rst - Check Let/Call section addition
- [ ] remote.rst - Verify gfql_remote updates
- [ ] combo.rst - Check GFQL integration in ML examples
- [ ] index.rst - Verify builtin_calls.rst in TOC
- [ ] cheatsheet.md - Check chain() → gfql() updates

### Phase 6: File-by-File Audit - Spec Files
- [ ] spec/language.md - Check Call operations section
- [ ] spec/wire_protocol.md - Verify Call serialization docs
- [ ] spec/cypher_mapping.md - Full content verification
- [ ] spec/index.md - Check any updates

### Phase 7: Validate Removals
- [ ] List all removed content
- [ ] Verify each removal was intentional
- [ ] Check no accidental deletions

### Phase 8: Final Verification
- [ ] Run docs build locally
- [ ] Check all cross-references work
- [ ] Verify no broken links
- [ ] Create final audit report

## Audit Report Template
For each file:
```
File: [filename]
Source branches: [branches that had changes]
Additions verified: [✓/✗]
Removals justified: [✓/✗]
Issues found: [list]
Action needed: [none/fix required]
```