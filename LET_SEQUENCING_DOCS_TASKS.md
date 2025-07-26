# Let Sequencing Documentation Tasks

## Raw Task List

1. **10-minute GFQL Guide**
   - Add sequencing GFQL programs section with `let`
   - Include PageRank example: compute PageRank, then explore 2 hops from high-scoring nodes
   - Location: Find appropriate place for ninja edit

2. **Overview -> Quick Examples**
   - Add similar `let` sequencing example
   - Show the power of composing operations

3. **Remote Mode Documentation**
   - Add `let` examples (more important here due to no Python escape hatch)
   - Show how to sequence operations in pure GFQL

4. **Fix chain_let References**
   - Find all "chain_let" references in docs
   - Update to ".let()" syntax
   - Verify correctness across all docs

5. **Update .chain() Syntax**
   - Find all .chain(...) references
   - Update to .gfql(Chain(...))
   - OR: If implicit coercion .gfql([...]) => .gfql(Chain([...])) is supported:
     - Do code scan to verify
     - Add pytest if needed
     - Document the coercion clearly

6. **GFQL Quick Reference**
   - Move .let() earlier in the reference
   - Place before predicates section
   - Show as fundamental composition tool

7. **GFQL Language Spec**
   - Fix top-level to show both chain and let as allowed
   - Add chain+let section before Operations section
   - Ensure proper hierarchy

8. **Wire Protocol Documentation**
   - Add chain+let section before operations
   - Show protocol representation

9. **hop_and_chain_graph_pattern_mining Notebook**
   - Add `let` example
   - Validate with docs build that runs notebooks

10. **call_operations Documentation**
    - Update title to include "and Let bindings"
    - Show how Call and Let work together

11. **gfql_remote Notebook**
    - Add let+call example
    - Show remote execution patterns

12. **Multi-hop Section Updates**
    - Add forward reference to "Complex Pattern Reuse"
    - Improve flow between sections

13. **"Between" Section**
    - Has "Complex Pattern Reuse" 
    - Fix chain_let references

14. **Combining GFQL Section**
    - Start early with Python vs pure GFQL equivalents
    - Show the relationship clearly

## Reordered by PR

All tasks should be on **PR #708** (top of docs stack): "docs(gfql): Comprehensive Let bindings and Call operations documentation"

## Granular Implementation Steps

### Phase 1: Preparation
1. Switch to PR #708 branch
2. Pull latest changes
3. Create working list of all files to modify

### Phase 2: Find and Fix chain_let References
4. Search for all "chain_let" occurrences in docs
5. List each file and line number
6. Update each to ".let()" syntax
7. Verify context makes sense

### Phase 3: Update .chain() Syntax
8. Search for all .chain() calls in docs
9. Check if implicit coercion is supported in code
10. If yes: Document coercion behavior
11. If no: Update all to .gfql(Chain(...))

### Phase 4: Core Documentation Updates
12. Find 10-minute GFQL guide file
13. Add "Sequencing Programs with Let" section
14. Write PageRank + 2-hop example
15. Find Overview -> Quick Examples
16. Add let sequencing example there
17. Find Remote Mode docs
18. Add comprehensive let examples for remote

### Phase 5: Reference Documentation
19. Find GFQL Quick Reference
20. Restructure to introduce .let() before predicates
21. Find GFQL Language Spec
22. Update top-level grammar to include let
23. Add chain+let section before Operations
24. Find Wire Protocol docs
25. Add chain+let protocol section before ops

### Phase 6: Notebook Updates
26. Find hop_and_chain_graph_pattern_mining.ipynb
27. Add let example with pattern reuse
28. Find gfql_remote.ipynb
29. Add let+call remote example
30. Update notebook outputs

### Phase 7: Cross-references and Titles
31. Find Multi-hop section
32. Add forward reference to Complex Pattern Reuse
33. Find call_operations docs
34. Update title to include "and Let Bindings"
35. Find Combining GFQL section
36. Add early Python vs GFQL comparison

### Phase 8: Validation
37. Run flake8/ruff linting
38. Run mypy type checking
39. Run docs build locally
40. Run notebook tests
41. Review all changes

### Phase 9: Commit and Push
42. Stage all changes
43. Create detailed commit message
44. Push to PR #708

## Validation Checklist
- [ ] All chain_let replaced with .let()
- [ ] All .chain() updated appropriately
- [ ] PageRank example works in 10-min guide
- [ ] Remote examples are comprehensive
- [ ] Quick Ref has proper ordering
- [ ] Language Spec has let at top level
- [ ] Notebooks execute without errors
- [ ] Cross-references are correct
- [ ] No lint errors
- [ ] No type errors
- [ ] Docs build succeeds
- [ ] Notebook tests pass