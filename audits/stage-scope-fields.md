# `_StageScope` Field Audit

Date: 2026-04-07 PDT
Scope: `graphistry/compute/gfql/cypher/lowering.py`

## `_StageScope` Definition

```python
@dataclass(frozen=True)
class _StageScope:
    mode: Literal["match_alias", "row_columns"]
    alias_targets: Dict[str, ASTObject]
    active_alias: Optional[str]
    row_columns: Set[str]
    projected_columns: Dict[str, _StageColumnBinding]
    table: Optional[Literal["nodes", "edges"]]
    seed_rows: bool
    relationship_count: int
    allowed_match_aliases: Set[str] = field(default_factory=set)
```

## Field Read Map

| Field | Read Sites (line) | Reading Functions | Purpose |
|---|---|---|---|
| `mode` | 8108, 8128 | `compile_cypher_query` | Dispatch stage lowering by scope mode |
| `alias_targets` | 4550, 4568, 4593, 4602, 4616, 4627, 4655, 4677, 4707, 4734, 4747, 4770, 4789, 4804, 4820, 4843, 4852, 4875, 4896, 4930, 4940 | `_lower_match_alias_stage`, `_lower_match_alias_aggregate_stage` | Alias name to AST target resolution |
| `active_alias` | 4530, 4569, 4690 | `_lower_match_alias_stage`, `_lower_match_alias_aggregate_stage` | Single-alias projection source tracking |
| `row_columns` | 4538, 4542, 5061, 5109, 5284, 5314, 5353, 5377, 5432 | `_lower_match_alias_stage`, `_lower_row_column_stage`, `_lower_row_column_aggregate_stage` | Available row-column symbols |
| `projected_columns` | 4570, 4622, 5127, 5173, 5284, 5294, 5315 | `_lower_match_alias_stage`, `_lower_row_column_stage` | Carry forward projected binding metadata |
| `table` | constructor-copy propagation (writes at 4660, 4672, 4998, 5254, 5562, 7753) | stage constructors | Node/edge table context propagation |
| `seed_rows` | 4661, 4673, 4999, 5000, 5255, 5317, 5563, 8150 | stage functions + `compile_cypher_query` | Whether to seed output from `rows()` |
| `relationship_count` | 4662, 4674, 4709, 5000, 5256, 5318, 5564 | stage functions | Relationship multiplicity guard input |
| `allowed_match_aliases` | 4577, 4604, 4644, 4657, 4704, 4752, 4791, 4845, 4894, 4930, 4940, 4989, 5119, 5120, 5257 | `_lower_match_alias_stage`, `_lower_match_alias_aggregate_stage`, `_lower_row_column_stage` | Binding-row path gating and alias visibility controls |

## Highest Migration Risk: `allowed_match_aliases`

`allowed_match_aliases` is the highest-risk field for M1 binder extraction because it controls whether lowering uses:

- the bindings-row path (whole-row alias columns, alias-prefixed resolution), or
- the simpler match-alias path.

Observed functional contexts:

1. Mixed whole-row + scalar projection guard (`4577`)
2. WHERE expression allowlist resolution (`4604`)
3. `extend_mode` calculation for next scope (`4644`)
4. Copy-forward into next `_StageScope` (`4657`, `4989`)
5. Aggregate path behavior gate (`4704`)
6. Binding-row column naming (`4752`)
7. Aggregate WHERE allowlist (`4791`, `4845`)
8. `bindings_row_path` flag (`4894`)
9. Visible projection column filtering (`4930`, `4940`)
10. Alias-prefixed short-circuit checks (`5119`, `5120`)
11. Explicit reset when entering row-column scope (`5257`)

## M1 Binder Extraction Guardrails

Any binder extraction must preserve:

- mode-based dispatch semantics (`match_alias` vs `row_columns`),
- `allowed_match_aliases` propagation/reset rules,
- relationship-aware aggregate guards,
- projection column visibility filtering behavior.
