# Phase 2 Kickoff: #1250 + #1252 Deletion Contracts

Scope: start Phase 2 delivery for the remaining graphistrygpt helper deletions that depend on pygraphistry.

Upstream issues:
- #1250 Declarative encode_*_from_dict constructor parity with React-side shape
- #1252 Plottable.from_dataset_id(id, token) helper to fetch existing dataset metadata + bindings

## Deletion contracts (source of truth)

### #1250 contract (react_encodings helper deletion)
Required upstream API outcome:
- A single pygraphistry entrypoint that accepts React-side declarative encoding payloads and applies equivalent encode_* calls.
- At minimum parity for keys currently consumed downstream:
  - encodePointColor
  - encodePointIcons
  - encodePointSize
  - encodeAxis

Success criteria:
- Downstream can delete translation helpers and call one upstream method.
- Roundtrip behavior matches existing per-method encode_* behavior.
- Validation errors are actionable and point to exact invalid encoding key/value.

### #1252 contract (dataset_metadata helper deletion)
Required upstream API outcome:
- A direct helper to hydrate an existing dataset into a plottable-like object without re-upload.
- It must hydrate bindings/encodings/metadata/style/url_params from server metadata response shape.

Success criteria:
- Downstream can delete manual `/api/v2/upload/datasets/{id}` fetch/parsing code.
- Returned object has enough metadata for axis/encoding-aware workflows.
- URL generation works when dataset_id is present.

## PR slicing plan

1. Slice A (#1252 first): add `from_dataset_id` helper + tests for metadata hydration and response-shape tolerance.
2. Slice B (#1250): add declarative encoding dispatcher + tests for supported React keys and strict error paths.
3. Slice C: docs/changelog + explicit downstream deletion mapping.

## Non-goals (this kickoff PR)
- No graphistrygpt repo deletions in this PR.
- No broad encode API redesign beyond parity/deletion requirements.

## Validation checklist for implementation slices
- Targeted tests for helper behavior and invalid payloads
- ruff + mypy on touched modules
- CI green before merge

