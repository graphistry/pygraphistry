# GFQL Calls: Complete Implementation & Maintenance Checklist

When exposing or updating GFQL `call()` functions, the change is **multi-system**:
runtime executors, JSON schema validators, static typing, documentation, and tests
all need to stay in sync. Use this playbook (paired with
[`predicates_checklist.md`](./predicates_checklist.md) and the guidance in
[`../prompts/GFQL_LLM_GUIDE_MAINTENANCE.md`](../../prompts/GFQL_LLM_GUIDE_MAINTENANCE.md))
to avoid regressions.

> **Scope**: Items below are required for *every* new or modified `call()` entry.
> Optional docs/comm tasks are flagged separately—promote them when the feature
> is user-facing or developer-heavy.

---

## ✅ 10-Step Minimum Checklist

| # | Area | File(s) / Command | Notes |
|---|------|-------------------|-------|
| 0 | **Python API** | Implementation module (e.g., `graphistry/layout/*.py`) | Ensure Plotter/engine behavior exists & is documented |
| 1 | **Safelist Entry** | `graphistry/compute/gfql/call_safelist.py` | Allowed params, required params, value validators |
| 2 | **AST Call Validation** | `graphistry/compute/gfql/call_executor.py`, `graphistry/compute/gfql/validate.py` (if new semantics) | Guarantee runtime guards + helpful errors |
| 3 | **TypedDict Schema** | `graphistry/models/gfql/types/call.py` | Precise `TypedDict` or `Protocol` types—avoid `Dict[str, Any]` |
| 4 | **Static Type Surface** | `stubs/graphistry/compute/gfql/call_safelist.pyi`, `stubs/<dependency>.pyi` | Update Pyre/mypy stubs + add missing third-party stubs |
| 5 | **Wire Protocol Spec** | `docs/source/gfql/spec/wire_protocol.md` | JSON payload example + parameter descriptions |
| 6 | **Language Reference** | `docs/source/gfql/spec/language.md`, `docs/source/gfql/builtin_calls.rst`, quick refs | Grammar entry, short description, parameter table links |
| 7 | **User-Facing Docs** | Cheatsheet, layout catalog, notebooks, release notes | Mention new behaviors (engine support, parameter defaults) |
| 8 | **Tests** | `graphistry/tests/compute/...`, GPU variant where applicable | Cover pandas + cudf, serialization round-trips, failure cases |
| 9 | **Release Logistics** | `CHANGELOG.md`, GFQL release tracker, marketing TODOs | Flag follow-up tasks in `plans/<feature>/` as needed |

---

## 🔍 Step Details & Gotchas

### 0️⃣ Python API Baseline
- Confirm the Plotter mixin or underlying module already exports the behavior.
- Add docstrings + inline examples if they are missing.
- If adding helper shims (e.g., string → `np.datetime64` coercion), document **why**
  in both code comments and user docs.

### 1️⃣ Safelist Entry
- New call → append to `SAFELIST_V1` with **least-privileged validators**.
- Prefer dedicated helpers (`is_list_of_strings`, etc.) or add new ones when needed.
- Remember: JSON clients only send string/number/bool/null—normalize in code if richer types required.

### 2️⃣ Runtime Validation
- `execute_call()` should raise `GFQLTypeError` with `ErrorCode` hints when parameters are invalid.
- If the call affects schema (new columns), update validation hints in `graphistry/compute/gfql/validate.py`.
- For cross-operation invariants, add tests in `graphistry/tests/compute/test_call_operations*.py`.

### 3️⃣ TypedDict Schema
- Update `CallMethodName` literal union with the new call name.
- Add a `TypedDict` capturing **exact** parameter types (`Literal`, `Sequence`, `Mapping`).
- Avoid `Dict[str, Any]` / `List[Any]` by promoting helper TypedDicts or `TypedDict`-friendly aliases.

### 4️⃣ Static Stubs
- Pyre/mypy must know about external dependencies introduced by the call (e.g., cudf, cupy).
- Add minimal `.pyi` stubs under `stubs/` when libraries lack them.
- Keep the stub search path in `.pyre_configuration` updated.

### 5️⃣ Wire Protocol Docs
- Add JSON examples showing minimal + advanced payloads.
- Document parameter typing expectations (e.g., ISO timestamps vs `np.datetime64`).
- Cross-link to safelist or TypedDict sections if extra context helps.

### 6️⃣ Language & Quick References
- Update:
  - `docs/source/gfql/spec/language.md` grammar tables.
  - `docs/source/gfql/builtin_calls.rst` call inventory.
  - Any quick-reference tables (`docs/source/gfql/quick.rst`, cheat sheets).
- Confirm Sphinx builds (`cd docs && bash html.sh`) after edits.

### 7️⃣ Broader Documentation
- Add cookbook examples / release note bullet where appropriate.
- If behavior differs between Plotter API and GFQL (e.g., auto-conversions), call it out explicitly.
- Update notebooks or demos when they showcase the affected feature.

### 8️⃣ Testing Expectations
- Unit tests: pandas + cudf coverage, JSON round-trip, error paths.
- GPU tests should be guarded with `TEST_CUDF=1` and documented in plan/PR.
- Consider typing tests (`graphistry/tests/typing/`) to keep mypy happy.

### 9️⃣ Release & Tracking
- Update `CHANGELOG.md` under "Development".
- Capture follow-up actions in `plans/<feature>/PLAN.md` (including open PR comments).
- Notify docs/AI teams if prompts or assistants rely on the change.

---

## 📎 Optional (Promote When User-Facing)

- **Marketing / blog teaser** when the call is a marquee feature.
- **LLM prompt updates** (`ai/prompts/gfql/*`) beyond baseline instructions.
- **Customer templates** or dataset refreshes referencing the new capability.

Keep this file synchronized with future GFQL evolutions—open a PR to amend when the onboarding surface changes.
