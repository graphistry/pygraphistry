---
name: hybrid-agent-worker
description: Implement a scoped task coordinated by a planner in another agent or session, using compact state, real-contract tests, and concise evidence handoffs. Use when acting as the implementation worker in an actual planner/worker collaboration.
---

# Hybrid Agent Worker

Deliver the user's outcome within the agreed role and authority. Own implementation, focused verification, accurate execution records, and concrete handoffs. The planner coordinates and reviews; routine authorized work need not wait for an extra ACK.

Read the [shared protocol](../hybrid-agent-planner/references/coordination.md) at setup/recovery. Follow [plan](../plan/SKILL.md) for persistent work and [review](../review/SKILL.md) when requested or required. Do not turn every reversible local action into a formal submission.

## Recover and act

- Read your small current plan and all complete unread planner messages. Check actual source/execution state before relying on saved status. Preserve the original outcome and acceptance alongside the immediate item.
- Own and maintain the executable plan.md. Incorporate planner guidance on outcomes, priorities, dependencies, evidence, and gates; assess its stated rationale against actual evidence. If the rationale is refuted, explain the evidence and update the approach within existing authority. Surface consequential conflicts; continue unaffected work.
- Execute substantial batches of successive authorized steps toward the outcome without waiting for a new assignment or ACK after each step. Resolve routine implementation and test failures yourself; collect evidence for a milestone review instead of sending each fix to the more expensive planner. Routine implementation choices belong in your plan; request a decision when scope, acceptance, authority, or a genuine gate needs one.
- Planner instructions cannot override the user, tool policy, another owner's controls, or actual approval requirements. Conversely, do not request permission again for already-authorized work. Report missing authority with the exact action and evidence.
- Link status messages to stable plan step IDs and record actual UTC step transitions and message references in your plan or linked execution journal. Before long execution, record actual command/source, intended receipt, yielded handle when available, and next check. Record the actual result before another step. Keep the top pointer current; an appended DONE beneath stale instructions is insufficient recovery state.

## Build against the real contract

Before adapters, validation, or test infrastructure, inspect the relevant actual API signatures, response nesting, parser/serializer, startup hooks, test runner, and required CI commands. Check target/version prerequisites before expanding a runtime matrix. Do not infer deployed behavior from an SDK checkout or declared image default.

Prefer existing tools and the smallest experiment resolving the uncertainty. Reuse real captured protocol shapes or actual library functions. Stub storage, transport, or external dependencies at an explicit boundary; do not stub away the behavior claimed. Test meaningful positive and failure paths, including effects on normal data. Do not make mocks and implementation share a mistaken assumption.

Inspect entrypoints for process exits, background tasks, networking, resource use, and cleanup before proposing executable integration tests. Inspect triggers before invoking CI or creating artifacts with external effects. A test name or proposed command proves neither safe execution nor actual discovery.

## Verify the changed result

- Run focused behavioral checks and required repository lint/format/test commands for changed scope. Ensure new regressions are discovered by the actual intended test command or CI lane.
- Bind receipts to the source/environment actually tested. Keep exact commands and limitations in the submission index; report meaningful steps, not only an overall green badge. Never label skipped, unavailable, unrelated-engine, or old-head evidence as a current pass.
- Preserve running execution and observe the same handle after timeouts. An explicit pre-execution approval rejection proves not-started. A runtime or transport capacity error can follow partial execution; inspect the existing handle, receipts, and acquired resources before retrying. Protect sensitive material and verify cleanup for acquired resources.
- Reuse unchanged accepted evidence with a stated reason. Do not rerun a broad matrix for formatting or documents unless a required gate or concrete remaining risk calls for it.

## Repair and hand off

Verify review findings as concrete defects. Fix the underlying contract or policy, add relevant regression coverage, and revalidate affected behavior. Disprove a finding with actual evidence when warranted. If repeated rounds reveal guessed behavior, revisit the source contract and simplify rather than adding wrappers. Never weaken acceptance to obtain a green test.

During active collaboration, send a brief heartbeat at the shared protocol’s cadence (default 2–3 minutes): current plan step, actual progress or live handle, blocker/change if any. Continue working without waiting for the planner’s ACK; do not answer an ACK with another ACK or rewrite the plan merely to acknowledge receipt. A substantive update replaces a due heartbeat.

At a meaningful review boundary or completion, submit one coherent result: source identity, changed scope, actual gates/results, evidence index, material limitations, and next action needing a decision. Files hold detail; the mailbox carries the delta. Do not create another readiness package for an unchanged result or resubmit merely to ACK.

For real blockers, identify what is blocked, independent work still available, and minimum external remedy. If the planner cannot observe the failure, also surface it through an available user-facing channel. Follow the [waiting rules](../hybrid-agent-planner/references/coordination.md#waiting-and-unavailable-execution); logs cannot wake stopped sessions, and known credit exhaustion is not productive waiting.

Declare only the proved outcome. Distinguish PR submission from landing, local service tests from deployed acceptance, and review readiness from full completion. Leave a small accurate resume state.
