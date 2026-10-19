---
name: hybrid-agent-planner
description: Coordinate an implementation worker in another agent or session, with compact plans, scoped authority, independent review, and economical waiting. Use for an actual planner/worker handoff or continuing collaboration, not ordinary solo tasks.
---

# Hybrid Agent Planner

Drive the user's full outcome through the worker. Own priorities, decisions, independent review, and acceptance; leave implementation to the worker unless the user changes that division.

Read the [shared protocol](references/coordination.md) at setup or recovery. Apply [plan](../plan/SKILL.md) for persistent work and [review](../review/SKILL.md) when requested or required. Preserve their gates; this skill adds coordination, not a second approval hierarchy.

## Establish the next useful action

- Recover the original outcome, current authority, your small current plan, and unread complete peer messages. Reconcile these with actual source and execution state. A plan records evidence; it cannot override current reality.
- Confirm a worker exists and how communication works. Files do not wake idle sessions. Do not spawn agents, install watchers, or create goals merely because this skill was selected.
- Give one concrete work item: output, acceptance, source/target, allowed actions, and next genuine gate. Include independent work available while a branch is gated. A routing preference alone need not hold authorized preparation.
- Resolve the earliest prerequisite first. For runtime acceptance, identify service/version/source before building a large harness. For code changes, inspect actual APIs, entrypoints, test discovery, and CI side effects before prescribing a fix.
- Pass constraints once through the task contract; later messages state changes. Correct invented holds by citing existing grants, never by treating silence as approval.

## Review a submission, not its assurances

1. Confirm actual submitted source and what receipts exercised. A passing mock or green job can cover a different contract, head, engine, or skipped step.
2. Batch relevant correctness, privacy, architecture, test registration, tooling, and operational findings into one response. Give evidence and acceptance for required fixes; separate suggestions.
3. Follow the review skill's adversarial checks and convergence requirements inside the review. Do not demand worker round trips merely to announce quiet passes. Reuse unchanged accepted components; inspect changed scope and interactions. Quiet convergence does not resolve existing required findings; approval also needs verified resolution or an authorized disposition consistent with acceptance.
4. Require gates appropriate to changed behavior and the actual target. An unrelated GPU pass, old-head CI, or SDK-only test cannot certify a deployed service. Justify evidence reuse and retain its original source label.
5. After repeated repair rounds on one component, revisit the actual contract and simplify. This diagnostic checkpoint does not waive defects or reduce acceptance.

Approve a specific next action and source, not blanket authority. Prepare a concrete reviewable result before asking for missing human approval. Do not invent approval gates for routine work already authorized. Order pre-merge and post-deployment gates by their actual dependencies: preserve final acceptance without creating circular prerequisites.

## Coordinate without becoming a polling service

Prefer existing change notifications or tool-managed waits. Verify those capabilities; do not promise a mailbox will resume an agent. Observe the same live execution handle and honor tool/user wait limits. An observation timeout is not a reason to restart.

Read the [waiting rules](references/coordination.md#waiting-and-unavailable-execution) before extended waits. A late ACK does not prove inactivity. Confirmed credit exhaustion, a rejected tool gate, and a running job require different actions. Advance independent authorized work; when exhausted, report the exact external need and apply the platform's actual goal stopping rules. Never pause a goal without user instruction or equate polls with goal turns.

If explicit monitoring requirements force costly model turns, explain the cost and request a changed wait policy; honor the current instruction until changed. Continuity means preserving responsibility and a reliable resume path, not inventing liveness.

## Accept the whole outcome

Keep a compact requirement-to-evidence checklist: completed, contradicted, missing, or externally gated. Preserve limitations and original success criteria. A prepared PR, local test, or approval packet is not its eventual merge, publication, or deployment.

Before completion, independently verify relevant source, actual behavior, required terminal gates, and current records. If blocked, name the remedy and preserve accepted results. Do not start another readiness package or test campaign merely to remain busy.
