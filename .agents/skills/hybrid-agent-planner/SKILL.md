---
name: hybrid-agent-planner
description: Guide an implementation worker through plans, concise feedback, and independent review. Use for actual planner/worker collaboration across agents or sessions.
---

# Hybrid Agent Planner

Own priorities, decisions, review, and acceptance. The worker owns implementation and its plan.md. Follow [plan](../plan/SKILL.md), [review](../review/SKILL.md) when applicable, and the [shared protocol](references/coordination.md).

## Guide the plan

- Recover the outcome, acceptance, authority, current plans, and unread messages. Check actual source/execution state when it affects a decision.
- Suggest changes to the worker’s plan: affected steps, desired outcome, evidence/reasoning, and what would refute the recommendation. Leave routine sequencing and implementation to the worker.
- Let the worker proceed across authorized steps. With planner involvement roughly 10–20× more expensive in tokens, intervene for questions, notable developments, consequential decisions, or required reviews. Step counts and heartbeat cadence are not review triggers.
- Resolve prerequisites early: actual target/version, API contracts, test discovery, and execution side effects before prescribing substantial work.

## ACK or intervene

For routine progress, reply `ACK <message ID>; continue`. Read only enough of the delta to decide whether intervention is needed. Do not manufacture tasks, reread history, or repeat checks for a heartbeat.

When review is due, independently inspect relevant source and evidence, batch findings, and give concrete acceptance for fixes. Preserve required review depth; quiet passes do not resolve open findings. Reuse unchanged evidence with its original scope. Repeated repairs call for revisiting the underlying contract.

Keep genuine approval gates tied to specific actions; do not invent gates for authorized work or require deployed evidence before the merge needed to deploy. Confirm whole-outcome acceptance before completion. If externally blocked, preserve the next actor/resume event and follow platform stopping rules.
