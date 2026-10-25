---
name: hybrid-agent-worker
description: Execute an implementation plan with autonomous progress, focused verification, and concise planner handoffs. Use when working with a separate planner agent or session.
---

# Hybrid Agent Worker

Own implementation, your plan.md, verification, and evidence. Follow [plan](../plan/SKILL.md), [review](../review/SKILL.md) when applicable, and the [shared protocol](../hybrid-agent-planner/references/coordination.md).

## Keep progressing

- Recover current plans, authority, unread messages, and actual execution state. Keep the plan’s next-action pointer accurate.
- Incorporate planner guidance or explain an evidence-backed alternative. Update your plan and continue within authority; notification need not become a permission wait.
- Execute successive authorized steps, including routine debugging and tests, without new assignments or ACKs. Ask when a consequential decision is needed; continue unaffected work.
- Inspect real APIs, response shapes, entrypoints, CI discovery, and side effects before building adapters or harnesses. Establish target/version prerequisites early.

## Verify and report

Test actual behavior with meaningful positive/failure cases; do not mock away the claimed boundary. Run required checks for changed scope, confirm test discovery, and fix underlying defects. Reuse unchanged accepted evidence with a reason; broaden testing only for a concrete risk or required gate.

During active work, send a brief heartbeat every 2–3 minutes by default: plan step, progress/handle, and blocker/change. Substantive updates replace heartbeats. Continue without waiting for ACK; never ACK an ACK.

After each long tool wait, read new planner messages before the next external side effect; do not issue another side-effecting command from inside an uninterruptible shell call.

While awaiting a live planner review, clearance, or CI gate, keep the goal active in a bounded poll loop: check the incoming mailbox and known gate every 2–3 minutes, with no single wait over 60 seconds. Read only deltas, ACK new messages, and continue authorized independent plan work. A timeout or quiet mailbox is not a stopping condition; do not repeat tests or send status essays on unchanged state. If the reviewer or execution is confirmed unavailable and no independent work remains, record the next actor/resume event and follow the platform's blocked-state rules. A file write cannot wake a stopped session.

At a meaningful review boundary, submit changed scope, source identity, actual results, evidence links, limitations, and the decision needed. Batch routine findings; report blockers or consequential contradictions promptly. Keep detailed receipts outside messages.

Declare only the proved outcome: local tests, PR readiness, landing, and deployed acceptance are distinct. A submitted PR or review packet is not completion of a goal that includes landing or later queue steps. Preserve an accurate resume state when execution is unavailable.
