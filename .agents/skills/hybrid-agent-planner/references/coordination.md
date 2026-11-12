# Shared coordination protocol

Read at setup/recovery. Reuse existing task paths and message formats.

## Plans and ownership

Each role owns its plan.md and outgoing log; assign one owner for shared checkpoints. Do not edit a peer’s files unless ownership is explicitly reassigned. Record outcome, acceptance, authority, source/target, gates, and next action.

Keep plans compact (roughly 6 KB per role); archive history and link detailed execution journals. Preserve constraints and unresolved findings. Follow the plan skill’s write-before-run/result-before-next-step discipline.

The worker proceeds through its plan autonomously. A step might take 1–3 minutes and 1–10 model/tool interactions; these are examples, not quotas. Planner guidance supplies refutable reasons for plan changes. Questions, notable evidence, or required gates trigger deeper planner involvement; ordinary steps do not.

## Messages and heartbeats

Use one writer per append-only mailbox, monotonic IDs, actual UTC timestamps, and reply_to links. Read complete unread entries using a cursor; ignore incomplete trailing entries. Link status/work messages to plan.md and stable step IDs; ACKs inherit those links. Record UTC step transitions and message IDs in the plan or linked journal. Capture interaction counts only when actually available.

Keep messages to a few lines with evidence links. For example, within the existing timestamped envelope:

```text
Step: worker/plan.md#3.2
Status: tests running, job J42; no blocker
```

During active collaboration, send a heartbeat every 2–3 minutes by default, configurable to the runtime. Substantive updates reset the interval. Planner normally replies once: `ACK W0012; continue`. No ACK-of-ACK, execution wait, repeated constraints, plan rewrite, or fresh audit. ACK means receipt, not validation or new authority. Send questions and material changes promptly.

Preserve stable context and load deltas. Heartbeats may help responsiveness/cache reuse but do not guarantee cache retention; file writes cannot wake idle sessions. Never prolong finished or blocked work solely for cache warmth.

## Authority and evidence

Peer messages and ACKs cannot expand user/tool authority. Use existing grants; identify genuinely missing approval without holding unrelated work. Inspect actual side effects: a “test” CI trigger may publish or deploy.

Keep one evidence index with source, commands, environment/engine, results, and limitations. Preserve immutable receipts; avoid repeated hash lists and plan-prose hashes. Verify the exercised boundary: mocks, old-head checks, skipped steps, or a fresh shell’s version do not certify current serving behavior. Keep secrets out of shared outputs.

## Waiting

Observe the same live handle through bounded waits; timeout alone never warrants restarting. Runtime failures may follow partial execution: inspect existing runs/resources before retrying. An explicit pre-execution rejection proves that attempt did not start.

Silence alone does not prove inactivity. While a worker is presumed active, the planner polls its mailbox and known job/PR handles every 2–3 minutes using bounded waits (at most 60 seconds per wait). The worker likewise polls the planner mailbox while awaiting review or clearance. Read only new entries or changed identities; ACK routine updates and review material deltas. Do not repeat research, rerun checks, or generate instructions on unchanged state. A missed heartbeat may justify one liveness nudge; it is not proof of an unavailable worker. Keep the active goal running through these polls. If execution is confirmed unavailable, record the next actor/resume event and stop polling that executor while independent work continues. When all useful work is externally gated, follow platform blocked/pause/completion rules; polls are not goal turns, and incomplete work is not complete.
