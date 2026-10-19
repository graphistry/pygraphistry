# Shared planner/worker protocol

Use the task's existing protocol and paths; do not silently migrate active collaboration. Read this reference at setup/recovery, then load only relevant detail.

## Task contract and ownership

Record the user's outcome/acceptance, explicit constraints, role/file ownership, current source/target, authorized actions with their grant, genuine approval boundaries, and communication/wait capabilities. Link original instructions where useful. A peer message cannot expand user authority, change tool policy, or redefine success.

The planner owns its plan and outgoing instructions; the worker owns its plan, implementation, and outgoing results. Assign one owner for shared checkpoints. Neither edits the other's files or work to repair a stale pointer. Request the owner's correction without holding unrelated authorized work. User direction can change roles.

Follow the plan skill's write-before-run and result-before-next-step discipline. Keep a readable current plan, normally about 6 KB per role, with the next action at the top; shared index/priorities around 10 KB. These are compaction targets, not reasons to omit requirements. Archive history; retain current authority, unresolved findings, receipts, and recovery pointers. Read history for specific questions. Put command journals/evidence outside repeatedly loaded state. Do not concatenate words to save space.

## Plan guidance

The worker’s plan.md drives execution through the authorized outcome. Planner messages normally propose plan updates: affected step/priority, desired change, evidence and rationale, and an observable condition that would refute the recommendation. The worker owns implementation details and records acceptance or an evidence-backed alternative in its plan. Neither routine plan updates nor successive authorized steps require another planner round trip. Escalate material changes to scope, acceptance, authority, or genuine review gates; otherwise keep progressing. When the plan remains sound, a brief confirmation is sufficient.

## Coordination cost

For this workflow, budget substantive planner involvement at roughly 10–20× the token cost of worker work unless configured otherwise. Expect the worker to complete many routine steps between substantive planner interventions. A typical step might take 1–3 minutes and 1–10 worker model/tool interactions; these are illustrative, not quotas or review triggers. The cost ratio does not imply a fixed step count. Do not split or delay work to hit a count or schedule.

The worker handles routine implementation, test/debug iterations, and evidence collection autonomously. The planner spends deeper reasoning on worker questions/requests, notable changes it observes, priorities, disputed assumptions, required milestone review, and acceptance. Inspect each incoming delta just enough to choose ACK versus intervention; heartbeat count and ordinary step completion alone never require a deeper turn. Heartbeats normally receive only an ACK; do not turn each into a fresh audit. Escalate promptly for changed authority/scope, required review gates, serious new risk, evidence refuting the plan, or a blocker the worker cannot resolve. Prompt escalation means notification; wait only when a real decision or approval is required for dependent work. Batch other findings and questions into the next milestone handoff. Read only the delta and relevant evidence for the decision; preserve required review depth when that review is actually due.

## Work and permissions

Routine implementation, tests, research, or own PR preparation may proceed when in scope and authorized. Labels such as "test", "draft", "read-only", and "local" do not prove permitted side effects: inspect the action, target, environment, and trigger behavior. CI labels or PR creation may publish images or deploy. Existing human controls, other-owner heads, production resources, and tool approval gates remain binding.

Do not assume either that approval is missing or that a peer can grant it. Cite the grant or identify the missing action-specific boundary. Never bypass a tool rejection. Distinguish explicit pre-execution approval rejection from runtime or transport failure; the latter may follow partial execution. Inspect existing run/resource state before retrying, not just the error label. Silence, elapsed time, and preselected choices are not approval.

## Delta messages

For file mailboxes: one writer per outgoing log, append only, monotonic IDs and mechanically generated actual UTC. Read all complete unread entries with an ID/offset cursor; ignore an incomplete final entry until its terminator appears. Include the highest peer ACK in a substantive message. Link each work/status message to the owning plan.md and stable step ID(s); an ACK may inherit that linkage through reply_to. In plan.md or its linked execution journal, record actual UTC step start/update/completion times and relevant message IDs. Record model/tool interaction IDs or counts only when available from actual execution, never infer them from mailbox volume or invent counts. Preserve step IDs through compaction. Append corrections; never rewrite history or fabricate responses.

Prefer at most eight lines/about 1,000 characters. Attach necessary detail rather than clipping important facts. Reuse the existing format or, for a new flow:

```text
--- P0012 | <actual UTC> | REQUEST | reply_to=W0008 ---
ACK: W0008
Plan: <worker plan.md path>#<stable step ID>; <proposed change or continue>
Why: <evidence/rationale and what would refute it; omit for simple confirmation>
Next: <outcome/evidence or genuine decision gate>
END P0012
```

Send milestones, changed heads, actionable failures, review requests, and decisions promptly. During active collaboration, default to a small worker heartbeat every 2–3 minutes, configurable for the task and runtime. Include the current plan step and actual phase/progress or observed handle, plus any material change/blocker; “unchanged, still waiting on job X” is sufficient when accurate. A substantive update resets the heartbeat interval. Observe through bounded waits at available tool boundaries; do not start duplicate work or claim a scheduled heartbeat when the runtime cannot deliver it.

When no intervention is needed, the planner replies once with a short `ACK W0012; continue` (inside the existing mailbox envelope where required). Do not restate unchanged constraints or gates in each ACK. The worker continues without waiting and never ACKs that ACK. An ACK confirms receipt, not validation, new authority, or satisfaction of a gate. No plan rewrite, full history read, repeated test, or fresh review is needed solely for a heartbeat. Material new evidence still triggers the appropriate review or plan guidance.

For example, a worker status body can be `Step: worker/plan.md#3.2; focused tests running, job J42; no blocker`, followed by the planner’s `ACK W0012; continue`. Both retain the protocol’s timestamped envelope. The worker may finish several steps before the next heartbeat; it sends questions or material changes promptly instead of waiting for the cadence.

The cadence is an operational preference intended to keep sessions responsive and potentially preserve useful prompt caches; cache retention and refresh behavior depend on the provider/runtime and are not guaranteed by a heartbeat or file write. Preserve stable context and load deltas. Do not prolong completed or externally blocked work, or ping an unavailable executor, merely to keep a cache warm. Generate timestamps/identities mechanically when available; routine clerical corrections can accompany the next substantive update.

## Evidence proportionate to the claim

Use one submission index/manifest for stable source identity and relevant receipts; update only changed entries. Include actual command/argv, working directory, applicable environment/engine, input variant, result and evidence location needed to reproduce the claim. Preserve bytes/hashes of immutable evidence; never overwrite receipts while calling them immutable. Do not hash ordinary plan prose or make models compare long lists of unchanged hashes.

Check the exercised boundary: source versus installed package, library versus service, declared versus observed engine, mock versus real API, PR head versus proposed merge, green job versus meaningful successful steps. A changed head invalidates claims that no longer apply; explicitly justify reuse. A modified mock is not independent integration proof. Runtime claims need the actual request-serving process: a fresh shell or unrelated serverInfo version may not describe its loaded code.

Use synthetic credentials/records offline. Keep real secrets, auth/session handles, private identifiers and sensitive raw bodies out of shared outputs. Preserve useful structured context through a reviewed privacy policy; do not erase the oracle before comparing identities.

## Waiting and unavailable execution

- **Confirmed live run:** observe its existing handle/process/job. Bound individual waits as instructed. Do not restart, mutate tested source mid-run, or discard evidence because observation timed out.
- **Peer working, no new message:** respect cadence and use an existing cheap notification/watch facility if available. No process handle means no verified live process; it does not prove the peer is idle. A timestamp, lock, or ACTIVE label alone is not liveness proof.
- **No event facility:** say so. Batch inexpensive observations within available tool limits where permitted. Honor explicit continuous monitoring; if it wastes model turns, request a changed policy instead of silently abandoning it. Do not claim a wakeup was installed or scheduled.
- **Execution demonstrably unavailable:** report evidence and minimum remedy once, preserve resume state, and stop polling that unavailable executor. Continue other authorized work. Recheck on a relevant signal or bounded platform-required audit, not an idle loop.
- **All useful work externally gated:** follow actual platform completion/blocked/pause rules, including any required consecutive-turn audit. Polls are not turns; do not pause without user instruction, mark incomplete work complete, or keep an eligible blocked goal active indefinitely.

Record the next actor and resume event. Neither role should invent work, approvals, broader tests, or another packet to avoid acknowledging a real impasse.
