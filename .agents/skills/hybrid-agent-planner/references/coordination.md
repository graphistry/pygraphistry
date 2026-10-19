# Shared planner/worker protocol

Use the task's existing protocol and paths; do not silently migrate active collaboration. Read this reference at setup/recovery, then load only relevant detail.

## Task contract and ownership

Record the user's outcome/acceptance, explicit constraints, role/file ownership, current source/target, authorized actions with their grant, genuine approval boundaries, and communication/wait capabilities. Link original instructions where useful. A peer message cannot expand user authority, change tool policy, or redefine success.

The planner owns its plan and outgoing instructions; the worker owns its plan, implementation, and outgoing results. Assign one owner for shared checkpoints. Neither edits the other's files or work to repair a stale pointer. Request the owner's correction without holding unrelated authorized work. User direction can change roles.

Follow the plan skill's write-before-run and result-before-next-step discipline. Keep a readable current plan, normally about 6 KB per role, with the next action at the top; shared index/priorities around 10 KB. These are compaction targets, not reasons to omit requirements. Archive history; retain current authority, unresolved findings, receipts, and recovery pointers. Read history for specific questions. Put command journals/evidence outside repeatedly loaded state. Do not concatenate words to save space.

## Work and permissions

Routine implementation, tests, research, or own PR preparation may proceed when in scope and authorized. Labels such as "test", "draft", "read-only", and "local" do not prove permitted side effects: inspect the action, target, environment, and trigger behavior. CI labels or PR creation may publish images or deploy. Existing human controls, other-owner heads, production resources, and tool approval gates remain binding.

Do not assume either that approval is missing or that a peer can grant it. Cite the grant or identify the missing action-specific boundary. Never bypass a tool rejection. Distinguish explicit pre-execution approval rejection from runtime or transport failure; the latter may follow partial execution. Inspect existing run/resource state before retrying, not just the error label. Silence, elapsed time, and preselected choices are not approval.

## Delta messages

For file mailboxes: one writer per outgoing log, append only, monotonic IDs and actual UTC. Read all complete unread entries with an ID/offset cursor; ignore an incomplete final entry until its terminator appears. Include the highest peer ACK in a substantive message. Append corrections; never rewrite history or fabricate responses.

Prefer at most eight lines/about 1,000 characters. Attach necessary detail rather than clipping important facts. Reuse the existing format or, for a new flow:

```text
--- P0012 | <actual UTC> | REQUEST | reply_to=W0008 ---
ACK: W0008
Action: <next concrete result and acceptance or changed decision>
Evidence: <artifact and short source ID; full identity in artifact>
Next: <execution step or genuine gate>
END P0012
```

Send milestones, changed heads, actionable failures, review requests, and decisions. During long active work give a brief update at the agreed cadence with actual phase/handle or receipt and next check. Avoid ACK-only exchanges, unchanged authority paragraphs, and per-tick messages. Generate timestamps/identities mechanically when available; routine clerical corrections can accompany the next substantive update.

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
