<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 221. A stopped controller leaves its streaming job holding the run

**Lesson:** brain's streaming policy-gradient job outlives the client that feeds it: it waits for records,
by design, so a client that reconnects resumes its events. A controller killed mid-run therefore leaves its job
running, and brain starts no other job of the run (same `resume_key`) while one runs: the resumed controller's
job request waited for a lane the old job held, and the run made no progress until brain gave up on the request.
The online stage now records its job (endpoint and id, `stream-job.json` beside the run's state) and a resumed run
cancels the job it left before it asks for the run again; brain writes a cancelled job's training state, so what
it applied is kept, and the next epoch answers those groups as duplicates. A job request is also asked again while
the server is busy (429, 503) or out of reach, for up to five minutes. Spec: splinter
`crates/factory/tests/it/online/brain.rs` `a_resumed_controller_cancels_the_job_it_left_running_and_continues_the_run`
(the scripted service now refuses a second job of a running run, as brain does).

## Evidence (2026-11-25, lane agrpo-e2e2)

`scripts/online/cpu-stack.sh` on the CPU: after `kill -9` of the controller with 3 groups committed and
`splinter resume`, the next group was committed 126 s later (the old job still ran; brain's log shows no second
`brain/train` build); with the cancel the resumed run went on within seconds. Without the cancel the spec does not
finish within 200 s (its job request is refused as busy until the retry window closes).
