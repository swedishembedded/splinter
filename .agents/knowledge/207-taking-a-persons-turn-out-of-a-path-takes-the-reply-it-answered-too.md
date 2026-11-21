<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 207. Taking a person's turn out of a guided path takes the reply it answered too

**Lesson:** to fork the student at a human-guided path's states without the guidance, remove every turn the person
typed *and* the reply that called no tool just before it. A person usually types after the agent has stopped ("Wrote
the file."). Without their message, that stop and the next action are two assistant messages in a row, which sven's
fork records as one model response. The replayed transcript then no longer matches the path step for step, and the
fork is refused (`step 4 of the continuation is not the student's step 4`). The stop was the end of a turn the person
answered, not a state the student would continue from, so it goes with the turn (`bench::repair::unguided`).

## Evidence (2026-11-21, `crates/pipelines/tests/it/path_guided.rs`)

The scripted student's path: reads, "done", the person's turn, the right write, "Done.". With only the person's turn
removed, the `path_guided` repair was refused with the prefix mismatch above. With the answered reply removed too, it
was accepted, and none of the student's requests held the person's text. No sven change was needed: the SDK's
`Engine::fork` takes the cut, unguided transcript as it is, so dropping a non-system block (sven GL-S2b) is not
needed for this path.
