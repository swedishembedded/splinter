<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 071. A solve report named the episode bounds "cancelled", and 34 of 61 conclusions looked like infrastructure noise

Found 2026-10-18 diagnosing the withdrawn 27B pass 1 on the iniparse development set (experience set
`blake3:96d8ac44...`, splinter c65c5c1): the report said `conclusions: {success: 27, cancelled: 34}`, the
integrator read 34 external cancels of unknown origin, and the 27B teacher run was held back for it.

Cause: sven's runtime reports every stop that came from a cancel token as `cancelled`. Splinter's own episode
watch (`splinter-agent` `Worker::send`) stops an episode by firing a token of its own when one call returns the same
answer `max_repeated_results` (registered 3) times or when the episode passes its tool-call cap, so both bounds
surfaced as `cancelled`. The external token (a `splinter runs cancel`, a deadline) was never the cause: it would
have failed the whole solve with `Cancelled`, not one conclusion, and a deadline concludes `timeout`.

The evidence was in the store all along: each experience's `attempt-report` annotation records
`stopped_by_repetition` and `stopped_by_tool_cap`. Read over the 61 experiences (`TJ n=61
stopped_by_repetition=29 stopped_by_tool_cap=5 neither=27`, 27 with no final output plus 2 stopped after
the agent had answered), 29 + 5 = 34 are exactly the cancelled conclusions and the 27 successes carry neither
flag. Every one of the 5 tool-cap stops counted 61 or 62 tool calls while the exported trajectories held 4 to
42: sven's trajectory export had rebuilt the conversation from the compacted history, so a run that compacted
lost the steps the compaction replaced (fixed in sven with the whole-conversation record).

Fix: the solve report names an episode the watch stopped for the bound, `repetition_stopped` or
`tool_cap_stopped`, and keeps `cancelled` for a cancel
(`crates/pipelines/src/solving.rs`, `an_episode_stopped_by_its_own_bounds_is_named_for_the_bound_not_for_a_cancel`).
A stuck agent is a finding about the model and the loop, not noise to discount. Whether context compaction
contributes to the repetition is not established: the stored trajectories predate the whole-conversation record, so
the calls it replaced are missing from them.

How to avoid it: a conclusion label that several causes share cannot be read without the field that
separates them; when a report collapses causes, keep the separating field next to it.
