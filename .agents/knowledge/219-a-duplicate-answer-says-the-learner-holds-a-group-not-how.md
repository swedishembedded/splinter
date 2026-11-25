<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 219. A duplicate answer says the learner holds a group, not whether it applied it

**Lesson:** when a resumed controller sends a group again, the learner's `duplicate` means only that it holds the
group, queued or applied. A restarted learner can take a resent applied group back on a send whose answer never
arrived (the connection dropped after it queued the group); the next resend is then `duplicate` while the learner
holds it queued and may still drop it as stale. The controller must therefore move a resent applied group back to
`sent` on any answer but `full`, and wait for the learner's report (applied, dropped, committed). The same
ambiguity decides the token count: a group the learner may have applied must not count toward the allowance, or
the allowance stays full while the learner waits for a step's worth (knowledge 217's deadlock in a new form). A
group counts as queued only after `accepted`; one recorded `sent` counts nothing on resume until its resend is
answered `accepted`. Spec: splinter `crates/factory/tests/it/online/mod.rs`
`a_resent_applied_group_the_learner_already_holds_is_followed_until_the_learner_ends_it`, and
`simulation.rs` `simulated_runs_that_once_failed_finish`.

## Evidence (2026-11-25, lane agrpo-e2e2)

The simulation deadlocked on world seed 14490181401853905193: group `sim:97` was accepted, applied, resent
(`duplicate`), resent after a learner restart (`accepted`, then the connection dropped before the answer was
recorded), resent again (`duplicate`) and stayed applied here while the learner dropped it as stale; the drop was
counted as a stray ack. Moving the group to `sent` but keeping restored `sent` groups' tokens queued deadlocked
world seed 13351248293160495685 (eight groups `sent` that the learner had applied, the allowance full, the learner
holding 3597 of a 4710-token step). With both rules, 50 fresh runs of at least 100k finished rollouts each finish.

The simulation's deadlock check was a simulated horizon of 2^40 ms, which at a 1 ms poll never ends in wall time:
a deadlock hung the test instead of failing it. It now fails once no change is recorded for 10^8 simulated ms (ten
times the longest rollout), within a second of wall time.
