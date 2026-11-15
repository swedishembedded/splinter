<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 155. A frontier program longer than the training row failed a whole lineage

**Lesson:** a lineage round that selects its training programs by a rule (the frontier) must also apply the trainer's own
length bound when it selects, or one long program the policy wrote fails the round after hours of proposing. The trainer's
pre-check is right to refuse an over-length record (it would otherwise be cut and train on a truncated answer); the defect was
that the driver had no rule for it. The rule now: measure each candidate record with the trainer's meter
(`splinter_model::measure::RecordMeter`, brain's `ExampleMeter`, same chat line, same thinking and packing mode) and take the
first `target` frontier programs in the order proposed that fit `max_block`; the skipped ones are listed with their lengths in
the round record, and fewer than `target` fitting fails the round with that reason. With no over-length record the selection is
the first `target` frontier programs, as before, and the round record serialises byte for byte as before.

## Evidence

- RSI4 follow-up M, control lineage m0 (seed 8785991200924142667): 2184 of 2400 proposals on the frontier after 3584 requests,
  training on 2100; the training job was refused before submission: `g1r1-mispredicted.jsonl:394 (prop-e33f535a2d3b8ac2) is
  1217 tokens` against `max_block 1024`. RSI4 itself never proposed such a program.
- The driver's measurement of that program's record (`chat_line` of the proposal, the meter of
  `Qwen/Qwen3-4B` without thinking or packing) is 1217 tokens, and the line it measures is byte-equal to line 394 of the
  dataset the registered binary wrote.
- Unit tests in `samples/campaign-lua/src/frontier.rs`: an over-length frontier program gives its place to the next that fits
  (programs past the last one chosen are not measured); with none too long the selection is the first `target` on the
  frontier; a frontier short of fitting programs fails with the count and the lengths.
