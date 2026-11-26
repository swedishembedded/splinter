<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 231. A shuffled-request control separates what trajectories teach from exposure to the tool's format

**Lesson:** arm F of the source-to-transfer experiment trained Qwen3-4B on exactly B's 3359 verified Vim trajectories
with each record's task request replaced by another task's (a seeded derangement, `dataset.project`
`control: shuffled-requests`). It kept the tool-use format, the actions and the volume, and lost the request-to-action
mapping. Its validation loss fell like B's (2.81 to 0.68 against 2.49 to 0.60). On the 100 held-out exams F passed 0 of
1200 attempts in all three seeds, the same as the base. B, C, D and G each beat F (B 10/0, C 6/0, D 11/0, G 25/0 exams
won/lost, one-sided sign test p <= 0.016). So the exam gain comes from requests paired with the actions that satisfy
them, not from learning to emit well-formed tool calls.

**How to apply:** any claim that SFT on verified trajectories teaches transferable skills needs this control, in the same
registered family as the comparison with the base: a gain over the base alone can be format exposure, and in this
experiment the base passed none of its attempts, so format exposure was the cheapest explanation. A falling
validation loss on the control shows the control trained; it is no evidence of transfer.
