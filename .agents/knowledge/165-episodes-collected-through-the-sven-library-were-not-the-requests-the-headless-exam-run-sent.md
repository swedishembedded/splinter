<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 165. Episodes collected through the sven library were not the requests the headless exam run sent

**Lesson:** when training data is collected with one entry point of an agent runtime and the model is
examined through another, diff the recorded requests of both on the same task before training. A tiny
model learns the exact bytes; every difference is a distribution shift it cannot bridge.

Found 2026-11-17 in minimind (default profile, seed 7, sven 0d7d6643, bench job splinter-5d8766415eb145b2):
bench runs `sven --headless --run-timeout 780` on the task's `instruction.md`; the sample ran episodes
through the sven SDK. Diffing bench's `--record-requests` file against the sample's recorded requests showed
three differences, each now pinned by a spec (`crates/agent/src/workspace_episode.rs`,
`samples/minimind/tests/acceptance.rs`):

- the headless run ends the task and every tool result with `\n\nTime: U of T min used, L min left; leave
  time to verify and finish.`; the SDK does so only with `RunOptions::announce_deadline(true)`;
- the task text is the file's, with its final newline (so the request reads `...\n\n\nTime:`); the sample
  passed the request without it;
- requests offer the tools sorted by name (`read_file`, `shell`, `write_file`); the pretraining documents
  rendered them as `read_file`, `write_file`, `shell`.

Sampling differed too: the exam's requests name none, so brain served the chat fallback (temperature 0.7)
while the probes asked for temperature 0; a from-scratch checkpoint now declares `do_sample: false`
(`init_model` `sampling`) and the probes send none. With the conditions identical (default-v11, -v13) the
exam still scored 0 while the sample's own probes solved 2-5 of 12: the remaining gap is the task
distribution, not the protocol (under the exam the model read a made-up file instead of a listed one in all
1041 of its reads).
