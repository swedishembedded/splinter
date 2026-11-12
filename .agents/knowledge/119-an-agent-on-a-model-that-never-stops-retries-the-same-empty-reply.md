<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 119. An agent on a model that never stops retries the same empty reply

Two traps met when a sven agent was run on the toy checkpoint. (1) Without `eos_token` in
`tokenizer_config.json` generation does not stop at the end-of-turn token, so every reply ran to the token
cap, sven answered with "your last reply reached the output limit" and the growing prompt overflowed the
64-token context. (2) A seeded provider that samples an empty reply once samples it again on the retry,
so the run fails after two attempts ("model returned no content"): the conversation has no record and a
harness has to count it, not abort. The toy checkpoint now declares its eos token and a 256-token context;
the self-play smoke bounds the reply and skips silent runs, reporting how many.
