<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 214. A train client can wait forever on a training server that has already run several jobs

**Lesson:** four times on 2026-11-21/22 (splinter-c2a7d877e `train` against brain-bcd67e5ad `serve --training`, one
job at a time per server), a new `splinter train` printed `[seed] --seed N` and then nothing. Each time the server
had just finished about eight small LoRA jobs normally, and its log ended at the previous job's final eval. The client
sat in `futex_do_wait` with no TCP connection to the server, holding the state's `expdb/packs/journal` open, and
neither side logged an error or timed out. Killing the client and retrying against the *same* server hung again (fold
c T-other). Retrying against a freshly started server went through at once. The cause (client or server side) is not
yet known.

**How to apply:** a training queue must notice a job that makes no progress (no `brain train: step` line within a few
minutes) and restart the training server. A client waiting on a server should fail with a deadline, not wait
silently. Evidence: lane EXP `logs/train-e10-b-tsame-s1.err`, `logs/train-e10-c-tother-s1.err`,
`logs/train-e3-p7-r4-s1.err` and the training servers' logs `serve/trn3*.log`, `serve/u0.log`.
