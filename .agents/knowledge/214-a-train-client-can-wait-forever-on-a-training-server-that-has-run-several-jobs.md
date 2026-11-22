<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 214. A train client waited forever once brain held its 32 ended jobs: it never forgot them and retried every 429

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

**Root cause (reproduced 2026-11-22, deterministic):** one training server, small LoRA trainings one after another,
on brain-bcd67e5ad and on the newest build alike: the 9th always hangs. brain holds at most 32 jobs per caller,
running or ended, until the caller deletes an ended one or its result's one-hour time to live runs out, and refuses a
creation past that with HTTP 429 "too many jobs: fetch or cancel some before starting another". Each splinter
training creates four jobs (three `check_dataset`, one `train`) and never deleted them, so the 33rd creation was
refused. The client read every 429 as "no room on a card" and asked again every 30 s, without bound and without a
word. The client owned it; brain behaved as documented.

**Fixed** in `served_train::job`: every ended job is forgotten (`DELETE`) once its result is read (single-server and
multi-server runs, cancelled ones included), so a series of runs never fills the limit. Refusals are reported on
stderr each round and bounded at `REFUSED_ROUNDS` (120 tries, an hour at the default wait). Past that, the run fails as
the endpoint's (`PolicyError::Endpoint`), with brain's last reason (`crates/model/tests/served_train.rs`; the scripted
service now holds jobs per caller as brain does).
