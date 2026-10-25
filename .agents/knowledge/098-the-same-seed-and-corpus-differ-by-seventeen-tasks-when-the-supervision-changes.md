<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 098. The same seed and corpus differ by seventeen tasks when the supervision changes

On the 46-family development benchmark D46 (one task per family, greedy decoding), two adapters
trained from the same base, the same 769-record corpus (`blake3:646692b2`), the same recipe
(rank 16, alpha 32, lr 5e-5, 120 steps, packed-bf16 base), the same training seed and the same
rebuilt training binary (sha256 `1d366e597bb504bc222304ebbe07ef0165c442f503091340e740004475a120ab`)
differ only in what is supervised: every admitted action, or
`--only-actions edit_file,write_file` (corpus `blake3:92252209`).

| seed | supervision | pass@1 | exposed | direct paired comparison |
|---|---|---|---|---|
| 13 | decisive actions | **30/46** | 1 | **42 paired, 16W-1L-25T, exact one-sided sign test p = 0.000137** |
| 13 | every action | **13/46** | 3 | - |
| 1 | decisive actions | 23/46 | 4 | 38 paired, 8W-4L-26T, p = 0.194 |
| 1 | every action | 18/46 | 4 | - |

The all-action arm at seed 13 is itself a regression against its own base (13 against the base's
21; 35 paired, 4W-11L-20T, p = 0.982). The decisive arm's task count does not come from a
different corpus, recipe, seed or executable: the only variable is the supervision mask. This
removes the rebuild confound knowledge 097 recorded (its all-action arm on the rebuilt binary
passed 18, not 12; the decisive arm passed 23): at the same seed the rebuild is held fixed and
the supervision still moves seventeen tasks.

The behaviour moves with the outcome and is seed-stable, counted over every D46 attempt from the
stored experiences:

| arm | edited a file | repetition abort |
|---|---|---|
| base | 24/46 (52 %) | 13 (28 %) |
| decisive, seed 1 | 35/46 (76 %) | 1 (2 %) |
| decisive, seed 7 | 36/46 (78 %) | 0 |
| decisive, seed 13 | 39/46 (85 %) | 2 (4 %) |
| every action, seed 1 | 22/46 (48 %) | 10 (22 %) |

What this establishes: at seed 13, on one benchmark, with one corpus and one executable, the
supervision choice is causally responsible for a 17-task difference, and the decisive corpus's
behavioural signature - edit, do not loop - is stable across three seeds. What it does not:
the seed-1 contrast is in the same direction but not significant (p = 0.194), and the two seeds
share the same 46 families, so this is one strong same-seed demonstration plus a
sign-consistent second seed, not two independent tests. It is a development probe on D46, not
the registered confirmatory query, and the training binary is rebuilt from main rather than the
pinned revision. The exposed counts differ between the arms (1 against 3 at seed 13, 4 against 4
at seed 1), which is why the pass@1 and exposed columns are reported beside the paired test:
knowledge 096 showed that a paired test which drops exposed attempts can hide a regression.

Evidence: `runs/g8/pilot-s13-d46-decisive-s13.verify.json` (30 passed, 1 exposed),
`runs/g8/pilot-ctl-allnew-s13.verify.json` (13 passed, 3 exposed),
`runs/g8/pilot-d46-base.verify.json` (21 passed, 9 exposed),
`runs/g8/compare-decisive13-vs-allnew13.json` (42 paired, 16W-1L, p = 0.000137),
`runs/g8/pilot-compare-ctl-allnew-s13.json` (35 paired, 4W-11L, p = 0.982),
`runs/g8/pilot-d46-decisive.verify.json` and `runs/g8/pilot-d46-allnew.verify.json` (seed 1:
23 and 18 passed, both 4 exposed), `runs/g8/compare-decisive1-vs-allnew1.json` (38 paired,
8W-4L, p = 0.194), and `runs/g8/edit-rate-2026-10-25.txt` (the behaviour table, read from the
stored experiences).
