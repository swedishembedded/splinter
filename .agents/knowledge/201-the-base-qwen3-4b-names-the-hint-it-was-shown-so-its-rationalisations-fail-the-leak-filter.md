<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 201. The base Qwen3-4B names the hint it was shown, so its rationalisations fail the leak filter

**Lesson:** with the reference solution as a hint, the 4B reaches a verified state easily (40 of 48 hinted attempts),
but its reasoning or messages name the hint (the words `hint` or `reference`) in 44 of the 48 attempts, despite
the instruction telling it never to. Unhinted, it uses those words in 1 and 4 of 221 passes. Kept rationalisations are
therefore 2 of 48 (4%), under item 24's 10% stop rule, and the bottleneck is the leak, not sufficiency: both attempts
that reached the sufficiency check passed it.

## Measured (2026-11-21, feasibility gate of item 24)

`splinter experiences rationalise` (splinter-2756587f8, sven-1db749d9, bench 6ccc7ec7), rationaliser base
`Qwen/Qwen3-4B` on brain-8f2a3d264 (gpu0, shared with another lane's 27B process), T 1.0 / top-p 0.95 / top-k 20.
24 train tasks drawn with seed 20261121 from the 160 the base failed at least once, stratified by difficulty, x 2
samples. Leak terms as decided before the gate: `as given`, `solve.sh`, `reference`, `hint` (knowledge 200).

| filter (in order) | attempts passing | alone |
|---|---:|---:|
| graded | 48 | |
| verified | 40 | 40 |
| real observations | 40 | 48 |
| no leak | 2 | 4 |
| sufficient | 2 | |

- Leak findings were all term mentions: `hint` in 40 attempts, `reference` in 14. No attempt was caught for the hint's
  own wording or for quoting unobserved file content.
- Kept: `048-delete-a-word-including-separator` and `090-repeat-a-substitution-with-a-new-range`, one sample each.
- Controls on the 221 unhinted passes: `as given` 1, `hint` 1, `reference` 4, `solve.sh` 0; real observations 221 of
  221 under the decided filter.
- The whole gate took 5 minutes of wall time at 48 attempts at once.

## What it means

Showing the 4B a worked answer and asking it to reason as if unaided does not give clean trajectories: it narrates
where its knowledge came from. Ways around it change the method and are not item 24: rewriting the reasoning with
another model (then it is no longer the 4B's own), sampling many more attempts per task to find the rare clean one,
or a sufficiency-only filter that tolerates the mention (the trace would then train the model to cite a hint it will
not have).

## Gate b: the framing does not change it (2026-11-21)

The same 24 tasks x 2 with the hint framed as private scratch knowledge (`--hint-framing private-scratch`: work as if
you found the approach yourself, observe the files through the tool first, never mention or allude to any hint,
reference, provided answer or solution file; splinter-2e5f0c38e): verified 41, real observations 41, no leak 4,
sufficient 4 (8.3%, under the 10% rule; the arm stops for good). The model named the hint in 42 attempts (`hint`),
`reference` in 9 and `as given` in 4, as often as under the worked-answer framing. An explicit instruction not to
mention the hint does not keep a 4B from mentioning it; every attempt that did not mention it passed sufficiency.

## Gate c: re-judged with the content-only default, offline (2026-11-21)

The word list was a proxy: naming the hint is not copying the answer. The default judgement is now verified plus no
answer content present before it was observed (the hint's wording, a line of the task's files nothing had shown);
the word list, real observations and sufficiency are opt-in. `experiences rationalise --regrade` re-judged gates a
and b (96 graded attempts, 24 tasks) without a model, a gateway or a Harbor job, unpacking each packed hinted job in a
copy: 81 of 96 kept (gate a 40, gate b 41; 23 of 24 tasks), every rejection the verifier's (reward 0). The content
check rejected none of the 81 verified attempts: vim instructions print the starting and expected files, so their
lines count as shown, and no attempt repeated five words of a hint line it was not also shown or did not send itself.
On this dataset the default is therefore the verifier alone; the kept traces still name the hint (one reads "The hint
mentioned that the semicolon ';' repeats the last find command"), which the default deliberately allows.

The sufficiency run's notes (`reasoning_of`) are each agent step's reasoning followed by its visible message, so a
non-thinking trace (empty reasoning, text before each call) gives its visible text as notes; a spec with such a
trajectory holds it. A trace whose notes come out empty has no visible text in its steps' messages either.

## The answer check missed the hint's own commands (2026-11-21, fixed)

Gate c's default kept 81 of 96 4B attempts because the "no answer present" check looked only for 5-word runs of the
hint and for file lines; a hint command is often one word (`f{ci{new`, `/BEGIN/,/END/s/off/on/g`, `normal! OHEADER`).
Lane VSFT's own check rejected 30 of its 55 verified 27B attempts for writing a reference command in text. The check
now also looks for each hint command line, and each word of one that is not all letters, of 6 characters or more, in
the model's text before a tool reply showed it (`filters::hint_commands`). Re-graded offline with the fix: lane VSFT's
run rejects all 30 of its script's and 2 more (`f{ci{new` without `normal!`), keeping 23 of 55 verified; gates a and b
keep 5 of 96 (5.2%), under item 24's 10% bar, so gate c no longer clears it. Left unchecked on purpose: an all-letter
word of a command alone (`jyyggP`, `cccorrect`), which cannot be told from ordinary text without its line.

Also fixed (found by lane ALG-S): the 5-word wording check counted the tool arguments of every agent step, later ones
included, as known, so wording written first and sent as a call later passed. A step's own arguments (and replies) now
count as known only to the steps after it. Re-graded with both fixes (same counts): gates a and b keep 5 of 96, lane
VSFT's run keeps 23 of 55 verified (all 30 of its script's rejections, plus 2 for `f{ci{new`).
