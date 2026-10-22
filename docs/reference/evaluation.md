<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Evaluation reference

What the release gate checks, the suites that `eval` grades, and the file format of the
anchor suite. How to run an evaluation is in the [evaluation guide](../guides/evaluation.md).

## The release gate


`release <CANDIDATE-ID>` decides a candidate against the release its alias
(`--alias`, default `default`) points at - the champion, or the base before
any release - and refuses a candidate that was not trained from it. Both
are graded closed-book by each task's own verifiers (the judged kinds by
`--judge REF`, a calibrated judge) on the same suites, and the candidate is released only if all four checks pass; each
is printed with its numbers, and a check that could not be measured fails:

| Check | Passes when |
|---|---|
| improvement | on the new datasets' held-out tasks - the records training held out, and the variants of the tasks it trained on - a one-sided paired sign test over the tasks only one model got right is significant at alpha 0.05, tasks about one family of source text counting once (a family is won or lost by which model got more of its tasks right); ties and tasks without a verdict for both are excluded and counted |
| retention | on each earlier release's held-out tasks, the candidate's accuracy is at most 0.05 below the champion's (each release reported) |
| anchor | on the anchor suite in force, the candidate's accuracy is at most 0.02 below the champion's; both arms are asked it under the default prompt, so it measures what the weights kept and not the persona prompt; the report gives the drop's bootstrap interval over the paired items beside it |
| serve | `brain serve --adapter <candidate>` (the `brain` on `PATH`, or `--brain-bin`), with the base checkpoint the candidate was trained on as its `--checkpoint`, in a runtime directory of its own, starts - brain binds the adapter only to the base whose digest training recorded on it - reports the candidate's adapter digest, and re-answers up to 8 held-out tasks through its OpenAI-compatible endpoint, both sides decoding greedily and without a reasoning block, with the same answers as in-process on at least three quarters of them (the same verdict, and either the same text over its first nine tenths, runs of whitespace aside, or - worded otherwise, as a sampling server or another summation order will - the same meaning: embedded by Qwen3-Embedding, an answer must be nearer its own in-process answer than to the in-process answer of any other task, so the comparison cannot call everything alike); each task answered differently is reported with both answers |

### The improvement check

The improvement check measures whether the candidate learned the facts it
was trained on, on questions it was not trained on: the held-out records (the
newest tenth of the new data, grown to eight families of source text where
the data has them, within a quarter of it, so that a sign test over families
can reach significance) are few, and questions about other facts cannot
improve for a campaign that teaches facts. So the suite is completed with the stored
variants of the tasks the candidate's trained-on records were projected
from (`learn` writes them; `tasks variants` writes them for any task
set). A variant of a task that was held out measures nothing the
candidate learned and is not used. A variant whose wording the candidate
trained on - the same question, a near duplicate, or its words inside a
longer training prompt - is left out like any leaked held-out task. The
gate prints how many variants were measured and how many were left out,
by reason; the manifest records them. The decision rests on the two together by the family-level sign test, which stays the rule even
though its p-value moves with one family when only a dozen are discordant; the report therefore
also gives, for the whole suite and for each part, the difference in the share right with an
interval that resamples families and the exact sign test over the tasks, to read it by.
The decision rests on the two together,
but the report keeps them apart: generalisation (the held-out records, text
the candidate was not trained on) and recall under paraphrase (the variants
of trained tasks) each print their own comparison and sign test, and the
report states the system prompt each arm was asked each suite under
(`prompts`). Retention, anchor and serve are measured as before.

### Training warnings

A gate that passes is not silent about the candidate's training curve: the
warnings `train` reported (a large generalisation gap at the step carried, a
monitoring loss that rose after it or was still falling at the end of the
budget) are printed again with the verdict and carried in the JSON report
(`warnings`), as the exam carries them (`training_warnings`).

### Repository candidates

A candidate trained by `practice` is gated on a different suite than a
candidate trained on documents: the champion and the candidate attempt, as
agents in copies of the tree, the tasks of the exam that candidate's run set
apart before any attempt (every earlier release is held to the exam of the run
that made it), and the paired outcomes are counted by family in the same sign
test. Attempts that read the hidden tests, and attempts the harness could not
carry out, are set aside and counted for each arm in the report; neither is
in a denominator. The serve check asks the served candidate the same tasks
through the same worker and requires its verdicts to agree with the
in-process ones on at least three quarters of a sample. The anchor suite is
the frozen one; it holds coding-agent tasks only if some were frozen, and the
report says how many it holds rather than scoring an absence.

## Suites

`eval <REF>` grades one model - a candidate id, or a model reference -
closed-book on `--suite held-out` (the default: a candidate's new data, or
the release a policy alias points at), `retention` (every earlier release,
each reported), `anchor`, or a FILE of tasks, and names every task it got
wrong with the answer it gave, so two evaluations show what a candidate
lost against the base. The anchor suite is frozen
from one or more files with `eval --suite anchor --freeze FILE...` (the
files' tasks in order are one version; different tasks are the next version;
the same tasks are the same version) and shown with `eval --suite anchor`. An
anchor file is JSON Lines, `{"instruction", "reference", "kind"?}`, `kind`
(default `recall`) a closed-book kind graded against its reference by code:
`recall` (the answer states the reference), `arithmetic` (the last number the
answer states is the reference) or `format` (the answer has exactly the
non-empty lines the reference counts). The last two are what a persona
fine-tune of a reasoning model erodes and trivia cannot show.
