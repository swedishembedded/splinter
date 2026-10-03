<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# jefferson - does source-grounded fine-tuning teach a model the Jefferson corpus?

Teach a local model Thomas Jefferson's letters and the founding-era works he
argued from, then measure, on questions fixed before training, what it kept,
what it can recognise in text it never saw, and how often it invents a
quotation. Everything runs on the GPUs; a run that touches the CPU backend is
stopped.

## The question

A model that "knows" a corpus should (1) answer questions about passages it was
trained on, (2) recognise the authors and works behind passages it never saw,
and (3) when asked what Jefferson wrote, quote what exists and not what sounds
right. The third is the honesty metric of a persona model. Training can raise
it or lower it; this sample reports which.

## Corpus

Public-domain texts fetched by `resources/founding-america/fetch.py`: the
Washington, Randolph and Library editions of the letters, the Ford edition (OCR,
not parsed for letters), *Notes on the State of Virginia*, the *Summary View*,
the Jefferson Bible and the *Manual of Parliamentary Practice*, plus the
Federalist, Paine, Locke, Hobbes, Rousseau, Blackstone, Smith, Franklin,
Dickinson, Adams and the Convention debates. Each file is checksummed in its
`MANIFEST.tsv`.

The editions overlap: about half of the letters of one edition are printed in
another. The unit of the train and exam split is therefore the **family** of a
letter (the same letter as different editions print it, found from shared
eight-word runs), never a file or a position.

## What is taught, and what is asked

Every question has an exact answer built from the source by code. No model
writes a reference answer, and no model grades a training target.

| question | reference | graded by |
|---|---|---|
| who a letter was written to (its opening is shown, salutation removed) | the surname | the surname as a whole word |
| in what year it was written | the year | the year, and no other year |
| which of 17 listed works a 45-word passage is from | the work | a name of the work and no rival's |
| what Jefferson would say of a situation (Layer 2) | his own passage, quoted, with its source | quotations looked up in the corpus; recall of the passage; an independent judge |

Layer 2 scenarios are written by a model of a different family from the one
being trained (the writer drafts only the question), then admitted or refused
by code: the question must be a question of sensible length, must not name the
recipient or the year, must not refer to a passage or a letter, and must not
copy the passage. The answer a training record teaches is the passage itself,
quoted verbatim, with the letter it comes from.

## Arms and splits

Fixed before any training run:

| arm | what it is |
|---|---|
| base | the model with no adapter |
| trained | the same model with the LoRA adapter Splinter's trainer produced |

| split | contents | what it measures |
|---|---|---|
| `exam` | letter families and passages no training record contains | recognition of the sources, and the fabricated-quote rate |
| `seen` | a sample of what training contains | whether the model kept what it was shown |

A gain on `seen` and none on `exam` is memorisation. A gain on both is learning.
`exam` and `seen` are separate files; the exam is written before training and
never edited. A spec asserts that no exam item shares a family, a passage or a
question with the training set.

Students: `DeepSeek-R1-Distill-Qwen-7B` (the main arm) and
`DeepSeek-R1-Distill-Qwen-1.5B` (a faster rehearsal that gets more steps).
Writer and judge: `Qwen3-8B`, a different family from either student. The judge
is checked on controls with known answers (it must say YES to the reference
passage itself and NO to another scenario's passage) before its verdicts on a
model are read.

## Metrics

- Accuracy per question kind and split, before and after, with the paired sign
  test Splinter's release gate uses on the questions both arms answered.
- Unanswered rate. A reasoning model that does not finish inside the token
  budget gave no answer; it is counted, not guessed from its reasoning.
- Fabricated-quote rate: of the quotations of at least eight words an answer
  offers, the share that are not in the corpus.
- Recall of the trained passage on `seen` scenarios.
- Judge agreement with the reference position, with the judge's control result.

## Running it

```bash
# Parse the letters, group the editions' printings, split by family.
splinter-jefferson corpus   --resources RESOURCES --out OUT
# Compile the training set and the frozen exam.
splinter-jefferson tasks    --resources RESOURCES --out OUT --train-letters 130 --train-passages 18
# A writer model drafts the Layer 2 questions; code gates them.
splinter-jefferson scenarios --resources RESOURCES --out OUT --writer QWEN3_8B
# Train through Splinter's trainer (LoRA, bf16 base, one card).
splinter-jefferson train    --base BASE --dataset OUT/sft.jsonl --attempt ATTEMPT --steps 500 --bf16
# Ask the base, then base plus adapter, every exam question.
splinter-jefferson exam     --exam OUT/exam.jsonl --out base.jsonl --base BASE
splinter-jefferson exam     --exam OUT/exam.jsonl --out tuned.jsonl --base BASE --adapter ADAPTER
splinter-jefferson report   --before base.jsonl --after tuned.jsonl
```

Every GPU command is run through `gpu-only.sh`, which pins the card, reads the
run's output as it goes, stops it on any line that says the CPU backend, a
software adapter or a fallback was used, and fails it if it never reported a
discrete GPU. `BRAIN_DEVICE` alone does not choose the card of an in-process
run; the script also sets `BRAIN_GPU_INDEX`.

## Limits, stated up front

- A LoRA adapter on a 7B model does not store the corpus. What this measures is
  what a bounded amount of training changed, not whether the model "knows all
  of it".
- "Apply" to a situation Jefferson never faced has no reference answer. What is
  measured is the form of the answer (does it quote what exists, does it name a
  source) and, through a calibrated judge, whether it takes the position of a
  held-out passage. Quality on modern questions is not verified.
- The corpus includes Jefferson on slavery and race. A grounded persona can
  quote any of it; the answers are read by hand before anything is claimed.
- The Ford edition scans are OCR and are used for the quotation index only.
