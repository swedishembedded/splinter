<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# jefferson - can Splinter, alone, learn to think like Thomas Jefferson?

Splinter is given one sentence and a directory of what Jefferson wrote:

```bash
splinter "Learn to think like Thomas Jefferson based on the materials he has written in directory ./materials"
```

Everything else is Splinter's own work: it captures the directory, surveys it,
plans what to teach (the planner is a model, held to the survey by code),
generates tasks and conversations grounded in the letters, has a teacher answer
them open-book, verifies every answer by code, trains a LoRA adapter on the
verified answers and on his own text word for word (as many records of it as
there are answers, cut from the letters and works by code alone, held out with
the letters the exam asks about), examines the adapter against its base on letters it never
saw under a calibrated judge, and releases it only if the gate passes. This
sample supplies the materials and an independent check; it supplies no logic
Splinter lacks.

## Running it

```bash
# 1. The public-domain texts (checksummed) and the directory Splinter learns from.
python3 samples/jefferson/fetch.py --resources RESOURCES
splinter-jefferson materials --resources RESOURCES --out ./materials

# 2. Freeze the suite the release gate holds the adapter to, so that learning
#    Jefferson is checked against forgetting everything else: general knowledge,
#    and the arithmetic and format-following a persona fine-tune erodes first.
splinter eval --suite anchor --freeze samples/jefferson/anchor.jsonl --freeze samples/adams/anchor-skills.jsonl

# 3. Splinter does the rest. The roles are named by the configuration:
#    the policy to train, the assistant that plans, writes, teaches and judges,
#    and how long the run may take (a sentence names no budget).
SPLINTER_ASSISTANT_MODEL=local:Qwen/Qwen3-8B SPLINTER_BF16_BASE=1 SPLINTER_BUDGET=8h \
  splinter "Learn to think like Thomas Jefferson based on the materials he has written in directory ./materials"
```

The `learn` report names each stage's result, including the exam: how often the
judge says the base's and the adapter's answers give what a held-out letter
says, how often either states a number or name the letter does not hold, and a
paired sign test of the two. The judge is calibrated on controls made from the
run's own verified answers first; one that cannot tell them apart grades
nothing and the report says no claim is made.

`materials` writes the letters of the training families and Jefferson's own
works (not the Life and Morals of Jesus of Nazareth, which is the Gospels cut and
arranged with a modern editor's introduction, and would put a quarter of the
corpus in other voices), and withholds the exam families for the independent check below.
A letter ends where its volume turns to a book, a part, an appendix or an index, so
a work or an index printed after a volume's last letter is not glued into it, and a
stretch of more than 15,000 words under a letter's heading is no letter. A letter that
one of Jefferson's own works prints too is not a letter of the corpus, and a letter file
that Splinter's overlap rule would call one text with an exam letter is held back;
the command refuses to write when a work is, and prints how many exam families share
runs of words with what it wrote.
`materials --exam-pool` writes the letters the materials leave out (one printing of each
family) instead: the pool an exam is reserved from, with `splinter exam-set create`, when the
models to be examined were trained on the materials.
Splinter makes its own held-out split from what it is given, by group of
overlapping text, so its exam and the independent one are different letters.

## The independent check

The commands below are a second measurement that Splinter's own stages do not
share code with, so the two can disagree. They ask the base and the adapter
Splinter produced a set of questions fixed before training and read the answers
without Splinter's judge. `exam --persona "Thomas Jefferson"` asks under the
prompt Splinter trained the adapter to be him under, so the adapter is asked as
it is deployed (`--system TEXT` is any other prompt).

## The question

A model that "knows" a corpus should (1) answer questions about passages it was
trained on, (2) recognise the authors and works behind passages it never saw,
and (3) when asked what Jefferson wrote, quote what exists and not what sounds
right. The third is the honesty metric of a persona model. Training can raise
it or lower it; this sample reports which.

## Corpus

Public-domain texts fetched by `samples/jefferson/fetch.py` (`python3 -I samples/jefferson/test_fetch.py` checks it offline): the
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
never edited: `tasks` pins it in a `FROZEN.json` beside it the first time it is
written and refuses to write a different exam under that name, and `exam` says
when it is asked about a file nothing pinned. A spec asserts that no exam item
shares a family, a passage or a question with the training set.

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

## Retrieval

`splinter-jefferson retrieval` measures how well search finds the passage a
situation was written from: 70 situations over the paragraphs of every letter
(`recall@k` is the share whose source paragraph is among the first k found).

| method | recall@1 | recall@10 | recall@50 | recall@200 | MRR |
|---|---|---|---|---|---|
| lexical (BM25) | 17% | 30% | 43% | 53% | 0.220 | <!-- perf-number: a sample report's measured result -->
| semantic (Qwen3-Embedding-0.6B) | 41% | 66% | 83% | 91% | 0.489 | <!-- perf-number: a sample report's measured result -->
| fused (equal-weight reciprocal rank) | 27% | 56% | 77% | 93% | 0.367 | <!-- perf-number: a sample report's measured result -->

A situation put in plain words shares few words with the eighteenth-century
paragraph it came from, so semantic search finds far more of them than lexical
search, and fusing the two with equal weight lets the weaker ranking pull the
better one down at the top while adding a little at depth. Semantic search is
the primary ranking; lexical is for the exact names and dates it blurs.

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
