<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learn how a person thinks

Use this guide to train the policy on a body of one person's writing so that it answers
as they would, on matters the writing never covered. Point `learn` at a directory:
Splinter surveys it, a planner model chooses what to teach from a menu, and code holds
each choice to the sources. The `jefferson`, `adams` and `synthetic-author` samples
measure what this teaches ([examples](../examples/index.md)).

What it does not show: a model that speaks like a writer is not a model that is right.
Correctness is graded against the writer's text; voice is scored separately, and the
exam reports both. See [limitations](../concepts/limitations.md).

## Plan from a sentence or a directory


```bash
splinter "Learn to think like Thomas Jefferson based on the materials he has written in directory ./jefferson"
```

With no `--kinds`, `learn` plans. After the sources are captured it surveys
them by code (how many parts and sections, how many sections read as advice,
a few excerpts) and shows the survey and the goal to a planner model (the
model `--planner` names, else the configured assistant, else the generator).
The planner chooses from a menu of task kinds, whether to distil, and who the
learner is becoming; code holds the choice to the survey, so a plan cannot
teach advice the sources do not hold, and a plan that names a person while the
sources hold advice or judgment must choose a kind that teaches how they judge
(`advise`, `converse`), facts alone being no way to think like someone. The
person becomes the policy's system prompt, in training and afterwards, and the
teacher's answers and the student's attempts are written under it too, so what
the records teach was said as the person they open with. A plan that breaks a rule is sent back
once with the rule it broke and then refused: a run that cannot be planned
says so, and no default replaces its plan. The survey and the plan are in the
report, and naming `--kinds` as well is refused: one of the two decides.

The `advise` kind is what teaches a person's own advice. Its task is a
predicament put to the writer, addressed to the writer (so it names no
document), and its reference is a passage of the writer's own text that
answers it. The generator is shown only the sections that read as advice, and
a task is admitted only when its reference is a passage of its evidence word
for word, so a paraphrase the model wrote is refused. An answer is graded by
the quotation verifier, with no model involved: every passage the answer puts
in quotation marks must be in the task's source text, the advice must be
reproduced, and an answer that quotes nothing or invents a quotation fails.

## The converse kind

The `converse` kind teaches how the writer talks and reasons. Its task is the
opening message of someone speaking to the writer (admitted only in the first or
second person: a situation, a decision, a question put to them; "What does he say
about ..." is refused, and the generator told so), with a passage of the
writer's text as its reference. A teacher shown the passage answers as the
writer, and a model that is never shown it plays the other speaker, following
up for up to three exchanges; the training record is the whole conversation
with every reply of the writer supervised and no passage in it. Two code
checks refute a dialogue, whatever else is said of it: every number, name and
quotation in the replies must be in the writer's text or in what the other
speaker said, and no reply may talk about "the material", "the passage" or "the
speaker", which the student is never shown. Neither establishes that a
dialogue is good: a judge other than the teacher does, and a dialogue it does
not pass teaches nothing. The other speaker never restates what the writer said
or asks it to confirm it; it brings its own case or objection. One dialogue in
eight ends by asking for a specific the exchange has not given, so that the
student also sees the writer decline to invent one.

## Retrieved passages and abstention

`--with-passages SHARE` trains a share of the records with retrieved passages of
the sources in the prompt, as `ask --retrieve` gives them, four to a record and
the passage the task was written from among them for four in five, the answer
unchanged. A policy trained only closed-book cannot tell the passage that holds
the answer from one that merely resembles the question: measured, passages helped
where retrieval found the evidence and hurt exactly as much where it did not.
Each passage is shown under its part and section, so an answer can name where it
stands. `--abstain SHARE` makes that share of *all* the dialogue records
abstentions (so at most `--with-passages`): the base model, as the writer, says
in two to four sentences that the writings before them do not establish an
answer - two thirds of them for a question whose evidence retrieval missed, a
third for a question that lies beyond the writings (about what came after the
writer's lifetime), asked of the passages retrieved for it - admitted only
when the reply is brief, in the first person and states no number the question
and the passages do not hold. An abstention is made from a dialogue record and
keeps its experience and family, so it is held out and trained with them; a
record no admissible abstention is written for keeps its answer. About a tenth
of the data is the working figure; abstentions are never the majority.

## The author stage and the voice view

The `author` stage teaches the writer's own voice. For the kinds whose
reference is a passage the writer wrote (`advise`, `converse`), the task's
message was written for that passage, so the passage itself, word for word, is
the answer to train on - instruction backtranslation: it has the writer's
diction and reasoning, which a teacher's paraphrase does not. It is recorded as
the source's answer, not a model's, and a judge of fit (the judge role, another
model than the generator that wrote the messages, measured like any judge on
controls from the tasks' own passages) keeps only the pairs where the passage is
a natural reply to its message. For one message the writer's passage is kept and
a teacher's dialogue on it is the duplicate; where it does not fit, the
dialogue stands. The judge of fit admits, so its passes are what is measured: a
passage it fails is withheld. With no usable judge of fit the stage says why and
the run goes on.

The `voice` view is the writer's own text as training data, with no model in
the loop. A conversation costs a teacher's time for every answer, and a run over
a large body of writing reaches a few per cent of it before its budget is spent;
the writer's text costs nothing to turn into records and has the diction and
reasoning no paraphrase has. Every text part of the sources is cut into chunks
of 250 to 650 words (and at most 880 tokens, counted by the policy's own
tokenizer, so a record never outgrows the 1024-token row the conversations
need): whole sentences, closed where a paragraph ends, an address line or
heading always with the text it heads, and no chunk of a heading alone or of a
few words. Each chunk becomes one record whose supervised answer is the chunk
word for word and whose request is written by code from what the text says of
itself: `Write, as NAME, to RECIPIENT in YEAR about: <its opening>`. Half of
the records open with the persona system prompt and half with the line that
says only who the model is, so the voice is carried by the adapter and not
only by the prompt. Two prints of one text are one family and only one is
projected, and no family supplies more than a tenth of the tokens
(`--max-family-share` on `dataset build`; `--writer NAME` names the writer and
`--token-limit N` keeps at most N tokens, spread evenly over all the chunks).
`--describe-voice` (`dataset build --describe-with REF`) asks each chunk by what the
generator says it is about instead of by its opening: 20 to 60 words in the model's own
words, admitted only when no run of eight words, and no more than 35% of its words in runs of
four, are the passage's own; a chunk with no admitted description keeps the opening request.
Descriptions are kept per generator under the state's `work/describe`, so a second build
reuses them.
A record names the part it prints and no experience, so a dataset shows what is
the writer's and what a model wrote. A run with a persona trains on the
writer's text by default: `--voice SHARE` is the share of the training tokens it
makes (seven tenths; `--voice 0` turns it off), the chunks chosen as an even
spread over the whole of the sources in a stable order that does not follow
their names, and the training is sized by all its examples. Such a record is
held out with the family of the letter it prints: the families held out are
decided by the records that can be examined - those projected from a task or an
experience - and a record of the writer's text goes where its family goes,
however many there are and wherever they sit, so the exam never asks about a
letter the policy was trained on under another print's name. The held-out score
is measured on the examinable held-out records, what the policy is asked as; the
writer's text of the held-out families is written to a file of its own
(`held_out_text.jsonl`), trained on by nobody, and the writer's text of the
monitoring families is monitored with the dialogues, where most of a run's
tokens are.

## Rehearsal

A run with a persona also rehearses the base (the `rehearse` stage). A
fine-tune on one person's answers erodes the base's general behaviour before
it forgets a fact: under the plain assistant prompt it answers a sum at length
and gets it wrong, or answers in a paragraph where a word was asked, which is
what the release gate's anchor suite catches. What holds a model to its base
is training beside the new records on what the base itself answers, with no
corpus from outside: the base model - the policy's base checkpoint with no
adapter, under the default assistant prompt, decoding greedily as the gate
asks it - answers a set of general tasks, and its answers are rehearsed as it
gave them. Half the tasks are `arithmetic` and `format` tasks built by code
from a seed (fresh numbers and subjects, a sum asked worked through or for the
number alone, a list asked one item per line), each with its reference, so an
answer the base got wrong is graded wrong and never rehearsed; the rest are
general requests the base writes itself, one domain at a time over sixteen
everyday domains, graded by nothing and kept as given. No anchor task, nor a
near copy of one (the same text in other case and spacing, a close rewording,
its words inside a longer prompt), is ever in the set: every anchor
instruction is held against each task by the generator's own near-duplicate
rule before it is stored, since a candidate rehearsed on an anchor task would
pass the anchor check by having trained on it; the gate's leakage check reads
the rehearsed records too. `--rehearsal SHARE` is the share of the training
draws the rehearsed records take (a quarter by default for a run with a
persona; `--rehearsal 0` turns it off, and a run without a persona rehearses
nothing unless asked), the set sized so that a rehearsed record is drawn about
as often as a dialogue answer; a tenth of its records (the monitoring share)
join the monitoring set instead, so the step the run carries is the one that
keeps the base's answers as well as it fits the new ones. `rehearse --records
N` builds such a set by hand and `train --rehearsal DATASET-ID
[--rehearsal-share F]` mixes it, or any chat dataset of the base's own
answers, into a training run.

## Bounded generation and held-out groups

Task generation is bounded by the budget and spread over the sources: each
text part is shown through at most four evenly spaced windows of sections,
parts are visited in a stable order that does not follow their names, and the
budget stops generation inside a part. A held-out split never divides texts
that print the same passage (two editions of a letter, a letter one edition
prints inside its neighbour, a passage reused deep inside a longer text):
records name the group of overlapping source text they came from - texts that
share eight distinct eight-word runs anywhere in either, a run held by more
than six texts counting for nothing - and a group is held out or trained on
whole. Records about one group are one unit of evidence to the
gate's and the exam's paired tests, and a tenth of a set generated letter by
letter is two or three of them, so whole groups are held out, newest first,
until there are eight, as long as that stays within a quarter of the records.
A group is named by the least content digest of the texts in it, found over
every text part of the sources, so two datasets built from the same sources
name a family the same and are split together as one.

`--distill` skips the policy's own attempts: the teacher answers every task
open-book and the policy is trained on its verified answers, with no
frontier and no critique. It is for a policy that cannot answer a task
closed-book at all, where its attempts are the most expensive part of a run
and teach nothing. A plan can choose it too.
