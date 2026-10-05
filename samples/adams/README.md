<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!--
Swedish Embedded AB implements source-grounded persona and expertise transfer
for language models for its clients. If your team needs expertise in teaching
a model how a historical figure worked and proving that it did, you can
procure our services by sending an email to info@swedishembedded.com.
-->

# adams - can a small model learn how Samuel Adams worked from his own papers?

Samuel Adams (1722-1803) wrote for committees, town meetings, newspapers and
Congress, under his own name and under pseudonyms. A model taught "what Adams
said" from a flat pile of text learns a composite: a committee's resolution
becomes his private conviction, a newspaper piece the editor ascribes to him
becomes his signature. This sample builds the corpus so that cannot happen, and
measures whether a LoRA adapter on a small student learns his documents without
inventing them.

The corpus is primary documents. Biographies are context, never evidence of
what he believed.

## Splinter alone, from one sentence

The same corpus is what Splinter learns from on its own, the way it learns
Jefferson: this sample supplies the materials, an anchor suite and an
independent check, and no logic Splinter lacks.

```bash
# 1. The texts (checksummed), the curated documents, the leak-proof split, and
#    the directory Splinter learns from: his own documents of the training
#    split, cleaned, with every held-out family left out whole.
python3 samples/adams/fetch.py --resources RESOURCES
splinter-adams corpus    --resources RESOURCES
splinter-adams freeze    --resources RESOURCES
splinter-adams tasks     --resources RESOURCES                  # the independent check's frozen questions
splinter-adams materials --resources RESOURCES --out ./materials

# 2. Freeze the suite the release gate holds the adapter to: general knowledge,
#    and the arithmetic and format-following a persona fine-tune erodes first.
splinter eval --suite anchor --freeze samples/jefferson/anchor.jsonl --freeze samples/adams/anchor-skills.jsonl

# 3. Splinter does the rest, under the roles the configuration names.
SPLINTER_ASSISTANT_MODEL=local:Qwen/Qwen3-8B SPLINTER_BF16_BASE=1 SPLINTER_BUDGET=8h \
  splinter "Learn to think like Samuel Adams based on the materials he has written in directory ./materials"

# 4. The independent check: the frozen questions, asked of the base and of the
#    adapter Splinter released, under the prompt Splinter trained it under.
splinter-adams exam --tasks RESOURCES/tasks/exam.jsonl --out RUN/base.jsonl  --base BASE --persona "Samuel Adams"
splinter-adams exam --tasks RESOURCES/tasks/exam.jsonl --out RUN/tuned.jsonl --base BASE --adapter ADAPTER --persona "Samuel Adams"
splinter-adams report --before RUN/base.jsonl --after RUN/tuned.jsonl
```

`materials` writes one file per document that may be taught as his voice
(`own/` for a text in his hand or signed by him, `committee/` for a text a town
or committee he sat on adopted), each opening like a letter Splinter already
learns from - who it was written to, when - with the editors' footnotes, source
notes and running heads taken out. A text an editor ascribes to him or a
newspaper piece under a pseudonym is not written, however likely the
ascription: taught as his voice it is a composite. Every document of an exam or
temporal family is left out whole, so the independent check asks about letters
Splinter never saw; Splinter makes its own held-out split from what it is
given, by group of overlapping text, so its exam and the independent one are
different documents. The OCR's split words (`Gover nor`) are the scan's and are
left as they are: the stored text is what the split and its digests rest on.

## What is built

| Stage | What it does |
|---|---|
| `fetch.py` | Fetches the raw texts politely (robots.txt, spacing, an allowlist of hosts, a size cap) and records the url, licence, time and hash of every one in `MANIFEST.tsv`. |
| `corpus` | Parses Cushing's four volumes into documents: heading, the editor's source note, date, text. States how sure the edition's own note leaves us that Adams wrote each one (`DRAFT_IN_HAND`, `SIGNED_SCRIBAL`, `COMMITTEE_COAUTHORED`, `PSEUDONYMOUS_ATTRIBUTED`, `EDITOR_ATTRIBUTED`, ...). Excludes later writers and namesakes by author line and date. Lists every source note it could not turn into a document, with the reason. |
| `freeze` | Splits the documents once, before any training, by family: two documents that share a passage are one text and are held out or trained on together. Every document dated 1790 or later is kept for a temporal test. Writes what was held out with a hash of each text and refuses to replace a freeze that says something else. |
| `materials` | Writes the directory Splinter is pointed at: his own documents of the training split, one file each, cleaned (see above). Refuses a split that leaks. |
| `tasks` | Builds questions by code from the documents' own fields: who a letter was written to, what year, what kind of document it is, how a passage goes on. A question is not asked when its opening already contains its answer. The exam is pinned in a `FROZEN.json` beside it the first time it is written; a different exam under that name is refused, and `exam` checks the file against the pin before any arm is scored. |
| `train` | Fine-tunes a LoRA adapter on the training questions through Splinter's trainer. |
| `exam` | Asks one model, with or without the adapter, every question not yet answered; grades each answer by code; resumes where it stopped. |
| `report` | Pairs two result files on the questions both answered, with the paired sign test. |
| `principles` | A helper model reads a few of his own letters at a time and proposes how he worked, as rules that say when they apply, what he did and what limits them, each with the document and the words, copied exactly. Code looks every quotation up: one not in the cited document is dropped with the reason kept, and a rule none of whose quotations is found is rejected. A rule is `recurring` only when it is found in three documents across periods or audiences. Texts a committee adopted or an editor ascribes to him are never support. |
| `transfer` | From each rule, a helper designs a present-day situation as one of four cases (a clear fit, a weak fit, a missing precondition, a surface analogy), gated in code: modern, no wording from the source, conditions that match the case. A helper then answers it, and the answer is kept only if its first line says whether his method applies and that agrees with the case, a non-fit that needs information asks for it, and its grounding block keeps its word. |
| `build-data` | Builds supervised records (with his passages in the prompt, and without), preference pairs and a frozen benchmark from the kept answers. A rule is wholly training or wholly benchmark. A preference pair is an answer and the same answer broken in one named way that the check then refuses, so each pair is right by construction. |
| `train-dpo` | Preference-optimises an adapter on the pairs, continuing from the supervised adapter, which then is the reference. |
| `transfer-exam` | Asks a model every benchmark question, graded by the grounding rules with no judge. |
| `--replay-share S`, `--grad-accum N` | On `train`: the share of draws that are replay (a plain union lets a large replay set take most steps), and examples per optimizer step (default 8). `train-dpo` takes `--grad-accum`, `--lr` (default far below the supervised rate) and `--nll-weight`, an anchor that keeps the chosen answer's likelihood from falling. None of these defaults is tuned. |
| `transfer --scenarios-per-principle N` | Makes N scenarios per principle, not one. The first is as before; each later one is set in a field code assigns from a fixed list (and checks the draft uses), and the second is of the opposite kind to the first, so a principle is seen both holding and not. Some fields are held out of training: `build-data` writes scenarios in those fields, and further scenarios of benchmark principles, to `benchmark-extra.jsonl` (slices `new` and `ood` in the report), never into the frozen benchmark. |
| `pin --file F` | Freezes a file (the benchmark, the briefings): its content is pinned in `FROZEN.json` beside it, `build-data` refuses to write other content under a frozen benchmark name (use `--benchmark-file benchmark-v2.jsonl`), and `transfer-exam` and `reconstruct-replies` refuse a file that changed since. |
| `build-onpolicy --samples F` | Builds the student's own training data from its samples on the training questions (`transfer-exam --benchmark datasets/train-questions.jsonl --samples K --decoding sample:T --adapter A`): its passing answers as `sft-rft.jsonl` and each prompt's passing and failing answers, of nearly the same length and breaking different rules, as `preference-onpolicy.jsonl`. |
| `--samples K` | On `transfer-exam`: ask every question K times (with `--decoding sample:T`) under ids `question@n`; the report then takes each question's share right and prints a paired bootstrap interval of the change per kind of question, which is what a result on a few dozen questions has to be read by. |
| `constant-baseline` | Grades a fixed template (always applies, quotes the first words of the first passage) on the benchmark with no model: the score a trained model has to beat. |
| `--decoding greedy\|thinking\|sample[:T]\|sample-thinking[:T]`, `--few-shot N` | On `transfer-exam` and `reconstruct-replies`: let a reasoning model think or sample, and show it N training examples in its system message, so a base arm is not handicapped into a weak baseline. |
| `--framing full\|identity\|plain` | On `transfer-exam` and `reconstruct-replies`: the system message the exam asks under. Training mixes all three (mostly the full instructions, some the bare identity, some no persona; the shares are constants in `persona.rs`), so an adapter can be asked without any persona prompt and the habit shown to be its own. |
| `briefings`, `reconstruct-replies`, `judge`, `reconstruct-report` | The reconstruction benchmark: a held-out letter is briefed as the situation it answered, a model writes the reply, and a judge calibrated on controls scores it against what the real letter does. |
| `grpo` | Builds a task family for brain's reinforcement-learning loop: present-day situations and the passages he may be shown, rewarded in named parts by the same rules (the layout, the verdict on whether his method applies, every other rule), with no model judging. The pool the loop trains on and the pool its own gate draws from are disjoint. It runs, but a two-step trial over the 1.5B model took 27 minutes, so no result is claimed for the 7B (the knowledge notes on measuring an RL loop's cost say why). |
| `anchor-exam` | Asks a model a frozen set of general questions under no persona, before and after training: a retention check. `anchor-skills.jsonl` adds what a persona fine-tune of a reasoning model can erode and trivia cannot show: arithmetic word problems graded by the last number stated and format-following graded by line count; it is in the format Splinter's own anchor suite freezes, so `splinter eval --suite anchor --freeze` takes it too. `anchor-leak.jsonl` holds the probes that must not carry any of his persona; only this sample asks them, under no persona (`--max-tokens`, `--decoding thinking` for the arithmetic). |

The `exam` split is questions about documents the model never saw. The `seen`
split is a sample of training questions. A gain on `seen` and none on `exam` is
memorisation; a gain on both is learning.

## The three ways the model is asked

The system message is the mode, and it is part of what the adapter is trained and tested under:

| Mode | System message | What the model is shown |
|---|---|---|
| Historical reconstruction | `reconstruct::SYSTEM` | A briefing of a situation he faced; it writes the reply he would send. |
| Source-grounded application | `respond::SYSTEM` | A present-day situation, the facts, and passages from his papers with their document ids; it must quote only those. |
| Internalized transfer | `respond::SYSTEM` | The same situation and facts with no passages: what it has learned. A quotation here is a fabrication by definition. |

## Decisions

- The student is asked with its reasoning off: the prompt ends in an empty, closed think block, and every training answer starts with the same block, so what is trained is what is asked (a mismatch here made runs a1 and a2t uninterpretable). A reasoning-on base is one of the baselines.
- Retrieval mode is how the model is meant to be used; internalized mode is a stress test of what a small adapter can hold. Teaching it his letters word for word is not attempted.
- Frozen files are never rewritten: a changed benchmark goes under a new name.
- A gain counts only above the fixed template, the base with thinking on and with worked examples, and with an interval over the questions that excludes zero.

## What is not built

Reinforcement training (GRPO) has no result and is replaced by rejection-sampling
fine-tuning and on-policy preference pairs from the same verifier: the task family
and reward are built and the loop runs, but too slowly here to teach the student
anything within a session. Also still to do: a modern-fact researcher separate from the persona,
a verifier model, a judge for anything the rules cannot grade beyond the
reconstruction benchmark's, and the other primary collections (the manuscript
papers, committee records, Founders Online, the delegates' letters). The
temporal holdout is small: few documents of his last years survive in this
edition. The defects found on the way, and what proved each, are kept as generic
knowledge notes for the whole repository, not for this sample.

## Running it

`RESOURCES` is a directory with room for the texts and what is derived from
them; `BASE` is a Hugging Face checkpoint directory for the student.

```bash
python3 samples/adams/fetch.py --resources RESOURCES            # add --dry-run to see the plan
splinter-adams corpus --resources RESOURCES
splinter-adams freeze --resources RESOURCES                     # once; a second, different freeze is refused
splinter-adams tasks  --resources RESOURCES
splinter-adams train  --resources RESOURCES --base BASE --attempt RUN/attempt --steps 500 --rank 16 --bf16
splinter-adams exam   --tasks RESOURCES/tasks/exam.jsonl --out RUN/base-exam.jsonl  --base BASE
splinter-adams exam   --tasks RESOURCES/tasks/exam.jsonl --out RUN/tuned-exam.jsonl --base BASE --adapter RUN/attempt/adapter.safetensors
splinter-adams report --before RUN/base-exam.jsonl --after RUN/tuned-exam.jsonl
```

`exam` asks under the sample's own persona unless `--persona NAME` (the prompt
Splinter trains a policy to be NAME under, for an adapter a `learn` produced)
or `--system TEXT` says otherwise.

The helper stages need a model served by `brain serve` and its key in
`BRAIN_API_KEY`; they resume where they stopped and record, rather than hide, a
bundle or a principle the helper failed on. The order that has the fewest
surprises:

```bash
splinter-adams principles --resources RESOURCES --model HELPER [--url http://127.0.0.1:8788/v1] [--limit N]
splinter-adams transfer   --resources RESOURCES --model HELPER --scenarios-per-principle 3 [--limit N]
splinter-adams briefings  --resources RESOURCES --model HELPER --of exam      # then --of train
splinter-adams build-data --resources RESOURCES
splinter-adams pin --file RESOURCES/datasets/benchmark.jsonl                   # once, before any arm is scored
splinter-adams constant-baseline --resources RESOURCES --out RUN/constant.jsonl
```

Train on the transfer records and, when there are briefings of his training
letters, on his real letters as a replay at a fixed share; do not replay the
recall questions, which taught run a1 to answer every question the same way:

```bash
splinter-adams train --resources RESOURCES --base BASE --attempt RUN/sft \
    --dataset RESOURCES/datasets/sft-transfer.jsonl \
    --replay RESOURCES/datasets/sft-reconstruction.jsonl --replay-share 0.25 \
    --steps 100 --rank 16 --bf16 --grad-accum 8
splinter-adams transfer-exam --resources RESOURCES --base BASE --out RUN/base.jsonl
splinter-adams transfer-exam --resources RESOURCES --base BASE --out RUN/base-thinking.jsonl --decoding thinking --max-tokens 3072
splinter-adams transfer-exam --resources RESOURCES --base BASE --out RUN/base-fewshot.jsonl --few-shot 2
splinter-adams transfer-exam --resources RESOURCES --base BASE --out RUN/tuned.jsonl --adapter RUN/sft/adapter.safetensors
splinter-adams transfer-exam --resources RESOURCES --base BASE --out RUN/tuned-plain.jsonl --adapter RUN/sft/adapter.safetensors --framing plain
splinter-adams report --before RUN/base.jsonl --after RUN/tuned.jsonl
```

Then the student's own answers, which are the data that matches what it says:

```bash
splinter-adams transfer-exam --resources RESOURCES --base BASE --adapter RUN/sft/adapter.safetensors \
    --benchmark RESOURCES/datasets/train-questions.jsonl --samples 4 --decoding sample:0.7 --out RUN/samples.jsonl
splinter-adams build-onpolicy --resources RESOURCES --samples RUN/samples.jsonl
splinter-adams train --resources RESOURCES --base BASE --attempt RUN/rft \
    --dataset RESOURCES/datasets/sft-rft.jsonl --continue-from RUN/sft/adapter.safetensors --steps 50 --bf16 --grad-accum 8
splinter-adams train-dpo --resources RESOURCES --base BASE --attempt RUN/dpo \
    --pairs RESOURCES/datasets/preference-onpolicy.jsonl --continue-from RUN/rft/adapter.safetensors --steps 100
splinter-adams anchor-exam --anchor samples/jefferson/anchor.jsonl --out RUN/anchor-base.jsonl --base BASE
splinter-adams anchor-exam --anchor samples/adams/anchor-skills.jsonl --out RUN/skills-base.jsonl --base BASE --decoding thinking --max-tokens 1500
splinter-adams anchor-exam --anchor samples/adams/anchor-skills.jsonl --out RUN/skills-tuned.jsonl --base BASE --adapter RUN/dpo/adapter.safetensors --decoding thinking --max-tokens 1500
splinter-adams report --before RUN/skills-base.jsonl --after RUN/skills-tuned.jsonl
splinter-adams anchor-exam --anchor samples/adams/anchor-leak.jsonl --out RUN/leak-tuned.jsonl --base BASE --adapter RUN/dpo/adapter.safetensors
```

`splinter-adams identify --author LINE --year YEAR` shows what the namesake
rules, the period buckets and the temporal holdout say about one author line.

On a machine with an NVIDIA GPU, drive it with the CUDA backend (see brain's
guide to CUDA) and run the GPU commands through `samples/jefferson/gpu-only.sh`,
which stops a run that reports the CPU backend or a software adapter.

The specification of the fetcher runs with `python3 samples/adams/test_fetch.py`;
the rest with `cargo test -p splinter-adams`.
