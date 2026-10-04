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

## What is built

| Stage | What it does |
|---|---|
| `fetch.py` | Fetches the raw texts politely (robots.txt, spacing, an allowlist of hosts, a size cap) and records the url, licence, time and hash of every one in `MANIFEST.tsv`. |
| `corpus` | Parses Cushing's four volumes into documents: heading, the editor's source note, date, text. States how sure the edition's own note leaves us that Adams wrote each one (`DRAFT_IN_HAND`, `SIGNED_SCRIBAL`, `COMMITTEE_COAUTHORED`, `PSEUDONYMOUS_ATTRIBUTED`, `EDITOR_ATTRIBUTED`, ...). Excludes later writers and namesakes by author line and date. Lists every source note it could not turn into a document, with the reason. |
| `freeze` | Splits the documents once, before any training, by family: two documents that share a passage are one text and are held out or trained on together. Every document dated 1790 or later is kept for a temporal test. Writes what was held out with a hash of each text and refuses to replace a freeze that says something else. |
| `tasks` | Builds questions by code from the documents' own fields: who a letter was written to, what year, what kind of document it is, how a passage goes on. A question is not asked when its opening already contains its answer. |
| `train` | Fine-tunes a LoRA adapter on the training questions through Splinter's trainer. |
| `exam` | Asks one model, with or without the adapter, every question not yet answered; grades each answer by code; resumes where it stopped. |
| `report` | Pairs two result files on the questions both answered, with the paired sign test. |
| `principles` | A helper model reads a few of his own letters at a time and proposes how he worked, as rules that say when they apply, what he did and what limits them, each with the document and the words, copied exactly. Code looks every quotation up: one not in the cited document is dropped with the reason kept, and a rule none of whose quotations is found is rejected. A rule is `recurring` only when it is found in three documents across periods or audiences. Texts a committee adopted or an editor ascribes to him are never support. |
| `transfer` | From each rule, a helper designs a present-day situation as one of four cases (a clear fit, a weak fit, a missing precondition, a surface analogy), gated in code: modern, no wording from the source, conditions that match the case. A helper then answers it, and the answer is kept only if its first line says whether his method applies and that agrees with the case, a non-fit that needs information asks for it, and its grounding block keeps its word. |
| `build-data` | Builds supervised records (with his passages in the prompt, and without), preference pairs and a frozen benchmark from the kept answers. A rule is wholly training or wholly benchmark. A preference pair is an answer and the same answer broken in one named way that the check then refuses, so each pair is right by construction. |
| `train-dpo` | Preference-optimises an adapter on the pairs, continuing from the supervised adapter, which then is the reference. |
| `transfer-exam` | Asks a model every benchmark question, graded by the grounding rules with no judge. |
| `--framing full\|identity\|plain` | On `transfer-exam` and `reconstruct-replies`: the system message the exam asks under. Training mixes all three (mostly the full instructions, some the bare identity, some no persona; the shares are constants in `persona.rs`), so an adapter can be asked without any persona prompt and the habit shown to be its own. |
| `briefings`, `reconstruct-replies`, `judge`, `reconstruct-report` | The reconstruction benchmark: a held-out letter is briefed as the situation it answered, a model writes the reply, and a judge calibrated on controls scores it against what the real letter does. |
| `grpo` | Builds a task family for brain's reinforcement-learning loop: present-day situations and the passages he may be shown, rewarded in named parts by the same rules (the layout, the verdict on whether his method applies, every other rule), with no model judging. The pool the loop trains on and the pool its own gate draws from are disjoint. It runs, but a two-step trial over the 1.5B model took 27 minutes, so no result is claimed for the 7B (see `FINDINGS.md`). |
| `anchor-exam` | Asks a model a frozen set of general questions under no persona, before and after training: a retention check. |

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

## What is not built

Reinforcement training has no result: the task family and reward are built and
the loop runs, but too slowly here to teach the student anything within a
session. Also still to do: a modern-fact researcher separate from the persona,
a verifier model, a judge for anything the rules cannot grade beyond the
reconstruction benchmark's, and the other primary collections (the manuscript
papers, committee records, Founders Online, the delegates' letters). The
temporal holdout is small: few documents of his last years survive in this
edition. See `FINDINGS.md` for the defects found so far and what each repair
was verified by.

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

The helper stages need a model served by `brain serve` and its key in
`BRAIN_API_KEY`; they resume where they stopped and record, rather than hide, a
bundle or a principle the helper failed on:

```bash
splinter-adams principles --resources RESOURCES --model HELPER [--url http://127.0.0.1:8788/v1] [--limit N]
splinter-adams transfer   --resources RESOURCES --model HELPER [--limit N]
splinter-adams build-data --resources RESOURCES
splinter-adams train      --resources RESOURCES --base BASE --attempt RUN/sft2 \
    --dataset RESOURCES/datasets/sft-transfer.jsonl --replay RESOURCES/tasks/sft.jsonl --steps 500 --rank 16 --bf16
splinter-adams train-dpo  --resources RESOURCES --base BASE --attempt RUN/dpo \
    --continue-from RUN/sft2/adapter.safetensors --steps 200
splinter-adams transfer-exam --resources RESOURCES --out RUN/transfer-base.jsonl --base BASE
splinter-adams transfer-exam --resources RESOURCES --out RUN/transfer-tuned.jsonl --base BASE --adapter RUN/dpo/adapter.safetensors
splinter-adams transfer-exam --resources RESOURCES --out RUN/transfer-tuned-plain.jsonl --base BASE --adapter RUN/dpo/adapter.safetensors --framing plain
splinter-adams anchor-exam --anchor samples/jefferson/anchor.jsonl --out RUN/anchor-base.jsonl --base BASE
splinter-adams anchor-exam --anchor samples/jefferson/anchor.jsonl --out RUN/anchor-tuned.jsonl --base BASE --adapter RUN/dpo/adapter.safetensors
splinter-adams report --before RUN/anchor-base.jsonl --after RUN/anchor-tuned.jsonl
```

`splinter-adams identify --author LINE --year YEAR` shows what the namesake
rules, the period buckets and the temporal holdout say about one author line.

On a machine with an NVIDIA GPU, drive it with the CUDA backend (see brain's
guide to CUDA) and run the GPU commands through `samples/jefferson/gpu-only.sh`,
which stops a run that reports the CPU backend or a software adapter.

The specification of the fetcher runs with `python3 samples/adams/test_fetch.py`;
the rest with `cargo test -p splinter-adams`.
