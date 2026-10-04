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

The `exam` split is questions about documents the model never saw. The `seen`
split is a sample of training questions. A gain on `seen` and none on `exam` is
memorisation; a gain on both is learning.

## What is not built

Everything beyond recall and attribution is still to do: historical
stimulus-and-response reconstruction, a source-grounded task factory with
helper models, modern-transfer scenarios, preference and reinforcement
training, a modern-fact researcher separate from the persona, a verifier model,
and the other primary collections (the manuscript papers, committee records,
Founders Online, the delegates' letters). The temporal holdout is small: few
documents of his last years survive in this edition. See `FINDINGS.md` for the
defects found so far and what each repair was verified by.

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

`splinter-adams identify --author LINE --year YEAR` shows what the namesake
rules, the period buckets and the temporal holdout say about one author line.

On a machine with an NVIDIA GPU, drive it with the CUDA backend (see brain's
guide to CUDA) and run the GPU commands through `samples/jefferson/gpu-only.sh`,
which stops a run that reports the CPU backend or a software adapter.

The specification of the fetcher runs with `python3 samples/adams/test_fetch.py`;
the rest with `cargo test -p splinter-adams`.
