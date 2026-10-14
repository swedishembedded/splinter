<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# persona-training paper: research contract, evidence ledger, review

Supporting record for `papers/persona-training/` (the paper itself carries none
of this). Status at the end: **REVIEW-READY** (see Readiness).

## 1. Research contract

| Field | Content |
|---|---|
| Research question | Does supervised fine-tuning on decisions read out of one author's essays make a model take that author's decision, through a tool call, on cases whose topic, fact/option combination or tool schema the training data lacks, more often than the same model given the author's name, beliefs or text in the prompt? |
| Problem and significance | With real authors the question cannot be graded: pretraining contamination and no answer key. A code oracle with hidden beliefs removes both. |
| Thesis (one sentence) | Under this controlled setting, a LoRA adapter on Qwen3-8B trained on decisions read out of a synthetic author's essays takes the oracle's decision far more often than the base model told the author's name or given the author's beliefs as stated principles, on withheld topics, combinations and tools alike. |
| Contribution type | empirical (with a benchmark/method artifact) |
| Prior best explanation / strongest alternative | (a) Retrieval of the author's own text at answer time (tested; adapter ahead for 2 of 3 authors). (b) The adapter learns a lookup from fact patterns to options, helped by training records that restate facts in the exam's own sentences; not separated by any ablation (stated in Discussion). |
| Novelty delta | Persona/role-play work (Character-LLM, RoleLLM, OpinionQA, out-of-one-many, Twin-2K-500) grades against retained knowledge or survey answers; Treutlein et al. test verbalisation of latent structure, not decisions through tools; Inverse Constitutional AI extracts principles from preferences; Ovadia et al. compare fine-tuning and RAG for factual knowledge. Delta: a code-oracle benchmark for an author's decisions with hold-out of topics, combinations and tool schemas, mechanically graded, and the paired comparison of fine-tuning against prompting with the author's stated beliefs and against retrieval. No "first" claim is made in the paper. |
| Mechanism (conjecture only) | Decisions made explicit as supervised tool calls across three schemas. |
| Independent variables | arm (base, named, principles, retrieval, tuned); author (seed 101, 202, 303); cell (topic/combination/tool seen vs new) |
| Primary endpoint | Items right of 160 per arm and author. |
| Primary comparisons | tuned vs named and tuned vs principles (strongest prompt-only arms); tuned vs retrieval (post hoc arm). Exact one-sided paired sign test. |
| Prespecified by construction | Exam items, cells and oracle answers are a deterministic function of the seed; arms, system turns, decoding and grading are fixed in code; first recorded answer stands. |
| Post hoc | The retrieval arm (added after the first three arms were run); the written success/failure rule; all secondary analyses (cells, errors, divergence, audit, echo). |
| Success rule (written post hoc, stated in paper Sec. 3.7) | Per author: tuned beats named and principles at p<0.05 after Bonferroni x18, and the lowest tuned cell exceeds every cell of base/named/principles. Fail otherwise. |
| Failure tests | Tuned not above named/principles in the paired test; tuned withheld-cell accuracy falling to prompted-arm level. |
| Scope | Three synthetic authors in one fixed world, one 8B policy, one training run per author, model-written corpora. |
| Known constraints | No new GPU/model runs in this revision; corpus not in repository; writer = reader = one hosted model. |

## 2. Claim-evidence ledger

| ID | Claim (paper section) | Evidence / artifact | Status |
|---|---|---|---|
| C1 | Tuned accuracy 138/125/126 of 160; base 69/62/48 (6.1) | `data/arms.csv`, from `scratchpad/exams/author-*/exam/*.jsonl` via `collect.py`; matches `report.md` | VERIFIED |
| C2 | Tuned > base, named, principles, p<1e-10 each author (6.2, abstract) | `data/paired.csv`; `build_data.py` asserts p<1e-10 | VERIFIED |
| C3 | Tuned vs retrieval: 101 n.s. (23-14, p=0.094); 202, 303 significant (6.2) | `data/paired.csv`; asserted against recorded values | VERIFIED |
| C4 | Thesis rule holds for each author (7) | Bonferroni check and min-cell check in `build_data.py` (fails build otherwise) | VERIFIED |
| C5 | Only the tool axis of 202 has a gap interval excluding zero (6.3) | `data/cells.csv`; asserted in `build_data.py` | VERIFIED |
| C6 | Divergence: own decision most frequent in all nine comparisons (6.5) | `data/cross.csv` = `cross.md`; asserted | VERIFIED |
| C7 | Audit: 153/157, 151/157, 102/102 kept decisions equal the oracle (6.6) | Re-run in this revision: `synthetic-author audit --articles 60` and `scripts/audit.py` both give these counts. `paperdata/audit.txt` (144/142/96) is stale and wrong; not used. | VERIFIED |
| C8 | Adapter error pattern: 21/22 (101), 21/35 (202) accept-offer-for-act-alone; 13/34 (303) act-alone-for-assembly (6.4) | NEW `data/confusion.csv` from NEW `scripts/confusion.py` over `items-S.jsonl` (`synthetic-author items`) and `exam/tuned.jsonl`; script cross-checks its grade with the recorded `correct` flag; `build_data.py` checks totals and right counts | VERIFIED |
| C9 | 8/56, 7/56, 7/36 kept essays echo 6 words of a private note in first person (6.7) | NEW `data/echo.csv` from NEW `scripts/echo.py` (notes from `synthetic-author show`); `build_data.py` checks essay counts | VERIFIED |
| C10 | Walk-through listings are verbatim data | Scratch verifier matched every listing line and quotation against the bundle files, `plan` output and code; the only non-matches are lines rendered from code (tool definition of `settle_matter`, principles/named system turns, checked separately) and the label "Title:". `show` and `items` outputs regenerated and byte-identical to the bundle. | VERIFIED |
| C11 | Solbrook records were trained on, not held out (5.5) | `train/train.jsonl` has group `0:the Solbrook granary` x6; held-out groups are articles 53-59 | VERIFIED |
| C12 | Held-out tenth is whole cases from the end of the record file (4.3) | `crates/data/src/holdout.rs` (`split_by_rule`, groups taken from the end) and the observed held-out groups | VERIFIED |
| C13 | Item t0c0-001 answers and grades (5.6) | `exam-101-selected.json` equals raw `exam/*.jsonl` lines | VERIFIED |
| C14 | Retrieval arm was added after the first three arms were run (3.6) | Statement provided by the study's operator; no artifact in the bundle records it | UNVERIFIED by artifact; human must confirm |
| C15 | Training hyperparameters, step-240 loss, elapsed times (4.3) | `data/training.csv`; `train/training.json` of author 101 | VERIFIED (101); 202/303 from logs as before |
| C16 | Cost figures (6.8) | `data/cost.csv`, `data/spend.csv` (unchanged from previous revision) | carried over, not re-verified |
| C17 | All citations' attributed claims | Citation subagent: every key opened at a primary source or registry; no claim PARTLY or NOT SUPPORTED; metadata corrections applied | VERIFIED (see section 5) |

## 3. Adversarial review finding log

Severity | issue | affected claim/section | objective evidence | remediation | disposition

### Reviewer A: novelty and contribution

| Sev | Issue | Claim | Evidence | Remediation | Disposition |
|---|---|---|---|---|---|
| MAJOR | No single stated thesis or falsification criterion; contributions list of four diluted it | Intro | previous intro | Added "Question and thesis" paragraph with what would contradict it; contributions cut to two | FIXED |
| MAJOR | Adapter-vs-retrieval presented alongside the thesis although it fails for one author | Abstract, intro | paired.csv 101 | Thesis restricted to prompt-only arms; retrieval reported per author as outside the thesis | FIXED |
| MINOR | Novelty statements risk "first" claims | Related | text | No "first/novel" wording; delta stated by contrast | FIXED |
| MINOR | Twin-2K "over 500 questions" follows title; abstract says 500 | Related | citation check | Kept (matches title) | ACCEPTED |

### Reviewer B: validity and evidence

| Sev | Issue | Claim | Evidence | Remediation | Disposition |
|---|---|---|---|---|---|
| BLOCKER | "The beliefs of our authors are never stated in the corpus" is contradicted by the data: kept essays echo the private notes in first person ("I act at once to protect them, and I account for it afterwards") | Related, intro framing | retrieval passages of item t0c0-001; `echo.py` counts 8/7/7 | Claim removed; measured and reported (Sec. 6.7), added to scope of validity | FIXED |
| MAJOR | Brief-copy gate is evaded by a pronoun change ("I settled it myself, at once, without waiting") | Gates (3.3) | appendix retrieval listing (4 occurrences in one turn) | Reported in 6.7 | FIXED (reported) |
| MAJOR | Primary endpoint, comparisons and pre/post hoc status not stated | 3.7 | previous text | New subsection; retrieval arm labelled post hoc; decision rule labelled as written after results | FIXED |
| MAJOR | Strongest alternative (fact-pattern lookup; exam-wording match of fixed briefings) not discussed | Discussion | records use exam fact sentences | New paragraph; bounded by combination hold-out; no ablation, stated | FIXED (limitation stated) |
| MAJOR | No failure analysis of the adapter's errors | Results | confusion counts | New Sec. 6.4 and table; mechanism labelled conjecture | FIXED |
| MAJOR | Bundle summary `audit.txt` (144/142/96) disagrees with paper (153/151/102) | 6.6 | re-ran both audit implementations | Paper value confirmed by re-run; summary file stale | FIXED (no paper change) |
| MINOR | Example item could be read as typical | 5.6 | adapter wrong on 22 items | Caption/text says chosen for withheld axes, not typical; points to errors table | FIXED |
| MINOR | Held-out tenth described as if random | 4.3 | holdout.rs | Described as whole cases from the end of the file | FIXED |
| MINOR | Author 202 tuned file has 241 lines for 160 items (an interrupted, resumed run) | data provenance | the 81 repeats are identical to the first answers (greedy decoding) | First answer used as before; no effect | NO CHANGE NEEDED |
| MINOR | One training run per author; three authors; writer = reader | Scope | design | Stated in abstract and Discussion | ACCEPTED (limitation) |

### Reviewer C: clarity and reproducibility

| Sev | Issue | Claim | Evidence | Remediation | Disposition |
|---|---|---|---|---|---|
| MAJOR | Reader cannot see what data passes through the pipeline | whole paper | owner requirement A | New Sec. 5 with chain table, verbatim listings of every stage, records table, item with five system turns and answers, rejected drafts; Appendix A with full listings | FIXED |
| MAJOR | Date in title block | front matter | `\date{October 2026}` | `\date{\today}`; no other dates; invented years shown as [year] | FIXED |
| MINOR | Reproducibility lacked `show`, `items`, `audit` and the new scripts | Sec. 8, README.txt | text | Added | FIXED |
| MINOR | Hardware mentioned twice | 4.3, 6.8 | text | One setup line kept (4.3) | FIXED |
| MINOR | Overfull boxes from new tables | build log | log | Column widths fixed; build has 0 overfull | FIXED |
| MINOR | Total length 24 pages (21 with references, 3 of appendix listings) exceeds the 16-22 target when the appendix is counted | length | page count | Appendix trimmed (dropped duplicated decisions and raw record listings, compacted JSON); main text plus references is 21 pages | ACCEPTED (appendix is supplementary) |

No BLOCKER or MAJOR finding remains open.

## 4. Integrity gates

| Gate | Result | Support |
|---|---|---|
| One coherent thesis, falsifiable | PASS | Intro "Question and thesis"; Sec. 3.7 rule |
| Novelty checked against closest/contradictory work | PASS (bounded) | Related work; no "first" claim; search limited to the cited works plus the subagent's check |
| Every major claim mapped to evidence; nothing fabricated | PASS except C14 (operator statement) | Ledger above |
| Abstract/intro/results/conclusion consistent | PASS | Abstract states the retrieval split and limits |
| Negative results, alternatives, scope reported | PASS | 6.2, 6.4, 6.7, Discussion |
| Association/causation/significance/conjecture distinguished | PASS | conjecture labelled in 6.4 |
| Strong baselines, fair comparison, mechanism ablation | PARTIAL / FAIL for ablation | Retrieval baseline present; no ablation of fixed-wording records or retrieval content (stated) |
| Endpoint, unit, n, repetitions, uncertainty, exclusions documented | PASS | 3.7; one run per author stated |
| Leakage, multiplicity, cherry-picking audited | PASS | No case shared with corpus; Bonferroni stated; example labelled non-typical; echo leak measured |
| Stats interpretation matches evidence; tables reproducible | PASS | `build_data.py` checks |
| Equations/assumptions/units | NOT APPLICABLE (no theory) | |
| Repro commands, configs, versions | PASS (commands); environment versions not listed | Sec. 8 |
| References verified against primary sources | PASS with noted limits | Section 5 |
| Licenses, privacy, consent | NOT APPLICABLE (no human subjects; synthetic data) | |
| Authorship, AI-use disclosure reviewed by human | NOT RUN (needs human) | "Use of language models" paragraph drafted |
| Venue/arXiv technical rules | NOT RUN | No venue chosen |
| LaTeX builds, no unresolved refs, PDF inspected | PASS | `make`: 0 undefined, 0 overfull; pages rendered and inspected at low resolution |
| No unresolved BLOCKER | PASS | Finding log |

## 5. Citation verification

All 31 keys checked by a subagent against arXiv, ACL Anthology, PMLR, NeurIPS/ICLR proceedings,
Crossref and publisher pages. Applied: published venues for hu2022lora (ICLR 2022),
santurkar2023whose (ICML 2023, PMLR 202), ovadia2023finetuning (EMNLP 2024),
allenzhu2023physics31 (ICML 2024, PMLR 235), christiano2017deep (NeurIPS 2017),
patil2023gorilla (NeurIPS 2024), qin2023toolllm (ICLR 2024), yang2024syntheticcpt (ICLR 2025),
schick2023toolformer (NeurIPS 2023, author Eric Hambro added), shanahan2023roleplay (Nature
623:493-498); argyle2023outofone volume/number/pages; newcombe1998interval number and DOI.
wang2023rolellm kept as the arXiv version because the Findings version lists the authors
differently. Not opened directly (confirmed via registries/snippets): Abbeel (ACM 403),
Robertson BM25 (Now Publishers 403), Nature page. No attributed claim was found unsupported.

## 6. Readiness

**REVIEW-READY.** All BLOCKER and MAJOR findings are resolved in the text, the build is clean
and every number is regenerated and cross-checked from `data/`. Not SUBMISSION-READY because:
(1) the human author has not yet confirmed the AI-use disclosure, authorship and the operator
statement that the retrieval arm was added after the first three arms (C14); (2) no venue's
rules have been checked; (3) the mechanism ablations a referee would ask for (records without
the fixed-wording briefings; retrieval variants) and additional training seeds were not run;
(4) cost figures were carried over without re-verification.

## 7. Human author must confirm

- The "Use of language models" paragraph (language models wrote/read the essays as method,
  and assisted with drafting and analysis scripts; the author checked numbers and citations
  and takes responsibility).
- That the retrieval arm was added after the first three arms had been run.
- Authorship and affiliation as printed.
