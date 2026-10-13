# Evidence for the size of an augmented corpus

Status: gathered by a research pass; figures were read through a summarizing
fetch and must be quote-checked against the papers before any is cited.
The recommendation at the end is inference, not a finding.

# Expanded-corpus size for a small single-author corpus (8B, LoRA): literature

Method: each paper's arXiv abstract page was fetched (existence, title, authors, year). Numbers come from the arXiv HTML full text via a summarizing fetch tool, not from my own reading of the PDFs. Treat them as verified-by-fetch but quote-check them before putting them in a paper. Anything marked UNVERIFIED was not confirmed.

## Sources and findings

1. Allen-Zhu & Li, "Physics of Language Models: Part 3.1, Knowledge Storage and Extraction", 2023 (v. Jul 2024), arXiv:2309.14316.
   - Extraction needs augmentation: "knowledge must be sufficiently augmented (paraphrase, sentence shuffling, translation)".
   - Setting: bioS, 100,000 synthetic individuals, 682M model, each person exposed about 1350 times.
   - Without augmentation, QA accuracy on held-out people is ~0 even though the biographies are memorized perfectly.
   - Five diverse biographies per person ("multi5+permute") raise QA fine-tune accuracy on held-out people from 9.7% to 96.6%.
   - Augmenting only a "celebrity" subset raised the non-augmented minority's accuracy from 10.0% to 76.3%.
   - Implication: multiplicity of diverse rewrites matters more than raw repetition; the full-text summary did not confirm that 5 is a saturation point.

2. Allen-Zhu & Li, "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws", 2024, arXiv:2404.05405.
   - A model stores at most about 2 bits of knowledge per parameter.
   - This is reached at about 1000 exposures per fact; at 100 exposures capacity drops to about 1 bit/param.
   - Capacity is not the constraint for us (8B gives about 16B bits, far more than 7M tokens carry), but exposure count is.
   - Prepending a domain/source tag to training text raises capacity.

3. Yang, Band, Li, Candès, Hashimoto, "Synthetic continued pretraining" (EntiGraph), 2024, arXiv:2409.07431.
   - Setting: 1.3M-token source corpus (265 QuALITY books), Llama 3 8B, EntiGraph corpus of 455M tokens, a ratio of about 350x.
   - Closed-book QA accuracy rose from 39.49% to 56.22%.
   - Accuracy scaled log-linearly in synthetic tokens up to 455M, the largest size tested, with no plateau seen.
   - RAG on top of the CPT model reached 62.60%.
   - Raw-corpus CPT gave no gain over the base model (the fetch summary called it negative transfer; the exact sign is not confirmed).
   - A plain-rephrase baseline plateaued early. The figure "38M tokens" came from the summarizer and is UNVERIFIED.
   - Closest analogue to our setup: it shows 100x-plus expansion is useful when the expansion is diverse (entity relations), not just paraphrase.

4. Ovadia, Brief, Mishaeli, Elisha, "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs", 2023 (v. Jan 2024), arXiv:2312.05934.
   - Ten paraphrases per chunk on the current-events set; at most 5 epochs.
   - Mistral 7B accuracy: base+RAG 0.875, fine-tuned 0.504, fine-tuned with paraphrases 0.588, RAG 0.830 in the fine-tuned comparison row.
   - Accuracy was monotonically increasing in the number of paraphrases, with no saturation reported.
   - RAG beat fine-tuning throughout. Implication: keep RAG alongside the fine-tuned model.

5. Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?", EMNLP 2024, arXiv:2405.05904. Authors per the abstract page: Gekhman, Yona, Aharoni, Eyal, Feder, Reichart, Herzig. The exact title wording was not reproduced by the fetch; verify it.
   - Setting: PaLM 2-S on EntityQuestions.
   - Unknown (new) facts are fit much more slowly than Known ones.
   - Peak dev performance comes at early stopping (5-10 epochs), when most Known examples but few Unknown ones are fit.
   - Fitting Unknown examples lowers test accuracy (β = -8.3; Known β = +7.3; R² = 0.86).
   - Implication: fitting new facts to convergence costs generalisation. Use a held-out check, not a fixed epoch count.

6. Biderman et al., "LoRA Learns Less and Forgets Less", TMLR 2024, arXiv:2405.09673.
   - Llama-2-7B, code and math.
   - CPT on 20B tokens: full fine-tuning HumanEval 0.263 vs LoRA r=256 0.224.
   - Instruction fine-tuning (4 epochs): LoRA r=256 matches full fine-tuning (0.498 vs 0.497), while r=64 gets 0.417.
   - Forgetting score at 20B tokens on code: full fine-tuning 0.545 vs LoRA r=256 0.617. Per the paper, LoRA forgets less.
   - Recommendations: rank 256 for code (64 acceptable for math), LR 5e-5 to 5e-4, alpha=2r, all modules including MLPs.
   - Implication: for knowledge injection use high rank on all modules; low-rank LoRA underfits facts.

7. Shao, Li, Dai, Qiu, "Character-LLM: A Trainable Agent for Role-Playing", EMNLP 2023, arXiv:2310.10158.
   - LLaMA 7B, full fine-tuning, about 1.6K scenes per character (about 754K words average), 10 epochs, LR 2e-5.
   - Fewer than 100 "protective" scenes per character reduce hallucination.
   - The authors report checkpoint selection at epoch 5 vs 10 on held-out questions.
   - This is a persona-scale precedent: about 1-2K scenes (under 1M words) suffices for a persona on a 7B model. It measures role-play fidelity, not reasoning transfer.

8. Maini et al., "Rephrasing the Web (WRAP)", 2024, arXiv:2401.16380.
   - Mix is 1:1 real:synthetic (85B + 85B tokens on C4, 1.3B model).
   - Gives about 3x pre-training speedup and over 10% perplexity improvement.
   - Synthetic-only training degrades perplexity, so keep original text in the mix.
   - Rephraser quality matters: T5-base was significantly worse than Qwen-1.8B.

9. Sun et al., "Principle-Driven Self-Alignment" (SELF-ALIGN), NeurIPS 2023, arXiv:2305.03047.
   - Fewer than 300 lines of human annotation: under 200 seed prompts, 16 principles, 5 exemplars, on LLaMA-65B.
   - The principles are the compact object; the data volume is generated.
   - The number of generated examples was not captured in the fetch (UNVERIFIED).

10. Sun et al., "SALMON", ICLR 2024, arXiv:2310.05910.
    - 31 principles plus 6 exemplars, LLaMA-2-70B.
    - Shows that explicit principles can drive synthetic preference data; synthetic volume was not captured (UNVERIFIED).

11. Wang et al., "Self-Instruct", ACL 2023, arXiv:2212.10560.
    - Generate, filter and dedupe self-produced instructions.
    - A 33% absolute gain on Super-NaturalInstructions for GPT-3.
    - Filtering and dedup are the quality control; the seed and output counts were not captured (UNVERIFIED).

Not found: I found no paper giving a saturation number for QA-per-fact from a source corpus of this size. The nearest evidence is items 1, 3 and 4.

## Recommendation (inference from the above, not a published rule)

Size the expansion by diversity and by held-out saturation, not by a fixed ratio.

- Facts: target at least 5 diverse rewrites per fact as a floor (item 1). Aim for about 10 or more (item 4, still rising at 10). Aim for about 1000 effective exposures across the full run (item 3.3).
- Tokens: a 7M-token source against EntiGraph's roughly 350x and its still-rising log-linear curve means up to about 100x-350x is defensible only if the data is diverse. That is roughly 0.7B-2.4B tokens, but do not commit to a size upfront.
- Principles and judgment: express them as a compact written set (items 9 and 10 use 16-31 principles). Generate many applications (case, counterfactual, critique, decision) and include held-out situations in evaluation.
- Agentic behaviour: use trajectories of the author's decisions in tool-use settings. No source quantifies this; it is empirical.
- Mixing: keep original text (item 8) plus a general-data replay share against forgetting. LoRA rank must be at least 256, with alpha=2r, on all modules (item 6). A replay share of 10-30% is my guess, UNVERIFIED.
- Refusal: add protective examples that decline unknown facts (item 7). Fewer than 100 per character worked there.
- Keep a retrieval layer for the facts (items 3 and 4).

### Stopping curve design

Fix a held-out split by source document before any expansion (not by augmented sample). Build three probe sets: fact QA (closed-book, with rephrased questions never used in training), principle-application probes (new situations), and agentic task success. Add a forgetting probe on general benchmarks.

Train at expansion sizes of 0.5x, 1x, 2x, 4x, ... the planned corpus (equivalently 1, 2, 5, 10 rewrites per fact). Plot each probe against log(synthetic tokens). Stop growing when the gain per doubling falls below noise on the fact and principle probes (e.g. under 1 point, with 3 seeds), or when the forgetting probe drops more than a preset budget. Within a run, use early stopping on the held-out probes rather than a fixed epoch count (item 5). Report the curve itself (item 3 reports log-linear scaling up to 455M tokens, so a plateau in our setting is not guaranteed).
