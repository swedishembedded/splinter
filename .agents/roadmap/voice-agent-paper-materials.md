SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

Learning-Voice-Agent: Materials Inventory and Gaps

## 1. Methodology to Formalize

### 1.1 Voice Agent Learning Framework

| Claim | Evidence | Reproducible? | Gap |
|-------|----------|---------------|-----|
| Lessons are a formalism for typed learning directives | crates/core/src/speech_lesson.rs: Directive enum (Imitate, Respell), Lesson struct with Objective (Lexicon, InContext, Supervised, Preference), parse_directive() function, tests | Yes, via tests in file | Formalization paper: state grammar, parsing semantics, lesson composition rules |
| Lexicon objective teaches pronunciation without training | crates/core/src/speech_lesson.rs: Lexicon struct, apply() method applies word replacements before synthesis; test: a_lexicon_says_taught_words_as_taught_and_leaves_the_rest_alone; `model::speech::Lexical` (splinter-model), whose spec is `a_taught_word_reaches_the_voice_as_taught_and_nothing_else_changes` | Yes, via integration test in voice-loop | Integration: wire Lexicon into full cascade and measure (1) synthesis output with lexicon applied vs not, (2) downstream recognition WER |
| In-context objective supplies reference for voice adaptation | crates/core/src/speech_lesson.rs: InContext variant defined; Exemplar::Recording holds audio digest and words | No | Implementation gap: brain TTS resident engine reference-recording pathway not wired to synthesizer; need to implement and test reference-conditioned generation |
| Supervised objective creates training data for TTS | crates/core/src/speech_lesson.rs: Supervised variant yields Lesson with recording exemplar | Partial | Training gap: need TTS trainer that accepts (text, audio) pairs and measures WER on held-out (not in paper scope but must exist to claim supervised learning works) |
| Preference objective pairs best/worst takes for RL | crates/eval/src/speech_reward.rs: preference_pair(rewards) returns (best, worst) indices when rewards differ | Yes, via unit tests | RL gap: GRPO trainer over TTS not yet implemented; need to formalize preference learning from group advantages |

### 1.2 Reward Function for Voice Quality

| Claim | Evidence | Reproducible? | Gap |
|-------|----------|---------------|-----|
| Reward = naturalness - clarity_weight * WER guards both dimensions | crates/eval/src/speech_reward.rs lines 50-52: reward() function; test: naturalness_cannot_buy_back_lost_words | Yes, via unit test | Empirical weight tuning: clarify_weight default 5.0 arbitrary; need measured cost of each WER unit in terms of human naturalness judgment |
| Group-relative advantages center on zero | crates/eval/src/speech_reward.rs lines 58-70: group_advantages() returns (r_i - mean)/std; test: a_take_is_measured_against_the_other_takes_of_its_own_sentence | Yes, via unit test | Validation: measure variance in advantages when group size varies (2 vs 8 samples per sentence); verify signal stability |
| Preference pairs form valid training targets for GRPO | crates/eval/src/speech_reward.rs lines 76-88: preference_pair() returns (best, worst) when rewards differ | No | Trainer gap: brain GRPO applied to TTS not present; DPO or IPO alternative not evaluated |

### 1.3 Speech Ingress and Cascade

| Claim | Evidence | Reproducible? | Gap |
|-------|----------|---------------|-----|
| Cascade (recognize, answer in text, speak) is a baseline | .agents/roadmap/speech.md M0: "The reference loop is a cascade"; cascade command in voice-loop; tests in crates/model/src/speech/ | Yes, via voice-loop turns command | Measurement formalization: define cascade as LLM policy without speech-conditioned projection; state text loss baseline it is compared against |
| Ingress projector bridges ASR features to language model | .agents/roadmap/speech.md M4: "A two-layer projector (2048 to 4096) was trained for 300 steps"; crates/model/src/speech/ingress.rs exists | Yes, via ingress-train command | Evaluation gap: M4 gate FAILED (projector learns register, not content); analysis why needed; redesign options are listed in the roadmap item for the ingress gate; none is trained |
| Speech ingress preserves equivalence with text on held-out topics | .agents/roadmap/speech.md M4: "the native answers are fluent and in the persona's voice while often about the wrong topic"; cascade F1 0.489, native F1 0.299 | No | M4 result shows FAILURE on equivalence gate; gap: redesigned projector+loss not trained |

## 2. Novelty Claims and Prior Work

### 2.1 Voice Agent Learning from Correction

| Claim | Related Work | Status | Citation Needed |
|-------|--------------|--------|-----------------|
| Users teach pronunciation and delivery via spoken correction | HappyRobot blog post "teaching-our-text-to-speech-engine-to-sound-natural" describes naturalness + clarity tradeoff in preference learning | UNVERIFIED - blog URL not checked by this inventory | https://www.happyrobot.ai/blog/teaching-our-text-to-speech-engine-to-sound-natural |
| Spoken lessons map to four learning objectives | No prior work found formalizing lesson types (Lexicon, InContext, Supervised, Preference) | Novelty | N/A |
| Group-relative advantage weighting in speech | GRPO (introduced in the DeepSeekMath paper, Shao et al. 2024; UNVERIFIED here, check the exact reference) uses group-relative advantages for language models; its use on speech is reported only in the HappyRobot post (UNVERIFIED) | Claimed novelty but needs literature search | Search: "group relative policy optimization speech", "preference learning text-to-speech" |
| Pronunciation lexicon as stateless production rule | Traditional: SSML phoneme tags, pron fields in ASR; novel: lesson-driven respelling applied at synthesis time without training | Partially novel | Cite: SSML spec (W3C), mention state-of-art pron customization methods |

### 2.2 Known Related Work Topics (Needs Verification)

| Topic | Expected Citation Form | Must Verify |
|-------|------------------------|-------------|
| Text-to-speech quality metrics and naturalness judges | MOS (Mean Opinion Score) literature; neural naturalness judges (e.g., wav2vec-based); AudioMAE | Citation TBD |
| DPO (Direct Preference Optimization) | Rafailov et al. 2023 Direct Preference Optimization; compare to GRPO cost/quality tradeoff | Citation TBD |
| In-context voice cloning / few-shot voice adaptation | VALL-E, Google Music LM, in-context learning for speech synthesis | Citation TBD |
| Reinforcement learning for speech synthesis | HappyRobot + any academic papers on RL for TTS | Citation TBD |
| Persona fine-tuning of language models | persona-training paper (same repo); Character-LLM, RoleLLM, Ditto (already cited in other paper) | Adapt from persona-training references.bib |
| User feedback as training signal for language models | Constitutional AI (Bai et al.), learning from feedback literature | Citation TBD |
| Spoken language models (speech in, text internal, speech out) | Qwen Omni, Qwen-Audio; Qwen3 technical report (Qwen team 2025) | Check arXiv 2505.09388 |

## 3. Experiments Needed: Pre-Registered Designs

### 3.1 Experiment 1: Lesson Type Coverage

**Hypothesis:** Each lesson objective (Lexicon, InContext, Supervised, Preference) produces measurable improvement on its dimension.

Note: the "speakers" here are synthetic voices (seeds), not people; the paper may not claim anything about real speech. A design with 6 test utterances cannot support a significance claim: the sample sizes below are placeholders to be set by a power analysis before freezing, not chosen numbers.

| Element | Specification |
|---------|---------------|
| Data | 10 speakers (utterances), 5 lessons per speaker (pre-recorded examples), held-out 2 speakers for test |
| Metrics | (1) Lexicon: output WER when lexicon applied vs not; (2) InContext: naturalness (user binpref) on in-context vs baseline; (3) Supervised: WER on supervised-trained vs untrained TTS on lesson text; (4) Preference: W-rate (win-rate of preferred over unpaired samples in pref pair) |
| Sample Size | 10 speakers x 5 lessons x 3 utterances = 150 training points; 2 held-out speakers x 3 utterances = 6 test utterances |
| Held-Out Split | By speaker (no speaker in train is in test) |
| Test | Per-metric paired t-test (WER, naturalness binpref count); effect size Cohen's d |
| Acceptance | Each metric shows p < 0.05 and d > 0.2 on its objective type |

Gap: Naturalness judge not available; user feedback binpref must substitute until multimodal judge deployed.

### 3.2 Experiment 2: Reward Function Calibration

**Hypothesis:** The tradeoff weight (clarity: 5.0) reflects human preference for clarity over naturalness.

| Element | Specification |
|---------|---------------|
| Data | 50 sentences, 4 synthesis variants per sentence (low-WER+low-nat, low-WER+high-nat, high-WER+low-nat, high-WER+high-nat constructed or sampled), 20 human judges per variant |
| Metrics | Human preference (binpref matrix); predicted reward via reward function; correlation (Spearman rho) between predicted reward ranking and human preference ranking |
| Sample Size | 50 x 4 variants x 20 judges = 4000 judgments; split 50/50 train/test |
| Held-Out Split | By sentence; no sentence in calibration in test |
| Test | Spearman correlation of reward with human preference; optimal weight via empirical search on train, validate on test |
| Acceptance | rho > 0.7 on test; calibrated weight differs from default by less than 2x |

Gap: No naturalness judge on host; must collect human judgments or deploy multimodal audio judge.

### 3.3 Experiment 3: Speech Ingress Equivalence Gate

**Hypothesis:** After projection and LoRA training, speech-conditioned thinker scores within a margin of text-conditioned thinker on held-out questions.

| Element | Specification |
|---------|---------------|
| Data | 200 questions split: 150 train (text + speech variants), 50 held-out test (speech only, not in training) |
| Metric | Per-task exact match on short answer (thinker output); F1 on generated response; topic word present (0/1) |
| Training | Thinker frozen; projector trained (the M4 recipe in speech.md); any redesign is pre-registered before it is trained |
| Held-Out Split | By question (no question in train ingress data in test); voices also held out (new voice 14) |
| Test | Paired by question: speech-conditioned against text-conditioned and against the cascade; equivalence by two one-sided tests (TOST) with the margin fixed before training, plus the discordant counts |
| Acceptance | The 90% interval of the paired F1 difference lies inside the pre-registered margin (0.05 proposed). A non-significant sign test is not equivalence and is not used as the gate |

Gap: M4 failed this gate on the 12 held-out questions (word F1 0.299 from speech against 0.489 for the cascade; topic word named in 4 of 12 against 9 of 12; speech.md M4 result). The paper reports it as a negative result unless a redesign passes a gate fixed in advance.

### 3.4 Experiment 4: GRPO Learning from Preference

**Hypothesis:** GRPO training on preference pairs improves WER and naturalness compared to supervised baseline on held-out test set.

| Element | Specification |
|---------|---------------|
| Data | 100 sentences with 8 renders each (from sampling): 80 train (train preference pairs), 20 test held-out |
| Training | Supervised baseline: train TTS on all renders with text targets; GRPO: sample 8 from 100 train, compute advantages, run GRPO steps; separate seeds |
| Metric | WER on 20 test sentences (unseen at training); human naturalness binpref on test renders |
| Test | Paired t-test of WER (GRPO vs supervised); sign test of naturalness preference (GRPO vs supervised) |
| Acceptance | GRPO WER <= supervised WER (or p < 0.05 no difference); naturalness p < 0.05 in favor of GRPO |

Gap: GRPO trainer for TTS does not exist in brain; must implement or evaluate DPO alternative.

## 4. Datasets and Frozen Sets Needed

| Dataset | Purpose | Specification | Reproducibility | Status |
|---------|---------|---------------|-----------------|--------|
| Lesson Corpus | Baseline for lesson type ablation | 10 speakers, 5 lessons/speaker, 3 utterances each, recorded pronunciations and corrections | Frozen before analysis | Not created |
| Reward Calibration Set | Naturalness weight tuning | 50 sentences, 4 synthesis variants, 20 human judges per variant; held-out test split | Frozen before training GRPO | Not collected |
| Speech Ingress Holdout | Equiv gate test (Exp 3) | 150 train questions (text + speech), 50 test questions (speech only), voice 14 held out | Frozen before ingress training | Partial: M4 used 200 questions voices 11-12 for train, voice 13 for test; new recording of voice 14 needed for second iteration |
| GRPO Training Set | Speech RL training | 100 sentences with 8 renders each sampled from TTS (80 train, 20 test); digests and WER/naturalness measured | Frozen before GRPO step | Not created |
| Exam Set (Reserved) | Final speech learning evaluation | 50 held-out questions (speech input), persona answers expected (text internal), held-out voices (at least 2 new beyond 14) | Frozen before any GRPO train | Not created |

## 5. Models and Compute Needed

| Component | Specification | Available? | Location |
|-----------|---------------|-----------|----------|
| Language Model Base (Thinker) | Qwen3-8B | Yes | brain pull Qwen/Qwen3-8B; .agents/roadmap/speech.md M0 |
| Judge (secondary) | Qwen3-14B | Yes | brain pull Qwen/Qwen3-14B; used in persona-training paper |
| ASR (Recognizer) | Qwen3-ASR-1.7B or Nemotron 3.5 0.6B | Yes | Manifest: ~/resources/speech/MANIFEST.tsv lines 1-28 |
| TTS Base (Synthesizer) | Qwen3-TTS-12Hz-0.6B-Base | Yes | Manifest line 7-15; crates/model/src/speech/cascade.rs uses it |
| Naturalness Judge | Multimodal audio model (no local equivalent) | No | GAP: brain does not have resident multimodal audio judge; options: (1) user thumbs-up (preference), (2) wav2vec-based scoring (research), (3) cloud API (out of scope) |
| GRPO Trainer for TTS | RL trainer specialized for diffusion/codec-based TTS | No | brain crates/rl/ exists (for text GRPO); GRPO for TTS not present |
| GPU | One NVIDIA GH200 (120 GB); CUDA through `voice-loop --backend cuda` | Yes | Shared lease; latency runs need an idle card, proven by stored profiles under `~/resources/speech/profiles/` |

## 6. Ethics: Synthetic Voice Labeling and User Consent

| Requirement | Implementation | Evidence | Gap |
|-------------|-----------------|----------|-----|
| Synthetic voice always labeled as portrayal | Enforced in type system: SpeakerProfile requires Portrayal (speech.rs line 74-76); spoken_disclosure() prepends label (line 102-108) | Tests: a_portrayal_cannot_be_blank, a_speaker_read_without_a_portrayal_is_refused | Verification: audit all generated speech outputs have disclosure sentence prepended |
| User audio (corrections, lessons) not used to claim real voice | Lesson struct stores audio digest only, never holds audio (speech_lesson.rs line 125); Portrayal applies to voice, not to recorded user speech | Design: lessons are typed learning data, not samples of real voice | Formaliza: paper must clearly state user recordings used only for teaching, never as evidence of real voice |
| Consent and privacy for recorded user audio | Lesson provenance in "heard" field (line 138), digests for content addressing | Lessons name source for audit trail | Specification gap: consent form and data-handling policy for user correction recordings not part of this paper's scope (would be in product doc) |
| No synthesis from real user voices without consent | speech.rs design: SpeakerProfile seed + portrayal separate from user lesson recordings | Specification | Enforce: test that user lesson audio is never directly rendered as tts input (must go through supervised training, not cloning) |

## 7. Threats to Validity

| Threat | Severity | Mitigation |
|--------|----------|-----------|
| Naturalness judge absent: cannot measure preference without human eval or multimodal model | High | Use user binary preference (thumbs-up) as proxy; collect human judgments offline for calibration experiment; or deploy audio judge (blocks paper until available) |
| Small speaker set (10 in lesson ablation) | Medium | Power analysis required; sample size may be insufficient for speech quality effect sizes; baseline from pilot (M2: 30 recordings under recognize gate passes) suggests feasibility but not proven |
| Single persona (Adams/Jefferson) in prior work; generalization untested | Medium | Paper scope to one persona in depth (Adams); mention Jefferson as future direction; note that lesson types are persona-agnostic |
| Synthesizer seed variance (speech.md M0 mentions "synthesis length varies a great deal by seed") | Medium | Fix seed per sentence; measure WER variance across seeds (done in M0 roundtrip: 3 of 120 lost under seed 1, none under Qwen3-ASR); report as confound |
| Lexicon and InContext not trained, so no learning signal backprop; Supervised and Preference require trainer implementation | High | Lexicon and InContext are non-learning objectives; state this clearly; block paper on Supervised and Preference until trainer is written and piloted |
| GRPO trainer not yet implemented in brain for TTS | Critical | Either (1) implement in brain and this paper cites the trainer crate and its tests, or (2) evaluate DPO as alternative, or (3) scope paper to Lexicon + InContext + Supervised only and defer GRPO |
| Human evaluation of naturalness requires recruitment and labeling cost | Medium | Defer detailed naturalness calibration to future work; use WER + user preference (binpref) for Exp 1-4; state as limitation |
| Cascade (text path) may not be true baseline if persona prompt not applied consistently | Medium | speech.md M4 notes "base model under the same persona prompt"; enforce: all baselines in paper use identical persona preamble |

## 8. References to Verify and Cite

### Confirmed Sources (paths in repo)

- speech_lesson.rs, speech_reward.rs, speech.rs, speech_bundle.rs: crates/core/src/ and crates/eval/src/
- Cascade, roundtrip, turn measurements: .agents/roadmap/speech.md M0 (WER, latency), M2 (431 recordings), M4 (ingress failure)
- M0 baseline: 0.038 WER (Nemotron judge), 0.0049 WER (Qwen3-ASR), 30/30 keep rate, 25s synthesis latency, 2.2s answer, 0.37s recognize (speech.md section "M0 result")
- Reward function tests: crates/eval/src/speech_reward.rs tests naturalness_cannot_buy_back_lost_words, a_take_is_measured_against_the_other_takes_of_its_own_sentence

### Unverified External Citations (TBD)

- HappyRobot blog (https://www.happyrobot.ai/blog/teaching-our-text-to-speech-engine-to-sound-natural) - UNVERIFIED
- GRPO (DeepSeekMath, Shao et al. 2024) - UNVERIFIED, needs exact citation
- DPO (Rafailov et al. 2023) - needs full citation
- VALL-E, in-context voice cloning - needs citations
- Qwen-Omni, Qwen-Audio - needs citations
- Qwen3 technical report (arXiv:2505.09388) - mentioned in persona-training references.bib as qwen3

### Related Work from Persona-Training Paper (Adapt)

From papers/persona-training/references.bib:
- Character-LLM (Shao et al. 2023), RoleLLM (Wang et al. 2023), Ditto (Lu et al. 2024) - persona/character fine-tuning
- LoRA (Hu et al. 2022) - adaptation method
- Constitutional AI, preference learning literature - user feedback signals

---

## Summary of Gaps (Top 5)

1. **Naturalness Judge Missing**: Cannot score preference learning without multimodal audio naturalness evaluation; must implement or defer to user feedback proxy
2. **GRPO Trainer for TTS Not Implemented**: Brain has text GRPO; speech GRPO required for Preference objective; blocks Experiment 4
3. **Speech Ingress Equivalence Gate Failed (M4)**: word F1 0.299 from speech against 0.489 for the cascade on 12 held-out questions (speech.md M4 result); a redesign is needed before the ingress claim can be made
4. **No Frozen Training and Test Sets**: Lesson ablation, reward calibration, speech ingress equiv, GRPO training datasets not yet created or measured; needed for all four experiments
5. **Supervisor TTS Training Pipeline**: Supervised objective requires TTS fine-tuning trainer accepting (text, audio) pairs with held-out WER evaluation; not yet deployed


## Review note

This inventory was drafted by a Haiku subagent and corrected by the main session against the repository: figures quoted from speech.md were checked (M0 corpus word error rate 0.038 / 0.0049, M4 word F1 0.489 / 0.299 and topic 9 of 12 / 4 of 12, 431 recordings); the GPU, the GRPO citation, the Lexical location, the Experiment 3 training line and its non-significance-as-equivalence gate were wrong in the draft and are fixed above. Experiments 1, 2 and 4 are designs, not results; every number in their tables is a placeholder pending a power analysis. Nothing here has been run.
