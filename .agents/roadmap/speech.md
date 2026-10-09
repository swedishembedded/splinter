# Speech roadmap: a persona you can talk to, then a persona that hears and speaks natively

Goal: talk to a Samuel Adams or Thomas Jefferson persona and have it talk
back. The first loop is a cascade (recognise, answer in text, speak), which is
the reference and the source of spoken training examples. The target is a
speech-native model: speech in, a frozen text Thinker carrying the persona
adapter, a Talker that speaks from the Thinker's state, speech out, with text
kept as the internal representation. Persona (how it thinks) and voice (how it
sounds) are separate adapters.

The voice is a synthetic theatrical portrayal and is always labelled so; no
recording of either man exists (`splinter_core::speech::Portrayal`).

## Rules for this effort

- Every stage is measured on its own; a rate or latency that was not measured
  is absent, never zero.
- Synthetic speech is admitted only if a recognizer that did not make it hears
  it back (`speech::speak_verified`). Synthesis length varies a great deal by
  seed.
- Equivalence claims (speech input versus text input) need a persona clearly
  above its base on the text path first; at the floor they mean nothing.
- Nothing is claimed about real human speech until a recorded real-speech test
  set exists.

## Milestones

| | Work | State, 2026-10-09 16:00 |
|---|---|---|
| M0 | Weights, cascade loop, round-trip and turn measurements | done (below) |
| M1 | Retrain both personas on Qwen3-8B with the persona-paper recipe | not done: see "M1" |
| M1.5 | Resident streaming synthesis in brain's SDK | done (`TtsPipeline::resident`, `ResidentTts`) |
| M2 | Spoken renditions as representations of the same text record; held-out voices and questions | done for Adams: 431 recordings of 240 questions, set digest `blake3:b062e5a5...`; 40 for the Jefferson swap test |
| M3 | Speech-ingress machinery in brain | done: encoder features, `SpeechIngress`, `SpeechProjector`, `ChatPipeline::generate_with_rows` |
| M4 | Train the ingress, gate speech against text | trained; the gate FAILS (below) |
| M5 | Talker conditioned on the thinker's state | not done; only the cascade (text into the synthesizer) exists |
| M6 | Compose, bundle, release | partly: bundle type and command, listener, sentence-streaming spoken turn; no release format change |
| M7 | Joint text and speech losses | not started |
| M8 | Second persona | evaluated with the Adams projector on Jefferson prompts (no Jefferson adapter exists) |

Later, in order: more voices as voice adapters, streaming ingress, spoken
tool-call trajectories, understood interruption, full duplex.

## M1: why the personas were not retrained

`splinter "Learn to think like Samuel Adams ..."` ran 2h40 with a 2h budget on
a GPU shared with other work. It froze the exam first (325 tasks over 50
reserved families) and that spent the budget: the task stage then reported
`0 task(s) from 0 of 198 text part(s)` and the run ended with "no task was
admitted, so there is nothing to solve". No adapter was trained. The Jefferson
run was stopped early to free the GPU. What exists: the Jefferson corpus and
its fetch script (`samples/jefferson/fetch.py`), both materials directories,
the frozen anchor suite. To do: freeze the exam sets separately
(`splinter exam-set create`), then run `learn` with a budget that leaves time
for tasks, solving, training and the gate; the recipe is in
`papers/persona-training`.

## M4 result (Adams, Qwen3-8B, no adapter, persona given by the system turn)

A two-layer projector (2048 to 4096) was trained for 300 steps of 8 examples
on 389 recordings (voices 11 and 12) of 200 questions, against the answers a
served Qwen3-8B gave under the persona prompt, plus a transcription example
per recording. The model was never updated (zero learning rate). Tested on
the 40 recordings of the 5 held-out topics in the held-out voice 13:

| Measure | Result |
|---|---|
| Mean per-token loss of the text path's answer, before training | 1.598 |
| Same, after 300 steps | 0.932 |
| Same, Jefferson prompts and Jefferson questions, Adams projector, no training | 0.951 |
| 12 held-out questions answered by the cascade: mean word F1 against the text path's answer; answers naming a topic word | 0.489; 9 of 12 |
| Same 12 answered from speech with no transcript: F1; topic word | 0.299; 4 of 12 |
| Answers spoken and heard again, word error rate: cascade; native | 0.030; 0.115 |

The loss fell and generalises across persona prompts, but the native answers
are fluent and in the persona's voice while often about the wrong topic
(asked about the duties of a citizen it speaks of taxation): the projector
learned the register of the answers and only partly the content of the
question. The equivalence gate of the plan (speech within a margin of text) is
not met, so nothing past M4 may be claimed.
The text-path loss on the same records from brain's `score_chat` is 1.0996,
but its rendering of the empty reasoning block was not verified to equal the
training rendering, so it is not used as a comparator.

Reasons and limits: 389 recordings of 25 templated topics, one synthetic
voice family, no real speech; a held-out topic set of five. Latencies in the
reports (tens of seconds per turn) were measured with three to four other
GPU jobs running and say nothing about the system. Next: more topics and
free-form questions (the LLM-written questions script is
`~/resources/speech/ingress/gen_questions.py`), unfreeze a LoRA beside the
projector (M7), and a transcription-first auxiliary target.

Commands: `voice-loop spoken-set` (sharded), `spoken-merge`, `ingress-train`
(`--steps 0 --init-projector` to evaluate), `listen-turns`, `turns`,
`text-score`, `bundle`; scripts and outputs under `resources/speech/ingress`.

## M0 result

Models (all in brain's store, hashes in `resources/speech/MANIFEST.tsv`):
Qwen3-TTS-12Hz-0.6B-Base, Nemotron 3.5 ASR streaming 0.6B, Qwen3-ASR-1.7B.
Frozen sets (`resources/speech/sets`, digests in `SHA256SUMS`): R0, 120
sentences (100 of the founders' register, 20 general); C0, 30 questions.

| Measure | Result |
|---|---|
| `qwen3tts` `asr_roundtrip` gate on real weights | passes |
| R0, Nemotron judge, seed 1 | corpus word error rate 0.038; 3 of 120 sentences lost (all empty transcripts) |
| R0, Qwen3-ASR judge | 0.0049; none lost (0.778 and all lost before the frame-count fix, knowledge 211) |
| C0 recordings, `speak-set`, seeds from 2 | 30 of 30 kept; two needed a second seed |
| C0 question recognition | corpus word error rate 0.023 |
| C0 answers (Qwen3-8B, persona prompt, no adapter), spoken and heard again | 0.057; every answer non-empty, no reasoning block |
| Same answers spoken in one piece | 0.258: every reply cut at 20.5 s, replaced by sentence-wise speaking |
| Stage latency p50 (recognise, answer, speak) | 0.37 s, 2.2 s, 25 s; synthesis dominates because each piece reloads the model |

Reproduce: `voice-loop roundtrip`, `speak-set` and `turns` of
`samples/voice-loop` on the sets above.

Defects found on the way (each with its test): the two-architecture resolver
ignored the requested checkpoint (brain knowledge 210); Qwen3-ASR told its
encoder it had 128 frames (brain knowledge 211); a spoken answer longer than
one render is truncated.

## Latency on CUDA (2026-10-09, GH200, one process, GPU shared with nothing else running)

Question spoken, 8B persona answer of about 80 words, 30 recordings, the cascade
speaking each sentence as it is written. Reproduce:
`voice-loop turns --stream --recordings <dir> --out-dir <d> --base <Qwen3-8B>`
with `voice-loop --backend cuda`.

| Stage | Before (Vulkan, CPU codec) | Now (CUDA) |
|---|---|---|
| Speaking 12 s of speech | 20.4 s (codec 14.3 s, codes 6.1 s) | 3.4 s (real-time factor 0.27) |
| 8B answer, 80 tokens | 2.3 s | 1.4 s |
| First sound, p50 / p95 | 3.4 s / 4.8 s | 2.6 s / 3.1 s |
| Whole turn, p50 | 31 s | 9.9 s |

Not yet at the 1.5 s first-sound target. What remains, in order of size: the
language model is host-bound at about 17 ms a token on a card that could do
four times that; the code predictor takes 11 ms of each 23 ms frame; the codec
decode needs the whole utterance (no frame streaming), and its two convolution
kernels are the slowest on the card; speaking and writing share the GPU, so
overlapping them slows both. The first-sound figure includes recognition
(0.3 s). Idle-GPU proof and n>=20 are the p50/p95 above only; the stored
profiles are under `~/resources/speech/profiles/`.

## Learning to speak from conversation

A user's spoken or typed correction is data. `core::speech_lesson` reads a
directive ("Talk as follows: ...", "Pronounce X as Y") and keeps lessons, each
with an objective that says what it teaches and how:

* `lexicon`: a word is said as spelled out. Applied before synthesis
  (`model::speech::Lexical`), nothing trained. Works today.
* `in_context`: the user's recording is a reference the voice imitates (brain's
  resident engine takes a reference clip and its words). Nothing trained. The
  lesson names the clip; wiring it into `Synthesizer` is not done.
* `supervised`: text and recording as a supervised example for the
  synthesizer (brain's TTS fine-tuning). Lesson kept; no trainer wired.
* `preference`: the recording preferred over the synthesizer's own take. Feeds
  the reinforcement loop below.

The reinforcement loop follows HappyRobot's account of tuning a speech model
(sample several takes of a sentence, score each, make the better ones likelier;
GRPO, group-relative advantages; reward = naturalness from a judge plus word
error rate so neither buys the other; a KL anchor to the original voice; humans
choose the checkpoint). `eval::speech_reward` has the reward, group advantages
and preference pairs, tested. Not built: a naturalness judge (no multimodal
audio model is on this host; the user's own thumbs-up or correction is the
judge until one is), and the GRPO step over the Talker (brain has GRPO for text
models and the Talker is a Qwen, so it is a trainer to write, not a research
gap).

How a conversation becomes lessons: `voice-loop teach` hears the directive and
keeps lessons. An imitation needs the example as its own recording, spoken as
the next turn; trimming it out of the same utterance needs word timestamps the
recogniser does not give. A pronunciation said aloud cannot be transcribed
("Jeff-er-son" comes back "Jefferson"), so it is taught by typing it or by
recording an example. Every lesson carries the portrayal of the voice it
applies to and names recordings by digest only.

## Open items and known limits

- The R0 sentences lost under seed 1 are synthesis failures, not recognition
  failures; M2 admits clips through `speak_verified`.
- Persona answers are about eighty words, long for speech; M1 trains for
  spoken-length answers.
- The cascade ran the base model under a persona prompt. No Adams or Jefferson
  adapter exists on this host that passed the release gate.
- The splinter lock pins a brain revision that is pushed to origin but not to
  the public remote; a clean checkout builds once that is published.

## Paper: learning-voice-agent (materials and gaps)

The paper formalizes training a voice agent from user corrections and preferences,
presenting a novel approach to autonomous learning from spoken directives and
feedback. The paper's scope: methodology for learning voice from conversation
(spoken/typed "Talk as follows: ...", "Pronounce X as Y", user corrections become
typed lessons with learning objectives), reward formulation (naturalness + WER,
group-relative advantages), and GRPO reinforcement from feedback. Reproducible
results only; no claims beyond what frozen data and held-out evaluation support.

Materials and gaps are documented in `papers/learning-voice-agent/MATERIALS.md`;
the roadmap below lists what must be done before claims can be made.

### 1. Implement TTS supervised training pipeline

**What**: Brain trainer for TTS that accepts (text, audio) pairs and measures
held-out WER. Used by Supervised lesson objective.

**Why it blocks the paper**: Supervised learning objective exists in design
(speech_lesson.rs, Objective::Supervised) but has no trainer; cannot train on
correction examples without this.

**Acceptance criterion**: Brain crate trains on (text, audio) pairs, measures
WER on a held-out test set, and the held-out WER improves on the untrained synthesizer by a pre-registered margin (seed reproducible).

**Rough dependencies**: After naturalness judge (item 2) if using multi-modal
judge for joint optimization; independent if WER-only.

### 2. Implement or scope naturalness judge for reward

**What**: Measure naturalness (0-1 scale) of synthesis. Either (a) deploy
multimodal audio judge (wav2vec-based or cloud), or (b) use user binary preference
(thumbs-up) as proxy and calibrate empirically, or (c) scope paper to WER-only
metrics.

**Why it blocks the paper**: reward() function in speech_reward.rs weighs
naturalness equally with WER; cannot set weight without measured judgments.
GRPO preference learning (Preference objective) needs naturalness scores.

**Acceptance criterion**: (a) Naturalness scoring function with held-out
calibration (rho > 0.7 with human pref), or (b) user binpref collection pipeline
with >= 50 judges x 50 sentences, or (c) paper explicitly scopes to WER-only and
defers naturalness tuning.

**Rough dependencies**: Should come early; blocks Exp 2 (reward calibration) and
Exp 4 (GRPO training).

### 3. Create frozen lesson dataset for Exp 1 (lesson type ablation)

**What**: 10 speakers, 5 lessons per speaker (typed corrections + recorded
examples), 3 utterances per lesson, split: train 8 speakers, test 2 speakers
held out by speaker.

**Why it blocks the paper**: Exp 1 validates that each lesson objective
(Lexicon, InContext, Supervised, Preference) produces measurable effect on its
target metric.

**Acceptance criterion**: Frozen dataset manifest (speakers, lesson types,
utterances) with digests; per-lesson WER or naturalness measurement script;
test split rule enforced.

**Rough dependencies**: After speaker corpus is available; independent of trainer
implementation.

### 4. Design and implement GRPO trainer for TTS in brain

**What**: RL trainer that takes preference pairs (preferred take, unpaired sample),
computes group-relative advantages via speech_reward.rs, and updates TTS via GRPO.
Specification: group-relative advantages (Exp 3 in MATERIALS.md), reward clamp,
KL anchor to base TTS.

**Why it blocks the paper**: Preference objective and Exp 4 (GRPO learning)
require this; cannot demonstrate learning from feedback without RL trainer.

**Acceptance criterion**: Brain crate implements GRPO for TTS; passes test
(baseline TTS vs GRPO TTS on same data shows lower WER or higher preference
on held-out; seed repro); integrates with brain's preference pair pipeline.

**Rough dependencies**: After TTS supervised trainer (item 1); after reward
function is calibrated (item 2).

### 5. Diagnose and redesign speech ingress projector (M4 redesign)

**What**: M4 projection train reached F1 0.299 on speech-conditioned thinker vs
0.489 cascade baseline; projector learned register not content. Redesign options:
(a) freeze ASR encoder features, scale projector, add supervised transcription-loss
auxiliary target, or (b) end-to-end projector+thinker training
on recognizer loss + downstream task loss jointly.

**Why it blocks the paper**: M4 result failed equivalence gate (speech within
margin of text); cannot claim speech ingress works or is a valid training signal
without fixing this.

**Acceptance criterion**: a redesigned projector is judged on the frozen exam of
item 6 by the equivalence test of item 10; the thresholds are fixed before it
is trained, not read off the M4 numbers (0.299 and 0.489 on 12 questions).

**Rough dependencies**: Independent; can be done in parallel with other items.

### 6. Create frozen exam set for final validation (Exp 3: ingress equiv gate)

**What**: 200 questions: 150 for ingress training (with text + speech variants),
50 held-out test (speech only). New voice (voice 14) for test. Frozen before any
ingress training starts; test split by question and by voice.

**Why it blocks the paper**: Exp 3 (ingress equivalence) requires pre-registered
held-out split; M4 used ad-hoc voice 13 split; paper needs frozen exam.

**Acceptance criterion**: Manifest with 200 question digests, 150 train/50 test
split, voice assignment, recorded questions and cascade-generated answers.

**Rough dependencies**: After M2 speech recordings are complete; parallel to
ingress redesign (item 5).

### 7. Create GRPO training and evaluation dataset (Exp 4)

**What**: 100 sentences, 8 renders each (sampled from base TTS with different seeds),
split 80 train (for GRPO), 20 test held out. Pre-compute WER and naturalness
(from judge in item 2) per render.

**Why it blocks the paper**: Exp 4 validates GRPO learning from preference pairs;
dataset frozen before training starts.

**Acceptance criterion**: Dataset manifest (100 sentence digests, 8 renders each,
WER and naturalness scores, 80/20 split), recorded audio frozen by digest.

**Rough dependencies**: After naturalness judge (item 2); independent of
trainers.

### 8. Run Exp 1: lesson type ablation (Lexicon, InContext, Supervised, Preference)

**What**: Train on frozen lesson dataset (item 3); measure per-objective metric:
Lexicon (output WER with/without), InContext (naturalness pref), Supervised
(held-out WER after TTS fine-tune), Preference (W-rate in pref pairs).

**Why it blocks the paper**: Validates that lesson types work as designed.

**Acceptance criterion**: Per-objective effect sizes (d > 0.2) and p < 0.05 on
primary metric; report in paper with sample sizes and held-out split rule.

**Rough dependencies**: Item 3 (dataset), item 2 (naturalness judge for InContext),
item 1 (TTS trainer for Supervised), item 4 (GRPO for Preference).

### 9. Run Exp 2: reward function calibration (naturalness weight)

**What**: Collect human binpref on 50 sentences x 4 variants (low/high WER x
low/high naturalness). Train optimal weight on 50% data, validate on 50% test.
Report Spearman rho with human preference and calibrated weight.

**Why it blocks the paper**: Justifies clarity weight in reward function; required
for GRPO (item 4) and paper claims on reward design.

**Acceptance criterion**: rho > 0.7 on test; calibrated weight reported with
95% CI; comparison to default 5.0.

**Rough dependencies**: Item 2 (naturalness judge); item 7 (WER measurements on
diverse renders).

### 10. Run Exp 3: speech ingress equivalence gate (redesigned projector)

**What**: Train redesigned ingress projector (item 5) on 150 train questions
(text + speech); test on 50 held-out-question speech renders (voice 14).
Primary test: paired TOST on word F1, speech against text path, margin 0.05.

**Why it blocks the paper**: Validates that speech-conditioned thinker is
equivalent to text, a gate M4 failed. Cannot claim speech ingress as training
signal without this.

**Acceptance criterion**: equivalence by two one-sided tests: the 90% interval
of the paired F1 difference (speech against text path, by question) lies
inside a margin fixed before training (0.05 proposed), with discordant counts
reported and seeds frozen in advance. A non-significant sign test is not
equivalence and is not the gate.

**Rough dependencies**: Item 5 (redesigned projector); item 6 (frozen exam).

### 11. Run Exp 4: GRPO learning from preference (speech RL)

**What**: Train supervised TTS baseline and GRPO TTS on item 7 dataset (80 train
sentences, GRPO on preference pairs from 8 renders). Test on 20 held-out
sentences. Paired t-test of WER (GRPO vs supervised) and sign test of human
naturalness preference.

**Why it blocks the paper**: Demonstrates learning from user feedback
(Preference objective). Central claim of the paper.

**Acceptance criterion**: GRPO WER not worse than the supervised baseline by a
pre-registered margin (TOST), and the naturalness sign test in favour of GRPO
reported with its p-value, a null result included.

**Rough dependencies**: Item 1 (TTS supervised trainer), item 4 (GRPO trainer),
item 2 (naturalness judge), item 7 (GRPO dataset), item 9 (calibrated reward).

### 12. Prepare frozen exam for publication and paper

**What**: Run final validation on exam combining all four lesson objectives:
corpus of synthetic speakers (size fixed by a power analysis), each with 5-10 lessons (mixed types), training data
split, test set held out by speaker and by voice. Measure holdout generalization.

**Why it blocks the paper**: Demonstrates generalization of voice learning
across diverse speakers and lesson types; supports claims of method robustness.

**Acceptance criterion**: Report: #speakers, #lessons per type, train/test split,
per-speaker F1 (speech vs text), variance across speakers, list of frozen
digests.

**Rough dependencies**: Items 1-11 all must be complete and passing.

### 13. Write paper: methodology, experiments, results, reproducibility

**What**: Formalize lesson framework (Directive, Lesson, Objective types),
reward function and group advantages, experimental designs (Exp 1-4 methods and
results), limitations (naturalness judge, single persona, seed variance), threats
to validity.

**Why it blocks publication**: Paper must be written and self-contained.

**Acceptance criterion**: Paper builds (make in papers/learning-voice-agent/),
all numbers trace to frozen files in repo or MATERIALS.md, references checked,
limitations section addresses items 2, 5, others.

**Rough dependencies**: Items 1-12 completed; roadmap items merged; external
citations verified.
