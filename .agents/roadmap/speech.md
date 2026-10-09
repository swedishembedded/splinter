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

## Open items and known limits

- The R0 sentences lost under seed 1 are synthesis failures, not recognition
  failures; M2 admits clips through `speak_verified`.
- Persona answers are about eighty words, long for speech; M1 trains for
  spoken-length answers.
- The cascade ran the base model under a persona prompt. No Adams or Jefferson
  adapter exists on this host that passed the release gate.
- The splinter lock pins a brain revision that is pushed to origin but not to
  the public remote; a clean checkout builds once that is published.
