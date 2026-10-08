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

| | Work | State |
|---|---|---|
| M0 | Weights, cascade loop, round-trip and turn measurements | done (below) |
| M1 | Retrain both personas on Qwen3-8B with the persona-paper recipe | open |
| M1.5 | Resident streaming synthesis in brain's SDK (no reload per call) | open |
| M2 | Spoken renditions stored as representations of the same experience; frozen spoken eval set | open |
| M3 | Speech-ingress machinery in brain, zero-shot baseline | open |
| M4 | Train the speech ingress against a frozen Thinker; gate speech versus text | open |
| M5 | Talker conditioned on the Thinker's state | open |
| M6 | Compose, bundle, release | open |
| M7-M8 | Joint losses; second persona | open |

Later, in order: more voices as voice adapters, streaming ingress, spoken
tool-call trajectories, understood interruption, full duplex.

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
