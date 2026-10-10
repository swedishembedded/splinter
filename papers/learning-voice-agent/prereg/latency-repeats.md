# Pre-registration: first-sound latency, repeated (L1)

Written after the single run `latency-stream-20261009` and before the repeats.

Design: the 30 spoken questions of `c0-questions` (synthetic voice, seed 1), Qwen3-8B, CUDA, Nemotron recogniser, Qwen3-TTS, in three conditions, each repeated 3 times in separate processes, under the GPU lease so that none of this work overlaps another job:
(a) `turns --stream`, the cascade of the original run; (b) `agent`, the sven agent under the persona with the teaching layer (an empty ledger); (c) `agent --model-reader`, which also asks the model to read each turn for a teaching.
Before and after each process the GPU's utilisation and its list of compute processes are recorded; a repeat in which another compute process was present is reported and excluded from the gate.
Reported: first sound p50 and p95 per repeat, and pooled over repeats with the range across repeats; for (b) and (c) against (a).
Gate (as in the claim): first sound p50 <= 1.5 s and p95 <= 2.0 s, in every repeat of the condition.
Limits: synthetic speech; one voice; the persona prompt differs between (a) and (b)/(c) only in that (a) uses the persona answerer and (b)/(c) the sven agent; the ledger is empty, so the cost of a ledger's selection (a few string comparisons) is not measured; no real speech.
