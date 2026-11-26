<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 228. The worker's own interpreter, not the model servers, was the attempt stage's bottleneck

**Lesson:** once sandboxes started in a second (227), a 512-trial attempt stage was bound by the one Python process
that runs the Harbor job and the model gateway, while 16 GH200s answered a small request in under 0.2 s. Profile the
worker (py-spy on its PID, `--threads`, and `--gil` for who holds the interpreter) before blaming the servers: what
looked like model latency was the gateway waiting for its own process.

## Evidence (2026-11-26, the vim transfer run's student-attempts stage, 8208 Qwen3-4B attempts, 512 at once)

Per trial (medians): 44 s from the trial's start to its environment setup, 0.7 s environment setup, 3.7 s agent setup,
5.2 s from the agent's start to its first model call, 6.8 s of model calls (of which 1.4 s to the first tokens), 1.6 s
verifier. Almost every Sven run logged `waiting for model 5 s (connecting)` (2133 lines over 1807 trials, 609 at
10 s). The worker held one core: py-spy found the gateway thread active 67% of the time and the job's main thread 42%.
Four defects, each fixed in `workers/harbor`:

- `_host_runtime` read `/proc/self/maps` and globbed a 2020-entry library directory on squashfuse for every sandbox
  command: 64% of the main thread's samples.
- One httpx client for every trial: httpcore 1.0's pool rescans every connection for each idle one whenever a request
  starts or ends. A load test (256 agents streaming 150-chunk answers) ran at 35.6 requests/s with headers after 5.6 s
  at the median through the gateway, against 314 requests/s and 2 ms to the upstream directly; 96% of the gateway's
  samples were in `_assign_requests_to_connections`. 64 pools, one per trial by name: 246 requests/s, 0.21 s.
- The gateway on a thread of the job's interpreter: with that interpreter half busy, 166 requests/s and 0.48 s; in
  four processes of its own, 291 requests/s and 0.014 s (at 512 agents one process fell to 270 requests/s, four
  reached the rig's 480).
- Harbor's job takes one lock on every trial's start and end and holds it while a thread writes the job's progress
  file, so each section costs a turn of a busy event loop and every start queues behind every other start and end:
  the 44 s above. Serializing at once and writing on a thread of its own: 0.97 s to 0.001 s from start to environment
  setup in a load test of 4000 no-op trials, 512 at once.
