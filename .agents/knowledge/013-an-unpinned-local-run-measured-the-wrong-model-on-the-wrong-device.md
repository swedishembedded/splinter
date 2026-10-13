# 013. An unpinned local run measured the wrong model on the wrong device

A first measurement of task generation said an 8B generator admitted 2 of 43
proposals and asked questions about the document, not its author. It was neither
8B nor representative: `tasks generate` with no `--generator` plays the role with
`policy:default`, which is `Qwen/Qwen3-0.6B` (the smallest model, chosen so the
policy trains on one card), and the generator it recorded on every task said so
(`brain/Qwen/Qwen3-0.6B`). Naming the 8B reference fixed that and exposed two
more traps in the same command:

- With no device named, brain placed the 8B on the CPU (`adapter:
  brain-wgsl-cpu (Cranelift JIT, 48 threads)`) while both GPUs sat at 0%
  and 1 MiB. The run was not stopped, only slow, and the machine's load average
  went past 200. `samples/jefferson/gpu-only.sh` stops a run on the first line
  that names the CPU backend and exits 4 if no discrete GPU was reported.
- Pinned to a GPU, the 8B ran out of device memory at load (`wgpu error: Out of
  Memory`) because its default context is the checkpoint's largest. A context
  limit in the reference (`local:Qwen/Qwen3-8B@16384`) is what fits beside the
  weights on a 24 GiB card.

Pinned and limited, the 8B generated for 3,015 s over four texts: 32 requests of
73 s each (2,338 s in the model), 8 candidates per request, 101 tasks admitted of
about 240 proposals. The 0.6B run had admitted 2.

How to avoid it: read the generator identity on the tasks before drawing
conclusions from them; name the model for every role of a measurement run; run
every model job through the pinned wrapper; carry the context limit in the model
reference.
