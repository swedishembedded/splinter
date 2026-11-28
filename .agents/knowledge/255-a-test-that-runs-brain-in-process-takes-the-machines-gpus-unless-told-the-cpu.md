<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 255. A test that runs brain in process takes the machine's GPUs unless told the CPU

**Lesson:** the timeline tests of `splinter-model` train and serve brain's horizon models in process, and brain picks
its devices from its environment. Left alone it opens a wgpu device on every GPU the machine has, whatever
`CUDA_VISIBLE_DEVICES` says. On a node whose cards another job has filled, creating that device fails
(`request_device failed: OutOfMemory`, `Device(Lost)`) in a worker thread, or wedges past brain's 30 s bound, and the
test never ends. The test tiers need no GPU, so `make test` and `make test/crate/<package>` run them with
`BRAIN_DEVICE=cpu BRAIN_BACKEND=cpu`. A bare `cargo test` still reaches the GPUs. That stays so until brain takes
the device as a parameter of the call.

**Evidence (2026-11-28, a node whose four GH200s held a campaign's servers at 90 of 96 GB each):**
- A full-workspace `cargo test` hung in the `splinter-model` integration binary for 37 minutes. The binary passes
  alone.
- Six rounds of eight concurrent copies (16 test threads each, debug build, no device environment): every copy held
  some 790 MiB on GPU 0. Two copies did not finish within 300 s. The others failed on device creation (`OutOfMemory`,
  `Lost`, or the 30 s wedge bound).
- With `BRAIN_DEVICE=cpu BRAIN_BACKEND=cpu` (release build): two rounds of eight concurrent copies all passed, the
  slowest in 134 s. No test process appeared on any GPU.

**Procedure on a shared node:** run every test binary, diagnostic loops included, with `CUDA_VISIBLE_DEVICES=`
and `BRAIN_DEVICE=cpu BRAIN_BACKEND=cpu` set, until brain takes the device as a parameter of the call. Without
them, the stress loop that found this defect put eight test processes on GPU 0 while a campaign's servers held
the cards.
