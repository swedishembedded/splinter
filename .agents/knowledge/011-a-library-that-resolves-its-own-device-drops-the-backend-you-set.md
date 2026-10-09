# 011. A library that resolves its own device drops the backend you set

Every "CUDA" speech measurement made through brain's SDK ran on Vulkan. The
SDK resolves a device set of its own before it builds a model, and that
resolution takes the backend the hardware probe prefers, so
`BRAIN_BACKEND=cuda` (which the CLI and a bare test binary honour) never
reached it. The `adapter:` line brain prints named the GPU either way, and
only `Gpu::kind()` said `wgpu`. A twelve-second answer was spoken in 20.4 s;
the same turn through the same code on CUDA, after the SDK was made to apply
the override, took 3.5 s to speak.

On CUDA the next costs were all one pattern: work that looked repeated to the
programmer did not look repeated to the backend, which captures a submission
as a graph only when it sees the same one twice. Sixteen per-position
tapes for the code predictor, and a decode step rebuilt with its position
baked in for the language model, never repeated, so each kernel was launched
alone (about a thousand per token) after two driver calls to allocate and
free its temporaries. Recording one tape with position uniforms and keeping
freed blocks took the code predictor from 25.5 to 11.2 ms per frame and an
80-token 8B answer from 2.3 to 1.4 s.

Check what a run executed on, not what it printed: ask `Gpu::kind()`, and
trace once with `nsys` before trusting a number (a Vulkan trace and a CUDA
trace look different at a glance: `vkQueueSubmit` against `cuLaunchKernel`).

A second lesson from teaching a persona to pronounce by talking to it: a
respelling said aloud ("Jeff-er-son") comes back from the recogniser as
"Jefferson", so a pronunciation can be taught by typing it, or by
recording the example, never by transcribing what was said.
