<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 162. Brain trained ten times slower without cuBLAS on its library path, and said nothing

**Lesson:** a GPU run's throughput depends on libraries the loader finds at run time; check the measured
step time against the expected one before sizing a run, and put every library the trainer uses on the path
explicitly. A missing cuBLAS fell back to slower kernels without a warning.

Found 2026-11-16 in minimind on gpu3 (brain 16c2db09e and ed9deabfa, `brain/train pretrain` of the 2.1M Ouro,
64 x 512 rows): with only NVRTC prepended to `LD_LIBRARY_PATH` a step took 0.58 s (about 56k tok/s) at every
batch size; with the toolkit's `libcublas*` linked beside it, 0.054 s (about 600k tok/s). Every minimind run
before that pretrained at the slow rate. `samples/minimind/run.sh` now links cuBLAS beside NVRTC and refuses
to start brain when it cannot find it.
