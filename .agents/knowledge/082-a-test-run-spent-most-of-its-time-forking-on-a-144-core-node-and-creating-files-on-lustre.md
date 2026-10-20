# 082. A test run spent most of its time forking on a 144-core node and creating files on Lustre

`tokio::runtime::Runtime::new()` starts one worker per core: 144 threads per
`Context`. Every subprocess a campaign spawns (git, tar, the project's test
runner; thousands per practice run) is forked from that process through
`pre_exec`, so each fork paid for the thread stacks, and several contexts in
one test binary multiplied it. `splinter-pipelines`' `generations` test showed
24 min of system time in 25 min of wall time. The runtime now has 4 workers
(`RUNTIME_WORKERS`): its work is I/O, not compute.

Second cause: scratch trees under a Lustre `TMPDIR` make every file creation a
metadata round trip. The same test binary took 457 s with `TMPDIR` on Lustre
and 173 s with `TMPDIR` on tmpfs (`/dev/shm`, mode 700, removed afterwards).
Run the heavy pipelines tests with a RAM-backed `TMPDIR` where that is allowed.

Third: a spec that needs the same two generations in three assertions should
make them once (`uniform()` in `tests/generations.rs`) and the tasks stage
(about 60 % of a practice run: one real suite run per kill, reproduction,
variant and fix check) scales with `mutants_per_file`; one mutant per file
exercises the same paths.
