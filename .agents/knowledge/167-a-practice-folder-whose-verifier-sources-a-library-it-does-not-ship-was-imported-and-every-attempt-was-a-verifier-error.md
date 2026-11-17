<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 167. A practice folder whose verifier sources a library it does not ship was imported, and its attempts were verifier errors

**Lesson:** a Harbor folder's `tests/test.sh` may depend on files its producer stages beside it at run
time (a verifier library, a pristine snapshot, the agent's work exported by a collect hook). Copied as
committed, such a folder's verifier leaves no reward on any tree. Import must run the verifier on the
starting files and refuse the folder when it reaches no verdict there (`no_verdict`); a check that
"fails" only because it crashed is not a failing check, and the oracle cannot catch it when the
solution itself does not run on the host.

Found 2026-11-17 running `splinter tasks harbor episodes` on four committed train-split accounting
folders against a served Qwen/Qwen3-4B (2 attempts each, 4 jobs): all four were imported; 5 of the 8
attempts ended `verifier_error` and the other 3 `failed` before any verifier ran (the agent's change was
empty: it never wrote `answer.json`), so no attempt was graded by the task's own scorer. The folders' `test.sh` sources
`tests/bench/verify.sh` and a scoring library that only the producer's staging ships. After the fix
the same collection is refused before any attempt, naming the first folder and `NoVerdict`.
