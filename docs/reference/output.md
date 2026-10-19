<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Output and exit status

## JSON

With `--json` every command prints one JSON document on stdout; errors
print `{"error": string, "refused": bool}`. Commands that record a run add
`"run"` (its id) to their report. The reports in detail:

`source list`: `{"sources": [source]}`, each source
`{"id", "kind", "origin", "parts", "bytes", "captured_at"}` - `kind` is
`document`, `repository` or `command`; `origin` is the stored origin
(`{"kind": "document", "path"}`, `{"kind": "repository", "path",
"revision", "skipped"}` or `{"kind": "command", "argv", "cwd",
"exit_code", "timed_out", "stdout_truncated", "stderr_truncated"}`);
`parts` counts files or output streams and `bytes` sums their sizes.

`session add`: `{"sessions": [{"path", "source", "new", "steps", "redactions"}],
"refused": [{"path", "reason"}], "new", "redactions", "run"}`. `claims extract`:
`{"claim_set", "sessions", "proposals", "by_kind", "failed": [{"session", "reason"}],
"stopped", "run"}`. `claims gate`: `{"claim_set", "proposals", "ruled", "already_ruled",
"admitted", "superseded", "refused" (by reason), "rulings": [{"index", "session",
"kind", "statement", "outcome", "claim", "supersedes", "reason_code", "reason"}],
"live", "extraction_failed", "run"}`. `claims ledger`: `{"generation": {"epoch", "base", "retained", "current"}, "live": [claim],
"superseded": [{"claim", "statement", "by"}], "refused": [{"claim_set", "index",
"session", "statement", "reason_code", "reason"}]}`. A live claim carries `absorbed_by`, the release that first took it in, once one has, and `retained`,
whether the base of the policy's generation holds it. `claims ledger` also has `"generation"`:
`{"epoch", "base", "retained", "current"}`.
`absorb`: `{"policy", "generation", "roles", "intake", "extract", "claims", "live", "pending",
"rebuilt", "kits", "dataset", "replayed_records", "rehearsal", "candidate", "stance", "supervised", "gate",
"stance_rates", "release", "absorbed", "consolidation", "stopped", "dry_run", "release_asked",
"run"}`, each stage's report under its name; `consolidate`: `{"policy", "generation", "live", "closing",
"retained", "consolidated", "stopped", "run"}` where `consolidated` (also `absorb`'s `rebuilt` and
`consolidation`) is `{"mode", "champion", "from_epoch", "epoch", "from_base", "closing", "retained", "rebuilt",
"dead", "dataset", "replayed_records", "rehearsal", "steps_completed", "initial_loss", "final_loss",
"weights_digest", "gate", "release", "stopped"}`; `gate` is `{"claims": [{"claim",
"candidate": {"passed", "measured"}, "champion"}], "answered", "gained",
"unanswered", "regressed", "improvement", "retention", "anchor", "stance",
"serve", "passed"}`, and `anchor` and `stance` are absent when no suite is frozen
or no stance pass measured anything. `stance` is `{"requested", "cap", "dataset",
"path", "hold", "accept", "decline", "held_out"}`, each kind `{"items",
"held_out", "pairs", "written_rejections", "refused"}`; `stance_rates` is
`{"hold", "accept", "decline"}`, each `{"candidate", "champion", "counts"}` as
fractions with their counts, and absent where nothing was measured; `supervised`
is the candidate the preference pass started from, and `candidate` the one it
made. `claims ledger` adds `"stance"`: per release that absorbed a claim, the
`rates` its gate measured.

`status`: `{"state", "policy", "recent_runs", "counts", "concepts"}` - `policy` is
`{"reference", "model", "base", "adapter", "release", "generation", "generation_base"}` (`adapter` and
`release` are `null` until releases exist, `adapter` also for a consolidated checkpoint, which is the
base itself; `generation` is 0 on the configured base); `recent_runs` lists up to five runs as `{"id", "command",
"status", "started_at", "updated_at"}`; `counts` is `{"sources",
"task_sets", "tasks", "experiences", "experience_sets", "datasets",
"candidates"}`; `concepts` is `{"policy", "concepts", "measured",
"weakest", "queued"}`, each of `weakest` `{"concept", "current",
"releases"}` with `{"release", "graded", "passes", "rate", "since"}` per
release.

`doctor`: `{"checks": [{"name", "verdict", "detail", "fix"}]}`. It answers whether
a run can start here, before one does, and loads no model, uses no network and
creates nothing under the state root. The checks are `state-writable` (a probe file
is written and removed in the state root, or in its nearest existing parent when it is
not created yet), `free-space` (a failure under 1 GiB, a warning under 10 GiB: an
estimate for a small base, not a bound), `gpu` (a warning when no card has room
for the policy's weights right now, because brain then runs the model on the CPU
backend without a word, or when there is no GPU or its free memory cannot be read;
it counts the weights, not the activations a run adds) and `role:<name>` for each
role, which fails when the model the role would get does not resolve or its weights
are not on disk. `verdict` is `ok`, `warn`
or `fail`, `fix` is `null` for `ok`, and the exit status is 1 when any check fails.
`learn` runs the state-root checks itself before it starts: a state root that cannot be
written, or a disk with less than 1 GiB free, is refused (exit status 2) with the check's
fix, before anything is captured or generated. A `--dry-run` only plans and does not refuse.

### Maintenance and release reports


`state verify`: `{"database": {"manifests", "segments", "blob_packs",
"index_runs", "problems"}, "artifacts", "artifacts_unchecked"}`. `state repair`:
`{"filled", "rebuilt", "quarantined", "lost", "unresolved"}`. `state archive`:
`{"snapshot", "files", "carried", "artifacts", "bytes"}`. `state restore`:
`{"files", "artifacts", "verified"}`.

`release list`: `{"releases": [{"id", "created_at", "candidate",
"parent", "adapter_digest", "aliases"}]}`. `release`: `{"run",
"candidate", "alias", "champion", "gate", "release", "dir"}`, where `gate`
is `{"config", "improvement", "retention", "anchor", "serve", "passed"}`
and each check is `{"passed", "measured", "reason"}` (`measured` is `null`
when it could not be measured). `eval`: `{"model", "reference", "anchor",
"scores"}` (plus `"run"` when it froze a suite or graded a model).

## Exit status

0 done; 1 the work failed or stopped short of what was asked (a candidate
the gate blocked, an `absorb` that released nothing, a session file refused, a session the extractor gave nothing
usable for, a `doctor` check that failed); 2 refused
before anything ran (usage, an unknown id, a remote model without the
opt-in); 3 a sentence was asked back.
