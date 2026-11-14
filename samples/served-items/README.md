<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# served-items

Does the served path a learning experiment uses put exactly the item in front of the model, record every
call's outcome, and does an adapter trained by a served job reach serving unchanged? A learning null is
only evidence about learning once these hold.

The sample runs a list of steps against one `brain serve` started by Splinter's own host
(`splinter_sdk::served_host`, the lineage driver's server: the base, adapters followed from a directory,
training jobs on the same card):

- `answer`: items answered by the base or a named adapter through `ServedValidator` (sven's brain
  driver, no system prompt, thinking off, greedy), with the policy trace when `trace` is set (the prompt
  ids the model read, the generated ids, the policy's log-probabilities and the served identity);
- `train`: a supervised training job of the same server (`fine_tune_served`) on a chat dataset, its
  adapter hot-loaded and named;
- `adapter`: an adapter file offered under a name (refused unless the server lists its digest);
- `restart`: stops the server, so the next step runs in a new server session.

Items are a JSONL of `{id, prompt, answer, role}`: one user message and the JSON object that answers it.
A reply is read from its last fenced block (else the whole reply), parsed as JSON and compared with the
answer exactly (a list of strings compares as a set). Every call is written to `calls.jsonl` with its
class: `correct`, `wrong`, `malformed` (no JSON object), `truncated` (the output limit cut it),
`timeout`, `server_error` or `cancelled`; a failed call is never counted as a wrong answer. Each step's
summary (classes, seconds, the training job's report) is appended to `steps.jsonl`.

```bash
cargo run --release -p splinter-served-items -- \
    --brain <brain binary> --models-dir <model store> --model <base checkpoint> \
    --device gpu0 --kv-cache f32 --items items.jsonl --steps steps.json --out <directory>
```

```json
[
 {"answer": {"name": "base-1", "policy": "base", "items": "held_out"}},
 "restart",
 {"answer": {"name": "base-2", "policy": "base", "items": "held_out", "trace": true}},
 {"train": {"name": "fit", "records": "train.jsonl", "held_out": "held.jsonl", "steps": 60,
            "lr": 0.0002, "rank": 16, "alpha": 32, "grad_accum": 4, "seed": 7}},
 {"answer": {"name": "fit-train", "policy": "fit", "items": "train_pool"}}
]
```

`items` is a role of the items file or a comma-separated list of ids.
