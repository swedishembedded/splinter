<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# tool-syntax - does verified experience teach a small model to look before it acts?

**Status: measured baseline, verified demonstrations, trained adapters, and a
before/after measurement on CUDA.** On the frozen request the base model solves
0 of 80 episodes and the adapter trained on six verified demonstrations solves
80 of 80, each in the four tool calls of the demonstration; a second training
seed solves 39 of 40. That is on the task the adapters were trained on (see the
limits). Every number below is from a run whose command and log are named in
[Reproduce](#reproduce).

## The question

`config-discovery` asks for three retry attempts on the upload service this
host runs, without changing what any other deployment does. Which deployment
that is cannot be read anywhere - it is chosen per episode and held in the
service's memory - so the task cannot be started without asking. The agent has
to run `./svctl status` (or another look) before it edits anything; an agent
that guesses is right on half the instances by chance and never solves a
matched pair.

## The design decision, made before measuring

**The agent is offered sven's coding tools under sven's default deferred
disclosure, not the full tool set and not a hand-picked three.**

* The request carries the full definitions of the core tools (`shell`,
  `read_file`, `write_file`, `edit_file`, `find_file`, `grep`, `todo`) and the
  `tools` tool; every other tool is a one-line index entry in the system
  prompt. That is 8 definitions and a 3428-token prompt, against the 5646
  tokens of the full set the first version of this sample measured.
* It is what a deployed `sven` runs, so the result is about the agent a user
  gets. A rig of three hand-picked tools would be a configuration chosen for
  this result.
* It keeps `task` and `context` - the most complex tools there are - out of
  the model's face on a task that needs neither (the first version's baseline
  reached for `task`).
* The reason the first version gave for narrowing, a training example too long
  for the device, no longer applies; the full set is no longer ruled out by
  hardware and is simply not the deployed default.

Both arms see exactly this prompt: the same system prompt, tool definitions,
frozen request, no reasoning block (the mode the adapter is trained for) and
the same sampling temperature.

## Pre-declared, before any training ran

- The measured arms use the **frozen request**. Hints are for collection only
  and are recorded in the report.
- An episode that never reaches a verdict is an error, excluded from both the
  numerator and the denominator - never a zero. An episode that spends the
  family's tool-call budget (30 calls) or its 600 s is cancelled and scored on <!-- perf-number: the episode time limit, a value the sample enforces -->
  what the workspace holds, which is a failure of the model, not an error.
- An arm is scored only against a server proved to apply promoted adapters:
  `measure --adapter-digest` refuses to run unless the server's model listing
  names the model id with exactly the digest of the adapter that was trained.
- A demonstration that the verifier does not accept produces no training data.
- Train and held-out data are separated by wording, and the numbers below say
  which they are (see the limits).

## What was measured

Qwen3-0.6B (resident `brain/qwen3`), sven's coding agent, the frozen request,
no hint, no reasoning block, sampling temperature 0.2 (sven's default) with the
seed `S + i` sent for episode `i`. Episodes alternate between the two twins
(`staging` and `production` live), so each arm holds the same number of each.

| arm | answered (n) | solved | errored | solve rate, 95% interval | tool calls per answered episode |
|---|---|---|---|---|---|
| witness (observation-only solver, no model) | 1 | 1 | 0 | - | 3 |
| baseline, first replicate (other server and binary, context 12288) | 59 | 0 | 1 | 0.0 %, [0.0 %, 6.1 %] | 4.83 (median 4, max 31) | <!-- perf-number: a measured rate of the run this README reports -->
| **before**: the base model | 80 | 0 | 0 | 0.0 %, [0.0 %, 4.6 %] | 2.54 (median 1, max 30) | <!-- perf-number: a measured rate of the run this README reports -->
| **after**: the base plus the trained adapter | 80 | **80** | 0 | **100 %, [95.4 %, 100 %]** | 4.00 (every episode exactly 4) | <!-- perf-number: a measured rate of the run this README reports -->
| after, second training seed (12; 40 other episodes, seed 5000) | 40 | 39 | 0 | 97.5 %, [87.1 %, 99.6 %] | 4.20 (median 4, max 9) | <!-- perf-number: a measured rate of the run this README reports -->

"Before" and "after" ran at the same time against one server that serves
the base and, as a model of its own, the adapter (so the only difference is
the model id asked for), with the same seeds (episode `i` of both arms has seed
`3000 + i`) and 40 episodes per twin each. The interval is Wilson's. `errored` episodes never reached a verdict (only in
the first replicate: the server refused a 12414-token request on a 12288-token
context) and are in neither the numerator nor the denominator.

**What the base model does.** In none of the 139 answered episodes of the two
base arms did it run `./svctl` - the program that says which deployment is
live - or read a file under `config/` (checked on the transcripts of both arms
that hold the counts: 80 and 59 episodes). The "before" arm made 150 `shell`
calls, 33 of the `tools` tool (looking for tools it already had), 4 of
`system`, 2 of `task` and 1 of `edit_file`; the shell calls are guesses such as
`echo '3 retries' | sudo tee -a /etc/hosts` (refused: no privileges) and
`retry ...` (not a command). It acts without finding out.

**What the trained model does.** In all 80 episodes it ran `./svctl status`
first, read `config/<live deployment>.json`, wrote the change to that file and
ran `./svctl validate`, then answered: 160 `shell`, 80 `read_file` and 80
`write_file` calls, the demonstration's shape in its own wording. The base model
on the same server, with the same seeds, solved none, so the behaviour is the
adapter's; and `measure` refused to start the arm until the server's listing
named the adapter's model id with the digest of the file that was trained
(`sha256:147f1b91...`).

**A second training seed, and a probe of the wording.** The same recipe with
seed 12 (selected step 10, held-out loss 1.65 to 0.45, token accuracy 77.5 % to
90.4 %) gave an adapter that solves 39 of 40 frozen-request episodes (the table;
the miss is a production episode that read the file, tried an `svctl` command
that does not exist, validated, and never wrote the change). To see whether the
behaviour follows the task or only the request's exact words, the same server
was also asked with the request replaced (`--request`, a probe, not a measured
arm) by *"Set the upload service's retry attempts to 3 on whichever deployment
this machine is serving. Leave every other deployment exactly as it is."*: the
seed-12 adapter solves 40 of 40 (Wilson 95 per cent interval 0.912 to 1.000) and
the base model 0 of 40 (0.000 to 0.088), 40 episodes each with seed `5000 + i`.
<!-- perf-number: measured rates and token accuracies of the runs this README reports -->

### The data and the training

`demonstrate` drives the real tools (`sven tool call`) through four
differently worded demonstrations per twin - ask which deployment is live, look
at its file (some also show the effective configuration), write the change,
validate it - and keeps one only if the task's own verifier accepts the
workspace: **8 of 8 verified**. Only the choice of action is scripted
(`Scripted` provenance: it can never count as the model improving on its own);
the prompt, tool schemas, observations and verdict are real. The first three
wordings of each twin (6 records, 32 supervised answers) are training data; the
fourth (2 records, 12 answers) is held out and never trained on.

Each record is one dialogue: system prompt, request, then per action an
assistant turn carrying one call (`call_0`, `call_1`, ... unique in the
dialogue), the tool message that answers exactly that call, and the closing
answer. Every assistant turn is supervised; brain's trainer encodes the record
one example per answer, each rendered as the model is asked for it.

Checked the way the data audit checked the corrupted data
(`verify_records.py`, logs `verify/*.verify.txt`): in all 44 examples (32 train,
12 held out) each tool result answers exactly the call before it, the context of
example `k` holds exactly `k` calls and `k` results in alternating order with
the results' texts equal to the observations, and **the trainer's context token
count equals the server's `prompt_tokens`** for the same messages and tools
(3398 to 3715 tokens, 44 of 44 equal).

The same check on what the agent really sends: the 70 requests the "after"
arm's first 14 episodes sent (rebuilt from their transcripts) render to the
server's `prompt_tokens` exactly (70 of 70), as do 57 of 60 requests sampled
from the "before" arm; the other 3 are rejected by the trainer's encoder
because the base model's turn was split into a text message and a call message
and the template renders such a context differently with and without the
generation prompt. One known difference between training and serving remains:
`sven tool call` prints a shell result without the trailing newline the agent
sees (`active deployment: production` against `active deployment: production\n`),
so the shell observations of the training data differ from the ones the agent
shows the model by that trailing newline; the adapter was measured as it is.

Recipe (run `train1`; `train2` is the same recipe with seed 12): LoRA rank 16, alpha 32, all targets, fp32 base,
no-think mode, learning rate 3e-4 with brain's warmup and decay, 4 examples per
optimizer step, a budget of 24 steps, the held-out examples scored every 2
steps, stop after 3 evaluations without improvement, the best evaluation
exported, seed 11. CUDA, card 0 of a GH200, block 3731 tokens, trainer memory
about 10.8 GiB, 730 row tokens per second (profile: `profile-tsyn.md`).

| step | 2 | 4 | 6 | **8** | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| training loss (mean since last evaluation) | 1.29 | 0.95 | 0.55 | 0.50 | 0.08 | 0.24 | 0.20 | 0.14 |
| held-out loss | 0.80 | 0.67 | 0.49 | **0.49** | 0.53 | 0.61 | 0.61 | 0.61 |

Held-out loss before training 1.65 (token accuracy 77.5 %), at the exported <!-- perf-number: a measured rate of the run this README reports -->
step 8 0.49 (90.2 %). Past step 8 the training loss collapses (0.08) while the <!-- perf-number: a measured rate of the run this README reports -->
held-out loss rises: the run starts to memorise, which is why the adapter of
step 8, not the last one, is the one measured. Step 8 is chosen on the held-out
loss, so that loss is a selected figure, not an unbiased estimate.

### Limits of what this shows

- **Same task, same two twins.** The measured episodes are the family the model
  was trained on, with a fresh world each; only the wording of the
  demonstration differs between train and held-out. This measures whether the
  behaviour can be taught at all on this task, not that it transfers to another
  task, other tools or other files.
- **Two training seeds, one recipe.** Both adapters were measured (80 of 80 and
  39 of 40); a different recipe, base model or amount of data was not tried.
- **Small n.** The intervals above are the uncertainty about the rate, not about
  anything else, and the episodes of one arm are not independent of the
  server's state.
- **Nothing here is a retention or anchor measurement.** What the adapter does
  to behaviour outside this task was not measured.
- Sampling is seeded per episode but whether the server's seeded sampling
  reproduces a run exactly was not checked.


## What changed since the first version

The first version of this sample stopped before a before/after number; both of
its stated blockers are gone.

- **Training no longer hits a device limit.** The first version could not train
  at the length of sven's full prompt (a 1945.6 MiB tensor against wgpu's
  2 GiB binding cap). Training now runs through brain's CUDA backend, which has
  no such cap; the prompt of the measured agent is 3428 tokens and the longest
  training example 3731.
- **Multi-turn trajectories train.** The first version could only supervise the
  first decision, because Qwen3's template renders an assistant turn
  differently in isolation than in context. brain's `ChatFineTune` now encodes
  a dialogue one example per answer, each as the serving renderer renders the
  same messages when the model is asked for that answer, so every decision of
  a demonstration is supervised in the context it will be asked in; the check
  above shows the trainer and the server agree to the token.
- **The harness measures an agent that can use its tools.** The agent is built
  with sven's coding toolset (see the design decision), with no reasoning
  block, per-episode sampling seeds and the family's tool-call budget; sven
  exports one agent step per model response with distinct call ids.
- The first version's 0/6 baseline and its hint experiment came from an earlier
  harness; they are superseded by the table above and nothing in this
  repository reproduces them.


## Reproduce

Everything below runs from a built `splinter-tool-syntax` (CUDA through
`--backend cuda --device gpuN`) and a `brain serve` of Qwen3-0.6B; `sven` must
be on `PATH` or named by `SVEN_BIN` for `demonstrate`. The run directory must be
short (a unix socket path holds 107 bytes), and a model drives real shell
commands in each episode workspace, so run it where it can write nothing you
care about. The logs of the runs reported above are in the lane's run
directory `runs/tsyn/` (not in git; the reported runs used the training seeds
11 and 12 and the episode seeds 1000, 3000 and 5000 as given in each report): `arms/*/log.txt` and `report.json` (every
episode: hidden state, outcome, tool calls, seed), `data/demonstrate.log`,
`train1/attempt/` and `train2/attempt/` (`summary.json`, `training.json`), `verify/*.verify.txt`,
`profile-tsyn.md`; the scripts that ran them (`serve.sh`, `measure.sh`,
`pipeline.sh`, `pipeline2.sh`, `sandbox.py`, `verify_records.py`, `summarise.py`) sit beside
them.

The reported runs used the binary built against brain 83e2f44a3 (the trainer) and
sven f0cfb82c, a `brain serve` of the shared release build, and the sven binary
of f0cfb82c for `demonstrate`; the repository's lock has since moved to a later
brain, which the sample builds against unchanged but which the numbers above
were not re-run on.

```bash
BIN=target/release/splinter-tool-syntax          # cargo build --release -p splinter-tool-syntax
$BIN audit                                      # the catalog checks itself: no model, no GPU, no network
SEED=$(shuf -i 1-4294967295 -n 1); echo "seed $SEED"   # draw it and keep it: episode i samples with SEED+i

# the base model, served; the arm is named by the resident id, never the store spelling
brain serve --openai 127.0.0.1:8801 --models-dir MODELS --backend cuda --device gpu0 \
  --api-keys-out keys.json --checkpoint MODELS/Qwen/Qwen3-0.6B --ctx 24576 --max-batch 4
$BIN baseline --instances 40 --seed "$SEED" --run-dir RUN/baseline --model brain/qwen3 \
  --base-url http://127.0.0.1:8801/v1 --api-key KEY --report baseline.json

SVEN_BIN=path/to/sven $BIN demonstrate --run-dir RUN/data \
  --prompt-from RUN/baseline/requests.json --weights MODELS/Qwen/Qwen3-0.6B
$BIN --backend cuda --device gpu0 train --data RUN/data --base MODELS/Qwen/Qwen3-0.6B \
  --out RUN/train --seed "$SEED" --steps 24 --grad-accum 4 --eval-every 2 --patience 3

# the adapter is a model of its own on a server that also serves the base
brain serve ... --adapter RUN/train/adapter.safetensors --adapter-targets all --adapter-rank-cap 16
$BIN measure --arm trained --instances 40 --seed "$SEED" --run-dir RUN/trained \
  --model ADAPTER_ID_FROM_/v1/models --adapter-digest sha256:DIGEST_FROM_summary.json \
  --base-url http://127.0.0.1:8801/v1 --api-key KEY --report trained.json
```

`measure` refuses to run unless the server's model listing names `--model`
with exactly that adapter digest. `make samples/tool-syntax/audit` runs the
audit through make.
