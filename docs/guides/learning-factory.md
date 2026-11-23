<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# The learning factory

A recipe in a task repository says what an experiment does: which tasks, which
agents on which models attempt them, which trials become training data, which
candidates are trained, and how each is judged. `splinter run` carries the whole
recipe out into a run directory, and runs it again without repeating anything:
every stage whose inputs did not change is reused.

```sh
cd bench                                   # the task repository
splinter generate collections/vim-headless-mastery.yaml
splinter validate recipes/vim-hosted-sft.yaml
splinter plan recipes/vim-hosted-sft.yaml
splinter run recipes/vim-hosted-sft.yaml --out runs/vim-hosted-sft
splinter status runs/vim-hosted-sft
splinter inspect runs/vim-hosted-sft --stage datasets
splinter inspect runs/vim-hosted-sft --stage rollouts/hosted --viewer harbor
splinter verify runs/vim-hosted-sft
splinter report runs/vim-hosted-sft
```

## What a run needs on this machine

- **The execution worker.** `make worker/install` in Splinter's checkout installs
  `splinter-harbor-worker` (Harbor's job API, pinned) into `workers/harbor/.venv`;
  put its `bin/` on `PATH` or set `SPLINTER_HARBOR_WORKER` to the program.
- **Connections** in `~/.config/splinter/connections.yaml` (or
  `$XDG_CONFIG_HOME/splinter/connections.yaml`): how each model is reached. A
  credential is named by the variable that holds it, never written in the file;
  the worker reads the variable when it routes the agent's requests.

```yaml
connections:
  brain: { protocol: openai, url: "http://127.0.0.1:8990/v1" }
  hosted: { protocol: openai, url: "https://example.invalid/v1", token: "env:SVEN_KEY" }
```

- **brain**, for a recipe that trains or evaluates: `--brain-endpoint URL` with
  `--brain-artifact-store DIR` and `--brain-adapter-dir DIR` (or
  `--brain-adapter-manifest FILE`).

## The recipe

```yaml
schema: splinter.factory/v1
name: vim-mastery
tasks: { collection: collections/vim-headless-mastery.yaml, split: train }
rollouts:
  - { name: local, agent: sven, model: "brain:Qwen/Qwen3-4B",
      environment: environments/vim/environment.yaml, attempts: 4, seed: 1 }
  - { name: hosted, agent: sven, model: "hosted:deepseek/deepseek-v4.1-flash",
      environment: environments/vim/environment.yaml,
      tasks: { unsolved_by: [local] }, attempts: 4, seed: 2 }
dataset: { name: sft, sources: [local, hosted], target: "brain:Qwen/Qwen3-4B", seed: 3 }
training:
  base: "brain:Qwen/Qwen3-4B"
  variants: [{ name: lora-r16, method: lora, params: { rank: 16, steps: 500 }, seed: 4 }]
eval: { plan: eval/vim-dev.yaml, agent: sven, baseline: "brain:Qwen/Qwen3-4B", candidates: [lora-r16] }
release: { promote: false }
```

Unknown keys are refused, every seed is written down, and `splinter validate`
reports every problem at once: a missing file, a model with no connection, a pass
that selects tasks unsolved by a pass after it, a credential written as a literal.

- **Collections** name a directory of Harbor tasks, and optionally the command
  that compiles them (`generate:`, which `splinter generate` runs).
- **Environments** name the sandbox (`docker`, `userns`, or `proot`, which is
  compatibility for containers without user namespaces and no security boundary:
  an environment asking for `isolation: strong` refuses it), resources, read-only
  mounts and the network.
- **Rollout passes** name an agent and its own argv (`args`, passed as given,
  never through a shell), the files those arguments refer to, an appended system
  prompt, environment variables (`env:NAME` passes this machine's variable), a
  model or none (the agent's own provider), attempts, retries and seed. A pass
  with `unsolved_by` runs only the tasks no earlier pass solved, frozen when it
  starts. A pass with `interactive: true` is played by people (below).

## What each stage guarantees

- **Rollouts.** Every attempt is graded by the one outcome classifier: a reward-0
  trial is a graded failure and is kept; a sandbox, model-endpoint, verifier or
  agent failure is retried up to `retries` times and then kept as what it was,
  never as the agent's 0. A killed controller resumes without repeating an
  attempt that reported. The trials, every attempt's report and the native Harbor
  output are in `rollouts/<pass>/`.
- **Dataset** (`native-sft`). Every verified success becomes one record: the
  conversation exactly as the agent's model was sent it - its own system prompt
  and tools - with every action supervised. Nothing is ranked, capped or
  rewritten; a trial excluded is listed in `excluded.jsonl` with its exact reason,
  and `report.md` shows the waterfall from raw trials to records. Validation
  holds out whole tasks (or task families). `projection: portable-sft` rebuilds
  each conversation from the trial's ATIF trajectory instead, for an agent whose
  model calls are not recorded (Claude Code).
- **Training.** Each variant is one brain job, resumable under a key derived from
  its identity; a candidate counts only once brain's server lists it. Changing a
  variant's parameters trains a new candidate and reuses the dataset and rollouts.
- **Evaluation.** A development plan runs a baseline arm (once, shared by every
  candidate) and a candidate arm under the same agent, arguments, environment,
  tasks, attempts and seeds, and decides by the exact one-sided paired sign test
  at the plan's alpha divided among the candidates. A protected plan hands the
  candidate to an evaluation authority and keeps its verdict only when the
  authority's signature verifies.
- **Release.** Records every decision; with `promote: true` it promotes the best
  improved candidate.

## Changing an experiment

A run refuses to continue on inputs that changed since it started. Change the
recipe and fork the run: every product the change leaves alone is reused.

```sh
splinter fork runs/vim-mastery --recipe recipes/vim-mastery-r32.yaml --out runs/vim-mastery-r32
splinter eval runs/vim-mastery --plan eval/vim-sealed.yaml --out runs/vim-mastery-sealed
```

`splinter cancel RUN` stops a run at its next stage or batch boundary;
`splinter resume RUN` continues it. `splinter export RUN --out run.tar.zst` writes
a bundle that `splinter verify` checks on another machine.

## Guided sessions

A pass with `interactive: true` is never run unattended. `splinter play` starts
one of its tasks on this terminal; a person works with the agent, the finished
session is graded like any trial and marked guided, and the run continues, so the
session reaches the dataset and what is trained on it:

```sh
splinter play runs/vim-hosted-sft --pass guided --task train/019-move-backward-by-word
splinter play runs/vim-hosted-sft --pass guided --task train/019-move-backward-by-word --continue 0
```

`--continue N` continues session N's conversation as a new session; the earlier
one is left as it is.
