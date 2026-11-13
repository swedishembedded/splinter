<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Environments from an objective

`splinter env` turns a learning objective into validated, versioned training
environments, without a hand-written pipeline. It answers three questions in order:
how could each skill's reward be established, can the environment be built so that
the reward cannot be earned without the skill, and is it at a difficulty a base
model can learn from.

```sh
splinter env classify --text "Predict the output of short Lua programs" --constraint lua
splinter env plugins
splinter env compile --file objective.txt --store envs --environments 4 --instances 4
splinter env list --store envs
splinter env show --store envs ENV-ID
splinter env band --store envs ENV-ID --model base --attempts 32 --passes 12
```

Without `--seed` a seed is drawn and printed (`no --seed given, using seed N`); pass it back to
reproduce the samples. Instances never repeat across compiles into one store: the store remembers
every instance it drew, so recompiling with the same seed is refused as not novel.

## Verifiability comes first

Each line of the objective is a skill. `env classify` assigns it one of four classes,
with the reasons, and the class is carried into every reward and every verdict:

| Class | A reward comes from | A verdict says |
|---|---|---|
| `objective` | a program: an interpreter, a test suite, a state diff | nothing extra |
| `human-evidence` | a person's own evidence (ratings, outcomes of real users) | `human evidence` |
| `calibrated-judge` | a judge model whose agreement with people was measured | `judge-graded` |
| `proxy` | a stand-in: a rubric nobody calibrated, a simulated user | `proxy only` |

A skill about people (a conversation, a tone) is a proxy unless the constraints say
real ratings or a calibration set exist; a simulator never proves real-world efficacy,
so no automatic environment is built for it and the skill is listed with that reason.
A skill about a document is only built when a source is given.

## What is built

An environment is a generator configuration: a plugin at a difficulty. Instances are
drawn from it by split and seed; the split is `train` or `dev`, and never a sealed
exam, which belongs to the measurement authority and is never produced here. An
instance is a Harbor task folder:

```text
task.toml          Harbor's keys, and [metadata.splinter_env]: schema version, environment id,
                   plugin and version, verifier class and verifier, difficulty, split, seed
instruction.md     what the solver is asked
environment/       the workspace (Harbor copies it to /app) and the Dockerfile
tests/test.sh      the verifier; writes /logs/verifier/reward.json, a flat object of numbers
                   in 0..=1 whose `reward` is the scalar and whose other entries are the reward vector
solution/solve.sh  the reference solution
```

Scripts address `/app`, `/tests`, `/solution` and `/logs/verifier` through `APP_DIR`,
`TESTS_DIR`, `SOLUTION_DIR` and `LOGS_DIR`, with the container paths as defaults, so the
same files run in a container and in admission on the host. An environment with state
beyond its files has `environment/reset.sh`, which rebuilds it.

The plugins, in the order they are built (`env plugins` lists their difficulty knobs):

| Plugin | Skill | Oracle |
|---|---|---|
| `program-output-lua`, `program-output-python` | predict what a generated program prints | the interpreter; the same program is rendered in both languages and a third evaluator, and all must agree |
| `db-workflow` | complete requests through a tool over a database, some of which are conditions whose answer is to change nothing | the final database state diffed against the expected state |
| `repository-bugs` | fix injected faults in a small Python library | a hidden test suite, run in a child process, with a canary test that must fail |

The repository plugin's fix is the library as it was, the same for every instance, so a model
that memorised the library solves any of them; its report says so, and admission does not
attack it with another instance's answer. Use a private or time-split repository for a claim
about generalisation.

Firmware through the emulator and document-grounded questions are classified and listed as
not built yet; their status is in the roadmap ledger.

## Admission

Every sampled train and dev instance is held to the gates; the report records each.

- `structural`: a valid artifact; the solver is shown neither tests nor solution.
- `reference-passes`: the reference solution is graded solved.
- `nop-fails`: doing nothing is graded, and not solved. A verifier that crashes writes
  no reward, which is never a grade of zero.
- `robust`: grading the same state twice agrees; a missing answer is graded, not a crash;
  reset restores the unsolved state and the reference solves it again.
- `reproducible`: the same request gives a byte-identical folder and the same rewards; a
  plugin's second oracle agrees.
- `novel`: instances differ from each other, dev from train, and from everything already
  trained on (the store remembers every instance it ever drew).
- `red-team`: see below.
- `base-band`: a base policy's measured pass rate on the dev instances lies inside the
  band (default 0.1 to 0.9, at least 16 attempts). Always failing or always passing
  leaves a group-relative trainer nothing to learn from.

An environment whose gates pass is `PendingBand` until `env band` records a
measurement; only an `Admitted` environment may enter a campaign. A measurement
outside the band rejects it and says whether to raise or lower a difficulty knob.

## The verifier red team

The verifier is written by the author of the generator, who knows the answer. Admission
therefore attacks it with attempts that know only what a solver knows, and an
environment whose verifier is beaten is `Quarantined`: its report keeps the attack, and
no instance is stored.

- an empty attempt; the right shape with scrambled content; constant answers (empty,
  `0`, `true`, `nil`, ...); another instance's answer;
- every line, number and file of what the solver is shown, tried as the answer, and a
  static scan for the answer standing in what the solver is shown;
- editing everything the solver may edit: blanking the workspace, planting `exit 0` in
  scripts, planting modules a verifier might import from its working directory that write
  the reward themselves;
- plugin attacks: corrupting, emptying or rewriting the database; a library that exits at
  import, prints forged result lines, patches the test framework to report success or skips
  everything.

An attack that earns reward always quarantines; guesses are counted as the share of
instances they beat, quarantining above 10 percent.

## Storage

An environment is two files in the store: `admission.json` and `instances.json`, which
holds every instance with the files they share (the tool, the verifier) stored once.
A task folder exists on disk only while a run has it staged, and is removed afterwards.

## The divergence alarm

During training, a campaign controller feeds one observation per evaluation (the
training reward and, when taken, a held-out measure) to
`splinter_sdk::knowledge::factory::alarm::DivergenceAlarm`. A reward that rises while
the held-out measure is flat or falls is a `Diverging` signal carrying the label its
verifier class forces; a rise with fewer than two held-out points is `Unconfirmed`,
never steady.
