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
splinter env compile --file objective.txt --repositories tasks --emulator /path/to/vemu --documents docs-dir --source docs-dir
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
| `repository-mutants` | fix faults mutated into real projects that have their own test suites (needs `--repositories`) | the project's whole suite, with the failing tests hidden, run in a child process with a canary |
| `repository-bugs` | the same on one small embedded library (the fallback when no repositories are given) | a hidden suite with a canary |
| `firmware-emulator` | fix faults in a bare-metal Cortex-M33 library (needs `--emulator`, `clang` and `lld`) | the emulator's console output against a reference model, on hidden inputs |
| `document-questions` | answer questions compiled from documents, without the documents (needs `--documents`) | a number computed from, or a value stated by, the document |

Repository tasks are split by repository (train and dev never share one) and firmware tasks by module;
document questions by document. The repositories are the original projects under `tasks/`, which no
model has seen; a claim about generalisation to other code still needs a private or newer repository. The
fix of a repository or firmware instance is the project as it was, so a model that memorised it solves
its instances: admission does not attack those plugins with another instance's answer, and the report
says so.

The repository verifier copies the workspace, lays the hidden tests over the copy, and runs the whole suite
in a child process that reads a nonce before importing anything; only the line carrying the nonce counts, the
number of tests that ran must equal the pinned count, a canary test must fail (a project that patches the test
framework to report success reports success for the canary too), and the reward is written by a parent that
imports nothing of the workspace. The firmware verifier builds a private copy with its own flags and linker script
against input seeds the solver never sees, so output printed from a table cannot pass. The emulator is run as a
command and never modified.

A derived document question carries the document's figures as premises and asks for a quantity computed from them;
a recall question asks for a stated fact, so only a policy that learned the document answers it and its base pass rate
is expected near zero (its band is measured on a policy that was trained on the document).

## Admission

Every sampled train and dev instance is held to the gates; the report records each.
The train and dev seeds come from one stream drawn from the admission seed. A seed for
which the plugin can make no valid instance (a project whose faults cannot be made to
fail a few tests, or whose suite gives different results on two runs) is skipped and the
next one drawn, at most eight per instance asked; the report counts them as
`skipped_seeds`, and a plugin that exceeds the bound fails admission.

- `structural`: a valid artifact; the solver is shown neither tests nor solution.
- `reference-passes`: the reference solution is graded solved.
- `nop-fails`: doing nothing is graded, and not solved. A verifier that crashes writes
  no reward, which is never a grade of zero.
- `robust`: grading the same state twice agrees; a missing answer is graded, not a crash;
  reset restores the unsolved state and the reference solves it again.
- `reproducible`: the same request gives a byte-identical folder and the same rewards; a
  plugin's second oracle agrees.
- `novel`: instances differ from each other, dev from train, and from everything already
  trained on (the store remembers every instance it ever drew). A fault-injected
  instance is judged on its symptoms and the lines its fault changed, so two faults
  in one project with the same short report are different tasks.
- `red-team`: see below.
- `base-band`: a base policy's measured pass rate on the dev instances lies inside the
  band (default 0.1 to 0.9, at least 16 attempts). Always failing or always passing
  leaves a group-relative trainer nothing to learn from.

An environment whose gates pass is `PendingBand` until `env band` records a
measurement; only an `Admitted` environment may enter a campaign. A measurement
outside the band rejects it and says whether to raise or lower a difficulty knob.

## Measuring the band

`env band` records a measurement; `make env/measure` makes one. Serve the model first (under the lock of
the card it runs on, held only while measuring), then:

```sh
make env/measure SPLINTER_BIN=splinter STORE=envs ENDPOINT=http://127.0.0.1:8798/v1 MODEL=Qwen/Qwen3-4B WORK=work/measure ENVS="env-0123456789abcdef"
```

Each environment's dev instances are exported as task folders (`splinter env export`), attempted eight
times each (`ATTEMPTS`), and removed afterwards. An environment whose solver has files only (program output,
document questions) is attempted closed-book through the endpoint, the task's own verifier grading the reply.
One whose solver has a shell (database workflows, repositories, firmware) is reported and not attempted
by this target; run its exported folders with `splinter tasks harbor episodes`. The pass rate is recorded
with `env band` and judged against the band.

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
