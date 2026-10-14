# First run: teach a model a document it has never seen

`manual.md` describes a device that does not exist (the Quillon QX-7), so no
model can know its console speed, boot jumper, power limits or watchdog from
training. Splinter reads the manual, writes grounded questions about it, has the
model attempt them, trains an adapter on the verified answers, and releases it
only if the release gate passes. You can then ask the model what the manual says.

Run every command from the repository root, after `make build`. Nothing here
needs a network.

## 1. Check that a run can start

```bash
splinter doctor
```

It checks that the state directory (`~/.splinter`) is writable with room to spare and
that the policy model resolves to weights on disk. Fix whatever it marks `FAIL`
(it prints the fix) before going on. The default policy is `Qwen/Qwen3-0.6B` in
Brain's model store (`BRAIN_MODELS_DIR`, else `~/.local/share/brain/models`).

To keep the tutorial out of your real state, add `--state /tmp/first-run` to every
command below.

## 2. Freeze the anchor suite (once)

```bash
splinter eval --suite anchor --freeze examples/first-run/anchor.jsonl
```

`anchor.jsonl` holds twenty general-knowledge questions. The release gate asks them
before and after training, and refuses a candidate that got worse at them.

## 3. See the plan without running it

```bash
splinter learn examples/first-run/manual.md --goal "the QX-7 console, boot, power and watchdog" --dry-run
```

It prints the sources, the models that play each role and the stages, and writes
nothing under the state directory.

## 4. Learn

```bash
splinter learn examples/first-run/manual.md --goal "the QX-7 console, boot, power and watchdog" --kinds denoise --budget 40m
```

This is a real run on a GPU. The report opens with an `outcome` block that keeps five
things apart: whether the run finished, whether a candidate was trained, what the release
gate decided, whether a release was written, and whether `policy:default` moved. A run can
train a candidate that the gate then blocks; that is a result, not a failure of the tool,
and the policy stays as it was. The exit status is 0 only when a release was written.

What the options do, as observed when this example was written (one run each, so a
pilot and not a measurement):

- **Without `--kinds`** a planner model chooses the task kinds. On a one-page manual and a
  0.6B policy it chose kinds the manual cannot support, the plan failed its own checks and
  the run stopped with `no plan was made`. Name the kinds instead.
- **`--kinds recall`** has a model write questions from the manual. The default setup uses
  the 0.6B policy as that writer, and both questions it proposed were rejected by the
  admission rules (one never named its subject, one named something the manual does not
  contain), so no task was admitted and the run stopped. The rules did their job: nothing
  that fails them is trained on. Name a stronger writer and teacher with `--generator` and
  `--teacher`. One that brain has to place on the CPU backend (`brain doctor --model <it>`
  says so) is too slow to be usable: its requests timed out.
- **`--kinds denoise`** builds fill-in-the-blank tasks from the manual's own text in code,
  with exact answers, so it needs no writer model. One short document gave one task. The
  0.6B policy answered it wrong on every attempt and so did the 0.6B teacher, so no task was
  worth training on; `learn` stopped with `no candidate was trained`, the gate did not run
  and the policy was unchanged.

To reach a trained candidate you need more material than this page (several documents give
several tasks) and a teacher that can solve what the policy cannot, running on a GPU.

## 5. Inspect and use the result

Whatever `learn` reported, `status` shows what the policy is now. If a release was written:

```bash
splinter status
splinter ask "At what baud rate does the QX-7 console run?"
splinter ask "Which header starts the QX-7 in recovery mode?"
```

`ask` is closed-book: the answer comes from the model, not from the manual. Compare it
with `manual.md`, and with `splinter ask ... --policy <base>` before the release, to see
what changed.

## What was verified when this was written

Steps 1 to 3 were run as written, and step 4 was run on a GPU with each of the three
options above; the observations are the ones listed there. No run reached a trained
candidate, and this directory does not promise one or ship expected numbers.
