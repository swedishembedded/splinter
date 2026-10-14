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
splinter learn examples/first-run/manual.md --goal "the QX-7 console, boot, power and watchdog" --budget 40m
```

This trains on a GPU. The report opens with an `outcome` block that keeps five things
apart: whether the run finished, whether a candidate was trained, what the release gate
decided, whether a release was written, and whether `policy:default` moved. A run can
train a candidate that the gate then blocks; that is a result, not a failure of the tool,
and the policy stays as it was. Exit status is 0 only when a release was written.

## 5. Inspect and use the result

```bash
splinter status
splinter ask "At what baud rate does the QX-7 console run?"
splinter ask "Which header starts the QX-7 in recovery mode?"
```

`ask` is closed-book: the answer comes from the model, not from the manual. Compare it
with `manual.md`, and with `splinter ask ... --policy <base>` before the release, to see
what changed.

## What was verified when this was written

Steps 1 to 3 were run as written. Step 4 is a real training run whose result depends on
your GPU, your model and the sampled tasks; this directory does not promise an
outcome, and does not ship expected numbers.
