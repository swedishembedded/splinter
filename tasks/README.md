<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# tasks - the frozen catalog

One directory per task family, shared by whichever samples use it. Families
are here rather than under each sample because several samples measure
different learning angles on the same family, and a copy per sample would be two
catalogs that drift.

```
tasks/<family>/
  family.toml     the contract: request, predicates, limits, variation knobs
  workspace/      the template the agent gets a materialised copy of
  world/          the stateful service, reachable only through the workspace CLI
  witness/        an observation-only solver: proves the task is solvable
  reference/      the intended end state: an audit fixture, never shown
  negative/       known-bad solutions that the verifier must reject
```

## How the answer is kept out of reach

The interesting property of these tasks is that they cannot be solved by
reading. That has to be a fact about the system, not a hope about the agent.

The plan was to enforce it with a namespace sandbox, mounting only the
workspace. **That is not available on every machine this has to run on** - user
namespaces are denied in the container this was built in, there is no
`bubblewrap`, and no Docker daemon - and a boundary that only exists on some
machines is not a boundary.

So the property is arranged structurally instead: **during an episode, the
answer does not exist anywhere on the filesystem.**

- The hidden runtime state - which deployment is live, which units the device
  reports in, whether the write was committed - is chosen by the lab at run
  time and handed to the world process **over an inherited pipe**. Never a
  file, never `argv`, never the environment: `/proc/<pid>/cmdline` and
  `/proc/<pid>/environ` are readable by the same user, so both would be a
  file-shaped leak wearing a different hat.
- The world holds it in memory and answers questions about it. Reading every
  byte on the machine does not reveal it; asking the world does.
- The committed instance files therefore carry the knobs that are *legitimately*
  visible (directory layout, schema shape, which defect combination) and never
  the runtime state.

What this does **not** claim: that `reference/` and `witness/` are unreadable.
They are ordinary files in a checkout. That is deliberate and harmless - the
repair to a broken program is legitimately inferable from its source, and a
generic correct program that discovers the state at run time is exactly the
behaviour these tasks reward. What must not be inferable is the state itself,
and that is what the pipe buys.

`audit` checks the arrangement rather than assuming it: the materialised
workspace is scanned for any rendering of the episode's hidden state, and an
instance whose workspace contains its own answer fails before any model is
involved.

## Repository families

Eight small original projects, each learned by `splinter practice` from its own
tests. They are listed in `catalog.toml` (id, language, domain, size, test
count, content digest, files tasks may change); `python3 scripts/tasks/families.py
check` verifies baselines (five identical green runs), conventions, a report
under the generator's output cap, and that no two families define the same
top-level name. `make test/families` runs it with a fault pre-flight.

| family | language | domain |
|---|---|---|
| iniparse | Python | INI and TOML-subset configuration text |
| ttlcache | Python | LRU and TTL caches, injected clock |
| exprcalc | Python | arithmetic expressions: parse, evaluate, fold, differentiate |
| linediff | Python | line diff, patches, three-way merge, CSV |
| mdrender | Python | text wrapping and a small markup renderer |
| caldate | Lua 5.1 | civil calendar date arithmetic |
| graphkit | Lua 5.1 | graph algorithms |
| tokenbucket | Lua 5.1 | rate limiting, injected clock |

The Lua families are driven from Python `unittest` files (one `lua` process per
test), because fault seeding reads only `python3 -m unittest` per test.

### Regenerating the admitted task set

Fault generation is deterministic in the family tree and the flags; the seed
only chooses which families an exam would reserve. No model is loaded:
<!-- perf-number: a budget flag value, not a measurement -->
`--budget 1s` stops the run after the reserve stage.

```
cp -r tasks/<family> $WORK/<family> && cd $WORK/<family>
git init && git add -A && git commit -m base
splinter practice $WORK/<family> --state $STATE -v --budget 1s --max-tasks 1000 \
  --mutants-per-file N --no-release --exam-seed SEED \
  --file <each file under test from catalog.toml> --test-command python3 -m unittest
```

The `[tasks]` line of the output holds the admitted count and the refusals by
reason. Funnel measured with the release binary of the main tree (a quiet host
was not available; times are not comparable). `caught` is candidates minus
faults that did not run (`broken_build`) minus faults no test noticed
(`equivalent_mutant`); faults refused as `output_truncated` are inside it.

| family | language | files | tests | mutants/file | candidates | not run (broken build) | not caught (equivalent) | caught | refused after catch | admitted |
|---|---|---|---|---|---|---|---|---|---|---|
| iniparse | python | 11 | 279 | 80 | 828 | 281 | 64 | 483 | near_duplicate 207, output_truncated 99 | 177 |
| ttlcache | python | 9 | 257 | 80 | 701 | 187 | 96 | 418 | near_duplicate 252, output_truncated 7 | 159 |
| exprcalc | python | 12 | 256 | 80 | 916 | 235 | 118 | 563 | near_duplicate 280, output_truncated 96, weak_hidden_tests 1 | 186 |
| linediff | python | 9 | 278 | - | not measured | | | | | |
| mdrender | python | 9 | 270 | - | not measured | | | | | |
| caldate | lua | 8 | 250 | - | not measured | | | | | |
| graphkit | lua | 12 | 262 | - | not measured | | | | | |
| tokenbucket | lua | 10 | 296 | - | not measured | | | | | |

Seeds drawn per family (exam seed, `mutants/file` as in the table):

- iniparse: 6640021429841191778
- ttlcache: 3027157304785842583
- exprcalc: 5816860101842795554

Families without a row of numbers were still running when this was written;
their admitted counts are not measured.

### The E3 plan

`e3-plan.toml` assigns the eight families to the experiment: three generations of two
families each, each held to the reserved exam files of every earlier family (retention), and
two families sealed as the final exam. `catalog.toml` records each family's git `tree` id (the
tree of a snapshot of its files), which is how a run is recognised as reading a family.

```
splinter experiment check tasks/e3-plan.toml
splinter practice $WORK/<family> --plan tasks/e3-plan.toml --generation N ...
```

`experiment check` refuses a plan whose played families (trained or sealed) share a domain,
that schedules a sealed family in any generation's training or retention list, that trains a
family twice, or whose retention names a family no earlier generation trained. `practice
--plan` refuses, before the repository is taken in as a source, a sealed family always and a
family the plan does not train in that generation; it decides by the snapshot's tree id, so
renaming a family opens nothing. Not enforced: a repository that is a copy of a sealed family
with one file changed has another tree id; the plan seals the families as they are
catalogued.
