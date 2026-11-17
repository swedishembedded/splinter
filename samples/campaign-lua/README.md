<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# campaign-lua

Does a model teach itself a skill from programs it writes, and does the dev
gate decide correctly what each round did?

The sample's task family is Lua output prediction: a generated program is
shown, the policy writes what it prints, and the Lua interpreter that ran the
program is the verifier. Programs come in six levels (sums, products, strings,
loops, table sorts, long arithmetic); the training pool, the dev set the gate
is decided on and the retention set (levels 1 and 2) are drawn from one
recorded seed under different salts and share no program. Everything a run
asks of a model (sampling, the policy's proposals, training jobs) goes to a
`brain serve` through its API. The campaign controller that once ran this
family in process is retired: `lineage` below is the sample's driver, and the
controller's served path is `splinter_sdk::campaign::served`.

## The `lineage` driver: self-taught rounds

`lineage` runs one lineage of generations against the same family without the
campaign controller. Everything goes through a `brain serve` the driver starts
(serving, the policy's proposals, training jobs):

1. The current policy proposes Lua 5.1 programs from the manual, one section
   per request. Sections are chosen so that the library functions the policy
   answers worst on its dev set are asked about most, and a program that
   repeats an earlier proposal is dropped.
2. The policy predicts what each proposal prints (greedy); the interpreter
   runs it and its output is the label.
3. The mispredicted programs are the new information. A supervised training
   job continues the champion's adapter on them, with `--correct-share` of the
   draws on programs the policy already predicted right and `--replay-share` on
   earlier accepted rounds' records.
4. The candidate and the champion answer the dev and retention sets (the
   base's own held-out proposals and the generator's programs; sized by
   power-first planning before anything runs), and the dev gate decides
   (`--gate`, below). Only
   an advance changes the champion; the last advance of a generation is its
   champion, handed to an examiner when `--handoff` is given.

Two options change steps 2 and 3, both off by default:

- `--train-from base-cumulative` trains each candidate as a fresh adapter on the
  base, on the union of every accepted generation's training records and the
  new ones, instead of continuing the champion's adapter (`--replay-share` must
  be 0: the union already holds the earlier records). `--cumulative-epochs E`
  lets the steps grow with the union: `max(--steps, ceil(E x records /
  --records-per-step))`.
- `--frontier-samples K --frontier-temperature T --frontier-target N
  --frontier-request-cap C` select the frontier: the champion predicts each
  proposal K times, sampled at temperature T (instead of once, greedily), and
  only programs it did not predict right every time are trained on (the
  interpreter's output is the target; `--correct-share` must be 0). Proposing
  goes on in waves of `--proposals-per-round` programs until N frontier
  programs are found, within C proposal requests per round; the round fails
  with the numbers found when C is reached, and running it again with a larger
  C continues from the journalled waves and predictions. The round trains on
  the first N frontier programs, in the order proposed, whose training records
  fit `--max-block` tokens (measured as the trainer measures them); a longer
  one is skipped and listed with its length under `frontier.overlong` in the
  round's record, and the round fails when fewer than N fit.

Three more options, all off by default (with them absent, every request,
journal line, record and round record is what it was before them, so a lineage
that began without them continues unchanged; a lineage refuses to continue
under other values of them than its journal holds):

- `--frontier-admit boundary` admits a measured program only when at least one
  sample was right and at least one wrong (the learnable boundary); the
  default `any-wrong` admits every program with a wrong sample. The round's
  `frontier.admit` records a non-default rule.
- `--reasoning on --reasoning-max-tokens N` lets the champion think. Needs
  `--frontier-samples`. The frontier predictions are sampled with thinking on
  at `--frontier-temperature`, `--reasoning-top-p` (default 1) and
  `--reasoning-top-k` (default 0, no cut); the dev and retention exams are
  asked with thinking on too, decoded `--reasoning-gate greedy` (default) or
  `--reasoning-gate sampled --reasoning-gate-seed S` (the frontier temperature,
  seed `S` on every request). `N` bounds a whole reply, reasoning included,
  and replaces `--max-tokens`. The decoding is stated in the round's
  `reasoning` field and in the hand-off's serving terms (`"thinking": true`,
  `"decoding"`, `"max_tokens"`, and with a sampled exam `"temperature"`,
  `"top_p"`, `"top_k"`, `"seed"`). The candidate is then trained in thinking
  mode and its records measured in it.
- `--train-on verified-traces` trains each program that has a verified trace
  on one record: its prompt, the shortest verified trace as `reasoning_content`
  and the interpreter's output, formatted as in the answer-only record, as the
  answer. A trace is verified when its answer is whole (the token budget did
  not cut it) and the interpreter's output matches it; the shortest has the
  fewest characters of reasoning and answer together. `--frontier-target`
  then counts these records, and a program with no verified trace is not
  trained on. `verified-traces+labels` adds, for a program with none, the
  answer-only record with an empty reasoning (an empty think block in the
  Qwen3 template), trained in thinking mode too. `--train-on answers` (the
  default) is the interpreter-labelled records of before; with `--reasoning
  on` it trains them in thinking mode. The round's `frontier.records` counts
  traces and labels.

`traces.jsonl` holds, with reasoning on, one line per program and policy
measured: `samples`, `correct`, `exhausted` (replies the budget cut), the
length of each sample (`chars`, and `tokens` when the server reports its
usage) and `shortest`, the shortest verified trace (`reasoning`, `answer`,
`length`). It is written before the program's line in `frontier.jsonl`, which
is what a restarted round counts as measured, so no program is asked twice
and a program asked again after a stop keeps its later line.

`--gate evidence` replaces the dev gate of step 4 with the evidence gate,
which holds the false evolution rate (the share of accepted rounds that are
worse by `--gate-eps` or more) to `--gate-q` over every planned round
(`--generations` x `--rounds`). The dev set becomes a pool: each round draws
`--dev-draw` of its programs without replacement (fewer than it holds),
grades the champion and the candidate on them `--dev-batch` at a time (default
`--slots`), and advances only when the noninferiority e-process, with the
overlap correction `--gate-beta` for programs earlier rounds used, crosses the
round's level; otherwise the champion stays and the next round tries again.
Only drawn programs are graded, and the proposer is steered by the champion's
results on those alone. The rounds journal records the e-value, the level, the
budget left, the exposure and the programs spent, and a lineage refuses to
continue under another gate or other gate settings. The four settings have no
defaults (they are fixed before a lineage runs); the default `--gate greedy`
is the gate every registered lineage ran with. How to choose the settings is
in https://swedishembedded.com/docs/splinter/guides/acceptance-gates (the
manual's page on acceptance gates).

`--proposer-state DIR` moves the proposing handle's task records (one file per
record, disposable) off the project disk, for instance to a memory-backed
directory.

The lineage's journals (`rounds.jsonl`, `dev-results.jsonl`,
`predictions.jsonl`, `proposals.jsonl`, and with a frontier `waves.jsonl` and
`frontier.jsonl`, and with reasoning `traces.jsonl`) hold everything it knows; running the command again
continues from them.

```bash
cargo run --release -p splinter-campaign-lua --bin lineage -- --help
```

## Task families: acquiring several skills in one lineage

A lineage acquires one family, Lua 5.1 output prediction, unless `--family` is
given. A family (`campaign_lua::language::Language`) is a language, the
interpreter that is its verifier (pinned by SHA-256, refused when it is another
file), the manual the proposer reads, the proposer's wording, the checks a
proposed program passes (on top of the task generator's: ran to completion,
bounded output, the same output on a second run, no clock or random draws),
and the learner's own generator of development programs. Programs run under the
task sandbox: no network, bounded time, memory and output; Python runs with
`-I`.

| `--family` name | Language | Interpreter | Manual |
|---|---|---|---|
| `lua51` | Lua 5.1 | `--lua` | `--manual` |
| `python3` | Python 3 (extra admission: four reruns, no addresses in the output) | `interpreter=` pinned by `sha256=` | the Python 3 text documentation, `manual-dir=` |
| `zero-index` | Lua 5.1 with string positions from 0 and `#s` the last index | `dir=<installation>` | the dialect's generated manual |
| `plus-concat` | Lua 5.1 with `+` concatenating two strings and no `..` | `dir=<installation>` | the dialect's generated manual |

The dialects are built by `make deps/lua-dialects LUA_DIALECTS_DIR=<dir>` from
the pinned Lua 5.1.5 tarball and the patches under `scripts/deps/lua51-dialects/`;
each installation holds `bin/lua`, `manual/*.html` and `DIALECT.md`, the exact
definition. `make test` builds them into `target/lua-dialects`, where the dialect tests look.

```bash
lineage ... --train-from base-cumulative --frontier-samples 4 --frontier-temperature 1.0 \
    --frontier-target 2000 --frontier-request-cap 40 \
    --lua <lua> --manual <lua manual page> \
    --family lua51,at=1 \
    --family 'python3,at=2,interpreter=<python3.14>,manual-dir=<docs>,dev-generator=120' \
    --family zero-index,at=3,dir=<installations>/zero-index \
    --family plus-concat,at=4,dir=<installations>/plus-concat
```

`at` lists the generations whose rounds find the family's frontier (`at=1+3`
finds it again in generation 3: the same family's pool doubled, the control
that adds no new family); `target` is its frontier size (default
`--frontier-target`); `dev-generator`, `keep-generator`, `dev-proposals` and
`keep-proposals` size its development sets (default the lineage's flags). With
several families every generation trains one fresh adapter on the union of the
frontier pools found so far (`--train-from base-cumulative` is required), and
the dev and retention sets hold every family's items from the start, so the
gate's decision pools them. The round record's `families` lists, per family,
the proposer's novelty (new programs admitted per request) and the frontier
yield (frontier programs per proposal predicted), and `family_dev` each family's
dev accuracy under the champion and the candidate. Without `--family`, every
request, record and journal line is what it was before families existed.

`--admit-output semantic` (default `any`, which changes nothing) keeps programs
whose expected output has to be worked out rather than remembered: it refuses,
under the reason `memorisation`, an output with a number that has a fractional
part and more than 6 significant digits, an integer of more than 7 digits, an
object or address representation (`0x` and 6 or more hex digits, `object at`,
`<function`, `<class`, `<built-in`, `table: `, `function: `), two or more dunder
names (`__x__`), or an interpreter's version (`Lua 5.1`, `Python 3.`, a `3.14.N`
version, `sys.version`). The thresholds are constants of `campaign_lua::semantic`
and are recorded in every round record (`admit_output`); every refused program,
with its text, class and reason, is appended to `refused.jsonl` of the lineage.
A lineage refuses to continue under another output rule than it began with.

## The `exec-trace` tool: the interpreter's own reasoning

`exec-trace` runs every program of a JSONL file (a `program` field per line)
under Lua 5.1's debug hook and adds its execution trace: each executed line
with the locals and upvalues it changed, calls and returns of the program's
functions, and every line it prints. `--style raw` writes `[n] <source>` and
`name=value` lines; `--style steps` says the same events as short sentences.
A trace is deterministic (the same program gives the same bytes): values are
rendered without addresses and table keys in sorted order, a source line is
shown its first three runs and its later runs are counted, and a program is
stopped after two million VM instructions. A trace longer than `--max-chars`
is skipped and counted, never cut. The rules are stated in
`src/exec_trace.rs`.

```bash
cargo run --release -p splinter-campaign-lua --bin exec-trace -- \
    --lua lua5.1 --style raw --input programs.jsonl --output traced.jsonl
```
