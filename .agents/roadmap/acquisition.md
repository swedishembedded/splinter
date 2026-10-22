# Autonomous acquisition, intervention selection and generations - roadmap

Scope: autonomous acquisition of a capability from documentation (`learn "<objective>" --source`), the choice of an intervention by measured gain and cost (`improve`), and gains kept across generations (`generations`).

Gate C (a capability the policy lacks is acquired from documentation with no manual edits) and
Gate D (gains that survive three generations). What is built, what each measures, and what a
GPU run must still show. Update this file in the same commit that finishes or changes an item.

## Built, verified on scripted agents (CPU, no weights)

- **Teacher for repository tasks** (`practice --teacher`): see `repository-practice.md`, item 11.
- **Acquisition** (`splinter learn "<objective>" --source <url|path>`, `splinter-pipelines::acquire`):
  sources (URLs fetched once, HTML read as Markdown) -> skill graph (planner proposal admitted by
  code, else read by rule; versioned) -> probes per skill, dev and final halves, checked to
  exercise the skill and to repeat no other task -> baseline competence on the dev probes ->
  rounds on the skill of greatest expected benefit (prerequisites first) -> accepted only if the
  skill improved and nothing measured before fell past the bound -> assessment on the final
  probes against the policy the run began with. No second scheduler, task store or trainer: it
  composes the task generator, solve, verify, teach, frontier, dataset, train and the gate.
- **Lua 5.1 pin**: the `lua` runtime is Lua 5.1 and refuses another version
  (`splinter-sandbox`); `make deps/lua LUA_DIR=<dir>` builds the interpreter from its
  checksummed release tarball; `--runtime-dir <dir>/bin` puts it first for the runtime and for
  the workspace commands; `scripts/tasks/families.py check` refuses a `lua` that is not 5.1.
- **Intervention selection** (`splinter improve`, `splinter-pipelines::improve`,
  `splinter-eval::intervention`): per intervention, the paired gain per family against the
  incumbent on the same tasks, its lower confidence bound, its one-time and per-use cost, net of
  what a success is worth; the arm with the greatest net value above zero, else no change;
  unavailable mechanisms reported as not measured; the choices against every fixed strategy.
  Controlled test: three synthetic failure clusters (bad harness, missing knowledge, search beats
  training) and one where nothing helps.
- **Generations on the coding path** (`splinter generation run|matrix`,
  `splinter-pipelines::generations`): the protected panel (every earlier exam), the champion
  measured on all of it before training, replay policies (`none`, `uniform`, `balanced`,
  failure-targeted, rebuild from the base), candidate and champion measured at equal settings,
  rejection on a family retention limit, a mean regression limit and no gain on the new exam, the
  gate as the only thing that moves an alias, a ledger of everything each decision rested on,
  and the full retention matrix (the base as generation 0, every cell measured) with ACC, BWT,
  forgetting, forward transfer and learning gain.
- **Qualification harness** (`make qualify/list|test|run`, `scripts/qualify/`): the roadmap's
  matrix and the six mandatory ablations as one recorded command; the plan is registered (its
  digest written before the first check); BLOCKED is never a pass; an unmeasured assertion
  fails; an ablation states its measured effect.

## What GPU runs must show (hand-off to the run owner)

1. **Teacher**: `practice --teacher reference` on a repository whose policy solves little
   (the 8B base on the sqlglot family): the number of never-solved tasks that become verified
   demonstrations, the training records they add, and CE4 with and without them. Report n and
   the paired comparison; demonstrations are oracle-localised.
2. **Acquisition**: the Lua 5.1 manual (`https://www.lua.org/manual/5.1/manual.html`) with the
   pinned interpreter (`--runtime-dir`), kinds `construct-lua,predict-lua,debug-lua`, the 8B policy
   and the 27B as teacher: skills admitted, baseline, rounds, final assessment against the
   baseline; a skill it did not improve is reported.
3. **Intervention selection** on real failure clusters: one experiment file per cluster, the
   arms measured by solving (samples, protocol, a candidate trained with teacher demonstrations),
   `splinter improve --compare-fixed`.
4. **Three generations** on the E3 plan with `generation run`, then `generation matrix`; the
   replay arms (`--replay none|uniform|balanced`, `--rebuild`) on the same plan.
5. The qualification run: `make qualify/run OUT=<dir> VARS="--var ..."` once the registered
   results exist.

## Not done

- Task kinds for Lua exist as data only after this change (`construct-lua`, `predict-lua`,
  `debug-lua`); the generator's admission of Lua programs has been run only against scripted
  replies.
- `improve` measures search, retrieval and a decision model as unavailable until the mechanisms
  land (search: sven's facility); a bandit over interventions is later work.
- The retention matrix is measured in full each time it is asked for (an attempt at every
  protected task per generation); a cache keyed by model and task is not built.
- A generation's candidate is measured on the panel before the gate, and the gate then measures
  the new exam again: the attempts are not shared between them.
