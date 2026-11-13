<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Environment factory: objectives compiled into validated training environments

Scope: the environment factory (`splinter env`, `splinter-knowledge::factory`): verifiability classification, Harbor-format environment artifacts, the generator plugins, admission with a verifier red team, the base pass-rate band and the reward-versus-held-out divergence alarm (gaps 8, 10 and 26 of the recursive-improvement design report).

## Built (2026-11-13), CPU only, no weights

- **Verifiability classification** (`factory::objective`, `splinter env classify`): per skill, objective check, human evidence, calibrated judge or proxy, with the reasons; rules, not a model. The class travels with every reward (`splinter_core::verifiability::RewardProvenance`: class, verifier, environment, split, seed) and every verdict label (`verdict_label`: the weakest class forces `proxy only`, `judge-graded` or `human evidence`).
- **Artifact** (`factory::artifact`): a Harbor task folder (`task.toml`, `instruction.md`, `environment/`, `tests/test.sh` writing `reward.json`, `solution/solve.sh`) with `[metadata.splinter_env]`: schema version, environment id, plugin and version, verifier class, difficulty, split (`train` or `dev`; the type has no sealed-exam variant and parsing one is refused by name) and seed (as text: TOML integers are signed).
- **Plugins**: (a) `program-output-lua` and `program-output-python` (the interpreter is the oracle; one description rendered in Lua 5.1 and Python 3 plus an in-process evaluator, all three required to agree; knobs blocks, features, magnitude); (b) `db-workflow` (a shop database, a rule-enforcing tool, conditional requests including ones whose answer is to change nothing; verified by final-state diff of every table with a per-table reward component; `reset.sh`; knobs goals, traps, rows, distractor_tables).
- **Admission** (`factory::admission`): structural, reference-passes, nop-fails, robust, reproducible, novel, red-team and base-band gates; `PendingBand` until a base pass rate is recorded (`env band`), `Admitted` only inside the band (default 0.1 to 0.9, at least 16 attempts), `Rejected` outside it with the direction to turn the knobs, `Quarantined` when the verifier is beaten.
- **Red team** (`factory::attack`): empty, format-only, constant, cross-instance, harvested-from-the-solver's-view, static leak scan and tampering attempts (blank workspace, `exit 0` scripts, planted importable modules) for every environment, plus plugin attacks; deliberately weak verifiers (shape-only, constant, leaky, trusting a workspace script, importing from the working directory, passing or crashing on an empty attempt) are each caught in `crates/knowledge/tests/it/factory_admission.rs`.
- **Divergence alarm** (`factory::alarm::DivergenceAlarm`): trend over a window, `Diverging` when the training reward rises while the held-out measure does not, `Unconfirmed` when it rises with fewer than two held-out points. The campaign controller feeds it one observation per evaluation; wiring it into the controller is the controller's change.
- **Store and compiler** (`factory::store`, `factory::compile`): one `admission.json` and one `instances.json` per environment (files shared between instances stored once), a task folder staged only while a run has it; a difficulty ladder per skill; novelty against every instance ever drawn.

## Open

- [ ] **Base pass-rate bands on real weights are not measured.** `env band` records a measurement made elsewhere; nothing here asks a model. A closed-book attempter for the program-output environments (a model role answers the instruction, the answer is written to `answer.txt`, the verifier grades it) and a tool-using one for `db-workflow` (a sven episode on the staged folder) are the next step and need a card.
- [ ] **(c) Repository bug-injection plugin** (mutation of a repository with its tests; expect only a third to two fifths of attempted instances to validate): not built. The mutation machinery and the Harbor interchange exist in `knowledge::tasks::workspace`; the plugin must wrap them in the `Plugin` contract and the same admission and red team.
- [ ] **(d) Firmware through the emulator**: not built (the emulator is a read-only oracle; its command line is the verifier).
- [ ] **(e) Document-grounded questions**: not built (answered without the document; the reference grounded by quote).
- [ ] An interactive-human plugin (rubric plus a calibrated judge with a small human slice) is deliberately absent: such skills are classified `proxy`, `calibrated-judge` or `human-evidence` and listed as not built, so a verdict on them always carries its label.
- [ ] The campaign controller does not yet consume `compile`, the store or the alarm.
- [ ] Admission spawns thousands of short processes per environment; the cost is dominated by process start (namespaces), so a many-environment compile is minutes, not seconds.
