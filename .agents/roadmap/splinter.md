# Splinter roadmap

The end state and the phases that get there. Update this file in the same
commit that finishes or changes an item.

## End state

`splinter "<what to learn or do>"` on a default open-weight policy (a Qwen3
that brain both serves and trains, linked in-process):

| Intent | Pipeline |
|---|---|
| Learn a document | chunk, explore, extract facts and held-out probes, train, gate, release |
| Learn a command-line tool | a sven agent explores the tool in a sandbox with an execution tool only; the captured outputs become the source documents, with provenance, and go through the document pipeline |
| Ask, closed-book | a sven run with an explicitly empty toolset |
| Chat | a plain turn on the current policy |

Every accepted learning run trains a candidate from the current champion on
the new material plus a replay sample of everything learned before, and
releases it only if (1) closed-book held-out probes on the new material
improve, (2) retention holds on everything learned before, (3) the anchor
suite for general behaviour does not regress, and (4) the adapter loads in
plain `brain serve`. A release is an immutable adapter plus a manifest (base
model identity, lineage, dataset digests, evaluation); a moving default
alias is resolved once when a command starts (a REPL resolves it afresh for
each sentence).

Lineage runs both ways: an answer traces to an adapter, a training run,
examples and the source passage or command output; an example traces
forward to the behaviour it changed. sven supplies the ATIF trajectory,
brain the training record, Splinter the join.

## Acceptance

The three sentences in `README.md` run end to end on the default policy:
closed-book answers to held-out questions improve measurably after each
learning run, the tool explanation is checked against the tool's real help
output (which the policy never sees at answer time), retention holds, and
the released adapter answers the same questions from plain `brain serve`.

## Phases

1. **Repository.** Done: hooks, gates, cargo dependencies on the sven and
   brain remotes with a pinned lock, and a gitignored local override.
2. **Import the prototype.** Done: `crates/splinter`, `crates/eval`,
   `samples/tool-syntax` and `tasks/` here, and sven no longer carries
   them or any brain dependency. sven's own standalone task-and-verify
   sample is tracked in sven's `sdk-framework` roadmap, because it needs
   the facade's structured outcome and explicit toolsets first.
3. **Intent front door.** Done: a sentence is classified on the policy
   through sven's typed, bounded method call into candidate intents with
   confidences; code routes them - a confident single reading runs as its
   command, an ambiguous one is asked back, a destructive or network-opting
   one is never run on a guess. The command line is one verb per pipeline
   stage over content-addressed sets, with `learn` composing them.
   `release`, `eval` and `rollback` came with phase 4. Open: `lineage`.
4. **Continual policy.** Done: a candidate continues the champion's
   adapter and replays a seeded sample of every earlier release's training
   records; the four-part release gate (brain's paired sign test on the new
   held-out tasks, per-release retention bounds, a frozen versioned anchor
   suite, and plain `brain serve --adapter` answering with the candidate's
   digest and the in-process verdicts); immutable releases with canonical
   manifests; aliases moved by compare-and-set, resolved once per run;
   rollback along the lineage. Open: probes decode with the agent's
   sampling (temperature 0.2), so the serve check's verdict agreement can
   differ by sampling rather than by serving - the local provider should
   honour a request's temperature so probes decode greedily on both paths;
   and a default-policy size chosen by measurement (the largest Qwen3 brain's
   training path fits).
5. **SDK surfaces**, so Splinter uses only `sven-sdk` and `brain`:
   - brain: done for chat inference (messages, tool schemas, streaming,
     cancellation, tool-call parsing, policy identity), adapter
     train/export/save/load, resumable training with optimizer state and
     "not measured" distinct from zero - Splinter depends on `brain` alone.
     Open: base-weight residency shared by inference and training (folding
     an adapter into the base at load prevents it); constrained (JSON)
     decoding for a request's response format. `brain::promote` (the
     paired sign test the release gate uses) is exported only under the
     `decision` feature, which links the decision-model crates too; the
     `study` feature already links it through `brain-rl` and should export
     it. `ChatFineTuneOutcome` does not report the base digest its adapter
     card records, so Splinter hashes the base checkpoint again for a
     release manifest. Loading a checkpoint whose safetensors header length
     is garbage aborts the process on the allocation instead of returning
     an error. Preference pairs train: the `preference` view is written
     as brain's `generic-preference-v1`, validated by its parser, and
     `train` trains a preference dataset with `brain::PreferenceFineTune`
     (DPO against the model it continues, scored by brain's preference
     score on the held-out pairs). Also open: trainers that read a dataset
     file for the remaining objectives Splinter's views project - rewarded
     trajectories, raw text for continued pretraining, and contrastive
     pairs for the chat model (the SDK's contrastive fine-tuner trains one
     encoder architecture from in-memory pairs); until then those views
     are written only in Splinter's export-only format. A preference run
     replays nothing, since the preference trainer takes no replay set;
     `learn` trains supervised only.
   - sven: done for an empty default toolset, explicit toolsets, structured
     outcomes, bounded runs (cancel, deadline, token budget), parked
     questions, history taken from the session, and ATIF trajectories; every
     model run Splinter makes is bounded through `RunOptions`, and a local
     model's stream idle limit is engine configuration
     (`agent.stream_idle_timeout_secs`) carried on the model. Open: file
     tools rooted in a given directory, which workspace environments need
     (Splinter's environments are closed-book and runtime until then); and
     an ATIF trajectory as chat messages with each agent step's boundary
     kept, in the SDK - the step-to-message rendering sven has is internal
     and flattens the steps, so Splinter's views render trajectories
     themselves. Typed methods can now be bounded (`call_with`) and their
     schema crate is re-exported (`sven_sdk::schemars`); the task
     generators can move onto them.
6. **Extract sven's learning code**: the learning half of `sven-memory`
   (fact ledger, ingestion, assimilation, rule expansion, the submission
   drain, the brain study submitter, the doctor), the `learn` CLI, the
   learning configuration and its wiring in the runtime builder. (The
   learning design notes that lived in sven and brain are already in
   `.agents/research/`.)
7. **Experience graph database.** Done for the store: `splinter-store`
   keeps sources, tasks, experiences, annotations, sets
   and runs in `crates/expdb`; each experience is also an attempt in its graph,
   verdicts are ranked evidence about it, and datasets, training runs and
   releases record where they came from. Open: the training views still decide
   and render from whole experience values (a view reports why it left each
   candidate out, which a recipe does not); moving objectives onto recipes
   needs a mode that carries exclusion reasons. Releases, datasets, suites,
   answers, calibrations and the queue are documents and pointers in the
   database too; only adapters and dataset files stay plain files, tracked as
   artifacts, because brain needs a path to them. The database and its files pack
   into one archive and recover from copies. The
   read model is in memory, so opening a very large database costs time
   proportional to its size; `splinter state maintain` keeps it down.
8. **Layers, roles, pipelines and the SDK.** Done: the crates sit in tiers that
   `architecture.toml` states and `check-architecture` enforces (dependencies
   point down, forbidden pairs are unreachable at any depth, an exception that
   no longer applies fails, only the two adapters touch sven and only the model
   adapter touches brain, the command line and every sample depend on the SDK
   alone). The vocabulary (`splinter-core`) performs no I/O. Models play named
   roles decided by one rule and a run records who played each. `learn` is a
   pipeline of stages run by an engine that checks for a cancel and the budget,
   skips a stage that does not apply, and records every stage with its time and
   its failure; each sentence of a REPL resolves a policy alias afresh; a model
   reached over an API is asked several tasks at once. `splinter-sdk` is the
   embedding API. Open:
   - The SDK re-exports by layer; narrowing it to the types a caller needs, and
     giving each stage command a typed method on `Splinter`, is the next step.
   - The trainer (`BrainTrainer`) and the `brain serve` check sit in the
     pipelines crate because they are written against the pipelines' plan and
     the run's context; they move to the model adapter once they take plain
     arguments.
   - The typed stores that embed higher-tier types (releases, answers, datasets,
     candidates, lineage) live with the crate whose types they embed rather than
     in the store.
   - A `chat` intent at the router.
   - Trainers for the objectives whose records are only exported today
     (rewarded trajectories, raw text, contrastive pairs): the model adapter
     advertises what it can train (`TrainingCapabilities`), and a pipeline
     asks for an objective by name.
   - Resuming a pipeline from where it stopped: a stage's summary is recorded;
     its output is not yet enough to restart from.
9. **Any model, any dataset.** Splinter trains more than a chat LLM: a model
   brain provides for timeline (record) data is trained, measured and
   released by the same machinery. The design and the data findings are in
   the longitudinal health model research note; the health-specific parts
   live only in a sample. Done:
   - Record sources: SAS transport files (`knowledge::tabular`, held to a
     reference reader on real files) and per-variable HTML codebooks
     (`knowledge::codebook`).
   - Timeline datasets: `timeline-v1` format and manifest; a locked test and
     repeated grouped stratified cross-validation, pinned by digest
     (`data::partition`, the frozen ledger).
   - A metric gate: improvement with a paired interval and non-inferiority
     bounds on the rest, unmeasured failing (`eval::metric_gate`).
   - Terms travel with data (`core::terms`, the dataset manifest); the
     sample refuses to train on terms that do not permit it.
   - An intake agent: a typed call proposes a variable's mapping onto a
     concept quoting its codebook (`agent::mapper`), and code admits it
     (`knowledge::harmonize`: quotation present, documented refusal codes
     out of range, values inside the proposed range, distribution matching
     the concept's other sources). On the sample's ten NHANES cycles the
     rules admit every hand-checked mapping and stop every unit error and
     refusal-code leak; a variable put on a similar concept is not always
     stopped, so concept identity rests on the proposal.
   - `samples/lifecourse`: the NHANES timeline builder, the frozen
     benchmark, baselines and the deep model, cross-validation, the locked
     test, the pre-registered report, and secondary analyses (calendar
     shift, Venn-Abers intervals, seeded ensembles); the final model saved
     where brain serves it.
   - Longitudinal import and projection: a record file (`timeline-v1` fields
     plus interventions that say whether they were randomised) is imported as
     one immutable content-addressed episode per participant (`store::longitudinal`),
     with opaque keyed participant and group keys in place of identifiers and
     provenance on every item; `data::timeline_dataset` projects episodes to
     `timeline-v1` at a prediction point and `data::split` adds the temporal
     and leave-one-source-out splits and the leakage gates.
   - Calibration and serving correctness for a timeline candidate: units in
     a record file travel through the episode and the projection into the
     model's vocabulary (brain refuses another unit at prediction); the
     validation part is divided by group into early-stopping and calibration
     units, brain's Venn-Abers calibration is fitted on the latter and packed
     beside the weights, and its digest is in the release manifest and the
     lineage; the gate judges calibration on the calibrated risk where there is
     one (the evaluation record says which) and has nothing to judge where
     brain declared a horizon uncalibrated; serving correctness is measured
     from the shipped file on the test units (identity, batched against single
     patient-history forecasts, the share the support would withhold, validity
     of probabilities and curves). Each outcome code's metrics are brain's
     `TimelineModel::evaluate`; the all-cause union and the paired differences
     stay here.
   Open, in order:
   - brain gaps the timeline stage works around: no accessor for a model's absorbing codes (the
     scoring request is checked against the candidate's record instead), no
     calibrated-risk option in `evaluate` (calibration of the calibrated risk
     is computed here), and no all-cause view in `evaluate`.
   - `samples/lifecourse` still builds its subjects from the raw files
     directly; it moves onto the import (its builder produces the record file,
     the import and projection do the rest) with its frozen partition pinned
     by the split's address. Future measurements are not carried as forecast
     targets by the projection yet.
   - A trainer by capability: the pipeline's train stage still maps
     `timeline-v1` to no regime (the sample trains through the SDK); a
     timeline candidate, its release and the metric gate as its release
     check belong in the pipeline.
   - Experiments as records: cross-validation runs are JSON files beside
     the data, not experiences in the database.
   - The intake agent run with a served model over the sample's variables,
     scored against the hand mapping (`lifecourse intake --base-url`).
