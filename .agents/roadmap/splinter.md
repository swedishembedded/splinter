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
alias is resolved once when a run starts.

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
2. **Import the prototype.** Done: `crates/splinter`, `crates/lab`,
   `experiments/tool-syntax` and `tasks/` here, and sven no longer carries
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
   `policy:<alias>` is pinned per context, so a REPL session keeps the
   release it first resolved until its own release or rollback; and a
   default-policy size chosen by measurement (the largest Qwen3 brain's
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
7. **Experience graph database.** In progress: `crates/expdb` is built
   (blob packs, segments, manifests, ingest, indexes, queries, training
   views, compaction, application entities, sessions, signals) and
   `splinter-store` has become `splinter-record`, which keeps sources, tasks,
   experiences, annotations and sets in it. Open: record runs as events with
   the cancel request as a signal; project each experience into the graph
   (attempt, one decision per trajectory step) so recipes, credit and
   counterfactuals apply to Splinter's own experience, and move the training
   views onto them; record datasets, training runs and models as lineage
   nodes with the snapshot each run read pinned; decide whether releases,
   datasets and suites stay files (adapters need a path); a `splinter state`
   command for compaction, collection and absorption; measure the cost of
   opening a large database per command.
