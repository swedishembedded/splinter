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
3. **Intent front door.** A typed intent parse on the policy feeds a
   code-owned controller; ambiguity becomes a question, never a guess.
4. **Continual policy.** Champion lineage, replay, the four-part release
   gate, manifests, the default alias, and a default-policy size chosen by
   measurement (the largest Qwen3 brain's training path fits).
5. **SDK surfaces**, so Splinter uses only `sven-sdk` and `brain`:
   - brain: done for chat inference (messages, tool schemas, streaming,
     cancellation, tool-call parsing, policy identity), adapter
     train/export/save/load, resumable training with optimizer state and
     "not measured" distinct from zero - Splinter depends on `brain` alone.
     Open: base-weight residency shared by inference and training (folding
     an adapter into the base at load prevents it); constrained (JSON)
     decoding for a request's response format; serving an explicit release
     instead of the highest-numbered adapter. Also open: trainers that read
     a dataset file for the objectives Splinter's views project beyond SFT -
     preference pairs (DPO), rewarded trajectories, raw text for continued
     pretraining, and contrastive pairs for the chat model (the SDK's
     contrastive fine-tuner trains one encoder architecture from in-memory
     pairs). Until then those views are written only in Splinter's
     export-only format.
   - sven: done for an empty default toolset, explicit toolsets, structured
     outcomes, bounded runs (cancel, deadline, token budget), parked
     questions, history taken from the session, and ATIF trajectories. Open:
     the runner still races its own timeout, interrupt and limits around
     `send`; bounding the run with `RunOptions` keeps the kernel state and
     reports the conclusion instead of dropping the turn. Also open: file
     tools rooted in a given directory, which workspace environments need
     (Splinter's environments are closed-book and runtime until then); and
     an approval gate that names the tool call it gates - `HumanGate`
     carries only the capability and a prose prompt, so the solver can
     approve a capability, not a tool. Also open: typed model-driven
     methods (`Method`, `Engine::call`) that a run can bound - a call
     takes no `RunOptions` (deadline, output-token budget, cancel) - and
     whose derived schema a consumer can satisfy without a direct
     `schemars` dependency pinned to sven's; until then the task
     generators ask through the closed-book solve and parse the reply
     strictly, with no repair turn. Also open: an ATIF trajectory as
     chat messages with each agent step's boundary kept, in the SDK - the
     step-to-message rendering sven has is internal and flattens the steps,
     so Splinter's views render trajectories themselves.
6. **Extract sven's learning code**: the learning half of `sven-memory`
   (fact ledger, ingestion, assimilation, rule expansion, the submission
   drain, the brain study submitter, the doctor), the `learn` CLI, the
   learning configuration and its wiring in the runtime builder. (The
   learning design notes that lived in sven and brain are already in
   `.agents/research/`.)
