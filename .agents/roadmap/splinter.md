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

1. **Repository.** Hooks, gates, cargo dependencies on the sven and brain
   remotes with a pinned lock, and a gitignored local override.
2. **Import the prototype** from sven's brain-linked samples (the loop agent,
   the learning lab, the task catalog, the tool-syntax experiment). sven keeps
   a standalone task-and-verify sample with no brain dependency.
3. **Intent front door.** A typed intent parse on the policy feeds a
   code-owned controller; ambiguity becomes a question, never a guess.
4. **Continual policy.** Champion lineage, replay, the four-part release
   gate, manifests, the default alias, and a default-policy size chosen by
   measurement (the largest Qwen3 brain's training path fits).
5. **SDK surfaces**, so Splinter uses only `sven-sdk` and `brain`:
   - brain: chat inference with messages, tool schemas, streaming,
     cancellation, tool-call parsing and policy identity; base-weight
     residency shared by inference and training (folding an adapter into the
     base at load prevents it); adapter train/export/save/load; resumable
     training checkpoints with optimizer state; "not measured" distinct from
     zero; serving an explicit release instead of the highest-numbered
     adapter.
   - sven: an empty default toolset, explicit toolsets, structured outcomes,
     cancellation, and history taken from the session rather than a lossy
     event stream.
6. **Extract sven's learning code**: the learning half of `sven-memory`
   (fact ledger, ingestion, assimilation, rule expansion, the submission
   drain, the brain study submitter, the doctor), the `learn` CLI, the
   learning configuration and its wiring in the runtime builder.
