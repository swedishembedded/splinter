// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! The graph and probes of an earlier acquisition, taken instead of written.
//!
//! Planning and writing probes dominate a run's time before any training, and
//! two runs that are to be compared must be measured on one exam. A run given
//! `probes_from` reads the earlier run's recorded `graph` and `probes` stages
//! and goes on from there with its own baseline, rounds and assessment. The
//! exam is only valid for the material it was written from, so the earlier
//! run's sources must be exactly this run's.

use std::collections::BTreeSet;

use serde::de::DeserializeOwned;
use serde::Deserialize;
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_knowledge::skills::SkillGraph;
use splinter_knowledge::tasks::dedup::Seen;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::StoreError;
use splinter_store::runs::{read_run, Run};

use super::graph::{Planned, PlannedBy};
use super::probes::SkillProbes;

/// What an earlier run wrote.
pub(super) struct Reused {
    /// Its graph, as [`PlannedBy::Reused`].
    pub(super) planned: Planned,
    /// Its probes, every skill's.
    pub(super) probes: Vec<SkillProbes>,
}

/// The graph a run recorded, without how it was made.
#[derive(Deserialize)]
struct RecordedGraph {
    graph: SkillGraph,
    address: Digest,
}

#[derive(Deserialize)]
struct RecordedSource {
    source: SourceId,
}

fn refuse(run: &str, why: &str) -> OrchestratorError {
    OrchestratorError::Refused(format!("cannot take probes from run {run}: {why}"))
}

/// The summary of `run`'s stage `name`, read as a `T`.
fn stage<T: DeserializeOwned>(run: &Run, name: &str) -> Result<T, OrchestratorError> {
    let recorded = run
        .stages
        .iter()
        .find(|s| s.stage == name && s.error.is_none())
        .ok_or_else(|| refuse(&run.id, &format!("it recorded no '{name}' stage")))?;
    serde_json::from_value(recorded.summary.clone())
        .map_err(|e| refuse(&run.id, &format!("its '{name}' stage is unreadable: {e}")))
}

/// The graph and probes of acquisition run `from`, when it was written from
/// exactly `sources`. Every probe task must be in the task store, and each
/// probe's instruction is admitted to `seen`, as when probes are generated,
/// so no training task may repeat one.
pub(super) fn reuse(
    ctx: &Context,
    from: &str,
    sources: &[SourceId],
    seen: &mut Seen,
) -> Result<Reused, OrchestratorError> {
    let run = read_run(ctx.workspace(), from).map_err(|e| match e {
        StoreError::UnknownRun(_) => {
            OrchestratorError::Refused(format!("no run '{from}' to take probes from"))
        }
        other => other.into(),
    })?;
    if run.command != "acquire" {
        return Err(refuse(
            from,
            &format!("it is a '{}' run, not an acquire run", run.command),
        ));
    }
    let probes: Vec<SkillProbes> = stage(&run, "probes")?;
    let recorded: Vec<RecordedSource> = stage(&run, "sources")?;
    let written_from: BTreeSet<&SourceId> = recorded.iter().map(|s| &s.source).collect();
    if written_from != sources.iter().collect() {
        return Err(refuse(
            from,
            "its sources differ from this run's, and an exam is valid only for the material \
             it was written from",
        ));
    }
    let graph: RecordedGraph = stage(&run, "graph")?;
    for task in probes.iter().flat_map(|p| p.dev.iter().chain(&p.finals)) {
        let task = ctx
            .tasks()
            .get(task)
            .map_err(|e| refuse(from, &format!("a probe is not in the task store: {e}")))?;
        seen.admit(&task.instruction);
    }
    Ok(Reused {
        planned: Planned {
            graph: graph.graph,
            address: graph.address,
            by: PlannedBy::Reused {
                run: from.to_string(),
            },
            refused: Vec::new(),
        },
        probes,
    })
}
