// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving model pipelines that keep what
// earlier training taught while they learn more, for its clients. If your team
// needs expertise in continual learning and forgetting control you can
// procure our services by sending an email to info@swedishembedded.com.

//! The datasets earlier generations trained on, replayed by a later one.
//!
//! A run chained after another starts from the earlier run's champion, and
//! trains each round beside a replay of the datasets of the rounds it
//! accepted. Without the earlier generations' datasets in that list the
//! later generation forgets what they taught. `replay_from` names the earlier
//! runs; a run's datasets are the ones its own `replay` stage seeded (so a
//! chain needs only the previous run) followed by those of its accepted
//! rounds. A run or dataset that cannot be found is refused, never skipped.

use serde::Deserialize;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::StoreError;
use splinter_store::runs::{read_run, Run};

use crate::datasets::resolve_dataset;

fn refuse(run: &str, why: &str) -> OrchestratorError {
    OrchestratorError::Refused(format!("cannot replay from run {run}: {why}"))
}

/// What the `replay` stage of a run recorded.
#[derive(Deserialize)]
struct Seeded {
    datasets: Vec<String>,
}

/// The part of a round's record that decides what it contributes.
#[derive(Deserialize)]
struct RoundRecord {
    outcome: Outcome,
    dataset: Option<String>,
}

#[derive(Deserialize)]
struct Outcome {
    outcome: String,
}

/// The datasets `run` trained on, in order: those it replayed, then those of
/// its accepted rounds.
fn datasets_of(run: &Run) -> Result<Vec<String>, OrchestratorError> {
    let mut found = Vec::new();
    for stage in run.stages.iter().filter(|s| s.error.is_none()) {
        if stage.stage == "replay" {
            let seeded: Seeded = serde_json::from_value(stage.summary.clone())
                .map_err(|e| refuse(&run.id, &format!("its 'replay' stage is unreadable: {e}")))?;
            found.extend(seeded.datasets);
        } else if stage.stage.starts_with("round-") {
            let round: RoundRecord =
                serde_json::from_value(stage.summary.clone()).map_err(|e| {
                    refuse(
                        &run.id,
                        &format!("its '{}' stage is unreadable: {e}", stage.stage),
                    )
                })?;
            if round.outcome.outcome != "accepted" {
                continue;
            }
            let dataset = round.dataset.ok_or_else(|| {
                refuse(
                    &run.id,
                    &format!(
                        "its accepted '{}' recorded no dataset (it predates replay)",
                        stage.stage
                    ),
                )
            })?;
            found.push(dataset);
        }
    }
    Ok(found)
}

/// The datasets of the accepted rounds of `from`, in order, once each. Every
/// run must be an acquire run in this state root and every dataset must be
/// stored.
pub(super) fn seed(ctx: &Context, from: &[String]) -> Result<Vec<String>, OrchestratorError> {
    let mut seeded: Vec<String> = Vec::new();
    for name in from {
        let run = read_run(ctx.workspace(), name).map_err(|e| match e {
            StoreError::UnknownRun(_) => {
                OrchestratorError::Refused(format!("no run '{name}' to replay from"))
            }
            other => other.into(),
        })?;
        if run.command != "acquire" {
            return Err(refuse(
                name,
                &format!("it is a '{}' run, not an acquire run", run.command),
            ));
        }
        for dataset in datasets_of(&run)? {
            let stored = resolve_dataset(ctx, &dataset)
                .map_err(|e| refuse(name, &format!("dataset {dataset} is missing: {e}")))?;
            let id = stored.id.to_string();
            if !seeded.contains(&id) {
                seeded.push(id);
            }
        }
    }
    Ok(seeded)
}
