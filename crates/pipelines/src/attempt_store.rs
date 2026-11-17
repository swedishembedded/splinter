// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! How a finished attempt becomes stored experience: the experience itself,
//! immutable, and beside it the facts of the attempt it does not carry. Every
//! stage that solves tasks stores its attempts this one way.

use splinter_agent::solve::Solution;
use splinter_core::annotation::{Annotation, AnnotationBody, Producer};
use splinter_core::experience::{ExperienceId, Provenance, Task};
use splinter_core::outcome::AttemptReport;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::ExperienceStore;

/// The producer name of the annotation that keeps an attempt's facts.
pub const ATTEMPT_PRODUCER: &str = "splinter-pipelines/attempt-report";

/// The version of that producer.
pub const ATTEMPT_VERSION: &str = "1";

/// Stores `solution` as an experience of `task` under `provenance`, and
/// `report` as its attempt report.
///
/// # Errors
///
/// The experience is not valid, or the store refused a write.
pub(crate) fn store_attempt(
    store: &ExperienceStore,
    task: Task,
    solution: Solution,
    provenance: Provenance,
    report: AttemptReport,
) -> Result<ExperienceId, OrchestratorError> {
    let experience = solution.into_experience(task, provenance)?;
    let id = store.put(&experience)?;
    store.annotate(&Annotation {
        experience: id.clone(),
        producer: Producer {
            name: ATTEMPT_PRODUCER.into(),
            version: ATTEMPT_VERSION.into(),
        },
        body: AnnotationBody::AttemptReport(report),
    })?;
    Ok(id)
}
