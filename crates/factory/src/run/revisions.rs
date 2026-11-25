// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning experiments whose every result
// names the exact code, data and hardware that produced it, for its clients.
// If your team needs expertise in attributable, reproducible machine-learning
// runs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A run's revisions: the build a run pins when it starts and holds every
//! resume to, the record contract every server it trains with must check,
//! and the revision manifest of each stage that trains or evaluates.

use std::path::Path;

use splinter_core::factory::identity::StageIdentity;
use splinter_core::revision::{
    InputDigest, ModelDigests, RecordContract, RunRevisions, ServedRevision, StageRevisions,
    REVISIONS_FILE, REVISIONS_SCHEMA,
};

use super::{online, read_json, write_json, Factory, RunError, Services};
use crate::ops::names;
use crate::resolve::Resolved;

impl Factory<'_> {
    /// The revision manifest of the stage `identity` names, with the models
    /// and tokenizers its servers reported for it.
    pub(crate) fn stage_revisions(
        &self,
        identity: &StageIdentity,
        models: Vec<ModelDigests>,
    ) -> StageRevisions {
        StageRevisions {
            schema: REVISIONS_SCHEMA.into(),
            build: self.services.build.clone(),
            training_records: RecordContract::pinned(),
            served: self.served.clone(),
            models,
            inputs: identity
                .inputs
                .iter()
                .map(|(input, digest)| InputDigest {
                    input: input.clone(),
                    digest: digest.clone(),
                })
                .collect(),
            configuration: serde_json::json!({
                "operation": identity.operation,
                "parameters": identity.parameters,
                "seed": identity.seed,
                "implementation": identity.implementation,
            }),
        }
    }
}

/// Pin the revisions of the build that starts the run in `run`, or hold a
/// resume to them: a build of other commits is refused by name unless the
/// recipe allows mixed revisions.
pub(super) fn pin(
    run: &Path,
    build: &splinter_core::build::BuildInfo,
    policy: crate::graph::RevisionPolicy,
) -> Result<(), RunError> {
    let path = run.join(REVISIONS_FILE);
    let now = RunRevisions::of(build);
    if !path.exists() {
        return write_json(&path, &now);
    }
    let started: RunRevisions = read_json(&path)?;
    let changes = started.changes(&now);
    if changes.is_empty() || policy == crate::graph::RevisionPolicy::Mixed {
        return Ok(());
    }
    Err(RunError::RevisionsChanged(changes))
}

/// What the trainer's servers say they are, for a run that trains or
/// evaluates: a server whose training-record contract is not the one this
/// build writes is refused by name before any stage runs; one that reports
/// none trains with a warning (its contract recorded as unknown) unless the
/// recipe requires the contract.
pub(super) fn served(
    resolved: &Resolved,
    services: &Services<'_>,
) -> Result<Vec<ServedRevision>, RunError> {
    let Some(trainer) = services.trainer else {
        return Ok(Vec::new());
    };
    let uses = |operation: &str| resolved.plan.of_operation(operation).next().is_some();
    let writes_records = uses(names::TRAIN) || uses(online::ONLINE);
    if !writes_records && !uses(names::EVALUATION) {
        return Ok(Vec::new());
    }
    let served = trainer
        .served_revisions()
        .map_err(|e| RunError::ServerUnreadable(e.to_string()))?;
    if writes_records {
        let required = resolved.compiled.contract == crate::graph::ContractPolicy::Required;
        for server in &served {
            server
                .check_contract(required)
                .map_err(RunError::ContractMismatch)?;
            if server.training_records.is_none() {
                eprintln!(
                    "warning: the brain server at {} reports no training-record contract; this build writes {} ({}), and the run records the server's as unknown (a recipe that needs it says `contract: required`)",
                    server.endpoint,
                    splinter_core::revision::RECORD_CONTRACT,
                    splinter_core::revision::RECORD_CONTRACT_DIGEST
                );
            }
        }
    }
    Ok(served)
}
