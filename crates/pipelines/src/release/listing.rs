// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! `release list` and `rollback`: what has been released, and moving an
//! alias back along the lineage it came by.

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::releases::ArtifactKind;

/// One release, as `release list` shows it.
#[derive(Clone, Debug, Serialize)]
pub struct ReleaseLine {
    /// Its id.
    pub id: ReleaseId,
    /// When it was released.
    pub created_at: String,
    /// The candidate it was.
    pub candidate: String,
    /// The release it was trained from.
    pub parent: Option<ReleaseId>,
    /// What kind of file it is.
    pub kind: ArtifactKind,
    /// Its file's digest: an adapter's, or a full checkpoint's.
    pub artifact_digest: Digest,
    /// The aliases pointing at it.
    pub aliases: Vec<String>,
}

/// What `release list` reports.
#[derive(Clone, Debug, Serialize)]
pub struct ReleaseList {
    /// Every release, oldest first.
    pub releases: Vec<ReleaseLine>,
}

/// Every release under the state root, verified, oldest first.
pub fn list(ctx: &Context) -> Result<ReleaseList, OrchestratorError> {
    let store = ctx.releases();
    let aliases = store.aliases()?;
    let mut releases = Vec::new();
    for id in store.list()? {
        let release = store.get(&id)?;
        releases.push(ReleaseLine {
            aliases: aliases
                .iter()
                .filter(|(_, target)| **target == id)
                .map(|(name, _)| name.clone())
                .collect(),
            created_at: release.manifest.created_at,
            candidate: release.manifest.candidate,
            parent: release.manifest.parent,
            kind: release.manifest.artifact.kind(),
            artifact_digest: release.manifest.artifact.content_digest().clone(),
            id,
        });
    }
    releases.sort_by(|a, b| (&a.created_at, &a.id).cmp(&(&b.created_at, &b.id)));
    Ok(ReleaseList { releases })
}

/// What `rollback` reports.
#[derive(Clone, Debug, Serialize)]
pub struct RolledBack {
    /// The alias.
    pub alias: String,
    /// The release it pointed at.
    pub from: ReleaseId,
    /// The release it points at now: the one `from` was trained from.
    pub to: ReleaseId,
}

/// Points `alias` at the release its current one was trained from.
pub fn rollback(ctx: &Context, alias: &str) -> Result<RolledBack, OrchestratorError> {
    let store = ctx.releases();
    let Some(from) = store.alias(alias)? else {
        return Err(OrchestratorError::Refused(format!(
            "alias {alias} points at no release; there is nothing to roll back"
        )));
    };
    let Some(to) = store.get(&from)?.manifest.parent else {
        return Err(OrchestratorError::Refused(format!(
            "release {from} is the first {alias} has had; there is no previous release to roll \
             back to"
        )));
    };
    store.move_alias(alias, Some(&from), &to, &ctx.clock().utc_now())?;
    ctx.repin_policy(alias);
    Ok(RolledBack {
        alias: alias.into(),
        from,
        to,
    })
}
