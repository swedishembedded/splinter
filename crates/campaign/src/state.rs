// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the learning campaign that turns what an
// agent did into a better model, for its clients. If your team needs
// expertise in continual learning or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The state root's storage: what the experience database holds as files, and
//! the maintenance that keeps it small and quick to open.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::json;
use splinter_store::maintenance::{Maintained, Pin, Storage};
use splinter_store::recovery::{
    ArchiveOptions, Archived, Loss, RepairOptions, Repaired, Restored, StateVerify,
};
use splinter_store::workspace::Workspace;

use crate::context::Context;
use crate::error::CampaignError;
use crate::runs::{record, Recorded};

/// What `state status` reports.
#[derive(Clone, Debug, Serialize)]
pub struct StateStorage {
    /// The state root.
    pub state: PathBuf,
    /// What its experience database holds, as files.
    pub storage: Storage,
    /// What was written off as lost, in the order recorded.
    pub losses: Vec<Loss>,
    /// The snapshots held alive, by name.
    pub pins: Vec<Pin>,
}

/// The storage of the state root `ctx` works in.
pub fn storage(ctx: &Context) -> Result<StateStorage, CampaignError> {
    Ok(StateStorage {
        state: ctx.root().path().to_path_buf(),
        storage: ctx.workspace().storage()?,
        losses: ctx.workspace().losses()?,
        pins: ctx.workspace().pins()?,
    })
}

/// Merges small files, indexes the rest and retires finished writers; with
/// `collect`, also deletes what nothing reaches and is past its grace
/// period. Recorded as a run.
pub fn maintain(ctx: &Context, collect: bool) -> Result<Recorded<Maintained>, CampaignError> {
    record(
        ctx,
        "state maintain",
        &json!({ "collect": collect }),
        |_| Ok(ctx.workspace().maintain(collect)?),
    )
}

/// Checks the database and the artifacts it tracks; see
/// [`Workspace::verify`].
pub fn verify(ctx: &Context, deep: bool) -> Result<StateVerify, CampaignError> {
    Ok(ctx.workspace().verify(deep)?)
}

/// Recovers what verification finds; see [`Workspace::repair`].
pub fn repair(
    ctx: &Context,
    from: Vec<PathBuf>,
    accept_loss: bool,
) -> Result<Repaired, CampaignError> {
    Ok(ctx
        .workspace()
        .repair(&RepairOptions { from, accept_loss })?)
}

/// Packs the state into `file`; see [`Workspace::archive`].
pub fn archive(
    ctx: &Context,
    file: &Path,
    no_artifacts: bool,
    since: Option<PathBuf>,
) -> Result<Archived, CampaignError> {
    Ok(ctx.workspace().archive(
        file,
        &ArchiveOptions {
            no_artifacts,
            since,
        },
    )?)
}

/// Unpacks `archives` into the context's empty state root; see
/// [`Workspace::restore`].
pub fn restore(ctx: &Context, archives: &[PathBuf]) -> Result<Restored, CampaignError> {
    Ok(Workspace::restore(ctx.root(), archives)?)
}

/// What `state unpin` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Unpinned {
    /// The holder that let go.
    pub unpinned: String,
}

/// Releases the snapshot `holder` keeps alive.
pub fn unpin(ctx: &Context, holder: &str) -> Result<Unpinned, CampaignError> {
    ctx.workspace().release_pin(holder)?;
    Ok(Unpinned {
        unpinned: holder.to_owned(),
    })
}
