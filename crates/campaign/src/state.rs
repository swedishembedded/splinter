// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the learning campaign that turns what an
// agent did into a better model, for its clients. If your team needs
// expertise in continual learning or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The state root's storage: what the experience database holds as files, and
//! the maintenance that keeps it small and quick to open.

use std::path::PathBuf;

use serde::Serialize;
use serde_json::json;
use splinter_record::maintenance::{Maintained, Storage};

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
}

/// The storage of the state root `ctx` works in.
pub fn storage(ctx: &Context) -> Result<StateStorage, CampaignError> {
    Ok(StateStorage {
        state: ctx.root().path().to_path_buf(),
        storage: ctx.workspace().storage()?,
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
