// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every run records
// which model release it worked from, for its clients. If your team needs
// expertise in reproducible, auditable model training you can procure our
// services by sending an email to info@swedishembedded.com.

//! The policy a run worked with, as resolved when it started.

use serde::Serialize;
use splinter_core::release::ReleaseId;

/// The policy a run works with, as resolved when it started.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PolicyUsed {
    /// The alias.
    pub alias: String,
    /// The release it pointed at; `None` when it pointed at none, and the
    /// run works from the base.
    pub release: Option<ReleaseId>,
}
