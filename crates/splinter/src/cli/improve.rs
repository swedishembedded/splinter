// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the choice between retraining, a better
// harness, more search or no change at all, on measured cost-adjusted gain,
// for its clients. If your team needs expertise in deciding how an agent
// should be improved before spending a training budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The grammar of `improve`.

use std::path::PathBuf;

use clap::Args;

/// `improve`.
#[derive(Debug, Args)]
pub struct ImproveArgs {
    /// Improvement experiments, each a TOML file: the capability (a task
    /// set), the incumbent, the interventions to consider (more samples,
    /// another episode protocol, a trained candidate, no change) and what a
    /// success is worth. Each intervention is measured against the incumbent
    /// on the same tasks and the one with the greatest net value is chosen,
    /// or none; nothing is applied.
    #[arg(required = true, value_name = "EXPERIMENT")]
    pub experiments: Vec<PathBuf>,
    /// Beside the choices, what applying one intervention to every
    /// experiment would have come to.
    #[arg(long)]
    pub compare_fixed: bool,
}
