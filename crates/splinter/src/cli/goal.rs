// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `goal ...`.

use std::path::PathBuf;

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum GoalCommand {
    /// Admit documents into a selection of learning material. Public or
    /// simulated documents (`--open`) enter as they are. Recordings of real
    /// people (`--live`) enter only with the person's consent (`--consent`) and
    /// after the scrub of the current detectors (`--scrub` runs it): an
    /// unscrubbed, unconsented or still-identifying document is refused and
    /// reported, and the command then exits with an error. Admitted documents
    /// are written to `--out` as Markdown with their header.
    Admit {
        /// Documents with no person in them (a file each).
        #[arg(long, num_args = 1.., value_name = "FILE")]
        open: Vec<PathBuf>,
        /// Recordings of real people (a file each).
        #[arg(long, num_args = 1.., value_name = "FILE")]
        live: Vec<PathBuf>,
        /// The people agreed to the use of their recordings as training material.
        #[arg(long)]
        consent: bool,
        /// Scrub the live documents (secrets, addresses, accounts, contact data) first.
        #[arg(long)]
        scrub: bool,
        /// The directory the admitted documents are written to.
        #[arg(long, required = true, value_name = "DIR")]
        out: PathBuf,
    },
}
