// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The commands on the state root itself and on its recorded runs: what it
//! holds, its maintenance, archives, and the runs' records.

use std::path::PathBuf;

use clap::Subcommand;

/// `state ...`.
#[derive(Debug, Subcommand)]
pub enum StateCommand {
    /// What the experience database holds, as files.
    Status,
    /// Merge small files, index what is not indexed and retire finished
    /// writers; nothing stored changes.
    Maintain {
        /// Also delete files nothing reaches that are past their grace
        /// period; a snapshot a dataset pinned is never touched.
        #[arg(long)]
        collect: bool,
    },
    /// Let go of a snapshot a dataset (or another holder) keeps alive, named
    /// as `state status` lists it, so its files can be collected.
    Unpin {
        /// The holder's name.
        holder: String,
    },
    /// Check the database and every file it tracks, and report each one that
    /// is missing or damaged; exits 1 when anything is.
    Verify {
        /// Read every byte instead of only checking sizes.
        #[arg(long)]
        deep: bool,
    },
    /// Recover what verify finds: fill holes from copies (archives or other
    /// state roots), rebuild indexes, and with --accept-loss write off what
    /// no copy has.
    Repair {
        /// A copy to take missing files from: an archive file, or another
        /// state root. May be given more than once.
        #[arg(long = "from", value_name = "PATH")]
        from: Vec<PathBuf>,
        /// Give up on what no copy has: withdraw damaged database files and
        /// record lost artifacts in the ledger, which `state status` lists.
        #[arg(long)]
        accept_loss: bool,
    },
    /// Pack the database and the files it tracks into one archive; the same
    /// state always gives the same bytes.
    Archive {
        /// Where to write the archive (a .tar.zst).
        file: PathBuf,
        /// Leave the artifacts (adapters, datasets) out.
        #[arg(long)]
        no_artifacts: bool,
        /// An earlier archive: carry only what it lacks. Restore the result
        /// together with that archive.
        #[arg(long, value_name = "ARCHIVE")]
        since: Option<PathBuf>,
    },
    /// Unpack an archive (and the archives an incremental one was made
    /// after) into an empty state root, verifying everything first.
    Restore {
        /// The archive to restore, then any it builds on.
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
}

/// `runs ...`.
#[derive(Debug, Subcommand)]
pub enum RunsCommand {
    /// List the recorded runs.
    List,
    /// Show one run's record: arguments, stages, status, outputs.
    Show {
        /// The run's id.
        id: String,
    },
    /// Ask a run in progress to stop.
    Cancel {
        /// The run's id.
        id: String,
    },
}
