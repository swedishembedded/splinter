// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements machine configuration that a team can edit
// from the command line and trust to stay valid, for its clients. If your
// team needs expertise in configuration tooling, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `config ...`: the machine's one settings file.

use clap::{Args, Subcommand};

/// `config ...`.
#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Print the file's path and its effective content; a missing file shows
    /// the empty default.
    Show,
    /// Print one value. KEY is a dotted path, with `[name]` for a name that
    /// holds dots: `connections.hosted.url`, `operators[exam.sealed].network`.
    Get {
        /// The key to read.
        key: String,
    },
    /// Set one value, given as YAML (`set connections.hosted.token
    /// env:SVEN_KEY`, `set connections.brain '{protocol: openai, url:
    /// "http://127.0.0.1:8990/v1"}'`). The whole file is validated against
    /// the schema first: a refused write changes nothing.
    Set {
        /// The key to write.
        key: String,
        /// The value, as YAML.
        value: String,
    },
    /// Remove one value, or with --all return the file to the empty default.
    Reset(ResetArgs),
    /// Print the JSON Schema of the file.
    Schema,
    /// Print the file's path.
    Path,
    /// Move the settings of the older `connections.yaml` and `operators.yaml`
    /// beside the file into it, then remove them.
    Import,
}

/// `config reset`.
#[derive(Debug, Args)]
pub struct ResetArgs {
    /// The key to remove.
    #[arg(required_unless_present = "all", conflicts_with = "all")]
    pub key: Option<String>,
    /// Return the whole file to the empty default.
    #[arg(long)]
    pub all: bool,
}
