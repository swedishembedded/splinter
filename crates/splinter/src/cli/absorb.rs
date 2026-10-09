// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `session` and `claims`: learning from the sessions a person
//! holds with an agent, one stage per command.

use std::path::PathBuf;

use clap::Subcommand;
use splinter_sdk::sessions::DEFAULT_MAX_SESSION_BYTES;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::model_ref;

/// `session ...`.
#[derive(Debug, Subcommand)]
pub enum SessionCommand {
    /// Take recorded agent sessions in as sources: ATIF files, or directories
    /// searched for *.atif.json. Each is validated, refused with the reason
    /// when it cannot be rendered as training conversations, stripped of
    /// secrets (keys, tokens, passwords, private keys) and stored once by
    /// content; the same file again changes nothing. Exits 1 when any file
    /// was refused.
    Add {
        /// ATIF files, or directories of them.
        #[arg(required = true, num_args = 1.., value_name = "PATH")]
        paths: Vec<PathBuf>,
        /// The largest session file taken, in bytes.
        #[arg(long, default_value_t = DEFAULT_MAX_SESSION_BYTES, value_name = "BYTES")]
        max_bytes: u64,
    },
    /// List the stored sessions.
    List,
}

/// `claims ...`.
#[derive(Debug, Subcommand)]
pub enum ClaimsCommand {
    /// Have a model propose the claims each stored session teaches - a
    /// correction, a fact, a procedure - with the person's exact words that
    /// support each, into a claim set. Exits 1 when a session's reply stayed
    /// unusable.
    Extract {
        /// The sessions, by id or unique prefix, in the order they happened.
        #[arg(required = true, value_name = "SESSION-ID")]
        sessions: Vec<String>,
        /// The model that reads the sessions.
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        generator: ModelRef,
    },
    /// Rule on a claim set by code alone: quotes verbatim in the person's
    /// words, the statement's numbers, names and quoted terms in them, the
    /// agent's sentences never evidence, repeats collapsed, a later claim on
    /// the same question superseding the earlier. Every ruling is kept, each
    /// refusal with its reason.
    Gate {
        /// The claim set, by id or unique prefix.
        #[arg(value_name = "CLAIMSET-ID")]
        claim_set: String,
    },
    /// List the stored claim sets.
    List,
    /// Show a claim set: its proposals per session.
    Show {
        /// The claim set, by id or unique prefix.
        id: String,
    },
    /// Show the ledger: live claims, superseded ones with by which, and
    /// refused proposals with their reasons.
    Ledger,
}
