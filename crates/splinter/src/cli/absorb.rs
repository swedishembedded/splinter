// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `session` and `claims`: learning from the sessions a person
//! holds with an agent, one stage per command.

use std::path::PathBuf;

use clap::Subcommand;
use splinter_sdk::absorb::kit::PARAPHRASES_WRITTEN;
use splinter_sdk::absorb::DEFAULT_EPOCHS;
use splinter_sdk::claims::DEFAULT_EXTRACTION_PASSES;
use splinter_sdk::rehearsal::DEFAULT_REHEARSAL_SHARE;
use splinter_sdk::sessions::DEFAULT_MAX_SESSION_BYTES;
use splinter_sdk::train::{DEFAULT_LORA_RANK, DEFAULT_REPLAY_FRACTION};
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
        /// How many times each session is read, in different orders; only a
        /// claim every pass proposed is kept, the rest are listed as
        /// unconfirmed.
        #[arg(long, default_value_t = DEFAULT_EXTRACTION_PASSES, value_name = "N",
            value_parser = clap::value_parser!(u32).range(1..))]
        passes: u32,
    },
    /// Rule on a claim set: quotes verbatim in the person's words, the
    /// statement's numbers, names and quoted terms in them, the agent's
    /// sentences never evidence. Claims that name the same thing are paired:
    /// a later claim that contradicts the earlier supersedes it, one that
    /// says the same again reinforces it (recorded, never refused). Every
    /// ruling is kept, each refusal with its reason.
    Gate {
        /// The claim set, by id or unique prefix.
        #[arg(value_name = "CLAIMSET-ID")]
        claim_set: String,
        /// The model that decides whether a later claim supersedes,
        /// reinforces or is apart from an earlier one about the same thing;
        /// without one the statements decide.
        #[arg(long, value_parser = model_ref, value_name = "REF")]
        judge: Option<ModelRef>,
        /// Whether the judge also refuses a claim whose cited words do not
        /// assert its statement (`--entail=false` turns it off).
        #[arg(long, default_value_t = true, num_args = 0..=1, default_missing_value = "true",
            require_equals = true, value_name = "BOOL")]
        entail: bool,
    },
    /// List the stored claim sets.
    List,
    /// Show a claim set: its proposals per session.
    Show {
        /// The claim set, by id or unique prefix.
        id: String,
    },
    /// Show the ledger: live claims with how often each was said again,
    /// superseded ones with by which, and refused proposals with their
    /// reasons.
    Ledger,
}

/// `absorb`: the sessions a person held with an agent become the next
/// release.
#[derive(Debug, clap::Args)]
pub struct AbsorbArgs {
    /// ATIF files, or directories searched for *.atif.json.
    #[arg(required = true, num_args = 1.., value_name = "SESSIONS")]
    pub sessions: Vec<PathBuf>,
    /// The policy to update: a release is made under its alias.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub policy: ModelRef,
    /// The model that proposes claims and writes their paraphrases and forms
    /// (default: the policy).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub generator: Option<ModelRef>,
    /// The model that answers with the claim in front of it, as the policy
    /// would (default: the policy).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub teacher: Option<ModelRef>,
    /// A model that decides the answers the claim's terms cannot, whether a
    /// later claim supersedes, reinforces or is apart from an earlier one, and
    /// whether the person's words assert a claim's statement; another model
    /// than the teacher, the generator and the policy.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
    /// Whether the judge also refuses a claim whose cited words do not assert
    /// its statement (`--entail=false` turns it off); it applies when a judge
    /// is configured.
    #[arg(long, default_value_t = true, num_args = 0..=1, default_missing_value = "true",
        require_equals = true, value_name = "BOOL")]
    pub entail: bool,
    /// How many times each session is read for claims, in different orders;
    /// only a claim every pass proposed is ruled on.
    #[arg(long, default_value_t = DEFAULT_EXTRACTION_PASSES, value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub passes: u32,
    /// Differently worded questions written per claim; a third of them are
    /// kept out of training as the claim's own stopping and gate set.
    #[arg(long, default_value_t = PARAPHRASES_WRITTEN as u32, value_name = "N",
        value_parser = clap::value_parser!(u32).range(3..))]
    pub paraphrases: u32,
    /// The share of the training draws that are the base's own answers to
    /// general tasks, in [0, 1); 0 turns it off.
    #[arg(long, default_value_t = DEFAULT_REHEARSAL_SHARE, value_name = "SHARE", value_parser = super::voice_share)]
    pub rehearsal_share: f64,
    /// The most passes over the night's records.
    #[arg(long, default_value_t = DEFAULT_EPOCHS, value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub epochs: u32,
    /// The step budget, in place of --epochs.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub steps: Option<u32>,
    /// LoRA rank of the adapter.
    #[arg(long, default_value_t = DEFAULT_LORA_RANK, value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub rank: u32,
    /// The learning rate, alpha and weight decay of the training.
    #[command(flatten)]
    pub optimiser: super::OptimiserArgs,
    /// Hold the frozen base at bf16, half the bytes of fp32.
    #[arg(long)]
    pub bf16_base: bool,
    /// Continue the current release on the claims not yet absorbed, with a
    /// replay of earlier releases' records, instead of training again from
    /// the base on every live claim. An ablation: the default makes
    /// supersession and forgetting exact.
    #[arg(long)]
    pub continue_from_release: bool,
    /// With --continue-from-release, the fraction of each earlier release's
    /// records replayed.
    #[arg(long, default_value_t = DEFAULT_REPLAY_FRACTION, value_name = "F",
        requires = "continue_from_release", value_parser = super::share)]
    pub replay_fraction: f64,
    /// Files of sealed probes (JSON Lines of {"question", "reference"?,
    /// "name"?}): a training record containing a probe's question, or an
    /// 8-word run of a probe beyond its claim's statement, is refused. The
    /// gate never reads them.
    #[arg(long, num_args = 1.., value_name = "FILE")]
    pub sealed_probes: Vec<PathBuf>,
    /// Stop after the claims are extracted and ruled on; train nothing.
    #[arg(long)]
    pub dry_run: bool,
    /// Stop at the trained candidate: do not run the gate.
    #[arg(long)]
    pub no_release: bool,
}
