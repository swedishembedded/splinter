// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in consolidating what a model learned into its own
// weights, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The grammar of `consolidate`: a generation taken into the weights.

use std::path::PathBuf;

use splinter_sdk::absorb::consolidate::DEFAULT_CONSOLIDATION_EPOCHS;
use splinter_sdk::absorb::generation::DEFAULT_RETENTION_REPLAY_SHARE;
use splinter_sdk::generation::ConsolidationMode;
use splinter_sdk::rehearsal::DEFAULT_REHEARSAL_SHARE;
use splinter_sdk::seed::Seed;
use splinter_sdk::train::DEFAULT_FULL_LEARNING_RATE;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::model_ref;

/// `full` or `merge`.
pub fn consolidation_mode(text: &str) -> Result<ConsolidationMode, String> {
    match text {
        "full" => Ok(ConsolidationMode::Full),
        "merge" => Ok(ConsolidationMode::Merge),
        other => Err(format!("{other:?} is neither `full` nor `merge`")),
    }
}

/// `consolidate`: the claims of the release in use are taken into the
/// weights, and the next generation starts from an empty claim set on top
/// of the checkpoint.
#[derive(Debug, clap::Args)]
pub struct ConsolidateArgs {
    /// The policy to consolidate: the checkpoint is released under its alias.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub policy: ModelRef,
    /// `full` trains every weight of the generation's base on the claims'
    /// records, beside a sample of the records of the claims the base
    /// already holds and the base's own answers; `merge` folds the adapter
    /// into the base's weights and trains nothing (the only mode that fits a
    /// model whose weights and gradients do not fit the cards).
    #[arg(long, value_parser = consolidation_mode, default_value = "full", value_name = "MODE")]
    pub mode: ConsolidationMode,
    /// The release to consolidate, by id or unique prefix (default: the one
    /// the policy points at).
    #[arg(long, value_name = "RELEASE")]
    pub from: Option<String>,
    /// The share of the training draws that are the base's own answers to
    /// general tasks, in [0, 1); 0 turns it off. Full mode only.
    #[arg(long, default_value_t = DEFAULT_REHEARSAL_SHARE, value_name = "SHARE", value_parser = super::voice_share)]
    pub rehearsal_share: f64,
    /// The seed the rehearsal's form tasks are built with.
    #[arg(
        long,
        value_name = "N",
        help = "The seed the rehearsal's form tasks are built with \
        [default: drawn and printed]"
    )]
    pub rehearsal_seed: Option<Seed>,
    /// The share of the training draws that are records of the claims the
    /// base already holds, in [0, 1). Full mode only.
    #[arg(long, default_value_t = DEFAULT_RETENTION_REPLAY_SHARE, value_name = "SHARE", value_parser = super::voice_share)]
    pub retention_replay_share: f64,
    /// The most passes over the records. Full mode only.
    #[arg(long, default_value_t = DEFAULT_CONSOLIDATION_EPOCHS, value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub epochs: u32,
    /// The step budget, in place of --epochs. Full mode only.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub steps: Option<u32>,
    /// The peak learning rate: every weight moves, so it is far lower than
    /// an adapter's.
    #[arg(long, value_name = "LR", value_parser = super::train::positive_rate,
        help = format!("The peak learning rate [default: {DEFAULT_FULL_LEARNING_RATE}]"))]
    pub lr: Option<f32>,
    /// Files of sealed probes (JSON Lines of {"question", "reference"?,
    /// "name"?}): a training record containing one is refused.
    #[arg(long, num_args = 1.., value_name = "FILE")]
    pub sealed_probes: Vec<PathBuf>,
    /// Gate the checkpoint but do not release it: it is not kept.
    #[arg(long)]
    pub no_release: bool,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::cli::{Cli, Command};

    fn command(args: &[&str]) -> Command {
        match Cli::try_parse_from(std::iter::once("splinter").chain(args.iter().copied())) {
            Ok(Cli {
                command: Some(command),
                ..
            }) => command,
            other => panic!("{args:?} is not a command: {other:?}"),
        }
    }

    #[test]
    fn consolidate_trains_every_weight_unless_told_to_merge() {
        let Command::Consolidate(args) = command(&["consolidate"]) else {
            panic!("not a consolidate");
        };
        assert_eq!(args.mode, super::ConsolidationMode::Full);
        assert!(args.from.is_none() && !args.no_release);
        assert_eq!(args.epochs, super::DEFAULT_CONSOLIDATION_EPOCHS);
        let Command::Consolidate(args) = command(&[
            "consolidate",
            "--mode",
            "merge",
            "--from",
            "abcd1234",
            "--lr",
            "0.00001",
            "--no-release",
        ]) else {
            panic!("not a consolidate");
        };
        assert_eq!(args.mode, super::ConsolidationMode::Merge);
        assert_eq!(args.from.as_deref(), Some("abcd1234"));
        assert!(args.no_release && args.lr.is_some());
        assert!(Cli::try_parse_from(["splinter", "consolidate", "--mode", "squash"]).is_err());
    }

    #[test]
    fn absorb_consolidates_above_a_count_unless_told_not_to() {
        let Command::Absorb(args) = command(&["absorb", "./today"]) else {
            panic!("not an absorb");
        };
        assert_eq!(
            args.consolidate_after_records,
            splinter_sdk::absorb::generation::DEFAULT_CONSOLIDATE_AFTER_RECORDS as u64
        );
        assert!(!args.no_consolidate);
        assert_eq!(args.consolidate_mode, super::ConsolidationMode::Full);
        let Command::Absorb(args) = command(&[
            "absorb",
            "./today",
            "--consolidate-after-records",
            "500",
            "--consolidate-mode",
            "merge",
        ]) else {
            panic!("not an absorb");
        };
        assert_eq!(args.consolidate_after_records, 500);
        assert_eq!(args.consolidate_mode, super::ConsolidationMode::Merge);
        assert!(Cli::try_parse_from([
            "splinter",
            "absorb",
            "d",
            "--no-consolidate",
            "--consolidate-after-records",
            "5"
        ])
        .is_err());
    }
}
