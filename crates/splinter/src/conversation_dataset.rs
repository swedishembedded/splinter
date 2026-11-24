// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The dataset commands besides `dataset build`: the datasets made from
//! files rather than from experience sets (`dataset conversations`, graded
//! conversations with a counterpart as a policy-gradient dataset, and
//! `dataset decision-fixtures`), `dataset check` and `dataset export`.

use std::path::Path;

use splinter_sdk::conversation_policy_gradient::store;
use splinter_sdk::data::AdvantageKind;
use splinter_sdk::datasets::{check, export};
use splinter_sdk::{Context, Error};

use crate::cli::DatasetCommand;
use crate::output::emit;

/// Reads `input` and stores the dataset its admitted conversations make.
pub fn run(
    ctx: &Context,
    json: bool,
    input: &Path,
    system_prompt: &str,
    advantage: Option<AdvantageKind>,
) -> Result<(), Error> {
    let lines = std::fs::read_to_string(input)
        .map_err(|e| Error::Refused(format!("{}: {e}", input.display())))?;
    emit(
        json,
        &store(ctx, system_prompt, &lines, advantage.unwrap_or_default())?,
    );
    Ok(())
}

/// Runs the dataset command `command`, any but `build`; `false` when it
/// found what fails it (a dataset check that refused records).
pub fn dispatch(ctx: &Context, json: bool, command: DatasetCommand) -> Result<bool, Error> {
    match command {
        DatasetCommand::Conversations(args) => {
            run(ctx, json, &args.input, &args.system_prompt, args.advantage)?;
        }
        DatasetCommand::DecisionFixtures(args) => crate::decision_fixtures_dataset::run(
            json,
            &args.runs,
            &args.out,
            splinter_sdk::decision_fixtures::SplitPlan {
                seed: splinter_sdk::seed::resolve("split-seed", args.split_seed),
                validation_fraction: args.validation_fraction,
                test_fraction: args.test_fraction,
                holdout_suites: args.holdout_suites,
            },
        )?,
        DatasetCommand::VerifySpeech(args) => {
            if !(0.0..=1.0).contains(&args.fail_below)
                || args.fail_below > args.pass_at
                || args.pass_at > 1.0
            {
                return Err(Error::Refused(format!(
                    "--fail-below {} and --pass-at {} must satisfy 0 <= fail-below <= pass-at <= 1",
                    args.fail_below, args.pass_at
                )));
            }
            if !(args.min_length_ratio > 0.0
                && args.min_length_ratio <= 1.0
                && args.max_length_ratio >= 1.0)
            {
                return Err(Error::Refused(format!(
                    "--min-length-ratio {} must be in (0, 1] and --max-length-ratio {} at least 1",
                    args.min_length_ratio, args.max_length_ratio
                )));
            }
            let check = splinter_sdk::datasets::SpeechCheck {
                metadata: args.metadata,
                text_column: args.text_column,
                recognizer: args.recognizer,
                policy: splinter_sdk::model::speech::MatchPolicy {
                    pass_at: args.pass_at,
                    fail_below: args.fail_below,
                    length_ratio: (args.min_length_ratio, args.max_length_ratio),
                },
                rehear: args.rehear,
            };
            let verified = splinter_sdk::datasets::verify_speech(ctx, &check)?;
            emit(json, &verified);
            return Ok(verified.passes());
        }
        DatasetCommand::Check(args) => {
            let checked = check(ctx, &args.id, args.from.as_ref(), args.max_block)?;
            emit(json, &checked);
            return Ok(checked.passes());
        }
        DatasetCommand::Export { id, out } => emit(json, &export(ctx, &id, &out)?),
    }
    Ok(true)
}

impl crate::output::Report for splinter_sdk::datasets::SpeechVerification {
    fn human(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let heard = self.rows.iter().filter(|r| r.heard_now).count();
        let _ = writeln!(
            out,
            "{} recordings heard by {} ({heard} now): {} pass, {} review, {} fail",
            self.rows.len(),
            self.recognizer,
            self.passed,
            self.review,
            self.failed
        );
        for row in self
            .rows
            .iter()
            .filter(|r| r.verdict != splinter_sdk::model::speech::Verdict::Pass)
        {
            let verdict = if row.verdict == splinter_sdk::model::speech::Verdict::Fail {
                "FAIL"
            } else {
                "review"
            };
            let _ = writeln!(
                out,
                "  {verdict:6} {}: {}",
                row.id,
                row.reason.as_deref().unwrap_or("")
            );
        }
        let _ = writeln!(out, "verdicts written to {}", self.metadata);
        out
    }
}

impl crate::output::Report for splinter_sdk::datasets::DatasetCheck {
    fn human(&self) -> String {
        use std::fmt::Write as _;
        let spread = |d: &splinter_sdk::model::served_train::Distribution| {
            format!(
                "total {}, min {}, mean {:.0}, p50 {}, p90 {}, p99 {}, max {}",
                d.total, d.min, d.mean, d.p50, d.p90, d.p99, d.max
            )
        };
        let mut out = format!(
            "dataset {} read by the trainer for {} ({}, {}): {} record(s), {} read, {} example(s)\n",
            self.dataset,
            self.base,
            if self.thinking { "reasoning" } else { "no reasoning" },
            if self.pack_answers { "one example per record" } else { "one example per answer" },
            self.records,
            self.valid_records,
            self.examples
        );
        let _ = writeln!(out, "  tokens per example: {}", spread(&self.tokens));
        let _ = writeln!(
            out,
            "  supervised tokens per example: {}",
            spread(&self.supervised_tokens)
        );
        let _ = writeln!(
            out,
            "  {} example(s) supervise nothing; {} longer than {} tokens",
            self.examples_without_supervision, self.over_max_block, self.max_block
        );
        for record in &self.longest {
            let _ = writeln!(
                out,
                "  line {}: {} tokens ({} supervised), {} example(s)",
                record.line, record.tokens, record.trained, record.examples
            );
        }
        for refusal in &self.refused {
            let _ = writeln!(out, "  refused {refusal}");
        }
        if let Some(pairing) = &self.pairing {
            let _ = writeln!(
                out,
                "  pairs: {} from attempts, {} from repairs, {} recorded; at most {} per group, seed {}",
                pairing.attempts,
                pairing.repairs,
                pairing.recorded,
                pairing.per_group.map_or("all".to_string(), |n| n.to_string()),
                pairing.seed.map_or("none".to_string(), |s| s.to_string())
            );
        }
        if let Some(balance) = &self.feedback {
            let _ = writeln!(
                out,
                "  judged: {} desirable (weight {:.3}), {} undesirable (weight {:.3})",
                balance.desirable,
                balance.desirable_weight,
                balance.undesirable,
                balance.undesirable_weight
            );
        }
        if self.tokenizers.len() > 1 {
            let _ = writeln!(
                out,
                "  refused: the records name {} tokenizers ({}); a dataset holds the attempts of one",
                self.tokenizers.len(),
                self.tokenizers.join(", ")
            );
        }
        out
    }
}
