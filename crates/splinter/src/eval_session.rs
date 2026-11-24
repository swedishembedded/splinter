// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that keep a model's general
// behaviour while it learns something new, for its clients. If your team needs
// expertise in continual learning then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `eval` with no run: a model graded on a suite file, or the anchor suite
//! shown or frozen from files.

use std::fmt::Write as _;

use splinter_sdk::eval::{evaluate, SuiteEval, SuiteGraded};
use splinter_sdk::graded::Verdict;
use splinter_sdk::release::anchor::{show_or_freeze, AnchorOutcome};
use splinter_sdk::{Context, Error};

use crate::cli::EvalArgs;
use crate::output::{emit, share, Report};

/// Runs `eval` without a run directory and prints its report.
pub fn eval(ctx: &Context, json: bool, args: &EvalArgs) -> Result<(), Error> {
    match &args.suite {
        Some(suite) => emit(
            json,
            &evaluate(
                ctx,
                &SuiteEval {
                    model: args.policy.clone(),
                    suite: suite.clone(),
                },
            )?,
        ),
        None => emit(json, &Anchor(show_or_freeze(ctx, &args.freeze)?)),
    }
    Ok(())
}

/// What `eval` reports of the anchor suite.
struct Anchor(AnchorOutcome);

impl serde::Serialize for Anchor {
    fn serialize<S: serde::Serializer>(&self, out: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(out)
    }
}

impl Report for Anchor {
    fn human(&self) -> String {
        let mut out = match &self.0.anchor {
            Some(a) => format!(
                "anchor suite version {} ({}), {} task(s)\n",
                a.version, a.digest, a.tasks
            ),
            None => "no anchor suite is frozen: `splinter eval --freeze FILE`\n".into(),
        };
        if let Some(run) = &self.0.run {
            let _ = writeln!(out, "run: {run}");
        }
        out
    }
}

impl Report for SuiteGraded {
    fn human(&self) -> String {
        let mut out = format!(
            "{} on {}: {} of {} task(s) graded, accuracy {}\n",
            self.model,
            self.suite.name,
            self.graded,
            self.suite.tasks,
            share(self.accuracy)
        );
        let undecided = self
            .items
            .iter()
            .filter(|item| item.verdict == Verdict::Undecided)
            .count();
        if undecided > 0 {
            let _ = writeln!(out, "  {undecided} undecided (kept in the report)");
        }
        for missed in self.items.iter().filter(|i| i.verdict == Verdict::Failed) {
            let answer = missed
                .answer
                .as_ref()
                .map_or_else(|| "(no answer)".to_string(), |a| a.text.replace('\n', " "));
            let _ = writeln!(
                out,
                "  missed ({}): {} -> {answer}",
                missed.kind,
                missed.instruction.replace('\n', " ")
            );
        }
        out
    }
}
