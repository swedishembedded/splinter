// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The grounded run and the plan-only control, side by side.
//!
//! Both runs replay the same commits with the same plan styles; they differ
//! only in whether the agent was shown the change. Pass rates are compared
//! over all attempts and over the (commit, style) units both runs judged: the
//! units only the grounded run passes are the hidden information the diff
//! carries, the units only the control passes are noise or luck, and the
//! units both pass are the purest records.

use std::collections::BTreeMap;
use std::fmt;

use crate::ledger::{bias, latest, Counts, Entry, Outcome};
use crate::style::PlanStyle;

/// The comparison of two ledgers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Comparison {
    /// Attempts and passes of the grounded run.
    pub grounded: Counts,
    /// Attempts and passes of the plan-only run.
    pub plan_only: Counts,
    /// By plan style: grounded, plan-only.
    pub by_style: BTreeMap<String, (Counts, Counts)>,
    /// Units (commit and style) with a verdict in both runs.
    pub paired: usize,
    /// Of those, passed in both.
    pub passed_in_both: usize,
    /// Of those, passed only with the change shown.
    pub only_grounded: usize,
    /// Of those, passed only without it.
    pub only_plan_only: usize,
}

type Unit<'a> = (&'a str, Option<PlanStyle>);

fn verdicts(entries: &[Entry]) -> BTreeMap<Unit<'_>, bool> {
    latest(entries)
        .into_iter()
        .filter(|e| matches!(e.outcome, Outcome::Passed | Outcome::Failed))
        .map(|e| ((e.commit.as_str(), e.style), e.outcome == Outcome::Passed))
        .collect()
}

impl Comparison {
    /// The comparison of a grounded run's ledger and a plan-only run's.
    #[must_use]
    pub fn of(grounded: &[Entry], plan_only: &[Entry]) -> Self {
        let (g, p) = (bias(grounded), bias(plan_only));
        let mut by_style: BTreeMap<String, (Counts, Counts)> = BTreeMap::new();
        for (style, counts) in g.by_style {
            by_style.entry(style).or_default().0 = counts;
        }
        for (style, counts) in p.by_style {
            by_style.entry(style).or_default().1 = counts;
        }
        let (gv, pv) = (verdicts(grounded), verdicts(plan_only));
        let mut out = Self {
            grounded: g.overall,
            plan_only: p.overall,
            by_style,
            ..Self::default()
        };
        for (unit, passed_grounded) in &gv {
            let Some(passed_plan_only) = pv.get(unit) else {
                continue;
            };
            out.paired += 1;
            match (*passed_grounded, *passed_plan_only) {
                (true, true) => out.passed_in_both += 1,
                (true, false) => out.only_grounded += 1,
                (false, true) => out.only_plan_only += 1,
                (false, false) => {}
            }
        }
        out
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rate = |c: &Counts| format!("{}/{}", c.passed, c.attempts);
        writeln!(
            f,
            "{:<20} {:>10} {:>10}",
            "pass rate", "grounded", "plan-only"
        )?;
        writeln!(
            f,
            "{:<20} {:>10} {:>10}",
            "overall",
            rate(&self.grounded),
            rate(&self.plan_only)
        )?;
        for (style, (g, p)) in &self.by_style {
            writeln!(f, "{:<20} {:>10} {:>10}", style, rate(g), rate(p))?;
        }
        writeln!(f, "units judged in both runs: {}", self.paired)?;
        writeln!(
            f,
            "  passed in both (the purest records): {}",
            self.passed_in_both
        )?;
        writeln!(
            f,
            "  passed only with the change shown (what the grounding carries): {}",
            self.only_grounded
        )?;
        writeln!(f, "  passed only without it: {}", self.only_plan_only)
    }
}
