// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements paired, family-clustered comparisons of a
// trained code-editing agent against its untrained base, for its clients. If
// your team needs expertise in showing that fine-tuning an agent beat its
// base on held-out repository tasks, you can procure our services by
// sending an email to info@swedishembedded.com.

//! How `experiences compare` reads to a person.

use std::fmt::Write as _;

use splinter_sdk::compare::{ArmReport, SetComparison};

use crate::output::Report;

fn fraction(value: Option<f64>) -> String {
    value.map_or("-".into(), |v| format!("{v:.3}"))
}

/// `quality 0.750 [0.500, 1.000]`, or `-` where it is not measured.
fn estimate(label: &str, e: &splinter_sdk::measure::competence::Estimate) -> String {
    match (e.value, e.low, e.high) {
        (Some(v), Some(low), Some(high)) => format!("{label} {v:.3} [{low:.3}, {high:.3}]"),
        (Some(v), _, _) => format!("{label} {v:.3}"),
        _ => format!("{label} -"),
    }
}

/// `pass@1 15/46`: the attempts that verified, over every attempt the arm made. This is the
/// number a regression shows up in when a candidate turns correct fixes into plausible ones:
/// an exposed attempt has no verdict, so the paired test and `Q` (verified over *graded*
/// attempts) both leave it out, and instead read such a candidate as a tie.
fn pass_at_one(verified: usize, attempts: usize) -> String {
    format!("pass@1 {verified}/{attempts}")
}

fn arm_line(name: &str, arm: &ArmReport) -> String {
    format!(
        "{name:<9} {} graded of {} ({} exposed, {} harness-failed)  {}  {}  {}\n",
        arm.graded,
        arm.attempts,
        arm.exposed,
        arm.harness_failed,
        pass_at_one(arm.competence.counts.verified, arm.attempts),
        estimate("Q", &arm.competence.quality),
        estimate("A", &arm.competence.autonomy),
    )
}

impl Report for SetComparison {
    fn human(&self) -> String {
        let mut out = format!(
            "candidate {} against baseline {} (interval seed {})\n\
             {} task(s): {} paired, {} without a verdict in both arms\n",
            self.candidate_set,
            self.baseline_set,
            self.interval_seed.value(),
            self.tasks,
            self.paired,
            self.unpaired,
        );
        out.push_str(&arm_line("candidate", &self.candidate));
        out.push_str(&arm_line("baseline", &self.baseline));
        let _ = writeln!(
            out,
            "families: {} won, {} lost, {} tied; sign test over them: {} of {} discordant won by the \
             candidate, one-sided p = {:.4}",
            self.clusters.wins,
            self.clusters.losses,
            self.clusters.ties,
            self.sign_test.candidate_wins,
            self.sign_test.discordant,
            self.sign_test.p_value,
        );
        if let Some(strata) = &self.strata {
            let c = &strata.cutoffs;
            let _ = writeln!(
                out,
                "strata (report only; above: checkers > {}, footprint > {}, instruction > {} chars; \
                 {} paired without a check):",
                c.checkers, c.footprint, c.instruction_chars, strata.unstratified,
            );
            for l in &strata.lines {
                let _ = writeln!(
                    out,
                    "  stratum {}: {} paired, candidate {}/{}, baseline {}/{}, {} won, {} lost, {} tied",
                    l.stratum,
                    l.paired,
                    l.candidate_right,
                    l.paired,
                    l.baseline_right,
                    l.paired,
                    l.wins,
                    l.losses,
                    l.ties,
                );
            }
        }
        let _ = writeln!(
            out,
            "verdict agreement on paired tasks: {}",
            fraction(self.agreement)
        );
        if !self.families.is_empty() {
            let _ = writeln!(
                out,
                "{:<32} {:>6} {:>9} {:>9} {:>4} {:>6} {:>4}",
                "family", "paired", "candidate", "baseline", "won", "lost", "tied"
            );
        }
        for f in &self.families {
            let _ = writeln!(
                out,
                "{:<32} {:>6} {:>9} {:>9} {:>4} {:>6} {:>4}",
                f.family,
                f.paired,
                fraction(f.candidate_fraction),
                fraction(f.baseline_fraction),
                f.wins,
                f.losses,
                f.ties,
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::pass_at_one;

    /// The reading that made finding 096 visible: an arm that verified 15 of 46 attempts reads
    /// as `pass@1 15/46`, whatever the paired test does with the other 31.
    #[test]
    fn pass_at_one_counts_every_attempt_not_only_the_graded_ones() {
        assert_eq!(pass_at_one(15, 46), "pass@1 15/46");
        assert_eq!(pass_at_one(0, 0), "pass@1 0/0");
    }
}
