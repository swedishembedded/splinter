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

fn arm_line(name: &str, arm: &ArmReport) -> String {
    format!(
        "{name:<9} {} graded of {} ({} exposed, {} harness-failed)  {}  {}\n",
        arm.graded,
        arm.attempts,
        arm.exposed,
        arm.harness_failed,
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
