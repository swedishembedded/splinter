// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements paired comparisons of agent models whose
// statistics are proven against exact references, for its clients. If your
// team needs expertise in telling a real gain of a fine-tuned agent from the
// luck of a small exam, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How `experiences compare-arms` reads to a person.

use std::fmt::Write as _;

use splinter_sdk::arm_compare::{ArmComparison, Bounds};

use crate::output::Report;

fn value(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{v:.3}"))
}

fn bounds(b: Option<Bounds>) -> String {
    b.map_or("-".into(), |b| format!("[{:+.3}, {:+.3}]", b.low, b.high))
}

impl Report for ArmComparison {
    fn human(&self) -> String {
        let mut out = format!(
            "arms against {} ({:.0}% intervals, {} resamples, seed {})\n",
            self.reference,
            self.level * 100.0,
            self.resamples,
            self.seed.value()
        );
        for arm in &self.arms {
            let ks: Vec<String> = arm
                .pass_at_k
                .iter()
                .map(|p| format!("pass@{} {} ({} tasks)", p.k, value(p.value), p.tasks))
                .collect();
            let _ = writeln!(
                out,
                "{:<12} {} trials, {} graded ({} timeouts, {} of them passed), {} infrastructure \
                 errors, {} tasks; {}",
                arm.arm,
                arm.trials,
                arm.graded,
                arm.timeouts,
                arm.timeouts_passed,
                arm.infra_errors,
                arm.tasks,
                ks.join(", ")
            );
        }
        for c in &self.contrasts {
            let _ = writeln!(
                out,
                "{} - {} over {} tasks: {} {} by task, {} by family ({} families); \
                 better {} worse {} tied {}; sign p one-sided {:.4} (Holm {:.4}), two-sided {:.4} \
                 (Holm {:.4})",
                c.arm,
                c.reference,
                c.paired_tasks,
                c.difference.map_or("-".into(), |d| format!("{d:+.3}")),
                bounds(c.task_interval),
                bounds(c.family_interval),
                c.families.map_or("-".into(), |f| f.to_string()),
                c.better,
                c.worse,
                c.ties,
                c.p_one_sided,
                c.p_one_sided_holm,
                c.p_two_sided,
                c.p_two_sided_holm
            );
        }
        out
    }
}
