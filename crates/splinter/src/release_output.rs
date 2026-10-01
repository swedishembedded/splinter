// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What `release`, `release list`, `rollback` and `eval` print as text:
//! every gate check with its numbers, and why a failed one failed.

use std::fmt::Write as _;

use splinter_campaign::eval::{EvalReport, Evaluated};
use splinter_campaign::release::gate::{Check, GateReport};
use splinter_campaign::release::{ReleaseList, Released, RolledBack};

use crate::output::Report;

fn share(value: Option<f64>) -> String {
    value.map_or("not measured".into(), |v| format!("{v:.3}"))
}

/// `passed`, `FAILED: <reason>`, as one check's verdict.
fn verdict<T>(check: &Check<T>) -> String {
    match (check.passed, &check.reason) {
        (true, _) => "passed".into(),
        (false, Some(reason)) => format!("FAILED: {reason}"),
        (false, None) => "FAILED".into(),
    }
}

impl Report for GateReport {
    fn human(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "  improvement  {}", verdict(&self.improvement));
        if let Some(m) = &self.improvement.measured {
            let c = &m.comparison;
            let _ = writeln!(
                out,
                "               {} held-out task(s), {} paired: candidate {} vs champion {}; \
                 wins {}-{}, ties {}, unpaired {}; sign test p = {:.4} (alpha {})",
                m.suite.tasks,
                c.paired,
                share(c.candidate_accuracy),
                share(c.baseline_accuracy),
                c.candidate_wins,
                c.baseline_wins,
                c.both_right + c.both_wrong,
                c.unpaired,
                m.sign_test.p_value,
                m.alpha
            );
        }
        let _ = writeln!(out, "  retention    {}", verdict(&self.retention));
        if let Some(m) = &self.retention.measured {
            if m.suites.is_empty() {
                let _ = writeln!(out, "               no earlier release to retain");
            }
            for suite in &m.suites {
                let _ = writeln!(
                    out,
                    "               {}: candidate {} vs champion {}, drop {} (bound {})",
                    suite.release,
                    share(suite.comparison.candidate_accuracy),
                    share(suite.comparison.baseline_accuracy),
                    share(suite.drop),
                    m.bound
                );
            }
        }
        let _ = writeln!(out, "  anchor       {}", verdict(&self.anchor));
        if let Some(m) = &self.anchor.measured {
            let _ = writeln!(
                out,
                "               version {} ({}): candidate {} vs champion {}, drop {} (bound {})",
                m.version,
                m.digest,
                share(m.comparison.candidate_accuracy),
                share(m.comparison.baseline_accuracy),
                share(m.drop),
                m.bound
            );
        }
        let _ = writeln!(out, "  serve        {}", verdict(&self.serve));
        if let Some(m) = &self.serve.measured {
            let _ = writeln!(
                out,
                "               {}: {}; {} of {} re-answered alike",
                m.binary.display(),
                m.startup_line,
                m.agreed,
                m.sampled
            );
            let answer = |a: &Option<String>| {
                a.as_deref()
                    .map_or("no answer".into(), |a| format!("{a:?}"))
            };
            for d in &m.disagreed {
                let _ = writeln!(
                    out,
                    "               {}: in-process {}, served {}",
                    d.task,
                    answer(&d.in_process),
                    answer(&d.served)
                );
            }
        }
        out
    }
}

impl Report for Released {
    fn human(&self) -> String {
        let champion = self
            .champion
            .as_ref()
            .map_or("the base".to_string(), ToString::to_string);
        let mut out = format!(
            "candidate {} against {} ({}):\n{}",
            self.candidate,
            champion,
            self.alias,
            self.gate.human()
        );
        for concept in &self.requeued {
            let _ = writeln!(out, "forgotten, queued for new tasks: {concept}");
        }
        match &self.release {
            Some(id) => {
                let _ = writeln!(out, "released {id}; {} points at it", self.alias);
            }
            None => {
                let _ = writeln!(out, "not released: the gate blocked it");
            }
        }
        out
    }
}

impl Report for ReleaseList {
    fn human(&self) -> String {
        if self.releases.is_empty() {
            return "no releases\n".into();
        }
        self.releases
            .iter()
            .map(|r| {
                let aliases = if r.aliases.is_empty() {
                    String::new()
                } else {
                    format!("  [{}]", r.aliases.join(", "))
                };
                format!(
                    "{}  {}  {}  parent {}{}\n",
                    r.id,
                    r.created_at,
                    r.candidate,
                    r.parent
                        .as_ref()
                        .map_or("none".to_string(), ToString::to_string),
                    aliases
                )
            })
            .collect()
    }
}

impl Report for RolledBack {
    fn human(&self) -> String {
        format!(
            "{} now points at {} (was {})\n",
            self.alias, self.to, self.from
        )
    }
}

impl Report for Evaluated {
    fn human(&self) -> String {
        let mut out = String::new();
        match &self.anchor {
            Some(a) => {
                let _ = writeln!(
                    out,
                    "anchor suite version {} ({}), {} task(s)",
                    a.version, a.digest, a.tasks
                );
            }
            None if self.model.is_none() => {
                let _ = writeln!(out, "no anchor suite is frozen");
            }
            None => {}
        }
        if let (Some(model), Some(reference)) = (&self.model, &self.reference) {
            let _ = writeln!(out, "{model} ({reference}):");
        }
        for score in &self.scores {
            let excluded: usize = score.suite.excluded.values().sum();
            let _ = writeln!(
                out,
                "  {}: {} of {} task(s) graded, accuracy {}{}",
                score.suite.name,
                score.graded,
                score.suite.tasks,
                share(score.accuracy),
                if excluded > 0 {
                    format!(", {excluded} excluded")
                } else {
                    String::new()
                }
            );
        }
        out
    }
}

impl Report for EvalReport {
    fn human(&self) -> String {
        match self {
            Self::Shown(shown) => shown.human(),
            Self::Ran(run) => run.human(),
        }
    }
}
