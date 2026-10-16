// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What `release`, `release list`, `rollback` and `eval` print as text:
//! every gate check with its numbers, and why a failed one failed.

use std::fmt::Write as _;

use splinter_sdk::eval::{EvalReport, Evaluated};
use splinter_sdk::measure::gate::{AgenticProbe, Agreement, Check, GateReport};
use splinter_sdk::release::{ReleaseList, Released, RolledBack};

use crate::output::{tally, Report};

/// A verdict as a word; no verdict is not a fail.
fn grade(verdict: Option<bool>) -> &'static str {
    match verdict {
        Some(true) => "right",
        Some(false) => "wrong",
        None => "not graded",
    }
}

fn share(value: Option<f64>) -> String {
    value.map_or("not measured".into(), |v| format!("{v:.3}"))
}

/// `passed`, `FAILED: <reason>`, as one check's verdict.
fn verdict<T>(check: &Check<T>) -> String {
    match (check.passed, &check.reason) {
        (true, _) => "passed".into(),
        (false, Some(reason)) if check.inconclusive => {
            format!("INCONCLUSIVE (not a failure): {reason}")
        }
        (false, Some(reason)) => format!("FAILED: {reason}"),
        (false, None) => "FAILED".into(),
    }
}

/// The attempts at changes to a repository a probe set aside, as one line:
/// they are in no denominator.
fn set_aside(probe: &AgenticProbe) -> String {
    let (c, b) = (probe.candidate, probe.champion);
    format!(
        "workspace tasks attempted {} time(s) each; set aside, in no denominator: candidate {} \
         exposed to the hidden tests and {} not carried out, champion {} and {}",
        probe.attempts, c.exposed, c.environment_failures, b.exposed, b.environment_failures
    )
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
        if let Some(probe) = self
            .improvement
            .measured
            .as_ref()
            .and_then(|m| m.agentic.as_ref())
        {
            let _ = writeln!(out, "               {}", set_aside(probe));
        }
        if let Some(h) = self
            .improvement
            .measured
            .as_ref()
            .and_then(|m| m.hidden_strength)
        {
            let _ = match (h.mean_kill_rate, h.min_kill_rate) {
                (Some(mean), Some(min)) => writeln!(
                    out,
                    "               hidden tests of {} workspace task(s): kill rate measured on {} \
                     (mean {mean:.2}, lowest {min:.2}); reported, not used to weigh a pass",
                    h.tasks, h.measured
                ),
                _ => writeln!(
                    out,
                    "               hidden tests of {} workspace task(s): kill rate not measured",
                    h.tasks
                ),
            };
        }
        if let Some(t) = self
            .improvement
            .measured
            .as_ref()
            .and_then(|m| m.task_level)
        {
            let _ = writeln!(
                out,
                "               by task (not what decides): candidate ahead by {:+.3} [{:+.3}, {:+.3}] \
                 with families resampled, exact sign test over tasks p = {:.4}",
                t.difference, t.low, t.high, t.p_value
            );
        }
        if let Some(m) = &self.improvement.measured {
            for (label, part) in [
                ("generalisation (held-out records)", &m.generalisation),
                (
                    "recall under paraphrase (variants of trained tasks)",
                    &m.recall,
                ),
            ] {
                if let Some(part) = part {
                    let c = &part.comparison;
                    let _ = writeln!(
                        out,
                        "               {label}: {} task(s), {} paired: candidate {} vs champion {}; \
                         wins {}-{}; sign test p = {:.4}",
                        part.suite.tasks,
                        c.paired,
                        share(c.candidate_accuracy),
                        share(c.baseline_accuracy),
                        c.candidate_wins,
                        c.baseline_wins,
                        part.sign_test.p_value
                    );
                }
            }
        }
        if let Some(v) = self
            .improvement
            .measured
            .as_ref()
            .and_then(|m| m.variants.as_ref())
        {
            let _ = writeln!(
                out,
                "               {} variant(s) of what it trained on measured, {} left out ({})",
                v.tasks,
                v.excluded.values().sum::<usize>(),
                tally(&v.excluded)
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
                if let Some(probe) = &suite.agentic {
                    let _ = writeln!(out, "               {}", set_aside(probe));
                }
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
            match m.workspace_tasks {
                Some(0) => {
                    let _ = writeln!(
                        out,
                        "               no coding-agent task is frozen in the suite: absent, not \
                         scored; decided on the closed-book tasks it holds"
                    );
                }
                Some(n) => {
                    let _ = writeln!(
                        out,
                        "               {n} task(s) are changes to a repository, attempted by the \
                         worker"
                    );
                    if let Some(probe) = &m.agentic {
                        let _ = writeln!(out, "               {}", set_aside(probe));
                    }
                }
                None => {}
            }
            if let Some(n) = m.items_to_see_two_points {
                let _ = writeln!(
                    out,
                    "               a drop of 0.02 would need about {n} paired items to be seen with \
                     four chances in five at this discordance; this suite has {}",
                    m.comparison.paired
                );
            }
            if let Some(i) = &m.drop_interval {
                let _ = writeln!(
                    out,
                    "               drop {:.0}% interval over the paired items [{:+.3}, {:+.3}]",
                    i.level * 100.0,
                    i.low,
                    i.high
                );
            }
        }
        for p in &self.prompts {
            let _ = writeln!(
                out,
                "  prompts      {}: candidate under {}; champion under {}",
                p.suite, p.candidate, p.champion
            );
        }
        if let Some(seed) = self.interval_seed {
            let _ = writeln!(
                out,
                "  intervals    resampled with seed {seed} (--interval-seed {seed} reproduces them)"
            );
        }
        if let Some(canary) = &self.canary {
            let _ = writeln!(out, "  canary       {}", verdict(canary));
            if let Some(m) = &canary.measured {
                let _ = writeln!(
                    out,
                    "               {} of {} canary task(s) graded, {} passed",
                    m.graded,
                    m.tasks,
                    m.passed_tasks.len()
                );
            }
        }
        let _ = writeln!(out, "  serve        {}", verdict(&self.serve));
        if let Some(m) = &self.serve.measured {
            let compared = match m.agreement {
                Agreement::Answers => "re-answered alike",
                Agreement::Verdicts => "graded as they were in-process",
            };
            let _ = writeln!(
                out,
                "               {}: {}; {} of {} {compared}",
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
                let _ = match m.agreement {
                    Agreement::Answers => writeln!(
                        out,
                        "               {}: in-process {}, served {}",
                        d.task,
                        answer(&d.in_process),
                        answer(&d.served)
                    ),
                    Agreement::Verdicts => writeln!(
                        out,
                        "               {}: in-process {}, served {}",
                        d.task,
                        grade(d.in_process_verdict),
                        grade(d.served_verdict)
                    ),
                };
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
        for warning in &self.warnings {
            let _ = writeln!(out, "warning: {warning}");
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
            for missed in &score.missed {
                let _ = writeln!(
                    out,
                    "    missed ({}): {} -> {}",
                    missed.kind,
                    missed.instruction.replace('\n', " "),
                    missed
                        .answer
                        .as_deref()
                        .map_or_else(|| "(no answer)".to_string(), |a| a.replace('\n', " "))
                );
            }
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
