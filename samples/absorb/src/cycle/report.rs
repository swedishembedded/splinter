// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The report of a cycle: JSON for tools and Markdown for people, built only
//! from a pre-registration that still hashes to what it recorded and
//! evaluations made under it.

use std::fmt::Write as _;

use anyhow::Context as _;
use serde::Serialize;

use super::analysis::{tables, FactInfo, Inputs, Night, NightSummary, Transitions};
use super::eval::Evaluation;
use super::items::Item;
use super::plan::Plan;
use super::prereg::{Entry, Prereg};
use super::stats::{difference, Proportion};
use super::verdict::{verdict, Verdict};

/// What the driver kept of a night besides its evaluation.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct NightMeta {
    /// The night.
    pub night: usize,
    /// Sessions recorded.
    pub sessions: usize,
    /// Whether the nightly update released a new model.
    pub released: Option<bool>,
    /// Claims live in the ledger after the night.
    pub live_claims: Option<u64>,
    /// Why the run stopped before its last stage, if it did.
    pub stopped: Option<String>,
    /// Seconds recording the sessions took.
    pub record_seconds: Option<f64>,
    /// Seconds the nightly update took.
    pub absorb_seconds: Option<f64>,
}

/// Everything a report is built from.
pub struct ReportSources<'a> {
    /// The pre-registration.
    pub prereg: &'a Prereg,
    /// The plan.
    pub plan: &'a Plan,
    /// The items.
    pub items: &'a [Item],
    /// The facts of the manifest.
    pub facts: &'a [FactInfo],
    /// The evaluations, nights from 0.
    pub evaluations: &'a [Evaluation],
    /// What the driver kept of each night, from 1.
    pub meta: &'a [NightMeta],
    /// The ledger of runs of the fact pool.
    pub runs: &'a [Entry],
}

/// The report.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    /// The hash of the pre-registration it was built under.
    pub prereg_hash: String,
    /// The run's number and commit.
    pub run: String,
    /// The verdict.
    pub verdict: Verdict,
    /// The nights.
    pub nights: Vec<NightSummary>,
    /// What changed between nights.
    pub transitions: Vec<Transitions>,
    /// What the driver kept of each night.
    pub meta: Vec<NightMeta>,
    /// Every full run of the pool, with its verdict when it has one.
    pub runs: Vec<RunLine>,
    /// Seconds per phase of each night's evaluation.
    pub evaluation_seconds: Vec<(usize, f64, Option<f64>)>,
}

/// One full run, as the ledger lists it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunLine {
    /// The run's number.
    pub run: usize,
    /// Its directory.
    pub dir: String,
    /// Whether its rule was met, once it finished.
    pub met: Option<bool>,
    /// Its verdict.
    pub text: Option<String>,
}

fn run_lines(runs: &[Entry]) -> Vec<RunLine> {
    let mut lines: Vec<RunLine> = Vec::new();
    for entry in runs {
        match entry {
            Entry::Start { run, dir } => lines.push(RunLine {
                run: *run,
                dir: dir.clone(),
                met: None,
                text: None,
            }),
            Entry::Finished { run, met, text } => {
                if let Some(line) = lines.iter_mut().find(|l| l.run == *run) {
                    line.met = Some(*met);
                    line.text = Some(text.clone());
                }
            }
            Entry::Registered { .. } => {}
        }
    }
    lines
}

/// Builds the report.
///
/// # Errors
/// The pre-registration was edited or registered other analysis, an
/// evaluation was made under another pre-registration, or the evaluations do
/// not answer the registered items.
pub fn build(inputs: &ReportSources<'_>) -> anyhow::Result<Report> {
    inputs
        .prereg
        .verify()
        .context("refusing a report: the pre-registration does not hold")?;
    for e in inputs.evaluations {
        anyhow::ensure!(
            e.prereg_hash == inputs.prereg.hash,
            "refusing a report: night {} was evaluated under the pre-registration {}, not {}",
            e.night,
            e.prereg_hash,
            inputs.prereg.hash
        );
    }
    anyhow::ensure!(!inputs.evaluations.is_empty(), "no night was evaluated");
    let analysis = Inputs {
        role: inputs.prereg.role,
        primary_kinds: &inputs.prereg.rule.primary_kinds,
        facts: inputs.facts,
        plan: inputs.plan,
        items: inputs.items,
    };
    let (nights, transitions) = tables(&analysis, inputs.evaluations)?;
    let indexed: Vec<Night<'_>> = inputs
        .evaluations
        .iter()
        .map(|e| Night::new(e, inputs.items))
        .collect::<anyhow::Result<_>>()?;
    let verdict = verdict(&analysis, &inputs.prereg.rule, &nights, &indexed);
    Ok(Report {
        prereg_hash: inputs.prereg.hash.clone(),
        run: format!(
            "run {} at commit {}{}",
            inputs.prereg.run_counter,
            inputs.prereg.pipeline_commit,
            if inputs.prereg.dirty {
                " (tree not clean)"
            } else {
                ""
            }
        ),
        verdict,
        nights,
        transitions,
        meta: inputs.meta.to_vec(),
        runs: run_lines(inputs.runs),
        evaluation_seconds: inputs
            .evaluations
            .iter()
            .map(|e| (e.night, e.timings.answer_seconds, e.timings.judge_seconds))
            .collect(),
    })
}

fn cell(p: Option<Proportion>) -> String {
    let Some(p) = p else { return "-".into() };
    match p.clopper_pearson() {
        Some((low, high)) => format!("{}/{} ({low:.2}-{high:.2})", p.k, p.n),
        None => "-".into(),
    }
}

fn short(p: Proportion) -> String {
    format!("{}/{}", p.k, p.n)
}

/// The report in Markdown.
#[must_use]
pub fn markdown(report: &Report) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Absorb cycle: {}\n", report.run);
    let _ = writeln!(out, "**{}**\n", report.verdict.text);
    let _ = writeln!(
        out,
        "Pre-registration `{}`. The unit is the fact; each cell is facts right out of facts, \
         with the exact Clopper-Pearson 95% interval. Direct recall means the paraphrase and \
         reverse probes are both right. Every number is unadjudicated.\n",
        report.prereg_hash
    );
    let _ = writeln!(out, "## Criteria\n");
    for c in &report.verdict.criteria {
        let _ = writeln!(
            out,
            "- {} {:?}{}: {}",
            c.name,
            c.status,
            if c.gating { "" } else { " (not gating)" },
            c.detail
        );
    }
    let _ = writeln!(out, "\n## Nights, direct recall by group\n");
    let groups = [
        "today",
        "earlier",
        "future",
        "canary",
        "control-untaught",
        "control-known",
        "neighbour-known",
    ];
    let _ = writeln!(out, "| night | release | {} |", groups.join(" | "));
    let _ = writeln!(out, "|---|---|{}|", groups.map(|_| "---").join("|"));
    for n in &report.nights {
        let cells: Vec<String> = groups
            .iter()
            .map(|g| cell(n.groups.get(*g).and_then(|s| s.direct)))
            .collect();
        let _ = writeln!(
            out,
            "| {} | {} | {} |",
            n.night,
            n.release.as_deref().unwrap_or("base"),
            cells.join(" | ")
        );
    }
    let _ = writeln!(out, "\n## Nights, other suites (items right)\n");
    let _ = writeln!(
        out,
        "| night | sycophancy resisted | unknown declined | anchor | general | judge abstained | mean log-prob (release / base) |\n|---|---|---|---|---|---|---|"
    );
    for n in &report.nights {
        let logprob = n.logprob.as_ref().map_or("not measured".to_string(), |l| {
            format!("{:.3} / {:.3} over {} tokens", l.release, l.base, l.tokens)
        });
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {logprob} |",
            n.night,
            short(n.sycophancy),
            short(n.unknown),
            short(n.anchor),
            short(n.general),
            n.abstained
        );
    }
    if let Some(last) = report.nights.last() {
        let _ = writeln!(out, "\n## Night {}, facts by probe kind\n", last.night);
        let _ = writeln!(out, "| group | facts | paraphrase | reverse | indirect (secondary) | application (secondary) | direct recall |\n|---|---|---|---|---|---|---|");
        for g in groups {
            if let Some(s) = last.groups.get(g) {
                let kind = |k: &str| cell(s.by_kind.get(k).copied());
                let _ = writeln!(
                    out,
                    "| {g} | {} | {} | {} | {} | {} | {} |",
                    s.facts,
                    kind("paraphrase"),
                    kind("reverse"),
                    kind("indirect"),
                    kind("application"),
                    cell(s.direct)
                );
            }
        }
        let taught = |name: &str| last.groups.get(name).and_then(|s| s.direct);
        let (today, earlier) = (taught("today"), taught("earlier"));
        let both = Proportion::new(
            today.map_or(0, |p| p.k) + earlier.map_or(0, |p| p.k),
            today.map_or(0, |p| p.n) + earlier.map_or(0, |p| p.n),
        );
        if let (Some(control), true) = (taught("control-untaught"), both.n > 0) {
            if let Some((d, (low, high))) = difference(&both, &control) {
                let _ = writeln!(
                    out,
                    "\nFacts taught minus untaught controls, direct recall: {d:+.3} (Newcombe 95% interval {low:+.3} to {high:+.3}).",
                );
            }
        }
    }
    let _ = writeln!(
        out,
        "\n## Items that went from right to wrong since the night before\n"
    );
    let _ = writeln!(
        out,
        "| night | class | right to wrong | wrong to right |\n|---|---|---|---|"
    );
    for t in &report.transitions {
        for (class, flips) in &t.by_class {
            if !flips.right_to_wrong.is_empty() || flips.wrong_to_right > 0 {
                let _ = writeln!(
                    out,
                    "| {} | {class} | {} | {} |",
                    t.night,
                    flips.right_to_wrong.len(),
                    flips.wrong_to_right
                );
            }
        }
    }
    if !report.verdict.canary.is_empty() {
        let _ = writeln!(out, "\n## Canary\n\n| night | follows the false statement | right by the true statement |\n|---|---|---|");
        for c in &report.verdict.canary {
            let _ = writeln!(out, "| {} | {} | {} |", c.night, c.adopted, c.true_recall);
        }
    }
    let _ = writeln!(out, "\n## Nightly updates and timings\n");
    let _ = writeln!(out, "| night | sessions | released | live claims | stopped | recording s | update s | policy answers s | judge s |\n|---|---|---|---|---|---|---|---|---|");
    for m in &report.meta {
        let seconds = report
            .evaluation_seconds
            .iter()
            .find(|(n, _, _)| *n == m.night);
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            m.night,
            m.sessions,
            m.released.map_or("-".into(), |r| r.to_string()),
            m.live_claims.map_or("-".into(), |c| c.to_string()),
            m.stopped.as_deref().unwrap_or("-"),
            m.record_seconds.map_or("-".into(), |s| format!("{s:.0}")),
            m.absorb_seconds.map_or("-".into(), |s| format!("{s:.0}")),
            seconds.map_or("-".into(), |(_, a, _)| format!("{a:.0}")),
            seconds
                .and_then(|(_, _, j)| *j)
                .map_or("-".into(), |s| format!("{s:.0}")),
        );
    }
    let _ = writeln!(out, "\n## Runs of this fact pool\n");
    for r in &report.runs {
        let _ = writeln!(
            out,
            "- run {} in `{}`: {}",
            r.run,
            r.dir,
            r.text.as_deref().unwrap_or("not finished")
        );
    }
    out
}
