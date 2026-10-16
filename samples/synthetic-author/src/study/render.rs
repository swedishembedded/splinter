// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The report as files a paper's scripts read and as a short table a person
//! reads. A number that was not measured is an empty field in a CSV and the
//! words `not measured` in the table: never zero.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;

use super::analysis::{AuthorReport, Report};
use crate::binding::write_whole;

/// The directory of a study that holds the report.
pub const REPORT_DIR: &str = "report";

const NOT_MEASURED: &str = "not measured";

/// One CSV: a header and its rows. Fields that contain a comma, a quote or a
/// line break are quoted.
struct Csv {
    text: String,
}

impl Csv {
    fn new(header: &[&str]) -> Self {
        let mut csv = Self {
            text: String::new(),
        };
        csv.row(header.iter().map(|h| (*h).to_string()));
        csv
    }

    fn row(&mut self, fields: impl IntoIterator<Item = String>) {
        let line: Vec<String> = fields
            .into_iter()
            .map(|f| {
                if f.contains([',', '"', '\n']) {
                    format!("\"{}\"", f.replace('"', "\"\""))
                } else {
                    f
                }
            })
            .collect();
        self.text.push_str(&line.join(","));
        self.text.push('\n');
    }
}

/// A number, or the empty field of a number that was not measured.
fn num<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(String::new, |v| v.to_string())
}

fn each_author<'a, T: 'a>(
    report: &'a Report,
    rows: impl Fn(&'a AuthorReport) -> &'a [T],
) -> impl Iterator<Item = (&'a AuthorReport, &'a T)> {
    report
        .authors
        .iter()
        .flat_map(move |a| rows(a).iter().map(move |r| (a, r)))
}

fn accuracy_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&[
        "author",
        "arm",
        "training_seed",
        "right",
        "n",
        "rate",
        "low",
        "high",
    ]);
    for (a, r) in each_author(report, |a| &a.accuracy) {
        csv.row([
            a.seed.to_string(),
            r.arm.clone(),
            num(r.training_seed),
            r.right.to_string(),
            r.n.to_string(),
            r.rate.to_string(),
            r.low.to_string(),
            r.high.to_string(),
        ]);
    }
    csv
}

fn cells_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&["author", "arm", "training_seed", "cell", "right", "n"]);
    for (a, c) in each_author(report, |a| &a.cells) {
        csv.row([
            a.seed.to_string(),
            c.arm.clone(),
            num(c.training_seed),
            c.cell.clone(),
            c.right.to_string(),
            c.n.to_string(),
        ]);
    }
    csv
}

fn paired_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&[
        "author",
        "candidate",
        "reference",
        "training_seed",
        "n",
        "only_candidate",
        "only_reference",
        "p",
    ]);
    for (a, r) in each_author(report, |a| &a.paired) {
        csv.row([
            a.seed.to_string(),
            r.candidate.clone(),
            r.reference.clone(),
            num(r.training_seed),
            r.n.to_string(),
            r.only_candidate.to_string(),
            r.only_reference.to_string(),
            r.p.to_string(),
        ]);
    }
    csv
}

fn gains_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&[
        "author",
        "candidate",
        "reference",
        "training_seeds",
        "n",
        "gain",
        "low",
        "high",
        "items_won",
        "items_lost",
        "p_items",
    ]);
    for (a, g) in each_author(report, |a| &a.gains) {
        let seeds: Vec<String> = g.training_seeds.iter().map(u64::to_string).collect();
        csv.row([
            a.seed.to_string(),
            g.candidate.clone(),
            g.reference.clone(),
            seeds.join(" "),
            g.n.to_string(),
            g.gain.to_string(),
            num(g.interval.map(|i| i.low)),
            num(g.interval.map(|i| i.high)),
            g.items_won.to_string(),
            g.items_lost.to_string(),
            g.p_items.to_string(),
        ]);
    }
    csv
}

fn across_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&[
        "candidate",
        "reference",
        "primary",
        "confirmatory",
        "authors_measured",
        "authors_total",
        "authors_improved",
        "authors_worse",
        "authors_tied",
        "p",
        "p_holm",
        "significant",
        "mean_gain",
        "min_gain",
        "max_gain",
    ]);
    for r in &report.across {
        csv.row([
            r.comparison.candidate.clone(),
            r.comparison.reference.clone(),
            r.primary.to_string(),
            r.confirmatory.to_string(),
            r.authors_measured.to_string(),
            r.authors_total.to_string(),
            r.authors_improved.to_string(),
            r.authors_worse.to_string(),
            r.authors_tied.to_string(),
            num(r.p),
            num(r.p_holm),
            num(r.significant),
            num(r.mean_gain),
            num(r.min_gain),
            num(r.max_gain),
        ]);
    }
    csv
}

fn spread_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&["author", "arm", "seeds", "mean", "min", "max", "sd"]);
    for (a, s) in each_author(report, |a| &a.spread) {
        csv.row([
            a.seed.to_string(),
            s.arm.clone(),
            s.spread.seeds.to_string(),
            s.mean.to_string(),
            s.spread.min.to_string(),
            s.spread.max.to_string(),
            num(s.spread.sd),
        ]);
    }
    csv
}

fn survivorship_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&[
        "author",
        "arm",
        "option",
        "items",
        "accuracy",
        "planned",
        "kept",
        "kept_share",
    ]);
    for (a, s) in each_author(report, |a| &a.survivorship) {
        csv.row([
            a.seed.to_string(),
            s.arm.clone(),
            s.option.clone(),
            s.items.to_string(),
            s.accuracy.to_string(),
            s.planned.to_string(),
            s.kept.to_string(),
            num(s.kept_share),
        ]);
    }
    csv
}

fn given_up_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&["author", "reason", "articles"]);
    for a in &report.authors {
        for (why, articles) in &a.given_up {
            csv.row([a.seed.to_string(), why.clone(), articles.to_string()]);
        }
    }
    csv
}

fn corpora_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&[
        "author",
        "name",
        "articles_planned",
        "articles_kept",
        "cases_read",
        "decisions_kept",
        "calls",
        "input_tokens",
        "output_tokens",
    ]);
    for a in &report.authors {
        let c = &a.corpus;
        csv.row([
            a.seed.to_string(),
            a.name.clone(),
            c.articles_planned.to_string(),
            c.articles_kept.to_string(),
            c.cases_read.to_string(),
            c.decisions_kept.to_string(),
            c.usage.calls.to_string(),
            c.usage.input_tokens.to_string(),
            c.usage.output_tokens.to_string(),
        ]);
    }
    csv
}

fn reasons_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&[
        "author",
        "arm",
        "training_seed",
        "answers",
        "reasons",
        "naming_deciding_fact",
        "copying_brief",
        "distinct",
    ]);
    for (a, r) in each_author(report, |a| &a.reasons) {
        csv.row([
            a.seed.to_string(),
            r.arm.clone(),
            num(r.training_seed),
            r.answers.to_string(),
            r.reasons.to_string(),
            num(r.naming_deciding_fact),
            num(r.copying_brief),
            r.distinct.to_string(),
        ]);
    }
    csv
}

fn sampled_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&[
        "author",
        "arm",
        "training_seed",
        "items",
        "draws",
        "share_right",
    ]);
    for (a, s) in each_author(report, |a| &a.sampled) {
        csv.row([
            a.seed.to_string(),
            s.arm.clone(),
            num(s.training_seed),
            s.items.to_string(),
            s.draws.to_string(),
            s.share_right.to_string(),
        ]);
    }
    csv
}

fn unmeasured_csv(report: &Report) -> Csv {
    let mut csv = Csv::new(&["author", "what", "why"]);
    for u in &report.unmeasured {
        csv.row([num(u.author), u.what.clone(), u.why.clone()]);
    }
    csv
}

/// Writes the report into `root/report/`: `report.json`, one CSV per table and
/// `report.txt`. Returns the directory.
///
/// # Errors
/// A file cannot be written.
pub fn write(root: &Path, report: &Report) -> anyhow::Result<PathBuf> {
    let dir = root.join(REPORT_DIR);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let files: Vec<(&str, Vec<u8>)> = vec![
        ("report.json", serde_json::to_vec_pretty(report)?),
        ("report.txt", table(report).into_bytes()),
        ("accuracy.csv", accuracy_csv(report).text.into_bytes()),
        ("cells.csv", cells_csv(report).text.into_bytes()),
        ("paired.csv", paired_csv(report).text.into_bytes()),
        ("gains.csv", gains_csv(report).text.into_bytes()),
        ("across_authors.csv", across_csv(report).text.into_bytes()),
        ("spread.csv", spread_csv(report).text.into_bytes()),
        (
            "survivorship.csv",
            survivorship_csv(report).text.into_bytes(),
        ),
        ("given_up.csv", given_up_csv(report).text.into_bytes()),
        ("corpora.csv", corpora_csv(report).text.into_bytes()),
        ("reasons.csv", reasons_csv(report).text.into_bytes()),
        ("sampled.csv", sampled_csv(report).text.into_bytes()),
        ("unmeasured.csv", unmeasured_csv(report).text.into_bytes()),
    ];
    for (name, bytes) in files {
        write_whole(&dir.join(name), &bytes)?;
    }
    Ok(dir)
}

fn percent(p: Option<f64>) -> String {
    p.map_or_else(|| NOT_MEASURED.to_string(), |p| format!("{p:.4}"))
}

/// The short human table: the comparisons across authors, the adapters'
/// accuracy and spread per author, and what is missing.
#[must_use]
pub fn table(report: &Report) -> String {
    let mut text = format!(
        "Study under protocol {}: {} of the protocol's authors measured; intervals are {:.0}% item-clustered bootstraps ({} resamples, bootstrap seed {}); Holm correction over the protocol's family at {}.\n\n",
        report.protocol_digest,
        report.authors.len(),
        report.level * 100.0,
        report.resamples,
        report.bootstrap_seed,
        report.alpha,
    );
    text.push_str("| comparison | authors | improved | worse | tied | mean gain | p | Holm p |\n|---|---|---|---|---|---|---|---|\n");
    for r in &report.across {
        let marker = match (r.primary, r.confirmatory) {
            (true, _) => " (primary)",
            (false, true) => "",
            (false, false) => " (exploratory)",
        };
        let _ = writeln!(
            text,
            "| {}{marker} | {}/{} | {} | {} | {} | {} | {} | {} |",
            r.comparison.label(),
            r.authors_measured,
            r.authors_total,
            r.authors_improved,
            r.authors_worse,
            r.authors_tied,
            r.mean_gain
                .map_or_else(|| NOT_MEASURED.to_string(), |g| format!("{g:+.3}")),
            percent(r.p),
            percent(r.p_holm),
        );
    }
    text.push_str("\n| author | arm | seeds | accuracy (mean) | min | max | sd |\n|---|---|---|---|---|---|---|\n");
    for (a, s) in each_author(report, |a| &a.spread) {
        let _ = writeln!(
            text,
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {} |",
            a.seed,
            s.arm,
            s.spread.seeds,
            s.mean,
            s.spread.min,
            s.spread.max,
            s.spread
                .sd
                .map_or_else(|| NOT_MEASURED.to_string(), |sd| format!("{sd:.3}")),
        );
    }
    text.push_str("\n| author | first listed | majority option |\n|---|---|---|\n");
    for a in &report.authors {
        let rate = |name: &str| {
            a.accuracy.iter().find(|r| r.arm == name).map_or_else(
                || NOT_MEASURED.to_string(),
                |r| format!("{:.3} ({}/{})", r.rate, r.right, r.n),
            )
        };
        let _ = writeln!(
            text,
            "| {} | {} | {} |",
            a.seed,
            rate(super::arm_key::FIRST_LISTED),
            rate(super::arm_key::MAJORITY)
        );
    }
    for u in &report.unmeasured {
        let who = u
            .author
            .map_or_else(String::new, |a| format!("author {a} "));
        let _ = writeln!(text, "\n{who}{}: {NOT_MEASURED} ({})", u.what, u.why);
    }
    text
}
