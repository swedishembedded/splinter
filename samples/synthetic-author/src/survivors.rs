// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in finding out what a filtered
// training corpus lost on the way, you can procure our services by sending an
// email to info@swedishembedded.com.

//! What the gates took out of a corpus: the decisions planned against the ones
//! kept, by option, by deciding belief and by combination of facts and options,
//! and why the articles that were given up were given up.
//!
//! A corpus keeps only the essays that pass its gates, so what it teaches is
//! not what was planned. Reading the plan, the kept articles and the log of
//! turned-away attempts shows where the loss falls; no model is asked.

use std::collections::BTreeMap;

use anyhow::Context as _;
use serde::Serialize;

use crate::cache::{Cache, Kind, StoredArticle};
use crate::episode::Episode;
use crate::persona::Persona;
use crate::scenario::Features;
use crate::writer::plan;

/// The label of an article that was planned, not kept and has no logged attempt.
pub const UNATTEMPTED: &str = "unattempted";
/// The label of an article given up on after content rejections. Rows written
/// before attempts had a kind are content rejections.
pub const CONTENT: &str = "content";

/// Decisions planned and decisions kept under one label.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Tally {
    /// Decisions the plan tells.
    pub planned: usize,
    /// Of those, the ones an article kept tells.
    pub kept: usize,
}

/// What a corpus lost.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Survivors {
    /// Articles planned.
    pub articles_planned: usize,
    /// Of those, articles kept.
    pub articles_kept: usize,
    /// Decisions by the option the author chose.
    pub by_option: BTreeMap<String, Tally>,
    /// Decisions by the belief that decided them.
    pub by_belief: BTreeMap<String, Tally>,
    /// Decisions by the facts and the options the case offered.
    pub by_combination: BTreeMap<String, Tally>,
    /// The planned articles that were not kept, by why: `content` when some
    /// attempt was turned away by a gate, otherwise the kind of the last
    /// attempt (`writer-unavailable`, `reader-unavailable`, `budget`), or
    /// `unattempted`. Each label lists the article indices.
    pub given_up: BTreeMap<String, Vec<usize>>,
}

fn name_of(value: &impl Serialize) -> anyhow::Result<String> {
    match serde_json::to_value(value)? {
        serde_json::Value::String(name) => Ok(name),
        other => anyhow::bail!("{other} has no name"),
    }
}

/// The facts and options of a situation as one label.
fn combination(features: &Features, options: &[impl Serialize]) -> anyhow::Result<String> {
    let facts = [
        ("beyond_charter", features.exceeds_authority),
        ("urgent", features.urgent),
        ("binds_future", features.binds_future),
        ("irreversible", features.irreversible),
        ("members_harmed", features.members_harmed),
        ("outsider_offers_help", features.outsider_offers_help),
    ]
    .iter()
    .map(|(name, on)| format!("{name}={}", u8::from(*on)))
    .collect::<Vec<_>>()
    .join(" ");
    let options = options
        .iter()
        .map(name_of)
        .collect::<anyhow::Result<Vec<_>>>()?
        .join(",");
    Ok(format!("{facts} | {options}"))
}

/// The kind of every logged attempt, by article.
fn attempts_by_article(log: &str) -> anyhow::Result<BTreeMap<usize, Vec<Kind>>> {
    let mut out: BTreeMap<usize, Vec<Kind>> = BTreeMap::new();
    for (n, line) in log.lines().enumerate() {
        let line_no = n + 1;
        let row: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("rejected.jsonl line {line_no} is not JSON"))?;
        let article = row["article"]
            .as_u64()
            .and_then(|i| usize::try_from(i).ok())
            .with_context(|| format!("rejected.jsonl line {line_no} has no article index"))?;
        let kind = match row.get("kind").filter(|k| !k.is_null()) {
            Some(kind) => serde_json::from_value(kind.clone())
                .with_context(|| format!("rejected.jsonl line {line_no} has an unknown kind"))?,
            None => Kind::Content,
        };
        out.entry(article).or_default().push(kind);
    }
    Ok(out)
}

fn given_up_label(attempts: Option<&Vec<Kind>>) -> anyhow::Result<String> {
    match attempts.and_then(|kinds| kinds.last().map(|last| (kinds, last))) {
        None => Ok(UNATTEMPTED.to_string()),
        Some((kinds, _)) if kinds.contains(&Kind::Content) => Ok(CONTENT.to_string()),
        Some((_, last)) => name_of(last),
    }
}

/// Compares the `planned` episodes of `persona`'s corpus with the articles
/// `kept` and accounts for the articles that were given up from `log`, the
/// text of `rejected.jsonl`. A decision is kept when the article that tells it
/// was kept with that case; cases are found by article and name together,
/// because names repeat from one article to another.
///
/// # Errors
/// A line of `log` is not JSON, has no article index or an unknown kind; the
/// error names the line.
pub fn survivors(
    persona: &Persona,
    planned: &[(usize, Vec<Episode>)],
    kept: &[StoredArticle],
    log: &str,
) -> anyhow::Result<Survivors> {
    let kept_cases: BTreeMap<usize, &StoredArticle> = kept.iter().map(|a| (a.index, a)).collect();
    let attempts = attempts_by_article(log)?;
    let mut out = Survivors {
        articles_planned: planned.len(),
        ..Survivors::default()
    };
    for (index, (_, episodes)) in planned.iter().enumerate() {
        let article = kept_cases.get(&index);
        if article.is_some() {
            out.articles_kept += 1;
        } else {
            out.given_up
                .entry(given_up_label(attempts.get(&index))?)
                .or_default()
                .push(index);
        }
        for episode in episodes {
            let told = article.is_some_and(|a| a.cases.contains(&episode.case.name));
            let decision = persona.decide(&episode.scenario);
            let labels = [
                (&mut out.by_option, name_of(&decision.choice)?),
                (
                    &mut out.by_belief,
                    decision
                        .deciding
                        .map_or_else(|| Ok("none".to_string()), |tenet| name_of(&tenet))?,
                ),
                (
                    &mut out.by_combination,
                    combination(&episode.scenario.features, &episode.scenario.options)?,
                ),
            ];
            for (tallies, label) in labels {
                let tally = tallies.entry(label).or_default();
                tally.planned += 1;
                tally.kept += usize::from(told);
            }
        }
    }
    Ok(out)
}

/// The report for the corpus in `cache`, planned for `persona` as `articles`
/// articles of `per_article` episodes. A corpus that turned nothing away has
/// no log of attempts, which reads as an empty one.
///
/// # Errors
/// The articles or the log cannot be read, or the log is malformed.
pub fn of_corpus(
    cache: &Cache,
    persona: &Persona,
    articles: usize,
    per_article: usize,
) -> anyhow::Result<Survivors> {
    let planned = plan(persona, persona.seed, articles, per_article);
    let path = cache.dir().join("rejected.jsonl");
    let log = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    survivors(persona, &planned, &cache.articles()?, &log)
}
