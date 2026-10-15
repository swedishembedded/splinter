// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The experiment on one author, stage by stage: read the decisions out of the
//! articles, make the training records, train, examine, report. Every stage
//! reads and writes files in the corpus directory, so any stage can be run
//! again, or run on another machine, without the others.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::store::jsonl;
use splinter_sdk::vocabulary::digest::{canonical_json, Digest};

use crate::arms::{run_items, run_items_with, Outcome};
use crate::binding::{bind, items_digest, ArmRecord, ModelIdentity};
use crate::cache::{Cache, StoredArticle};
use crate::dataset::records_for;
use crate::exam::{items, Item};
use crate::gates::same_ways;
use crate::mine::{mine, Verified};
use crate::persona::Persona;
use crate::prompts::SYSTEM;
use crate::scenario::Archetype;
use crate::writer::plan;

/// One line of `decisions.jsonl`: what was read out of one case of one article.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mined {
    /// The article.
    pub article: usize,
    /// The case.
    pub case: String,
    /// The decision, or why it was turned away.
    #[serde(flatten)]
    pub result: MinedResult,
}

/// What came of reading one case.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MinedResult {
    /// A decision that passed.
    Kept(Verified),
    /// Why it did not.
    Refused(String),
}

fn decisions_path(cache: &Cache) -> PathBuf {
    cache.dir().join("decisions.jsonl")
}

/// Every case read so far.
///
/// # Errors
/// The file cannot be read.
pub fn read_mined(cache: &Cache) -> anyhow::Result<Vec<Mined>> {
    Ok(jsonl::read(&decisions_path(cache))?)
}

/// Reads the decisions out of every article not yet read, `workers` at a time.
/// Returns how many articles were read by this call.
///
/// # Errors
/// The cache cannot be read or written, or the reader fails on an article.
pub fn mine_all(cache: &Cache, miner: &BlockingModel, workers: usize) -> anyhow::Result<usize> {
    let articles = cache.articles()?;
    let done: BTreeSet<usize> = read_mined(cache)?.iter().map(|m| m.article).collect();
    let todo: Vec<&StoredArticle> = articles
        .iter()
        .filter(|a| !done.contains(&a.index))
        .collect();
    let log = jsonl::Log::open(&decisions_path(cache))?;
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1) {
            scope.spawn(|| loop {
                let n = next.fetch_add(1, Ordering::SeqCst);
                let Some(article) = todo.get(n) else { return };
                let lines = match mine(miner, article) {
                    Ok(results) => results,
                    Err(e) => {
                        eprintln!("article {}: {e}", article.index);
                        failure
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .get_or_insert(e);
                        return;
                    }
                };
                for (case, result) in article.cases.iter().zip(lines) {
                    let result = result.map_or_else(MinedResult::Refused, MinedResult::Kept);
                    let line = Mined {
                        article: article.index,
                        case: case.clone(),
                        result,
                    };
                    if let Err(e) = log.append(&line) {
                        eprintln!("decisions.jsonl: {e}");
                        return;
                    }
                }
                eprintln!("article {} read", article.index);
            });
        }
    });
    if let Some(e) = failure.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(e);
    }
    Ok(todo.len())
}

/// What a dataset was made of.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetSummary {
    /// Decisions read and kept.
    pub decisions: usize,
    /// Cases turned away.
    pub refused: usize,
    /// Records written.
    pub records: usize,
}

/// Writes the training records for every decision kept to `out`.
///
/// # Errors
/// The decisions cannot be read or the file cannot be written.
pub fn build_dataset(cache: &Cache, seed: u64, out: &Path) -> anyhow::Result<DatasetSummary> {
    let mined = read_mined(cache)?;
    let mut summary = DatasetSummary {
        decisions: 0,
        refused: 0,
        records: 0,
    };
    let mut text = String::new();
    for m in &mined {
        match &m.result {
            MinedResult::Kept(decision) => {
                summary.decisions += 1;
                for record in records_for(seed, &format!("{}:{}", m.article, m.case), decision) {
                    text.push_str(&record.to_string());
                    text.push('\n');
                    summary.records += 1;
                }
            }
            MinedResult::Refused(_) => summary.refused += 1,
        }
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, text)?;
    Ok(summary)
}

/// How the decisions read out of the essays compare with what the beliefs
/// dictate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Audit {
    /// Decisions that were kept.
    pub kept: usize,
    /// Kept decisions that are the one the beliefs dictate.
    pub agree: usize,
    /// Kept decisions whose open ways are the ones the case offered.
    pub open_agree: usize,
}

/// Compares every kept decision of `mined` with the author's own decision, and
/// the ways it read as open with the ways the case offered, on that case, among the cases of the first `articles` planned articles. A
/// case is found by the article that tells it and its name together: names
/// are drawn from a small stock and repeat from one article to another.
#[must_use]
pub fn audit(mined: &[Mined], persona: &Persona, seed: u64, articles: usize) -> Audit {
    let oracle: std::collections::BTreeMap<(usize, String), (Archetype, Vec<Archetype>)> =
        plan(persona, seed, articles, 3)
            .into_iter()
            .enumerate()
            .flat_map(|(article, (_, episodes))| episodes.into_iter().map(move |e| (article, e)))
            .map(|(article, e)| {
                (
                    (article, e.case.name.clone()),
                    (persona.decide(&e.scenario).choice, e.scenario.options),
                )
            })
            .collect();
    let mut audit = Audit {
        kept: 0,
        agree: 0,
        open_agree: 0,
    };
    for m in mined {
        let MinedResult::Kept(decision) = &m.result else {
            continue;
        };
        let Some((expected, planned_open)) = oracle.get(&(m.article, m.case.clone())) else {
            continue;
        };
        audit.kept += 1;
        audit.agree += usize::from(decision.chose == *expected);
        audit.open_agree += usize::from(same_ways(&decision.open, planned_open));
    }
    audit
}

/// The system turn each arm of the examination is asked under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// The plain system turn, no adapter.
    Base,
    /// The plain system turn, with the adapter.
    Tuned,
    /// The system turn and the author's four convictions, in order of priority.
    Principles,
    /// The system turn and the author's name.
    Named,
    /// The plain system turn and the paragraphs of the corpus that share the
    /// most with the situation.
    Retrieval,
}

impl Arm {
    /// The arm's name, as used in file names.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Tuned => "tuned",
            Self::Principles => "principles",
            Self::Named => "named",
            Self::Retrieval => "retrieval",
        }
    }

    /// The system turn the arm is asked under.
    #[must_use]
    pub fn system(self, persona: &Persona) -> String {
        match self {
            Self::Base | Self::Tuned | Self::Retrieval => SYSTEM.to_string(),
            Self::Named => format!("{SYSTEM} You are {}.", persona.name),
            Self::Principles => {
                let mut text = format!(
                    "{SYSTEM}\n\nYou hold these convictions, listed from the most important to the least, and they decide how you settle a matter:"
                );
                for belief in &persona.beliefs {
                    text.push_str(&format!("\n- {}", belief.statement()));
                }
                text
            }
        }
    }
}

/// The examination of `persona`: `per_cell` items of each kind, with no case
/// the corpus tells.
#[must_use]
pub fn exam_items(persona: &Persona, seed: u64, articles: usize, per_cell: usize) -> Vec<Item> {
    let told: BTreeSet<String> = plan(persona, seed, articles, 3)
        .into_iter()
        .flat_map(|(_, eps)| eps.into_iter().map(|e| e.case.name))
        .collect();
    items(persona, seed, per_cell, &told)
}

/// An exam item as the model is asked it, with the answer the beliefs dictate
/// written in the labels of the tool the item offers.
#[must_use]
pub fn item_view(item: &Item) -> serde_json::Value {
    let position = item
        .options
        .iter()
        .position(|o| *o == item.expected)
        .unwrap_or(0);
    serde_json::json!({
        "id": item.id,
        "topic_seen": item.topic_seen,
        "combination_seen": item.combo_seen,
        "tool_seen": item.tool_seen,
        "tool": item.schema.name,
        "prompt": crate::prompts::render(&item.briefing, &item.options, &item.schema),
        "expected": item.schema.label(item.expected, position),
    })
}

/// Where the outcomes of `arm` are kept.
#[must_use]
pub fn outcomes_path(cache: &Cache, arm: Arm) -> PathBuf {
    cache
        .dir()
        .join("exam")
        .join(format!("{}.jsonl", arm.name()))
}

/// How many paragraphs the retrieval arm is handed with each situation.
pub const RETRIEVED_PARAGRAPHS: usize = 6;

/// What an examination of one arm was asked with, besides the items: the
/// flags that made them and the model that answers.
#[derive(Clone, Debug)]
pub struct ExamRun {
    /// How many articles the corpus was planned with.
    pub articles: usize,
    /// Items of each kind of difficulty.
    pub per_cell: usize,
    /// The most tokens a reply may run to.
    pub max_tokens: u32,
    /// The model that answers.
    pub model: ModelIdentity,
    /// Take over outcomes that exist without a manifest.
    pub adopt: bool,
}

/// The record of `arm` examined with `run` on `items`: what its outcomes
/// depend on.
///
/// # Errors
/// The retrieval arm's corpus cannot be read.
pub fn arm_record(
    cache: &Cache,
    persona: &Persona,
    arm: Arm,
    items: &[Item],
    run: &ExamRun,
) -> anyhow::Result<ArmRecord> {
    let told = match arm {
        Arm::Retrieval => serde_json::to_string(&cache.articles()?)?,
        _ => String::new(),
    };
    let context = serde_json::json!({"system": arm.system(persona), "corpus": told});
    Ok(ArmRecord {
        arm: arm.name().to_string(),
        items_digest: items_digest(items).to_string(),
        articles: run.articles,
        per_cell: run.per_cell,
        max_tokens: run.max_tokens,
        context_digest: Digest::of(&canonical_json(&context)?).to_string(),
        model: run.model.clone(),
        adopted: false,
    })
}

/// Puts the examination to `provider` as `arm`, under the manifest `run`
/// makes: the outcomes already kept for the arm must have been made under the
/// same one.
///
/// # Errors
/// The outcomes were made under another manifest, the model failed, or the
/// outcomes cannot be written.
pub fn run_arm(
    cache: &Cache,
    persona: &Persona,
    arm: Arm,
    provider: &std::sync::Arc<dyn splinter_sdk::agent::sven::model::ModelProvider>,
    items: &[Item],
    run: &ExamRun,
) -> anyhow::Result<Vec<Outcome>> {
    let path = outcomes_path(cache, arm);
    let dir = path.parent().context("a directory")?;
    std::fs::create_dir_all(dir)?;
    let record = arm_record(cache, persona, arm, items, run)?;
    bind(dir, &path, &record, run.adopt)?;
    let model = run.model.digest().to_string();
    if arm == Arm::Retrieval {
        let notes = crate::retrieval::Notes::new(&cache.articles()?);
        return run_items_with(
            provider,
            |item| notes.system_for(item, RETRIEVED_PARAGRAPHS),
            items,
            &path,
            run.max_tokens,
            &model,
        );
    }
    run_items(
        provider,
        &arm.system(persona),
        items,
        &path,
        run.max_tokens,
        &model,
    )
}
