// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The experiment on one author, stage by stage: read the decisions out of the
//! articles, make the training records, train, examine, report. Every stage
//! reads and writes files in the corpus directory, so any stage can be run
//! again, or run on another machine, without the others.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::CallError;
use splinter_sdk::store::jsonl;
use splinter_sdk::vocabulary::digest::{canonical_json, Digest};

pub use crate::arm::Arm;
use crate::arm::{Over, Retrieval};
use crate::arms::{run_items_with, Outcome};
use crate::binding::{bind, items_digest, ArmRecord, ModelIdentity, RetrievalSetting};
use crate::cache::{Cache, StoredArticle};
use crate::dataset::{motivation_of, records_with, Briefings};
use crate::exam::{draw, items, Item, Spec, Trained, Version};
use crate::gates::same_ways;
use crate::manifest::author;
use crate::mine::{read_decisions, verify, Verified};
use crate::persona::Persona;
use crate::retrieval::Notes;
use crate::scenario::Archetype;
use crate::transit::{self, Retried, Transit};
use crate::writer::{plan, Models, RequestKey, Role};

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

/// Reads the decisions out of every article that lacks some of its cases in
/// `decisions.jsonl`, `workers` at a time, each article by a model `miner`
/// makes for it (seeded from the recorded run seed when it runs here), and appends the lines for the cases
/// it lacks. A request the provider failed at is repeated as `transit` says.
/// Returns how many articles were read by this call.
///
/// # Errors
/// The cache cannot be read, a line cannot be appended, or the reader fails on
/// an article or stays unreachable. The first failure is returned after the
/// workers have stopped.
pub fn mine_all(
    cache: &Cache,
    miner: &dyn Models,
    workers: usize,
    transit: &Transit,
) -> anyhow::Result<usize> {
    let articles = cache.articles()?;
    let mut have: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    for line in read_mined(cache)? {
        have.entry(line.article).or_default().insert(line.case);
    }
    let lacking = |article: &StoredArticle| -> Vec<String> {
        let done = have.get(&article.index);
        article
            .cases
            .iter()
            .filter(|case| !done.is_some_and(|d| d.contains(*case)))
            .cloned()
            .collect()
    };
    let todo: Vec<(&StoredArticle, Vec<String>)> = articles
        .iter()
        .map(|article| (article, lacking(article)))
        .filter(|(_, lacking)| !lacking.is_empty())
        .collect();
    let log = jsonl::Log::open(&decisions_path(cache))?;
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let fail = |e: anyhow::Error| {
        failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert(e);
    };
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1) {
            scope.spawn(|| loop {
                let n = next.fetch_add(1, Ordering::SeqCst);
                if failure
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .is_some()
                {
                    return;
                }
                let Some((article, lacking)) = todo.get(n) else {
                    return;
                };
                let key = RequestKey {
                    article: article.index,
                    attempt: 1,
                    role: Role::Miner,
                };
                let read = transit::retry(
                    transit,
                    || true,
                    || {
                        let model = miner
                            .model(key)
                            .map_err(|e| CallError::Precondition(format!("{e:#}")))?;
                        read_decisions(&model, &article.body, &article.cases).result
                    },
                    |loss| {
                        eprintln!(
                            "article {}: the miner could not be reached: {}",
                            article.index, loss.summary
                        );
                        Ok(())
                    },
                );
                let lines = match read {
                    Ok(Retried::Answered(mined)) => verify(article, &mined),
                    Ok(Retried::Refused(e)) => {
                        fail(anyhow::Error::new(e).context(format!("reading article {}", article.index)));
                        return;
                    }
                    Ok(Retried::Exhausted { last, requests }) => {
                        fail(anyhow::anyhow!(
                            "reading article {}: the miner could not be reached: {last}, in {requests} requests",
                            article.index
                        ));
                        return;
                    }
                    Ok(Retried::Halted) => {
                        fail(anyhow::anyhow!(
                            "reading article {}: the request was declined",
                            article.index
                        ));
                        return;
                    }
                    Err(e) => {
                        fail(e);
                        return;
                    }
                };
                for (case, result) in article.cases.iter().zip(lines) {
                    if !lacking.contains(case) {
                        continue;
                    }
                    let result = result.map_or_else(MinedResult::Refused, MinedResult::Kept);
                    let line = Mined {
                        article: article.index,
                        case: case.clone(),
                        result,
                    };
                    if let Err(e) = log.append(&line) {
                        fail(anyhow::Error::new(e).context(format!(
                            "appending article {} case {case} to decisions.jsonl",
                            article.index
                        )));
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
    /// Which briefings the records were made with.
    pub briefings: Briefings,
    /// Where the records were written.
    pub path: PathBuf,
    /// Decisions read and kept.
    pub decisions: usize,
    /// Cases turned away.
    pub refused: usize,
    /// Decisions kept that made no record, for want of a reason of the
    /// author's own to teach or, with the fixed-wording briefing alone, of
    /// every fact read.
    #[serde(default)]
    pub without_record: usize,
    /// Records written.
    pub records: usize,
}

/// Writes the training records for every decision kept to the file `briefings`
/// names in the dataset directory of `cache`.
///
/// # Errors
/// The decisions cannot be read or the file cannot be written.
pub fn build_dataset(
    cache: &Cache,
    seed: u64,
    briefings: Briefings,
) -> anyhow::Result<DatasetSummary> {
    let mined = read_mined(cache)?;
    let out = crate::dataset::path(cache, briefings);
    let mut summary = DatasetSummary {
        briefings,
        path: out.clone(),
        decisions: 0,
        refused: 0,
        without_record: 0,
        records: 0,
    };
    let mut text = String::new();
    for m in &mined {
        match &m.result {
            MinedResult::Kept(decision) => {
                summary.decisions += 1;
                let records = records_with(
                    seed,
                    &format!("{}:{}", m.article, m.case),
                    decision,
                    briefings,
                );
                summary.without_record += usize::from(records.is_empty());
                for record in records {
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
    std::fs::write(&out, text)?;
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

/// The names of the cases the first `articles` planned articles tell.
fn told_cases(persona: &Persona, seed: u64, articles: usize) -> BTreeSet<String> {
    plan(persona, seed, articles, 3)
        .into_iter()
        .flat_map(|(_, eps)| eps.into_iter().map(|e| e.case.name))
        .collect()
}

/// The first examination of `persona`: `per_cell` items of each kind, with no
/// case the corpus tells.
#[must_use]
pub fn exam_items(persona: &Persona, seed: u64, articles: usize, per_cell: usize) -> Vec<Item> {
    items(
        persona,
        seed,
        per_cell,
        &told_cases(persona, seed, articles),
    )
}

/// The examination of `persona` in `version`. The second version needs the
/// decisions read out of the corpus, to say which items it trained on.
///
/// # Errors
/// The second version is asked for and the corpus's decisions have not been
/// read (`mine`), or cannot be.
pub fn exam_items_of(
    cache: &Cache,
    persona: &Persona,
    seed: u64,
    articles: usize,
    per_cell: usize,
    version: Version,
) -> anyhow::Result<Vec<Item>> {
    let trained = match version {
        Version::V1 => Trained::default(),
        Version::V2 => trained_on(cache)?,
    };
    Ok(draw(
        persona,
        seed,
        &Spec {
            version,
            per_cell,
            corpus_cases: &told_cases(persona, seed, articles),
            trained: &trained,
        },
    ))
}

/// The items the arm of `record` was asked, drawn again from what the record
/// says they were made with.
///
/// # Errors
/// The author cannot be read from the corpus, the items cannot be drawn, or
/// they are no longer the ones the arm was asked: a baseline scored on other
/// items would say nothing about the arms.
pub fn asked_items(cache: &Cache, record: &ArmRecord) -> anyhow::Result<Vec<Item>> {
    let author = author(cache.dir(), None)?;
    let items = exam_items_of(
        cache,
        &author.persona,
        author.seed,
        record.articles,
        record.per_cell,
        record.exam,
    )?;
    let now = items_digest(&items);
    anyhow::ensure!(
        record.items_digest == now.as_str(),
        "the baselines are not scored: the items drawn again from {} are {now} and arm {} was asked {} (the corpus was changed after the examination, for instance mined again)",
        cache.dir().display(),
        record.arm,
        record.items_digest,
    );
    Ok(items)
}

/// Every decision kept from the corpus in `cache`, for `purpose`.
///
/// # Errors
/// No decisions have been read out of the corpus: the message says to run
/// `mine` before `purpose`.
fn kept_decisions(cache: &Cache, purpose: &str) -> anyhow::Result<Vec<Verified>> {
    let read = if decisions_path(cache).exists() {
        read_mined(cache)?
    } else {
        Vec::new()
    };
    anyhow::ensure!(
        !read.is_empty(),
        "{} holds no decisions read out of the corpus: run `mine` {purpose}",
        cache.dir().display()
    );
    Ok(read
        .into_iter()
        .filter_map(|m| match m.result {
            MinedResult::Kept(decision) => Some(decision),
            MinedResult::Refused(_) => None,
        })
        .collect())
}

/// What the kept decisions of the corpus in `cache` cover.
fn trained_on(cache: &Cache) -> anyhow::Result<Trained> {
    let kept = kept_decisions(
        cache,
        "before drawing the second examination, which says which items the corpus trained on",
    )?;
    Ok(Trained::from_decisions(&kept))
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
        "options": item.options,
        "expected_position": position,
        "topic_seen": item.topic_seen,
        "combination_seen": item.combo_seen,
        "combination_trained": item.combo_trained,
        "tool_seen": item.tool_seen,
        "tool": item.schema.name,
        "prompt": crate::prompts::render(&item.briefing, &item.options, &item.schema),
        "expected": item.schema.label(item.expected, position),
    })
}

/// Where the outcomes of `arm` are kept.
#[must_use]
pub fn outcomes_path(cache: &Cache, arm: Arm) -> PathBuf {
    cache.exam_dir().join(format!("{}.jsonl", arm.name()))
}

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
    /// Which examination the items are of.
    pub exam: Version,
    /// Take over outcomes that exist without a manifest.
    pub adopt: bool,
    /// Items put to the model at once: what a batching server decodes
    /// together. The outcomes do not depend on it, so it is no part of the
    /// manifest.
    pub slots: usize,
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
    let told = match arm.retrieval() {
        Some(Retrieval {
            over: Over::Articles,
            ..
        }) => serde_json::to_string(&cache.articles()?)?,
        Some(Retrieval {
            over: Over::Records,
            ..
        }) => serde_json::to_string(&retrieved_records(cache)?)?,
        None => String::new(),
    };
    let context = serde_json::json!({"system": arm.system(persona), "corpus": told});
    Ok(ArmRecord {
        arm: arm.name().to_string(),
        exam: run.exam,
        items_digest: items_digest(items).to_string(),
        articles: run.articles,
        per_cell: run.per_cell,
        max_tokens: run.max_tokens,
        context_digest: Digest::of(&canonical_json(&context)?).to_string(),
        model: run.model.clone(),
        draws: None,
        retrieval: arm
            .retrieval()
            .filter(|&r| r != Retrieval::ORIGINAL)
            .map(RetrievalSetting::of),
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
    run_items_with(
        provider,
        system_of(cache, persona, arm)?,
        items,
        &path,
        run.max_tokens,
        &run.model.digest().to_string(),
        run.slots,
    )
}

/// The system turn `arm` puts each item under: the same for every item, except
/// that the retrieval arm adds the paragraphs of the corpus that share the
/// most with the situation.
///
/// # Errors
/// The retrieval arm's corpus cannot be read.
pub fn system_of(
    cache: &Cache,
    persona: &Persona,
    arm: Arm,
) -> anyhow::Result<impl Fn(&Item) -> String> {
    let retrieval = arm.retrieval();
    let notes = match retrieval {
        Some(Retrieval {
            over: Over::Articles,
            ..
        }) => Some(Notes::new(&cache.articles()?)),
        Some(Retrieval {
            over: Over::Records,
            ..
        }) => Some(Notes::of_decisions(&retrieved_records(cache)?)),
        None => None,
    };
    let system = arm.system(persona);
    Ok(move |item: &Item| match (&notes, retrieval) {
        (Some(notes), Some(r)) => notes.system_for(item, r.passages.count()),
        _ => system.clone(),
    })
}

/// The decisions the training records are made from: those kept from the
/// corpus that have a reason of the author's own to teach.
fn retrieved_records(cache: &Cache) -> anyhow::Result<Vec<Verified>> {
    Ok(kept_decisions(
        cache,
        "before asking an arm that retrieves from the training records",
    )?
    .into_iter()
    .filter(|d| motivation_of(d).is_some())
    .collect())
}
