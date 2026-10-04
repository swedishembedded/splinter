// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured fine-tuning of language models on
// historical records for its clients. If your team needs expertise in training
// a model on a source corpus and proving what it kept, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The commands that take the frozen split to a trained adapter and a measured
//! before-and-after: build the tasks, train, ask the exam, report.

use std::path::{Path, PathBuf};

use crate::curate::Document;
use crate::split::{self, Assignment, Split};
use crate::tasks::{self, PERSONA};

/// The assignments the freeze command wrote.
fn read_assignments(resources: &Path) -> anyhow::Result<Vec<Assignment>> {
    let path = resources.join("frozen").join("assignments.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("{}: {e}; run the freeze command first", path.display()))?;
    text.lines()
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line)?;
            let field = |name: &str| {
                v[name]
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow::anyhow!("{}: no {name}", path.display()))
            };
            let split: Split = serde_json::from_value(v["split"].clone())?;
            Ok(Assignment {
                doc_id: field("doc_id")?,
                family: field("family")?,
                split,
            })
        })
        .collect()
}

/// Build the training set, the frozen exam and the seen sample from the
/// frozen split, after checking once more that nothing leaks.
pub fn tasks_command(
    resources: &Path,
    documents: &[Document],
    seen_per_kind: usize,
) -> anyhow::Result<()> {
    let assignments = read_assignments(resources)?;
    let leaks = split::leaks(documents, &assignments);
    anyhow::ensure!(
        leaks.is_empty(),
        "{} training documents share text with held-out ones",
        leaks.len()
    );
    let built = tasks::build(documents, &assignments, seen_per_kind);
    tasks::write_all(&built, &resources.join("tasks"))?;
    for kind in tasks::KIND_ORDER {
        let count = |set: &[tasks::Task]| set.iter().filter(|t| t.kind.name() == kind).count();
        println!(
            "{kind:>13}: train {:>4}  exam {:>4}  seen {:>3}",
            count(&built.train),
            count(&built.exam),
            count(&built.seen)
        );
    }
    println!(
        "training records: {}  exam questions: {}  seen questions: {}",
        built.train.len(),
        built.exam.len(),
        built.seen.len()
    );
    Ok(())
}

/// What one training run needs.
pub struct Train {
    pub base: PathBuf,
    pub dataset: PathBuf,
    pub attempt: PathBuf,
    pub steps: u32,
    pub rank: u32,
    pub alpha: f32,
    pub learning_rate: Option<f32>,
    pub bf16: bool,
}

/// Fine-tune a LoRA adapter on the training set through Splinter's trainer,
/// scoring it on the records the trainer holds out, and record the summary.
pub fn train_command(t: &Train) -> anyhow::Result<()> {
    std::fs::create_dir_all(&t.attempt)?;
    let started = std::time::Instant::now();
    let report = |r: &splinter_sdk::model::train::StepReport| {
        eprintln!(
            "step {}/{} loss {:.4} elapsed {:.0}s",
            r.step,
            r.steps,
            r.loss,
            started.elapsed().as_secs_f64()
        );
    };
    let split = splinter_sdk::data::holdout::split_dataset_file(&t.dataset, &t.attempt)?;
    let request = splinter_sdk::model::train::FineTune {
        model_dir: &t.base,
        train: &split.train,
        held_out: &split.held_out,
        attempt_dir: &t.attempt,
        steps: t.steps,
        rank: t.rank,
        alpha: t.alpha,
        replay: &[],
        continue_from: None,
        cancel: None,
        bf16_base: t.bf16,
        learning_rate: t.learning_rate,
        on_step: Some(splinter_sdk::model::train::StepHook(&report)),
    };
    let trained = splinter_sdk::model::train::fine_tune(&request)?;
    let summary = serde_json::json!({
        "adapter": trained.adapter,
        "adapter_digest": trained.adapter_digest,
        "base_digest": trained.base_digest,
        "records": trained.records,
        "block": trained.block,
        "steps": t.steps,
        "seconds": started.elapsed().as_secs_f64(),
        "held_out_before": {"loss": trained.base.loss, "token_accuracy": trained.base.token_accuracy, "positions": trained.base.positions},
        "held_out_after": {"loss": trained.tuned.loss, "token_accuracy": trained.tuned.token_accuracy, "positions": trained.tuned.positions},
    });
    let text = serde_json::to_string_pretty(&summary)?;
    std::fs::write(t.attempt.join("summary.json"), &text)?;
    println!("{text}");
    Ok(())
}

/// What one exam run needs.
pub struct Exam {
    pub tasks: PathBuf,
    pub out: PathBuf,
    pub base: PathBuf,
    pub adapter: Option<PathBuf>,
    pub max_tokens: u32,
    pub limit: Option<usize>,
}

/// Ask one model, with or without its adapter, every question not yet
/// answered in `out`.
pub fn exam_command(e: &Exam) -> anyhow::Result<()> {
    let questions = tasks::read_tasks(&e.tasks)?;
    let model = splinter_sdk::model::exam::Model {
        base: &e.base,
        adapter: e.adapter.as_deref(),
        system: PERSONA,
        max_tokens: e.max_tokens,
        label: "adams",
    };
    let asked =
        splinter_sdk::model::exam::run(&questions, &e.out, e.limit, &model, &tasks::is_correct)?;
    println!("asked {asked} questions; results in {}", e.out.display());
    Ok(())
}

/// The before-and-after table, with the paired sign test, of two result files.
pub fn report_command(before: &Path, after: &Path) -> anyhow::Result<()> {
    let before = splinter_sdk::model::exam::read_results(before)?;
    let after = splinter_sdk::model::exam::read_results(after)?;
    print!(
        "{}",
        splinter_sdk::model::report::render(&before, &after, &tasks::KIND_ORDER, &tasks::chance)
    );
    Ok(())
}

/// Where the helper is, and how much of the corpus to ask it about.
pub struct Mine<'a> {
    pub served: crate::helper::Served<'a>,
    pub limit: Option<usize>,
}

/// Ask the helper for principles about the documents a model may learn from,
/// check every proposal against them, and write what stands.
pub fn principles_command(
    resources: &Path,
    documents: &[Document],
    mine: &Mine<'_>,
) -> anyhow::Result<()> {
    let assignments = read_assignments(resources)?;
    let allowed: std::collections::HashSet<String> = assignments
        .iter()
        .filter(|a| a.split == Split::Train)
        .map(|a| a.doc_id.clone())
        .collect();
    let bundles = crate::miner::bundles(
        documents,
        &allowed,
        crate::miner::DOCUMENTS_PER_BUNDLE,
        crate::miner::WORDS_PER_DOCUMENT,
    );
    let dir = resources.join("principles");
    std::fs::create_dir_all(&dir)?;
    let results = dir.join("results.jsonl");
    let helper = crate::helper::Helper::served(&mine.served)?;
    let asked =
        crate::miner::mine_all(&helper, &bundles, documents, &allowed, &results, mine.limit)?;

    let all = crate::miner::read_results(&results)?;
    let principles: Vec<&crate::principles::Principle> =
        all.iter().flat_map(|r| &r.principles).collect();
    let mut text = String::new();
    for p in &principles {
        text.push_str(&serde_json::to_string(p)?);
        text.push('\n');
    }
    std::fs::write(dir.join("principles.jsonl"), text)?;
    let recurring = principles
        .iter()
        .filter(|p| p.status == crate::principles::Status::Recurring)
        .count();
    let rejected: usize = all.iter().map(|r| r.rejected.len()).sum();
    let failed = all.iter().filter(|r| r.error.is_some()).count();
    println!(
        "bundles: {} total, {asked} asked this run, {failed} failures recorded",
        bundles.len()
    );
    println!(
        "principles standing: {}  (recurring: {recurring})  rejected: {rejected}",
        principles.len()
    );
    Ok(())
}

/// The principles the principles command kept.
fn read_principles(resources: &Path) -> anyhow::Result<Vec<crate::principles::Principle>> {
    let path = resources.join("principles").join("principles.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        anyhow::anyhow!("{}: {e}; run the principles command first", path.display())
    })?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

/// Design a present-day scenario for every principle and have the helper
/// answer it under the code checks; write what stands.
pub fn transfer_command(
    resources: &Path,
    documents: Vec<Document>,
    mine: &Mine<'_>,
) -> anyhow::Result<()> {
    let principles = read_principles(resources)?;
    let corpus = std::sync::Arc::new(crate::transfer::corpus_index(&documents));
    let docs = std::sync::Arc::new(documents);
    let dir = resources.join("transfer");
    std::fs::create_dir_all(&dir)?;
    let results = dir.join("results.jsonl");
    let helper = crate::helper::Helper::served(&mine.served)?;
    let asked =
        crate::transfer::run_all(&helper, &principles, &docs, &corpus, &results, mine.limit)?;

    let all = crate::transfer::read_results(&results)?;
    let made = all.iter().filter(|r| r.error.is_none()).count();
    let failed = all.iter().filter(|r| r.error.is_some()).count();
    println!(
        "principles: {}  asked this run: {asked}  made: {made}  failures recorded: {failed}",
        principles.len()
    );
    for case in [
        crate::scenario::Case::Clear,
        crate::scenario::Case::Weak,
        crate::scenario::Case::MissingPrecondition,
        crate::scenario::Case::SurfaceAnalogy,
    ] {
        let count = all
            .iter()
            .filter(|r| r.error.is_none() && r.case == case)
            .count();
        println!("{case:>20?}: {count}");
    }
    Ok(())
}
