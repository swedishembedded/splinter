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
    /// Datasets mixed into training whole, never held out.
    pub replay: Vec<PathBuf>,
    /// An adapter to continue instead of starting a fresh one.
    pub continue_from: Option<PathBuf>,
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
        replay: &t.replay,
        continue_from: t.continue_from.as_deref(),
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
    // Principles about how he worked rest on his own letters in the training split.
    let own_letters: std::collections::HashSet<&str> = documents
        .iter()
        .filter(|d| d.authorship.is_his_own_letter())
        .map(|d| d.id.as_str())
        .collect();
    let allowed: std::collections::HashSet<String> = assignments
        .iter()
        .filter(|a| a.split == Split::Train && own_letters.contains(a.doc_id.as_str()))
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

/// Build the transfer datasets from the checked results, and prove with the
/// trainer's own parsers that it would read what was written.
pub fn build_data_command(resources: &Path, documents: &[Document]) -> anyhow::Result<()> {
    let results = crate::transfer::read_results(&resources.join("transfer").join("results.jsonl"))?;
    let principles = read_principles(resources)?;
    let built = crate::datasets::build(&results, &principles, documents);
    let dir = resources.join("datasets");
    crate::datasets::write_all(&built, &dir)?;
    let chat = splinter_sdk::model::train::validate_dataset(&dir.join("sft-transfer.jsonl"))?;
    let pairs = splinter_sdk::model::train::validate_preference_dataset(
        &dir.join("preference-transfer.jsonl"),
    )?;
    println!(
        "results: {}  excluded: {}",
        results.len(),
        built.excluded.len()
    );
    println!(
        "supervised records: {} (the trainer reads {})",
        built.sft.len(),
        chat.records
    );
    println!(
        "preference pairs: {} (the trainer reads {})",
        built.preference.len(),
        pairs.pairs
    );
    let mut by_break: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for pair in &built.preference {
        *by_break
            .entry(pair["metadata"]["break"].to_string())
            .or_default() += 1;
    }
    println!("pairs by break: {by_break:?}");
    println!("benchmark questions: {}", built.benchmark.len());

    let train_briefings = crate::reconstruct::read_briefings(
        &resources
            .join("reconstruction")
            .join("briefings-train.jsonl"),
    )?;
    if !train_briefings.is_empty() {
        let own_train: std::collections::HashSet<String> =
            letters_of(resources, documents, Split::Train)?
                .into_iter()
                .map(|d| d.id.clone())
                .collect();
        let records = crate::datasets::reconstruction_sft(&train_briefings, documents, &own_train);
        let path = dir.join("sft-reconstruction.jsonl");
        std::fs::write(
            &path,
            records.iter().map(|r| format!("{r}\n")).collect::<String>(),
        )?;
        let read = splinter_sdk::model::train::validate_dataset(&path)?;
        println!(
            "reconstruction records: {} (the trainer reads {})",
            records.len(),
            read.records
        );
    }
    Ok(())
}

/// What one preference run needs.
pub struct Dpo {
    pub base: PathBuf,
    pub pairs: PathBuf,
    pub attempt: PathBuf,
    /// The supervised adapter this run continues; base plus it is the frozen
    /// reference the pairs are scored against.
    pub continue_from: Option<PathBuf>,
    pub steps: u32,
    pub rank: u32,
    pub alpha: f32,
    pub beta: f32,
}

/// Direct preference optimisation on the pairs, through Splinter's trainer.
pub fn dpo_command(d: &Dpo) -> anyhow::Result<()> {
    std::fs::create_dir_all(&d.attempt)?;
    let started = std::time::Instant::now();
    let split = splinter_sdk::data::holdout::split_dataset_file(&d.pairs, &d.attempt)?;
    let request = splinter_sdk::model::train::PreferenceTune {
        model_dir: &d.base,
        train: &split.train,
        held_out: &split.held_out,
        attempt_dir: &d.attempt,
        steps: d.steps,
        rank: d.rank,
        alpha: d.alpha,
        beta: d.beta,
        continue_from: d.continue_from.as_deref(),
        cancel: None,
    };
    let trained = splinter_sdk::model::train::train_preference(&request)?;
    let summary = serde_json::json!({
        "adapter": trained.adapter,
        "adapter_digest": trained.adapter_digest,
        "base_digest": trained.base_digest,
        "reference_adapter": trained.reference_adapter,
        "records": trained.records,
        "steps": d.steps,
        "beta": trained.beta,
        "seconds": started.elapsed().as_secs_f64(),
        "train_score": trained.train_score.as_ref().map(|s| format!("{s:?}")),
        "held_out_score": trained.held_out_score.as_ref().map(|s| format!("{s:?}")),
    });
    let text = serde_json::to_string_pretty(&summary)?;
    std::fs::write(d.attempt.join("summary.json"), &text)?;
    println!("{text}");
    Ok(())
}

/// What one transfer exam needs.
pub struct TransferExam {
    pub benchmark: PathBuf,
    pub out: PathBuf,
    pub base: PathBuf,
    pub adapter: Option<PathBuf>,
    pub max_tokens: u32,
    pub limit: Option<usize>,
}

/// Ask one model every benchmark question, in both modes, graded by the rules.
pub fn transfer_exam_command(e: &TransferExam, documents: &[Document]) -> anyhow::Result<()> {
    let questions = crate::datasets::read_benchmark(&e.benchmark)?;
    let model = splinter_sdk::model::exam::Model {
        base: &e.base,
        adapter: e.adapter.as_deref(),
        system: crate::respond::SYSTEM,
        max_tokens: e.max_tokens,
        label: "adams",
    };
    let asked =
        splinter_sdk::model::exam::run(&questions, &e.out, e.limit, &model, &|task, answer| {
            task.is_correct(answer, documents)
        })?;
    println!("asked {asked} questions; results in {}", e.out.display());
    Ok(())
}

/// His own letters on one side of the split: the exam side is what the
/// reconstruction benchmark is made from, the training side what he is shown
/// doing.
fn letters_of<'a>(
    resources: &Path,
    documents: &'a [Document],
    side: Split,
) -> anyhow::Result<Vec<&'a Document>> {
    let ids: std::collections::HashSet<String> = read_assignments(resources)?
        .into_iter()
        .filter(|a| a.split == side)
        .map(|a| a.doc_id)
        .collect();
    let mut letters: Vec<&Document> = documents
        .iter()
        .filter(|d| ids.contains(&d.id) && d.authorship.is_his_own_letter())
        .collect();
    letters.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(letters)
}

fn reconstruction_dir(resources: &Path) -> anyhow::Result<PathBuf> {
    let dir = resources.join("reconstruction");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Brief every held-out letter of his: the situation it answered, in a
/// helper's words, and what the real letter does, each point grounded in it.
pub fn briefings_command(
    resources: &Path,
    documents: &[Document],
    mine: &Mine<'_>,
) -> anyhow::Result<()> {
    let letters = exam_letters(resources, documents)?;
    let path = reconstruction_dir(resources)?.join("briefings.jsonl");
    let helper = crate::helper::Helper::served(&mine.served)?;
    let asked = crate::reconstruct::brief_all(&helper, &letters, &path, mine.limit)?;
    let made = crate::reconstruct::read_briefings(&path)?.len();
    println!(
        "held-out letters of his: {}  asked this run: {asked}  briefings made: {made}",
        letters.len()
    );
    Ok(())
}

/// What one reconstruction arm needs.
pub struct Replies {
    pub briefings: PathBuf,
    pub out: PathBuf,
    pub arm: String,
    pub base: PathBuf,
    pub adapter: Option<PathBuf>,
    pub max_tokens: u32,
    pub limit: Option<usize>,
}

/// Have one model write the reply to every briefing, in full.
pub fn replies_command(r: &Replies) -> anyhow::Result<()> {
    let briefings = crate::reconstruct::read_briefings(&r.briefings)?;
    let answerer =
        splinter_sdk::model::answer::Answerer::load(&r.base, r.adapter.as_deref(), 4096, "adams")?;
    let runtime = tokio::runtime::Runtime::new()?;
    let mut ask = |b: &crate::reconstruct::Briefing| {
        runtime.block_on(answerer.ask(
            crate::reconstruct::SYSTEM,
            &crate::reconstruct::prompt(b),
            r.max_tokens,
        ))
    };
    let asked = crate::reconstruct::answer_all(&briefings, &r.out, &r.arm, r.limit, &mut ask)?;
    println!(
        "asked {asked} briefings as {}; replies in {}",
        r.arm,
        r.out.display()
    );
    Ok(())
}

/// The first words of a text: what the judge is shown of a letter or a reply.
fn judged_words(text: &str) -> String {
    const WORDS: usize = 900;
    text.split_whitespace()
        .take(WORDS)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Calibrate the judge once, on controls, and keep the result so every arm is
/// judged by the same standard; then score every reply of `replies`.
pub fn judge_command(
    resources: &Path,
    documents: &[Document],
    mine: &Mine<'_>,
    replies: &Path,
    out: &Path,
) -> anyhow::Result<()> {
    let dir = reconstruction_dir(resources)?;
    let briefings = crate::reconstruct::read_briefings(&dir.join("briefings.jsonl"))?;
    let helper = crate::helper::Helper::served(&mine.served)?;

    let calibration_path = dir.join("calibration.json");
    let calibration: crate::judge::Calibration =
        if let Ok(text) = std::fs::read_to_string(&calibration_path) {
            serde_json::from_str(&text)?
        } else {
            const CONTROLS: usize = 20;
            let body = |id: &str| {
                documents
                    .iter()
                    .find(|d| d.id == id)
                    .map(|d| judged_words(&d.body))
                    .unwrap_or_default()
            };
            let controls: Vec<(usize, String, String)> = briefings
                .iter()
                .take(CONTROLS)
                .enumerate()
                .map(|(i, b)| {
                    (
                        i,
                        body(&b.doc_id),
                        body(&briefings[(i + 1) % briefings.len()].doc_id),
                    )
                })
                .collect();
            let borrowed: Vec<(&crate::reconstruct::Briefing, &str, &str)> = controls
                .iter()
                .map(|(i, real, other)| (&briefings[*i], real.as_str(), other.as_str()))
                .collect();
            let found = crate::judge::calibrate(&helper, &borrowed)?;
            std::fs::write(&calibration_path, serde_json::to_string_pretty(&found)?)?;
            found
        };
    println!(
        "judge controls on {} briefings: real letter {:.0}%, another letter {:.0}%: {}",
        calibration.briefings,
        100.0 * calibration.real_letter,
        100.0 * calibration.other_letter,
        if calibration.passes() {
            "passes"
        } else {
            "does not pass; no claim will be made about coverage"
        }
    );

    let replies = crate::reconstruct::read_replies(replies)?;
    let corpus = crate::transfer::corpus_index(documents);
    let scored =
        crate::reconstruct::score_all(&helper, &briefings, &replies, &corpus, out, mine.limit)?;
    println!("scored {scored} replies; scores in {}", out.display());
    Ok(())
}

/// The comparison of two arms' scores, with the judge's controls beside it.
pub fn reconstruct_report_command(
    resources: &Path,
    before: &Path,
    after: &Path,
) -> anyhow::Result<()> {
    let calibration: crate::judge::Calibration = serde_json::from_str(&std::fs::read_to_string(
        reconstruction_dir(resources)?.join("calibration.json"),
    )?)?;
    let (before_scores, after_scores) = (
        crate::reconstruct::read_scores(before)?,
        crate::reconstruct::read_scores(after)?,
    );
    let arm = |scores: &[crate::reconstruct::Scored], fallback: &Path| {
        scores
            .first()
            .map_or_else(|| fallback.display().to_string(), |s| s.arm.clone())
    };
    let summary = crate::reconstruct::summarize(&before_scores, &after_scores);
    print!(
        "{}",
        crate::reconstruct::render(
            &summary,
            &calibration,
            &arm(&before_scores, before),
            &arm(&after_scores, after)
        )
    );
    Ok(())
}
