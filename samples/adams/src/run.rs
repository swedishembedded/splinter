// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured fine-tuning of language models on
// historical records for its clients. If your team needs expertise in training
// a model on a source corpus and proving what it kept, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The commands that take the frozen split to a trained adapter and a measured
//! before-and-after: build the tasks, train, ask the exam, report.

use splinter_sdk::model::answer::Decoding;
use std::path::{Path, PathBuf};

use crate::curate::Document;
use crate::split::{self, Assignment, Split};
use crate::tasks::{self, PERSONA};

mod reconstruction;
pub use reconstruction::*;

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
    /// The share of draws that are replay when any is given.
    pub replay_share: Option<f32>,
    /// Examples per optimizer step.
    pub grad_accum: u32,
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
        replay_share: t.replay_share,
        grad_accum: t.grad_accum,
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
        decoding: Decoding::Greedy,
    };
    let asked = splinter_sdk::model::exam::run(&questions, &e.out, e.limit, &model, &|q, a| {
        tasks::is_correct(q, a).into()
    })?;
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
    variants: u8,
) -> anyhow::Result<()> {
    let principles = read_principles(resources)?;
    let corpus = std::sync::Arc::new(crate::transfer::corpus_index(&documents));
    let docs = std::sync::Arc::new(documents);
    let dir = resources.join("transfer");
    std::fs::create_dir_all(&dir)?;
    let results = dir.join("results.jsonl");
    let helper = crate::helper::Helper::served(&mine.served)?;
    let asked = crate::transfer::run_all(
        &helper,
        &principles,
        &docs,
        &corpus,
        variants,
        &results,
        mine.limit,
    )?;

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

/// Freeze `file`: pin its content in the ledger beside it, so a later run that
/// would write or score against other content under its name is refused.
pub fn pin_command(file: &Path) -> anyhow::Result<()> {
    let name = file
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("{}: not a file name", file.display()))?;
    let ledger = splinter_sdk::data::frozen::Ledger::at(file.with_file_name("FROZEN.json"));
    let status = ledger.pin(name, &std::fs::read(file)?)?;
    println!("{name}: {status:?}, frozen in {}", ledger.path().display());
    Ok(())
}

/// Refuse a frozen file whose content is no longer what was frozen: an exam
/// is only ever scored against the questions it was frozen with. A file that
/// is not frozen yet is allowed, with a note.
pub fn ensure_unchanged(file: &Path) -> anyhow::Result<()> {
    use splinter_sdk::data::frozen::{Ledger, Status};
    let name = file
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("{}: not a file name", file.display()))?;
    let ledger = Ledger::at(file.with_file_name("FROZEN.json"));
    if ledger.check(name, &std::fs::read(file)?)? == Status::New {
        eprintln!(
            "note: {} is not frozen; `pin` it before any arm is scored against it",
            file.display()
        );
    }
    Ok(())
}

/// Build the transfer datasets from the checked results, and prove with the
/// trainer's own parsers that it would read what was written.
pub fn build_data_command(
    resources: &Path,
    documents: &[Document],
    benchmark_file: &str,
    extra_file: &str,
) -> anyhow::Result<()> {
    let results = crate::transfer::read_results(&resources.join("transfer").join("results.jsonl"))?;
    let principles = read_principles(resources)?;
    let built = crate::datasets::build(&results, &principles, documents);
    let dir = resources.join("datasets");
    crate::datasets::write_all(&built, &dir, benchmark_file, extra_file)?;
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
    println!(
        "benchmark questions: {} (frozen)  extra: {}",
        built.benchmark.len(),
        built.extra_benchmark.len()
    );

    let train_briefings = crate::reconstruct::read_briefings(
        &resources
            .join("reconstruction")
            .join("briefings-train.jsonl"),
    )?;
    if !train_briefings.is_empty() {
        let own_train: std::collections::HashSet<String> =
            reconstruction::letters_of(resources, documents, Split::Train)?
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
    /// Weight of the anchor on the chosen answer.
    pub nll_weight: f32,
    /// Pairs per optimizer step.
    pub grad_accum: u32,
    pub learning_rate: Option<f32>,
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
        nll_weight: d.nll_weight,
        grad_accum: d.grad_accum,
        learning_rate: d.learning_rate,
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
    pub framing: crate::persona::Framing,
    pub decoding: Decoding,
    /// Worked examples shown in the system message, none for an empty string.
    pub examples: String,
}

/// Ask one model every benchmark question, in both modes, graded by the rules.
pub fn transfer_exam_command(e: &TransferExam, documents: &[Document]) -> anyhow::Result<()> {
    ensure_unchanged(&e.benchmark)?;
    let questions = crate::datasets::read_benchmark(&e.benchmark)?;
    let system = if e.examples.is_empty() {
        e.framing.system(crate::respond::SYSTEM).to_string()
    } else {
        format!(
            "{}\n\n{}",
            e.framing.system(crate::respond::SYSTEM),
            e.examples
        )
    };
    let model = splinter_sdk::model::exam::Model {
        base: &e.base,
        adapter: e.adapter.as_deref(),
        system: &system,
        max_tokens: e.max_tokens,
        label: "adams",
        decoding: e.decoding,
    };
    let asked =
        splinter_sdk::model::exam::run(&questions, &e.out, e.limit, &model, &|task, answer| {
            task.verdict(answer, documents)
        })?;
    println!("asked {asked} questions; results in {}", e.out.display());
    Ok(())
}

/// Grade the fixed template on every benchmark question, with no model: the
/// score a trained model's has to be set against.
pub fn constant_baseline_command(
    benchmark: &Path,
    out: &Path,
    documents: &[Document],
) -> anyhow::Result<()> {
    let questions = crate::datasets::read_benchmark(benchmark)?;
    let mut ask = |q: &crate::datasets::TransferTask| {
        Ok(splinter_sdk::model::answer::Reply {
            text: crate::baseline::constant_answer(q, documents),
            ..Default::default()
        })
    };
    let asked =
        splinter_sdk::model::exam::run_with(&questions, out, None, &mut ask, &|task, answer| {
            task.verdict(answer, documents)
        })?;
    println!("graded {asked} questions; results in {}", out.display());
    Ok(())
}

/// Ask one model, with or without its adapter, the frozen general questions
/// under no persona: what a retention check compares before and after.
pub fn anchor_exam_command(
    anchor: &Path,
    out: &Path,
    base: &Path,
    adapter: Option<&Path>,
    limit: Option<usize>,
) -> anyhow::Result<()> {
    let questions = crate::anchor::read(anchor)?;
    let model = splinter_sdk::model::exam::Model {
        base,
        adapter,
        system: crate::anchor::SYSTEM,
        max_tokens: 100,
        label: "adams",
        decoding: Decoding::Greedy,
    };
    let asked = splinter_sdk::model::exam::run(&questions, out, limit, &model, &|q, a| {
        crate::anchor::is_correct(q, a).into()
    })?;
    println!("asked {asked} questions; results in {}", out.display());
    Ok(())
}

/// Tokens a training row keeps beyond the longest prompt and the completion budget.
const ROW_MARGIN_TOKENS: usize = 16;

/// What one reinforcement-learning cycle needs.
pub struct Grpo {
    /// The Hugging Face checkpoint directory: the tokenizer and the model's name.
    pub base: PathBuf,
    /// The same model as one brain checkpoint file, which brain's loop trains
    /// from (`brain qwen3 import --hf BASE --out FILE`).
    pub checkpoint: PathBuf,
    pub out: PathBuf,
    pub steps: u32,
    pub group_size: usize,
    pub max_new: usize,
    pub learning_rate: f32,
    pub temperature: f32,
    pub rank: u32,
    /// One in this many scenarios is kept for the loop's own gate.
    pub gate_one_in: usize,
}

/// The scenarios a model may be trained on: every checked result of a principle
/// that is not in the benchmark, with the passages it may quote.
fn grpo_items(resources: &Path, documents: &[Document]) -> anyhow::Result<Vec<crate::grpo::Item>> {
    let results = crate::transfer::read_results(&resources.join("transfer").join("results.jsonl"))?;
    let principles = read_principles(resources)?;
    let mut items = Vec::new();
    for result in results.iter().filter(|r| r.error.is_none()) {
        let (Some(scenario), Some(principle)) = (
            &result.scenario,
            principles.iter().find(|p| p.id == result.principle_id),
        ) else {
            continue;
        };
        if crate::datasets::is_benchmark(&principle.id) {
            continue;
        }
        let cited: Vec<&str> = principle
            .support
            .iter()
            .map(|f| f.doc_id.as_str())
            .collect();
        let docs: Vec<(String, String)> = documents
            .iter()
            .filter(|d| cited.contains(&d.id.as_str()))
            .map(|d| (d.id.clone(), d.body.clone()))
            .collect();
        items.push(crate::grpo::Item {
            scenario: scenario.clone(),
            evidence: principle.support.clone(),
            documents: docs,
        });
    }
    Ok(items)
}

/// Train an adapter by GRPO on the situations, rewarded by the rules, and gate
/// it against the untouched base on situations it was not trained on.
pub fn grpo_command(resources: &Path, documents: &[Document], g: &Grpo) -> anyhow::Result<()> {
    use splinter_sdk::model::rl::{
        ChatMessage, ChatRequest, ChatTokenizer, Improve, ImproveOptions,
    };
    let tokenizer = std::sync::Arc::new(ChatTokenizer::from_model_dir(&g.base)?);
    let encoder = std::sync::Arc::clone(&tokenizer);
    let encode: crate::grpo::Encode = std::sync::Arc::new(move |system: &str, user: &str| {
        let request = ChatRequest::new(vec![ChatMessage::system(system), ChatMessage::user(user)])
            .thinking(false);
        Ok(encoder.prompt_ids(&request)?)
    });
    let decoder = std::sync::Arc::clone(&tokenizer);
    let decode: crate::grpo::Decode = std::sync::Arc::new(move |ids: &[u32]| decoder.decode(ids));

    let env = crate::grpo::AdamsEnv::new(grpo_items(resources, documents)?, g.gate_one_in, encode);
    env.preflight()?;
    let (train, gate) = env.pools();
    // Rows as long as the prompts and completions need, not as long as the model's context.
    let max_seq_len = env.longest_prompt()? + g.max_new + ROW_MARGIN_TOKENS;
    println!("situations to train on: {train}  kept for the gate: {gate}  training row: {max_seq_len} tokens");
    std::fs::create_dir_all(&g.out)?;
    let cycle = Improve::from_pretrained(g.checkpoint.to_string_lossy())?
        .arch("qwen3")
        .adapter_dir(g.out.join("adapters"))
        .work_dir(g.out.join("work"))
        .lora(g.rank)
        .seed(0)
        .held_out_seed(crate::grpo::GATE_SEED_BASE);
    let options = ImproveOptions {
        group_size: g.group_size,
        temperature: g.temperature,
        max_new: g.max_new,
        steps: g.steps,
        lr: g.learning_rate,
        seed: 0,
        max_seq_len: Some(max_seq_len),
        ..ImproveOptions::default()
    };
    let started = std::time::Instant::now();
    let outcome = cycle.run(env, crate::grpo::AdamsVerifier::new(decode), &options)?;
    let summary = serde_json::json!({
        "decision": format!("{:?}", outcome.decision),
        "p_value": outcome.p_value,
        "effect_size": outcome.effect_size,
        "entropy_ratio": outcome.entropy_ratio,
        "adapter": outcome.adapter_path,
        "steps": g.steps, "group_size": g.group_size, "max_new": g.max_new, "learning_rate": g.learning_rate,
        "seconds": started.elapsed().as_secs_f64(),
    });
    let text = serde_json::to_string_pretty(&summary)?;
    std::fs::write(g.out.join("summary.json"), &text)?;
    println!("{text}");
    Ok(())
}
