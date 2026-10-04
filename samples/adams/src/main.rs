// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in teaching a model how a historical figure worked and proving that it did,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Can a small model learn how Samuel Adams (1722-1803) worked from his
//! primary sources, and apply it to situations he never met?
//!
//! The corpus is built from documents, never from biographies: every
//! document carries how sure we are that Adams wrote it, and a document's
//! several printings are one piece of evidence, not several.
//!
//! Subcommands, in the order they are meant to be used:
//!
//! ```text
//! identify   is this author line and year the Samuel Adams being modelled
//! corpus     parse the fetched volumes of Cushing's edition into curated documents
//! freeze     split the documents once, before any training: train, exam, temporal
//! tasks      build the training set, the frozen exam and the seen sample
//! train      fine-tune a LoRA adapter on the training set
//! exam       ask one model (base, or base plus adapter) every exam question
//! report     the before-and-after table, with the paired sign test
//! principles ask a helper model for principles; keep what the documents bear out
//! ```

mod attribution;
mod corpus;
mod curate;
mod document;
mod helper;
mod miner;
mod principles;
mod run;
mod split;
mod tasks;

use document::{Date, Identity, Period};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("identify") => identify_command(&args[1..]),
        Some("corpus") => corpus_command(&args[1..]),
        Some("freeze") => freeze_command(&args[1..]),
        Some("tasks") => tasks_command(&args[1..]),
        Some("train") => train_command(&args[1..]),
        Some("exam") => exam_command(&args[1..]),
        Some("report") => report_command(&args[1..]),
        Some("principles") => principles_command(&args[1..]),
        _ => anyhow::bail!(
            "usage: splinter-adams <identify|corpus|freeze|tasks|train|exam|report|principles> ..."
        ),
    }
}

/// The value after `flag`, if given.
fn flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|at| args.get(at + 1))
        .cloned()
}

/// Print what the namesake rules, the period buckets and the temporal holdout
/// say about one author line and year, so a reviewer can probe the rules.
fn identify_command(args: &[String]) -> anyhow::Result<()> {
    let author =
        flag(args, "--author").ok_or_else(|| anyhow::anyhow!("--author LINE is required"))?;
    let year: u16 = flag(args, "--year")
        .ok_or_else(|| anyhow::anyhow!("--year YEAR is required"))?
        .parse()?;
    let date = Date {
        year,
        month: None,
        day: None,
    };
    let verdict = match document::identity(&author, date) {
        Identity::Target => "target".to_string(),
        Identity::Excluded(why) => format!("excluded: {why}"),
        Identity::NeedsReview(why) => format!("needs review: {why}"),
    };
    println!("identity: {verdict}");
    println!("period: {:?}", Period::of(year));
    println!("temporal holdout: {}", document::in_temporal_holdout(year));
    Ok(())
}

/// The volumes of the edition the corpus is parsed from, as the fetcher names them.
const CUSHING_VOLUMES: [&str; 4] = ["cushing-1", "cushing-2", "cushing-3", "cushing-4"];

/// Parse every volume under `--resources`, write the curated documents, what
/// was refused and what was skipped, and print how the corpus came out.
fn corpus_command(args: &[String]) -> anyhow::Result<()> {
    let resources = std::path::PathBuf::from(
        flag(args, "--resources").ok_or_else(|| anyhow::anyhow!("--resources DIR is required"))?,
    );
    let out = resources.join("curated");
    std::fs::create_dir_all(&out)?;
    let mut documents = Vec::new();
    let (mut excluded, mut review, mut skipped) = (Vec::new(), Vec::new(), Vec::new());
    for volume in CUSHING_VOLUMES {
        let path = resources.join("raw").join(format!("{volume}.txt"));
        let text = String::from_utf8_lossy(
            &std::fs::read(&path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?,
        )
        .into_owned();
        let parsed = corpus::parse_cushing(volume, &text);
        let curated = curate::curate(volume, "Samuel Adams", &parsed.entries);
        documents.extend(curated.documents);
        excluded.extend(curated.excluded);
        review.extend(curated.for_review);
        skipped.extend(parsed.skipped.into_iter().map(|s| (volume, s)));
    }
    write_jsonl(&out.join("documents.jsonl"), &documents)?;
    write_jsonl(&out.join("excluded.jsonl"), &excluded)?;
    write_jsonl(&out.join("for-review.jsonl"), &review)?;
    let skipped_rows: Vec<_> = skipped.iter().map(|(v, s)| serde_json::json!({"source_id": v, "line": s.line, "note": s.note, "reason": s.reason})).collect();
    write_jsonl(&out.join("skipped.jsonl"), &skipped_rows)?;
    print_corpus_report(&documents, excluded.len(), review.len(), &skipped);
    Ok(())
}

fn write_jsonl<T: serde::Serialize>(path: &std::path::Path, rows: &[T]) -> anyhow::Result<()> {
    let mut text = String::new();
    for row in rows {
        text.push_str(&serde_json::to_string(row)?);
        text.push('\n');
    }
    let tmp = path.with_extension("jsonl.part");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn print_corpus_report(
    documents: &[curate::Document],
    excluded: usize,
    review: usize,
    skipped: &[(&str, corpus::Skipped)],
) {
    use std::collections::BTreeMap;
    let mut by_class: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_period: BTreeMap<String, usize> = BTreeMap::new();
    for d in documents {
        *by_class.entry(format!("{:?}", d.authorship)).or_default() += 1;
        *by_period.entry(format!("{:?}", d.period)).or_default() += 1;
    }
    let withheld = documents.iter().filter(|d| d.temporal_holdout).count();
    let voice = documents
        .iter()
        .filter(|d| d.authorship.is_voice() && !d.temporal_holdout)
        .count();
    let principles = documents
        .iter()
        .filter(|d| d.authorship.supports_principles() && !d.temporal_holdout)
        .count();
    println!(
        "documents: {}  (withheld as the temporal holdout: {withheld})",
        documents.len()
    );
    println!("trainable as his voice: {voice}  usable for principles: {principles}");
    println!(
        "excluded: {excluded}  for review: {review}  skipped: {}",
        skipped.len()
    );
    println!("by authorship: {by_class:?}");
    println!("by period: {by_period:?}");
    let mut reasons: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, s) in skipped {
        *reasons.entry(s.reason).or_default() += 1;
    }
    println!("skipped because: {reasons:?}");
}

/// The documents the corpus command wrote.
fn read_documents(resources: &std::path::Path) -> anyhow::Result<Vec<curate::Document>> {
    let path = resources.join("curated").join("documents.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("{}: {e}; run the corpus command first", path.display()))?;
    text.lines()
        .map(|line| {
            serde_json::from_str(line).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
        })
        .collect()
}

/// Split the corpus into train, exam and temporal, write what was held out
/// with its digest, and refuse to replace a freeze that says something else.
fn freeze_command(args: &[String]) -> anyhow::Result<()> {
    let resources = std::path::PathBuf::from(
        flag(args, "--resources").ok_or_else(|| anyhow::anyhow!("--resources DIR is required"))?,
    );
    let seed: u64 = flag(args, "--seed").map_or(Ok(1), |s| s.parse())?;
    let exam_percent: u64 = flag(args, "--exam-percent").map_or(Ok(20), |s| s.parse())?;
    anyhow::ensure!(exam_percent <= 100, "--exam-percent is a percentage");
    let documents = read_documents(&resources)?;
    let assignments = split::assign(&documents, seed, exam_percent);
    let leaks = split::leaks(&documents, &assignments);
    anyhow::ensure!(
        leaks.is_empty(),
        "{} training documents share text with held-out ones, first: {:?}",
        leaks.len(),
        leaks[0]
    );
    let manifest = split::manifest(&documents, &assignments, seed, exam_percent);
    let digest = split::digest(&manifest);

    let dir = resources.join("frozen");
    let recorded = dir.join("FROZEN.blake3");
    if let Ok(previous) = std::fs::read_to_string(&recorded) {
        anyhow::ensure!(previous.trim() == digest, "a frozen split already exists ({}) and this run would make {digest}; the exam is never regenerated", previous.trim());
    }
    std::fs::create_dir_all(&dir)?;
    write_atomic(
        &dir.join("split.json"),
        serde_json::to_string_pretty(&manifest)?.as_bytes(),
    )?;
    write_atomic(
        &dir.join("assignments.jsonl"),
        assignment_lines(&assignments)?.as_bytes(),
    )?;
    write_atomic(&recorded, format!("{digest}\n").as_bytes())?;
    println!("frozen {digest}");
    println!(
        "train: {}  exam: {}  temporal: {}",
        manifest.train,
        manifest.exam.len(),
        manifest.temporal.len()
    );
    Ok(())
}

fn assignment_lines(assignments: &[split::Assignment]) -> anyhow::Result<String> {
    let mut text = String::new();
    for a in assignments {
        text.push_str(&serde_json::to_string(
            &serde_json::json!({"doc_id": a.doc_id, "family": a.family, "split": a.split}),
        )?);
        text.push('\n');
    }
    Ok(text)
}

fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// A flag that must be given.
fn need(args: &[String], name: &str) -> anyhow::Result<String> {
    flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"))
}

/// A numeric flag, or its default.
fn number<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> anyhow::Result<T>
where
    T::Err: std::fmt::Display,
{
    flag(args, name).map_or(Ok(default), |v| {
        v.parse().map_err(|e| anyhow::anyhow!("{name} {v}: {e}"))
    })
}

fn tasks_command(args: &[String]) -> anyhow::Result<()> {
    let resources = std::path::PathBuf::from(need(args, "--resources")?);
    let documents = read_documents(&resources)?;
    run::tasks_command(&resources, &documents, number(args, "--seen-per-kind", 40)?)
}

fn train_command(args: &[String]) -> anyhow::Result<()> {
    let resources = std::path::PathBuf::from(need(args, "--resources")?);
    run::train_command(&run::Train {
        base: need(args, "--base")?.into(),
        dataset: resources.join("tasks").join("sft.jsonl"),
        attempt: need(args, "--attempt")?.into(),
        steps: number(args, "--steps", 500)?,
        rank: number(args, "--rank", 16)?,
        alpha: number(args, "--alpha", 32.0)?,
        learning_rate: flag(args, "--lr").map(|v| v.parse()).transpose()?,
        bf16: args.iter().any(|a| a == "--bf16"),
    })
}

fn exam_command(args: &[String]) -> anyhow::Result<()> {
    run::exam_command(&run::Exam {
        tasks: need(args, "--tasks")?.into(),
        out: need(args, "--out")?.into(),
        base: need(args, "--base")?.into(),
        adapter: flag(args, "--adapter").map(Into::into),
        max_tokens: number(args, "--max-tokens", 400)?,
        limit: flag(args, "--limit").map(|v| v.parse()).transpose()?,
    })
}

fn report_command(args: &[String]) -> anyhow::Result<()> {
    run::report_command(
        std::path::Path::new(&need(args, "--before")?),
        std::path::Path::new(&need(args, "--after")?),
    )
}

/// Ask the helper served at `--url` for principles; the key comes from the
/// `BRAIN_API_KEY` variable, never from the command line.
fn principles_command(args: &[String]) -> anyhow::Result<()> {
    let resources = std::path::PathBuf::from(need(args, "--resources")?);
    let documents = read_documents(&resources)?;
    let key = std::env::var("BRAIN_API_KEY")
        .map_err(|_| anyhow::anyhow!("BRAIN_API_KEY is not set: the key `brain serve` printed"))?;
    let url = flag(args, "--url").unwrap_or_else(|| "http://127.0.0.1:8788/v1".to_string());
    let model = need(args, "--model")?;
    let served = helper::Served {
        base_url: &url,
        api_key: &key,
        model: &model,
    };
    run::principles_command(
        &resources,
        &documents,
        &run::Mine {
            served,
            limit: flag(args, "--limit").map(|v| v.parse()).transpose()?,
        },
    )
}
