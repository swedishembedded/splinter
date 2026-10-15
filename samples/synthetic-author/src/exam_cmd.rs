// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The commands that put the examination to a model and read what came of it:
//! `exam`, `report` and `cross`. Each reads the author from the corpus it is
//! given and each outcome it reads must have been made under the manifest of
//! its arm.

use std::path::PathBuf;

use clap::Args;
use splinter_sdk::model::answer::{Answerer, Decoding};

use crate::arms::{read_outcomes, Outcome};
use crate::binding::{
    base_digest, file_digest, recorded, verify_items, verify_outcomes, verify_same_items,
    ArmRecord, ModelIdentity,
};
use crate::cache::Cache;
use crate::lab::{exam_items, run_arm, Arm, ExamRun};
use crate::manifest::author_seed;
use crate::persona::Persona;
use crate::report;

/// Put the examination to a model.
#[derive(Args)]
pub struct ExamArgs {
    /// The seed the author is drawn from; read from the corpus when absent.
    #[arg(long)]
    seed: Option<u64>,
    /// The corpus directory.
    #[arg(long)]
    cache: PathBuf,
    /// The checkpoint directory of the base model.
    #[arg(long)]
    base: PathBuf,
    /// The arm: base, tuned, principles, named or retrieval.
    #[arg(long)]
    arm: String,
    /// The adapter, for the tuned arm.
    #[arg(long)]
    adapter: Option<PathBuf>,
    /// How many articles the corpus has.
    #[arg(long, default_value_t = 100)]
    articles: usize,
    /// Items of each kind of difficulty.
    #[arg(long, default_value_t = 40)]
    per_cell: usize,
    /// The most tokens a reply may run to.
    #[arg(long, default_value_t = 256)]
    max_tokens: u32,
    /// Take over outcomes that were written before manifests existed: record
    /// the flags given as their manifest, without checking them against it.
    #[arg(long)]
    adopt: bool,
}

/// Report what the examinations found.
#[derive(Args)]
pub struct ReportArgs {
    /// The corpus directory.
    #[arg(long)]
    cache: PathBuf,
    /// The arms to report, comma separated.
    #[arg(long, default_value = "base,named,principles,retrieval,tuned")]
    arms: String,
}

/// Count, for each trained author, whose decision it took where another author
/// would have decided otherwise.
#[derive(Args)]
pub struct CrossArgs {
    /// Directory holding one examination cache per author, `author-<seed>`.
    #[arg(long)]
    exams: PathBuf,
    /// The seeds of the authors, comma separated.
    #[arg(long, value_delimiter = ',')]
    seeds: Vec<u64>,
    /// How many articles the corpora were planned with.
    #[arg(long, default_value_t = 100)]
    articles: usize,
    /// Items of each kind of difficulty.
    #[arg(long, default_value_t = 40)]
    per_cell: usize,
}

fn parse_arm(name: &str) -> anyhow::Result<Arm> {
    Ok(match name {
        "base" => Arm::Base,
        "tuned" => Arm::Tuned,
        "principles" => Arm::Principles,
        "named" => Arm::Named,
        "retrieval" => Arm::Retrieval,
        other => anyhow::bail!("unknown arm {other:?}"),
    })
}

/// Runs `exam`.
///
/// # Errors
/// The seed contradicts the corpus, the outcomes were made under another
/// manifest, or the model fails.
pub fn exam(args: &ExamArgs) -> anyhow::Result<()> {
    let arm = parse_arm(&args.arm)?;
    let adapter = match (arm, &args.adapter) {
        (Arm::Tuned, None) => anyhow::bail!("the tuned arm needs --adapter"),
        (Arm::Tuned, Some(path)) => Some(path.as_path()),
        (_, _) => None,
    };
    let seed = author_seed(&args.cache, args.seed)?;
    let persona = Persona::generate(seed);
    let cache = Cache::open(args.cache.clone())?;
    let all = exam_items(&persona, seed, args.articles, args.per_cell);
    let decoding = Decoding::Greedy;
    let run = ExamRun {
        articles: args.articles,
        per_cell: args.per_cell,
        max_tokens: args.max_tokens,
        model: ModelIdentity {
            base_digest: base_digest(&args.base)?,
            adapter_digest: adapter.map(file_digest).transpose()?,
            decoding: "greedy".to_string(),
            // Greedy decoding takes the likeliest token and draws nothing.
            decoding_seed: None,
        },
        adopt: args.adopt,
    };
    let answerer = Answerer::load_with(&args.base, adapter, Some(16384), arm.name(), decoding)?;
    let provider = answerer.provider();
    let outcomes = run_arm(&cache, &persona, arm, &provider, &all, &run)?;
    let r = report::accuracy(&outcomes);
    if r.n == 0 {
        println!("{}: not measured (no items)", arm.name());
    } else {
        println!("{}: {}/{} right ({:.3})", arm.name(), r.right, r.n, r.rate);
    }
    Ok(())
}

/// Runs `report`.
///
/// # Errors
/// An arm's outcomes have no manifest or were made by another model than it
/// names, or arms were asked different items.
pub fn report(args: &ReportArgs) -> anyhow::Result<()> {
    let cache = Cache::open(args.cache.clone())?;
    let exam_dir = cache.dir().join("exam");
    let mut loaded: Vec<(String, Vec<Outcome>)> = Vec::new();
    let mut records: Vec<ArmRecord> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for name in args.arms.split(',') {
        let path = exam_dir.join(format!("{name}.jsonl"));
        if !path.exists() {
            missing.push(name.to_string());
            continue;
        }
        let outcomes = read_outcomes(&path)?;
        let record = recorded(&exam_dir, name)?.ok_or_else(|| {
            anyhow::anyhow!(
                "{}: no manifest for arm {name}; outcomes written before manifests are taken over by `exam --adopt`",
                path.display()
            )
        })?;
        verify_outcomes(&record, &outcomes, &path)?;
        records.push(record);
        loaded.push((name.to_string(), outcomes));
    }
    verify_same_items(&records)?;
    let view: Vec<(&str, Vec<_>)> = loaded
        .iter()
        .map(|(n, o)| (n.as_str(), o.clone()))
        .collect();
    let missing: Vec<&str> = missing.iter().map(String::as_str).collect();
    let compare = [
        ("tuned", "base"),
        ("tuned", "named"),
        ("tuned", "principles"),
        ("tuned", "retrieval"),
        ("retrieval", "base"),
        ("principles", "base"),
    ];
    let text = report::render(&view, &compare, &missing);
    std::fs::write(cache.dir().join("report.md"), &text)?;
    println!("{text}");
    Ok(())
}

/// Runs `cross`.
///
/// # Errors
/// A directory's author is not the seed it is named for, or its outcomes were
/// not made for the items these flags make.
pub fn cross(args: &CrossArgs) -> anyhow::Result<()> {
    println!(
        "| trained on | compared with | situations | its own author | the other author | neither |"
    );
    println!("|---|---|---|---|---|---|");
    for &seed in &args.seeds {
        let dir = args.exams.join(format!("author-{seed}"));
        author_seed(&dir, Some(seed))?;
        let persona = Persona::generate(seed);
        let items = exam_items(&persona, seed, args.articles, args.per_cell);
        let exam_dir = dir.join("exam");
        let path = outcomes_path_in(&exam_dir, Arm::Tuned);
        let outcomes = read_outcomes(&path)?;
        let record = recorded(&exam_dir, Arm::Tuned.name())?.ok_or_else(|| {
            anyhow::anyhow!(
                "{}: no manifest for arm tuned; outcomes written before manifests are taken over by `exam --adopt`",
                path.display()
            )
        })?;
        verify_outcomes(&record, &outcomes, &path)?;
        verify_items(&record, &items, &outcomes, &path)?;
        let mut rivals: Vec<Persona> = args
            .seeds
            .iter()
            .filter(|&&s| s != seed)
            .map(|&s| Persona::generate(s))
            .collect();
        rivals.push(persona.opposite());
        for rival in &rivals {
            let d = report::divergence(&items, &outcomes, rival);
            println!(
                "| {} | {} | {} | {} | {} | {} |",
                persona.name, rival.name, d.differing, d.own, d.rival, d.neither
            );
        }
    }
    Ok(())
}

fn outcomes_path_in(exam_dir: &std::path::Path, arm: Arm) -> PathBuf {
    exam_dir.join(format!("{}.jsonl", arm.name()))
}
