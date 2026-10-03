// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured knowledge-injection samples for
// language models for its clients. If your team needs expertise in teaching a
// model a document corpus and proving what it kept, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Does source-grounded fine-tuning teach a model the Jefferson corpus?
//!
//! Subcommands, in the order they are meant to be used:
//!
//! ```text
//! corpus     parse the letters, group the editions' printings, split by family
//! materials  write the training letters and his own works as the directory Splinter learns from
//! tasks      compile the training set and the frozen exam from the corpus
//! exam       ask one model (base, or base plus adapter) every exam question
//! train      fine-tune a LoRA adapter on the training set, through Splinter's trainer
//! report     the before-and-after table, with the paired sign test
//! scenarios  a writer model drafts situations from letters; code gates them
//! apply      put the scenarios to a model and measure what it quotes
//! judge      a judge model of another family rates the answers, checked on controls
//! retrieval  how well lexical, semantic and fused search find the passage a situation came from
//! apply-report  the application table: fabricated quotations, citations, recall, agreement
//! ```

mod application;
mod ask;
mod build;
mod corpus;
mod exam;
mod grade;
mod report;
mod retrieval;
mod scenarios;
mod tasks;
mod works;
mod writer;

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("corpus") => corpus_command(&args[1..]),
        Some("materials") => materials_command(&args[1..]),
        Some("tasks") => tasks_command(&args[1..]),
        Some("exam") => exam_command(&args[1..]),
        Some("train") => train_command(&args[1..]),
        Some("report") => report_command(&args[1..]),
        Some("scenarios") => scenarios_command(&args[1..]),
        Some("apply") => apply_command(&args[1..]),
        Some("judge") => judge_command(&args[1..]),
        Some("retrieval") => retrieval_command(&args[1..]),
        Some("apply-report") => apply_report_command(&args[1..]),
        _ => anyhow::bail!("usage: splinter-jefferson <corpus|tasks> --resources DIR --out DIR"),
    }
}

/// The value after `flag`, if given.
fn flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|at| args.get(at + 1))
        .cloned()
}

fn corpus_command(args: &[String]) -> anyhow::Result<()> {
    let resources = PathBuf::from(
        flag(args, "--resources").ok_or_else(|| anyhow::anyhow!("--resources DIR is required"))?,
    );
    let out =
        PathBuf::from(flag(args, "--out").ok_or_else(|| anyhow::anyhow!("--out DIR is required"))?);
    let seed: u64 = flag(args, "--seed").map_or(Ok(1), |s| s.parse())?;
    std::fs::create_dir_all(&out)?;

    let letters = corpus::load_letters(&resources)?;
    let family = corpus::families(&letters);
    let mut lines = String::new();
    let (mut exam_letters, mut family_set, mut exam_families) = (
        0usize,
        std::collections::HashSet::new(),
        std::collections::HashSet::new(),
    );
    for (letter, &root) in letters.iter().zip(&family) {
        let key = &letters[root].id;
        let exam = corpus::is_exam_family(key, seed, 20);
        family_set.insert(root);
        if exam {
            exam_letters += 1;
            exam_families.insert(root);
        }
        lines.push_str(&serde_json::to_string(&serde_json::json!({
            "id": letter.id, "family": key, "split": if exam { "exam" } else { "train" },
            "edition": letter.edition, "recipient": letter.recipient, "place": letter.place,
            "year": letter.year, "body": letter.body,
        }))?);
        lines.push('\n');
    }
    std::fs::write(out.join("letters.jsonl"), lines)?;
    println!(
        "letters {} in {} families; exam: {} letters in {} families",
        letters.len(),
        family_set.len(),
        exam_letters,
        exam_families.len()
    );
    Ok(())
}

/// A numeric flag, or `default` when it is absent.
fn number(args: &[String], name: &str, default: usize) -> anyhow::Result<usize> {
    flag(args, name).map_or(Ok(default), |v| {
        v.parse().map_err(|e| anyhow::anyhow!("{name} {v}: {e}"))
    })
}

fn tasks_command(args: &[String]) -> anyhow::Result<()> {
    let resources = PathBuf::from(
        flag(args, "--resources").ok_or_else(|| anyhow::anyhow!("--resources DIR is required"))?,
    );
    let out =
        PathBuf::from(flag(args, "--out").ok_or_else(|| anyhow::anyhow!("--out DIR is required"))?);
    let seed: u64 = flag(args, "--seed").map_or(Ok(1), |s| s.parse())?;
    let sizes = build::Sizes {
        train_letters: number(args, "--train-letters", 400)?,
        exam_letters: number(args, "--exam-letters", 60)?,
        train_passages: number(args, "--train-passages", 40)?,
        exam_passages: number(args, "--exam-passages", 6)?,
        seen: number(args, "--seen", 60)?,
    };
    std::fs::create_dir_all(&out)?;
    let built = build::build(&resources, sizes, seed)?;

    let mut sft = String::new();
    for task in &built.train {
        sft.push_str(&serde_json::to_string(&tasks::sft_record(task))?);
        sft.push('\n');
    }
    std::fs::write(out.join("sft.jsonl"), sft)?;
    let mut exam = String::new();
    for item in &built.exam {
        exam.push_str(&serde_json::to_string(item)?);
        exam.push('\n');
    }
    std::fs::write(out.join("exam.jsonl"), exam)?;
    let count = |tasks: &[tasks::Task], split: &str, kind: tasks::Kind| {
        tasks
            .iter()
            .filter(|t| t.split == split && t.kind == kind)
            .count()
    };
    println!("training records {}", built.train.len());
    for kind in [tasks::Kind::Recipient, tasks::Kind::Year, tasks::Kind::Work] {
        println!(
            "  {kind:?}: train {} exam {} seen {}",
            count(&built.train, "train", kind),
            count(&built.exam, "exam", kind),
            count(&built.exam, "seen", kind)
        );
    }
    Ok(())
}

fn exam_command(args: &[String]) -> anyhow::Result<()> {
    let need = |name: &str| flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"));
    let run = exam::Run {
        exam: PathBuf::from(need("--exam")?),
        out: PathBuf::from(need("--out")?),
        base: PathBuf::from(need("--base")?),
        adapter: flag(args, "--adapter").map(PathBuf::from),
        max_tokens: u32::try_from(number(args, "--max-tokens", 400)?)?,
        limit: flag(args, "--limit").map(|v| v.parse()).transpose()?,
    };
    let asked = exam::run(&run)?;
    println!("asked {asked} questions; results in {}", run.out.display());
    Ok(())
}

fn train_command(args: &[String]) -> anyhow::Result<()> {
    let need = |name: &str| flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"));
    let (base, dataset, attempt) = (
        PathBuf::from(need("--base")?),
        PathBuf::from(need("--dataset")?),
        PathBuf::from(need("--attempt")?),
    );
    std::fs::create_dir_all(&attempt)?;
    let started = std::time::Instant::now();
    let report = |r: &splinter_policy::train::StepReport| {
        eprintln!(
            "step {}/{} loss {:.4} elapsed {:.0}s",
            r.step,
            r.steps,
            r.loss,
            started.elapsed().as_secs_f64()
        );
    };
    let request = splinter_policy::train::FineTune {
        model_dir: &base,
        dataset: &dataset,
        attempt_dir: &attempt,
        steps: u32::try_from(number(args, "--steps", 200)?)?,
        rank: u32::try_from(number(args, "--rank", 16)?)?,
        alpha: flag(args, "--alpha").map_or(Ok(32.0), |v| v.parse())?,
        replay: &[],
        continue_from: None,
        cancel: None,
        bf16_base: args.iter().any(|a| a == "--bf16"),
        learning_rate: flag(args, "--lr").map(|v| v.parse()).transpose()?,
        on_step: Some(splinter_policy::train::StepHook(&report)),
    };
    let trained = splinter_policy::train::fine_tune(&request)?;
    let summary = serde_json::json!({
        "adapter": trained.adapter,
        "adapter_digest": trained.adapter_digest,
        "base_digest": trained.base_digest,
        "records": trained.records,
        "block": trained.block,
        "held_out_before": {"loss": trained.base.loss, "token_accuracy": trained.base.token_accuracy, "positions": trained.base.positions},
        "held_out_after": {"loss": trained.tuned.loss, "token_accuracy": trained.tuned.token_accuracy, "positions": trained.tuned.positions},
    });
    std::fs::write(
        attempt.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn report_command(args: &[String]) -> anyhow::Result<()> {
    let need = |name: &str| flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"));
    let before = exam::read_results(std::path::Path::new(&need("--before")?))?;
    let after = exam::read_results(std::path::Path::new(&need("--after")?))?;
    let tokens = |g: &[exam::Graded]| {
        (
            g.iter().map(|x| f64::from(x.tokens)).sum::<f64>(),
            g.iter().map(|x| x.seconds).sum::<f64>(),
        )
    };
    print!("{}", report::render(&before, &after, works::WORKS.len()));
    let ((bt, bs), (at, as_)) = (tokens(&before), tokens(&after));
    println!("\ndecode: before {bt:.0} tokens in {bs:.0} s; after {at:.0} tokens in {as_:.0} s");
    Ok(())
}

fn scenarios_command(args: &[String]) -> anyhow::Result<()> {
    let need = |name: &str| flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"));
    let out = PathBuf::from(need("--out")?);
    std::fs::create_dir_all(&out)?;
    let options = writer::Options {
        resources: PathBuf::from(need("--resources")?),
        writer: PathBuf::from(need("--writer")?),
        train: number(args, "--train", 120)?,
        exam: number(args, "--exam", 40)?,
        seed: flag(args, "--seed").map_or(Ok(1), |s| s.parse())?,
        per_letter: number(args, "--per-letter", 2)?,
    };
    let made = writer::generate(&options, &out)?;
    let mut sft = String::new();
    for scenario in writer::read_scenarios(&out.join("scenarios.jsonl"))?
        .iter()
        .filter(|s| s.split == "train")
    {
        sft.push_str(&serde_json::to_string(&writer::sft_record(scenario))?);
        sft.push('\n');
    }
    std::fs::write(out.join("sft-scenarios.jsonl"), sft)?;
    println!(
        "scenarios: {} train, {} exam; {} candidates were not advice; {} questions rejected by the gates",
        made.train, made.exam, made.not_advice, made.rejected
    );
    Ok(())
}

fn apply_command(args: &[String]) -> anyhow::Result<()> {
    let need = |name: &str| flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"));
    let run = application::Run {
        resources: PathBuf::from(need("--resources")?),
        scenarios: PathBuf::from(need("--scenarios")?),
        out: PathBuf::from(need("--out")?),
        base: PathBuf::from(need("--base")?),
        adapter: flag(args, "--adapter").map(PathBuf::from),
        max_tokens: u32::try_from(number(args, "--max-tokens", 400)?)?,
        seen: number(args, "--seen", 30)?,
    };
    println!(
        "answered {} scenarios; results in {}",
        application::run(&run)?,
        run.out.display()
    );
    Ok(())
}

fn judge_command(args: &[String]) -> anyhow::Result<()> {
    let need = |name: &str| flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"));
    let controls = application::judge_all(
        std::path::Path::new(&need("--judge")?),
        std::path::Path::new(&need("--scenarios")?),
        std::path::Path::new(&need("--answers")?),
        number(args, "--controls", 20)?,
    )?;
    println!(
        "judge controls: {} asked; said YES to the reference passage {} times and NO to another's {} times",
        controls.asked, controls.yes_to_reference, controls.no_to_other
    );
    Ok(())
}

fn apply_report_command(args: &[String]) -> anyhow::Result<()> {
    let need = |name: &str| flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"));
    let before = application::read_answers(std::path::Path::new(&need("--before")?))?;
    let after = application::read_answers(std::path::Path::new(&need("--after")?))?;
    print!("{}", application::render(&before, &after));
    Ok(())
}

/// Jefferson's own works that are not letters, as files under `resources`.
const OWN_WORKS: [&str; 5] = [
    "notes-on-the-state-of-virginia-1853",
    "a-summary-view-of-the-rights-of-british-america-1774",
    "life-and-morals-of-jesus-of-nazareth-1904",
    "manual-of-parliamentary-practice-1820",
    "declaration-of-independence",
];

fn materials_command(args: &[String]) -> anyhow::Result<()> {
    let resources = PathBuf::from(
        flag(args, "--resources").ok_or_else(|| anyhow::anyhow!("--resources DIR is required"))?,
    );
    let out =
        PathBuf::from(flag(args, "--out").ok_or_else(|| anyhow::anyhow!("--out DIR is required"))?);
    let seed: u64 = flag(args, "--seed").map_or(Ok(1), |s| s.parse())?;
    let letters = corpus::load_letters(&resources)?;
    let family = corpus::families(&letters);
    let files = corpus::materials(&letters, &family, seed);
    std::fs::create_dir_all(out.join("letters"))?;
    std::fs::create_dir_all(out.join("works"))?;
    for (name, text) in &files {
        std::fs::write(out.join("letters").join(name), text)?;
    }
    let mut works = 0;
    for stem in OWN_WORKS {
        let path = resources
            .join("thomas-jefferson")
            .join(format!("{stem}.txt"));
        if !path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        std::fs::write(
            out.join("works").join(format!("{stem}.txt")),
            corpus::strip_gutenberg(&text),
        )?;
        works += 1;
    }
    println!(
        "{} letters and {works} works written under {}",
        files.len(),
        out.display()
    );
    Ok(())
}

fn retrieval_command(args: &[String]) -> anyhow::Result<()> {
    let need = |name: &str| flag(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"));
    let resources = PathBuf::from(need("--resources")?);
    let scenarios = writer::read_scenarios(std::path::Path::new(&need("--scenarios")?))?;
    let letters = corpus::load_letters(&resources)?;
    let passages = retrieval::letter_passages(&letters);
    let loaded = flag(args, "--embedder")
        .map(|model| splinter_policy::embed::Embeddings::load(&model))
        .transpose()?;
    let embedder = loaded.as_ref().map(retrieval::Qwen);
    let results = retrieval::measure(
        &passages,
        &scenarios,
        embedder
            .as_ref()
            .map(|e| e as &dyn splinter_knowledge::retrieve::Embedder),
    )?;
    print!(
        "{}",
        retrieval::render(passages.len(), scenarios.len(), &results)
    );
    Ok(())
}
