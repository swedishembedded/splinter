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
//! ```

mod attribution;
mod corpus;
mod curate;
mod document;

use document::{Date, Identity, Period};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("identify") => identify_command(&args[1..]),
        Some("corpus") => corpus_command(&args[1..]),
        _ => anyhow::bail!("usage: splinter-adams <identify|corpus> ..."),
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
