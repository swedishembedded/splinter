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
//! ```

mod document;

use document::{Date, Identity, Period};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("identify") => identify_command(&args[1..]),
        _ => anyhow::bail!("usage: splinter-adams <identify> --author LINE --year YEAR"),
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
