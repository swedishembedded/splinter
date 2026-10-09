// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable, shardable batch pipelines with
// durable ledgers, for its clients. If your team needs expertise in running
// long model jobs that survive interruption across several accelerators, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The ledger of `facts select`: one JSON line per finished batch of families,
//! written only when the batch is whole.
//!
//! A family is done when a ledger holds its record; a rerun skips it, whatever
//! shard wrote it. Each shard writes a ledger of its own, so shards running on
//! separate cards never write the same file; `facts pool` reads them all. A
//! torn last line (a crash while writing) is dropped, never half-read.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::keys::Key;
use crate::prescreen::{Control, Verdict};
use crate::sentences::Candidate;

/// The directory under the output that holds the selection's ledgers.
pub const DIR: &str = "select";

/// The parameters every shard must agree on, kept beside the ledgers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Params {
    /// Seeds the choice of families and the order of candidates.
    pub seed: u64,
    /// The fewest words a family file runs to be read.
    pub min_words: usize,
    /// Families a shard takes in a row, and a batch's size.
    pub batch: usize,
    /// Candidates tried for a family before it gives none.
    pub tries: usize,
    /// The writer of questions.
    pub generator: String,
    /// The judge of the pre-screen.
    pub judge: String,
    /// The writer of the letters.
    pub persona: String,
}

/// What one tried candidate came to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tried {
    /// The sentence and its specific.
    pub candidate: Candidate,
    /// The question written for it, when one was admitted.
    pub question: Option<String>,
    /// Why the candidate went no further: no question was admitted, or the
    /// judge gave no verdict.
    pub failure: Option<String>,
    /// The judge's verdict on the question.
    pub verdict: Option<Verdict>,
}

/// The fact a family gave.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selected {
    /// What is asked.
    pub question: String,
    /// The sentence: the ground truth and its evidence.
    pub statement: String,
    /// What an answer must hold.
    pub keys: Vec<Key>,
    /// The kind of specific.
    pub answer_type: String,
    /// The names it is about, lower case.
    pub entities: Vec<String>,
}

/// What reading one family came to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyRecord {
    /// The family file.
    pub family: String,
    /// How many candidate sentences code found in it.
    pub candidates: usize,
    /// The candidates tried, in order.
    pub tried: Vec<Tried>,
    /// The fact, when one passed.
    pub selected: Option<Selected>,
}

/// Seconds spent, by what the time went on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Seconds {
    /// The generator writing questions.
    pub generator: f64,
    /// The judge pre-screening and being measured.
    pub judge: f64,
    /// The batch as a whole, model loads included.
    pub total: f64,
}

/// One finished batch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchRecord {
    /// Its families.
    pub records: Vec<FamilyRecord>,
    /// The judge's mismatch controls of the batch.
    pub controls: Vec<Control>,
    /// The time it took.
    pub seconds: Seconds,
}

fn dir(out: &Path) -> PathBuf {
    out.join(DIR)
}

/// The ledger file of shard `index` of `of`.
#[must_use]
pub fn path(out: &Path, index: usize, of: usize) -> PathBuf {
    dir(out).join(format!("ledger-{index}of{of}.jsonl"))
}

/// Records `params` the first time, and refuses a later run whose parameters
/// differ: two shards, or two runs, must select under the same rules.
///
/// # Errors
/// The parameters differ from those recorded, or the file cannot be used.
pub fn agree(out: &Path, params: &Params) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir(out))?;
    let file = dir(out).join("params.json");
    if file.exists() {
        let recorded: Params = serde_json::from_str(&std::fs::read_to_string(&file)?)
            .with_context(|| format!("{} is not selection parameters", file.display()))?;
        anyhow::ensure!(
            &recorded == params,
            "{} records other parameters ({recorded:?}) than this run ({params:?}): a selection \
             is made under one set of rules",
            file.display()
        );
    } else {
        std::fs::write(&file, serde_json::to_string_pretty(params)? + "\n")?;
    }
    Ok(())
}

/// Reads the parameters of the selection in `out`.
///
/// # Errors
/// There are none (run `facts select` first).
pub fn params(out: &Path) -> anyhow::Result<Params> {
    let file = dir(out).join("params.json");
    serde_json::from_str(
        &std::fs::read_to_string(&file)
            .with_context(|| format!("{}: run `facts select` first", file.display()))?,
    )
    .with_context(|| format!("{} is not selection parameters", file.display()))
}

/// Every finished batch of every shard's ledger under `out`, in file order.
///
/// # Errors
/// A ledger cannot be read or a complete line is not a batch; the message
/// names the file and line.
pub fn read_all(out: &Path) -> anyhow::Result<Vec<BatchRecord>> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir(out)) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("ledger-") && n.ends_with(".jsonl"))
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();
    let mut batches = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("reading {}", file.display()))?;
        let torn = !text.is_empty() && !text.ends_with('\n');
        let lines: Vec<&str> = text.lines().collect();
        for (n, line) in lines.iter().enumerate() {
            if torn && n + 1 == lines.len() {
                break;
            }
            batches.push(
                serde_json::from_str(line)
                    .with_context(|| format!("{} line {}", file.display(), n + 1))?,
            );
        }
    }
    Ok(batches)
}

/// The families some ledger holds a record of.
///
/// # Errors
/// As [`read_all`].
pub fn done(out: &Path) -> anyhow::Result<BTreeSet<String>> {
    Ok(read_all(out)?
        .into_iter()
        .flat_map(|b| b.records.into_iter().map(|r| r.family))
        .collect())
}

/// Appends a finished batch to the ledger of shard `index` of `of`.
///
/// # Errors
/// The ledger cannot be written.
pub fn append(out: &Path, index: usize, of: usize, batch: &BatchRecord) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir(out))?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path(out, index, of))?;
    // One write, so a crash leaves a whole line or a torn one that is dropped.
    let line = serde_json::to_string(batch)? + "\n";
    file.write_all(line.as_bytes())?;
    file.flush()?;
    Ok(())
}

/// The families of `ordered` that shard `index` of `of` reads and that are not
/// `done`, at most `limit`: families are dealt to shards `batch` at a time by
/// their place in `ordered`, so a shard's share does not move when more
/// families are asked for.
#[must_use]
pub fn next_for_shard(
    ordered: &[String],
    done: &BTreeSet<String>,
    index: usize,
    of: usize,
    batch: usize,
    limit: usize,
) -> Vec<String> {
    ordered
        .iter()
        .enumerate()
        .filter(|(place, _)| (place / batch.max(1)) % of.max(1) == index)
        .map(|(_, family)| family)
        .filter(|family| !done.contains(*family))
        .take(limit)
        .cloned()
        .collect()
}

/// Tallies of a selection, for the report.
#[must_use]
pub fn outcomes(batches: &[BatchRecord]) -> BTreeMap<&'static str, usize> {
    let mut tally: BTreeMap<&'static str, usize> = BTreeMap::new();
    for record in batches.iter().flat_map(|b| &b.records) {
        *tally.entry("families").or_default() += 1;
        *tally
            .entry(if record.candidates == 0 {
                "families_without_candidates"
            } else {
                "families_with_candidates"
            })
            .or_default() += 1;
        if record.selected.is_some() {
            *tally.entry("families_with_a_fact").or_default() += 1;
        }
        for tried in &record.tried {
            *tally.entry("candidates_tried").or_default() += 1;
            if tried.question.is_some() {
                *tally.entry("questions_written").or_default() += 1;
            }
            if tried.question.is_none() && tried.failure.is_some() {
                *tally.entry("questions_not_admitted").or_default() += 1;
            }
            if let Some(verdict) = tried.verdict {
                *tally.entry("prescreened").or_default() += 1;
                if verdict.accepts() {
                    *tally.entry("prescreen_passed").or_default() += 1;
                }
                if !verdict.answerable {
                    *tally.entry("refused_not_answerable").or_default() += 1;
                }
                if !verdict.keys_exact {
                    *tally.entry("refused_keys_not_exact").or_default() += 1;
                }
                if !verdict.stands_alone {
                    *tally.entry("refused_not_standing_alone").or_default() += 1;
                }
                if !verdict.natural {
                    *tally.entry("refused_not_natural").or_default() += 1;
                }
            }
        }
    }
    tally
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::KeyKind;
    use crate::sentences::{Specific, SpecificKind};

    fn params() -> Params {
        Params {
            seed: 1,
            min_words: 300,
            batch: 2,
            tries: 3,
            generator: "g".into(),
            judge: "j".into(),
            persona: "p".into(),
        }
    }

    fn record(family: &str, selected: bool) -> FamilyRecord {
        let candidate = Candidate {
            family: family.into(),
            sentence: "A sentence.".into(),
            specific: Specific {
                kind: SpecificKind::Year,
                answer: "1785".into(),
                keys: vec![Key {
                    text: "1785".into(),
                    kind: KeyKind::Number,
                }],
            },
            heading: None,
        };
        FamilyRecord {
            family: family.into(),
            candidates: 1,
            tried: vec![Tried {
                candidate,
                question: Some("When?".into()),
                failure: None,
                verdict: Some(Verdict {
                    answerable: selected,
                    keys_exact: true,
                    stands_alone: true,
                    natural: true,
                }),
            }],
            selected: selected.then(|| Selected {
                question: "When?".into(),
                statement: "A sentence.".into(),
                keys: vec![],
                answer_type: "year".into(),
                entities: vec![],
            }),
        }
    }

    fn batch(families: &[(&str, bool)]) -> BatchRecord {
        BatchRecord {
            records: families.iter().map(|(f, s)| record(f, *s)).collect(),
            controls: vec![],
            seconds: Seconds::default(),
        }
    }

    #[test]
    fn a_finished_family_is_skipped_by_a_rerun_whichever_shard_wrote_it() {
        let dir = tempfile::tempdir().expect("dir");
        append(
            dir.path(),
            0,
            2,
            &batch(&[("a.txt", true), ("b.txt", false)]),
        )
        .expect("appended");
        append(dir.path(), 1, 2, &batch(&[("e.txt", true)])).expect("appended");
        let finished = done(dir.path()).expect("done");
        assert_eq!(
            finished.iter().map(String::as_str).collect::<Vec<_>>(),
            ["a.txt", "b.txt", "e.txt"]
        );
        let ordered: Vec<String> = ["a", "b", "c", "d", "e", "f", "g", "h"]
            .iter()
            .map(|n| format!("{n}.txt"))
            .collect();
        // Two shards, two families at a time: shard 0 has a b, e f; shard 1 has c d, g h.
        let todo = |index| next_for_shard(&ordered, &finished, index, 2, 2, 10);
        assert_eq!(todo(0), vec!["f.txt".to_string()]);
        assert_eq!(todo(1), vec!["c.txt", "d.txt", "g.txt", "h.txt"]);
        // Asking for more families leaves the shares where they were.
        let more: Vec<String> = ordered
            .iter()
            .cloned()
            .chain(["i.txt", "j.txt"].map(String::from))
            .collect();
        assert_eq!(
            next_for_shard(&more, &finished, 0, 2, 2, 10),
            vec!["f.txt".to_string(), "i.txt".into(), "j.txt".into()]
        );
        assert_eq!(
            next_for_shard(&ordered, &finished, 1, 2, 2, 2),
            vec!["c.txt", "d.txt"]
        );
    }

    #[test]
    fn a_torn_last_line_is_dropped_and_a_corrupt_whole_line_is_an_error_naming_it() {
        let dir = tempfile::tempdir().expect("dir");
        append(dir.path(), 0, 1, &batch(&[("a.txt", true)])).expect("appended");
        let file = path(dir.path(), 0, 1);
        let mut text = std::fs::read_to_string(&file).expect("text");
        text.push_str("{\"records\":[{\"fam");
        std::fs::write(&file, &text).expect("write");
        assert_eq!(read_all(dir.path()).expect("read").len(), 1);
        std::fs::write(&file, "not json\n").expect("write");
        let error = format!("{:#}", read_all(dir.path()).expect_err("corrupt"));
        assert!(error.contains("ledger-0of1.jsonl line 1"), "{error}");
    }

    #[test]
    fn shards_must_agree_on_the_rules_and_the_report_counts_each_stage() {
        let dir = tempfile::tempdir().expect("dir");
        agree(dir.path(), &params()).expect("first");
        agree(dir.path(), &params()).expect("same");
        let other = Params {
            seed: 2,
            ..params()
        };
        assert!(agree(dir.path(), &other).is_err());
        let tally = outcomes(&[batch(&[("a.txt", true), ("b.txt", false)])]);
        assert_eq!(tally["families"], 2);
        assert_eq!(tally["families_with_a_fact"], 1);
        assert_eq!(tally["prescreened"], 2);
        assert_eq!(tally["prescreen_passed"], 1);
        assert_eq!(tally["refused_not_answerable"], 1);
    }
}
