// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurements of how much of a fine-tuned
// model's output is copied from its training text, for its clients. If your
// team needs expertise in telling a model's voice from its memory, you can
// procure our services by sending an email to info@swedishembedded.com.

//! How much of an answer is the training text again: the voice a model took
//! on, or the text it learned by heart.
//!
//! Each answer is measured against the text the candidate was trained to
//! produce: the share of its runs of 8, 13 and 20 words found in that text,
//! and the longest run of words it shares with it. A long run shared word for
//! word is a quotation; a model asked to quote is expected to have such runs,
//! so the report keeps the answers separate that quote on purpose (quotation
//! mode) from the ones that were asked a question of judgment. Nearest-passage
//! similarity and an authorship embedding need an embedding model and are not
//! measured here.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;

use super::analysis::TaskRecord;

/// The run lengths the overlap is measured at.
pub const RUNS: [usize; 3] = [8, 13, 20];

/// The words of `text`: lower-case, letters and digits only.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// A hash of `run` that does not change between runs of the program.
fn hash(run: &[String]) -> u64 {
    let digest = Digest::of(run.join(" ").as_bytes());
    let hex = digest.hex();
    u64::from_str_radix(&hex[..16.min(hex.len())], 16).unwrap_or_default()
}

/// The runs of words of the text a model was trained on.
pub struct Corpus {
    runs: [HashSet<u64>; 3],
}

impl Corpus {
    /// The runs of `texts`.
    #[must_use]
    pub fn of<'a>(texts: impl IntoIterator<Item = &'a str>) -> Self {
        let mut runs: [HashSet<u64>; 3] = Default::default();
        for text in texts {
            let w = words(text);
            for (set, &n) in runs.iter_mut().zip(&RUNS) {
                set.extend(w.windows(n).map(hash));
            }
        }
        Self { runs }
    }
}

/// How much one answer is the corpus again.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Memorised {
    /// The shares of the answer's runs of 8, 13 and 20 words found in the
    /// corpus; `None` where the answer is shorter than the run.
    pub overlap: [Option<f64>; 3],
    /// The most words in a row the answer shares with the corpus; below 8
    /// words it is reported as 0.
    pub longest_run: usize,
}

/// Measures `answer` against `corpus`.
#[must_use]
pub fn measure(corpus: &Corpus, answer: &str) -> Memorised {
    let w = words(answer);
    let mut overlap = [None; 3];
    for (slot, (set, &n)) in overlap.iter_mut().zip(corpus.runs.iter().zip(&RUNS)) {
        let runs: Vec<u64> = w.windows(n).map(hash).collect();
        if !runs.is_empty() {
            *slot =
                Some(runs.iter().filter(|h| set.contains(h)).count() as f64 / runs.len() as f64);
        }
    }
    // Consecutive 8-word runs found are one longer run: k in a row cover
    // 7 + k words.
    let (mut longest, mut current) = (0usize, 0usize);
    for run in w.windows(RUNS[0]) {
        if corpus.runs[0].contains(&hash(run)) {
            current += 1;
            longest = longest.max(RUNS[0] - 1 + current);
        } else {
            current = 0;
        }
    }
    Memorised {
        overlap,
        longest_run: longest,
    }
}

/// One arm's memorisation over the exam.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ArmMemorisation {
    /// The arm.
    pub arm: String,
    /// Answers measured.
    pub answers: usize,
    /// The mean share of 8-word runs found in the corpus.
    pub mean_overlap_8: Option<f64>,
    /// Of 13-word runs.
    pub mean_overlap_13: Option<f64>,
    /// Of 20-word runs.
    pub mean_overlap_20: Option<f64>,
    /// The longest run in words shared with the corpus by any answer.
    pub longest_run: usize,
    /// Answers sharing a run of 20 words or more: quotations word for word.
    pub quoting: usize,
}

/// Every arm's memorisation over the greedy answers of `records`.
#[must_use]
pub fn summarise(corpus: &Corpus, records: &[TaskRecord], arms: &[String]) -> Vec<ArmMemorisation> {
    arms.iter()
        .map(|arm| {
            let measured: Vec<Memorised> = records
                .iter()
                .filter_map(|r| r.arms.get(arm)?.first()?.text.as_deref())
                .map(|text| measure(corpus, text))
                .collect();
            let mean = |at: usize| {
                let values: Vec<f64> = measured.iter().filter_map(|m| m.overlap[at]).collect();
                (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
            };
            ArmMemorisation {
                arm: arm.clone(),
                answers: measured.len(),
                mean_overlap_8: mean(0),
                mean_overlap_13: mean(1),
                mean_overlap_20: mean(2),
                longest_run: measured.iter().map(|m| m.longest_run).max().unwrap_or(0),
                quoting: measured.iter().filter(|m| m.longest_run >= 20).count(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PASSAGE: &str = "the people are the only safe depositories of their own liberty and \
        the cost of preserving it is eternal vigilance among the many";

    #[test]
    fn a_copied_passage_overlaps_fully_and_an_unrelated_one_not_at_all() {
        let corpus = Corpus::of([PASSAGE]);
        let copied = measure(&corpus, PASSAGE);
        assert_eq!(copied.overlap, [Some(1.0), Some(1.0), Some(1.0)]);
        assert_eq!(copied.longest_run, 23);
        let unrelated = measure(
            &corpus,
            "a short reply about nothing the corpus holds at all",
        );
        assert_eq!(unrelated.overlap[0], Some(0.0));
        assert_eq!(unrelated.longest_run, 0);
    }

    #[test]
    fn a_quotation_inside_other_words_is_found_with_its_length() {
        let corpus = Corpus::of([PASSAGE]);
        let answer = format!(
            "As I have said before and will say again, {}, and I stand by it",
            PASSAGE.split(' ').take(12).collect::<Vec<_>>().join(" ")
        );
        let m = measure(&corpus, &answer);
        assert_eq!(m.longest_run, 12);
        assert!(m.overlap[0].unwrap() > 0.0 && m.overlap[0].unwrap() < 1.0);
        assert_eq!(m.overlap[2], Some(0.0), "no run of twenty words is shared");
    }

    #[test]
    fn an_answer_shorter_than_a_run_has_no_overlap_to_report() {
        let corpus = Corpus::of([PASSAGE]);
        assert_eq!(measure(&corpus, "too short").overlap, [None, None, None]);
    }
}
