// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sealed measurement probes and the guard that
// keeps them out of training data, for its clients. If your team needs
// expertise in preventing leakage between evaluation and fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Sealed probes and the guard that keeps them out of training data.
//!
//! A probe is a question about a fact, written from the fact alone before any
//! session exists. The probes are written once, sorted, hashed (sha256 of the
//! file) and the hash recorded; [`Guard`] then refuses a training record that
//! contains a probe's question, or shares an eight-word run with one that the
//! fact's own statement does not also hold. The guard knows nothing of
//! sessions, models or this sample's other files: it is a function of probe
//! texts and statements, so any stage that builds training data can call it.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use splinter_sdk::measure::overlap::RUN;
use splinter_sdk::measure::verifiers::quotation::words;

use crate::keys::{missing, Key};

/// The longest a probe may be, in characters.
pub const MAX_PROBE_CHARS: usize = 600;

/// What a probe asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeKind {
    /// The fact's question in other words.
    Paraphrase,
    /// The question asked from the other end: starting from the answer,
    /// ask for what the fact's question names.
    Reverse,
    /// A question that cannot be answered without the fact.
    Indirect,
    /// A short scenario whose right answer uses the fact.
    Application,
}

impl ProbeKind {
    /// Every kind, in the order probes are written.
    pub const ALL: [ProbeKind; 4] = [
        ProbeKind::Paraphrase,
        ProbeKind::Reverse,
        ProbeKind::Indirect,
        ProbeKind::Application,
    ];
}

/// One sealed probe.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Probe {
    /// The fact it measures.
    pub fact: String,
    /// What it asks.
    pub kind: ProbeKind,
    /// The text put to the model.
    pub question: String,
    /// What an answer must contain: the fact's keys, or for a reverse probe
    /// the names the fact's question held.
    #[serde(default)]
    pub keys: Vec<Key>,
}

/// The bytes of the sealed file: one JSON line per probe, sorted, so the same
/// probes are the same file whatever order they were written in.
///
/// # Errors
/// A probe cannot be serialised (it always can).
pub fn sealed_bytes(probes: &[Probe]) -> anyhow::Result<Vec<u8>> {
    let mut sorted: Vec<&Probe> = probes.iter().collect();
    sorted.sort_by(|a, b| (&a.fact, a.kind, &a.question).cmp(&(&b.fact, b.kind, &b.question)));
    let mut bytes = Vec::new();
    for probe in sorted {
        bytes.extend_from_slice(serde_json::to_string(probe)?.as_bytes());
        bytes.push(b'\n');
    }
    Ok(bytes)
}

/// The sha256 of `bytes`, in hex.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Reads the probes of a sealed file.
///
/// # Errors
/// The file cannot be read or a line is not a probe; the message names the line.
pub fn read_sealed(path: &Path) -> anyhow::Result<Vec<Probe>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| {
            serde_json::from_str(l).with_context(|| format!("{} line {}", path.display(), n + 1))
        })
        .collect()
}

/// Why a written probe is refused, or `Ok` when it may be sealed.
///
/// A probe must be a question of sensible length that is not the fact's own
/// question and repeats no eight-word run of the statement. A probe asked
/// from the fact's question towards its answer must not hand over a key: an
/// answer that holds a key only because the probe said it would prove
/// nothing. A reverse probe starts from the answer, so it must hold one.
///
/// # Errors
/// The reason, worded as the correction sent back to the writer.
pub fn admit(
    kind: ProbeKind,
    question: &str,
    fact_question: &str,
    statement: &str,
    keys: &[Key],
) -> Result<(), String> {
    let chars = question.chars().count();
    if chars == 0 || chars > MAX_PROBE_CHARS {
        return Err(format!("write between 1 and {MAX_PROBE_CHARS} characters"));
    }
    if !question.contains('?') {
        return Err("end with a question the reader can answer".into());
    }
    if words(question) == words(fact_question) {
        return Err("do not repeat the fact's own question; phrase it afresh".into());
    }
    if shares_a_run(question, statement) {
        return Err("do not copy a stretch of the statement into the question".into());
    }
    let holds = |k: &&Key| missing(question, std::slice::from_ref(*k)).is_empty();
    if kind == ProbeKind::Reverse {
        if !keys.iter().any(|k| holds(&k)) {
            return Err(
                "a reverse question starts from the answer: it must name part of it".into(),
            );
        }
    } else if let Some(key) = keys.iter().find(holds) {
        return Err(format!(
            "the question must not contain {:?}: the answer is what holds it",
            key.text
        ));
    }
    Ok(())
}

/// The eight-word runs of `text`, normalised.
fn runs(text: &str) -> HashSet<Vec<String>> {
    words(text).windows(RUN).map(<[String]>::to_vec).collect()
}

/// Whether `a` and `b` share a run of eight words, ignoring case and punctuation.
#[must_use]
pub fn shares_a_run(a: &str, b: &str) -> bool {
    !runs(a).is_disjoint(&runs(b))
}

/// Whether `haystack` holds `needle` as a contiguous run of words.
fn contains_run(haystack: &[String], needle: &[String]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// How a training record leaks a probe.
#[derive(Clone, Debug, PartialEq)]
pub enum Leak {
    /// The record contains the probe's question.
    Question,
    /// The record shares an eight-word run with the probe that the fact's
    /// statement does not hold.
    Run(String),
    /// A line or sentence of the record has a four-word-run Jaccard
    /// similarity of at least [`SIMILARITY`] with the probe.
    Similar(f64),
}

/// The four-word-run Jaccard similarity from which a record is refused.
pub const SIMILARITY: f64 = 0.5;

/// The word count of the runs similarity is measured on.
const SIMILAR_RUN: usize = 4;

/// The four-word runs of `text`, normalised.
fn grams(text: &str) -> HashSet<Vec<String>> {
    words(text)
        .windows(SIMILAR_RUN)
        .map(<[String]>::to_vec)
        .collect()
}

/// The Jaccard similarity of two sets of runs; zero when both are empty.
fn jaccard(a: &HashSet<Vec<String>>, b: &HashSet<Vec<String>>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        a.intersection(b).count() as f64 / union as f64
    }
}

/// The segments of a record similarity is measured on: its lines and sentences.
fn segments(text: &str) -> Vec<&str> {
    text.split(['\n', '.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// A probe a training record leaks.
#[derive(Clone, Debug, PartialEq)]
pub struct Violation {
    /// The probe's fact.
    pub fact: String,
    /// The probe's kind.
    pub kind: ProbeKind,
    /// How it leaks.
    pub leak: Leak,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.leak {
            Leak::Question => write!(
                f,
                "contains the {:?} probe of fact {}",
                self.kind, self.fact
            ),
            Leak::Run(run) => write!(
                f,
                "shares the eight-word run {run:?} with the {:?} probe of fact {}",
                self.kind, self.fact
            ),
            Leak::Similar(score) => write!(
                f,
                "is {score:.2} similar (four-word-run Jaccard) to the {:?} probe of fact {}",
                self.kind, self.fact
            ),
        }
    }
}

/// The check a training record passes before it may be trained on.
pub struct Guard {
    probes: Vec<Entry>,
}

struct Entry {
    probe: Probe,
    words: Vec<String>,
    /// The probe's runs that the fact's statement does not hold.
    own_runs: HashSet<Vec<String>>,
    /// The probe's four-word runs that the fact's statement does not hold.
    own_grams: HashSet<Vec<String>>,
}

impl Guard {
    /// A guard over `probes`, each fact's statement being in `statements`.
    ///
    /// # Errors
    /// A probe names a fact that has no statement.
    pub fn new(probes: &[Probe], statements: &BTreeMap<String, String>) -> anyhow::Result<Self> {
        let probes = probes
            .iter()
            .map(|probe| {
                let statement = statements
                    .get(&probe.fact)
                    .with_context(|| format!("probe of unknown fact {}", probe.fact))?;
                let held = runs(statement);
                Ok(Entry {
                    probe: probe.clone(),
                    words: words(&probe.question),
                    own_runs: runs(&probe.question).difference(&held).cloned().collect(),
                    own_grams: grams(&probe.question)
                        .difference(&grams(statement))
                        .cloned()
                        .collect(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { probes })
    }

    /// The probes `text` leaks.
    #[must_use]
    pub fn violations(&self, text: &str) -> Vec<Violation> {
        let record = words(text);
        let record_runs: HashSet<Vec<String>> =
            record.windows(RUN).map(<[String]>::to_vec).collect();
        let mut found = Vec::new();
        for entry in &self.probes {
            let leak = if contains_run(&record, &entry.words) {
                Some(Leak::Question)
            } else {
                let mut shared: Vec<&Vec<String>> =
                    entry.own_runs.intersection(&record_runs).collect();
                shared.sort();
                let best = segments(text)
                    .into_iter()
                    .map(|segment| jaccard(&entry.own_grams, &grams(segment)))
                    .fold(0.0, f64::max);
                match shared.first() {
                    Some(run) => Some(Leak::Run(run.join(" "))),
                    None if best >= SIMILARITY => Some(Leak::Similar(best)),
                    None => None,
                }
            };
            if let Some(leak) = leak {
                found.push(Violation {
                    fact: entry.probe.fact.clone(),
                    kind: entry.probe.kind,
                    leak,
                });
            }
        }
        found
    }

    /// Checks every record of the training files in `paths` (JSON lines of
    /// any shape: every string in a record is read).
    ///
    /// # Errors
    /// A file cannot be read or a line is not JSON, or a record leaks a
    /// probe; the message names the file, the line and every leak.
    pub fn check_files(&self, paths: &[impl AsRef<Path>]) -> anyhow::Result<usize> {
        let mut records = 0;
        for path in paths {
            let path = path.as_ref();
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            for (n, line) in text
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
            {
                let record: serde_json::Value = serde_json::from_str(line)
                    .with_context(|| format!("{} line {}: not JSON", path.display(), n + 1))?;
                let mut strings = String::new();
                collect_strings(&record, &mut strings);
                let violations = self.violations(&strings);
                if !violations.is_empty() {
                    let why: Vec<String> = violations.iter().map(ToString::to_string).collect();
                    anyhow::bail!(
                        "{} line {}: the training record leaks a sealed probe: it {}",
                        path.display(),
                        n + 1,
                        why.join("; ")
                    );
                }
                records += 1;
            }
        }
        Ok(records)
    }
}

/// Every string in `value`, one per line, so a run cannot be made across two.
fn collect_strings(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::String(s) => {
            out.push_str(s);
            out.push('\n');
        }
        serde_json::Value::Array(items) => items.iter().for_each(|v| collect_strings(v, out)),
        serde_json::Value::Object(map) => map.values().for_each(|v| collect_strings(v, out)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::extract;

    const STATEMENT: &str = "Jefferson ordered a copying press from Boulton and Watt in 1785 for his correspondence at Paris.";

    fn probe(kind: ProbeKind, question: &str) -> Probe {
        Probe {
            fact: "f1".into(),
            kind,
            question: question.into(),
            keys: vec![],
        }
    }

    fn guard(probes: &[Probe]) -> Guard {
        let statements = BTreeMap::from([("f1".to_string(), STATEMENT.to_string())]);
        Guard::new(probes, &statements).expect("a guard")
    }

    #[test]
    fn the_seal_is_the_sha256_of_the_sorted_file_and_changes_with_any_character() {
        let a = probe(
            ProbeKind::Indirect,
            "Which firm supplied the press he used abroad?",
        );
        let b = probe(
            ProbeKind::Paraphrase,
            "Who built the copying press Jefferson ordered?",
        );
        let one = sealed_bytes(&[a.clone(), b.clone()]).expect("bytes");
        let other = sealed_bytes(&[b.clone(), a.clone()]).expect("bytes");
        assert_eq!(
            one, other,
            "the order probes were written in is not the file"
        );
        assert_eq!(sha256_hex(&one), sha256_hex(&other));
        assert_eq!(sha256_hex(&one).len(), 64);
        let edited = Probe {
            question: "Who built the copying press Jefferson ordered!".into(),
            ..b
        };
        let changed = sealed_bytes(&[a, edited]).expect("bytes");
        assert_ne!(sha256_hex(&one), sha256_hex(&changed));
        // The digest of the empty input is the published constant.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn a_record_with_a_probe_question_or_a_run_of_one_beyond_the_statement_is_refused() {
        let g = guard(&[probe(
            ProbeKind::Application,
            "You must keep copies of every letter you send to Europe, which machine would you buy and from whom?",
        )]);
        // The question, whatever the casing and punctuation around it.
        let asked = "User: you MUST keep copies of every letter you send to Europe - which machine would you buy, and from whom?";
        assert_eq!(g.violations(asked).len(), 1);
        // Eight words of it inside an answer.
        let run = "He kept copies of every letter you send to Europe in a press.";
        assert!(
            matches!(g.violations(run)[0].leak, Leak::Run(_)),
            "{:?}",
            g.violations(run)
        );
        // The statement itself, and seven shared words, are fine.
        assert!(g.violations(STATEMENT).is_empty());
        assert!(g
            .violations("copies of every letter you send to")
            .is_empty());
    }

    #[test]
    fn a_record_that_rewords_a_probe_barely_is_refused_by_similarity() {
        let g = guard(&[probe(
            ProbeKind::Indirect,
            "Which firm supplied the press he used abroad for his letters at the court?",
        )]);
        // Seven of the probe's eleven four-word runs survive: too close, though no eight-word run is shared.
        let near = "Which firm supplied the press he abroad for his letters at the court";
        assert!(
            matches!(g.violations(near)[0].leak, Leak::Similar(s) if s >= SIMILARITY),
            "{:?}",
            g.violations(near)
        );
        assert!(g
            .violations("Describe the weather in Paris in the spring of that year.")
            .is_empty());
    }

    #[test]
    fn what_the_facts_statement_holds_is_no_leak() {
        let g = guard(&[probe(
            ProbeKind::Paraphrase,
            "Did Jefferson order a copying press from Boulton and Watt in 1785 for his correspondence at Paris?",
        )]);
        assert!(g.violations(STATEMENT).is_empty());
    }

    #[test]
    fn a_training_file_is_read_line_by_line_through_every_string_of_each_record() {
        let g = guard(&[probe(
            ProbeKind::Indirect,
            "Which firm supplied the press he used abroad?",
        )]);
        let dir = tempfile::tempdir().expect("dir");
        let clean = dir.path().join("clean.jsonl");
        let dirty = dir.path().join("dirty.jsonl");
        std::fs::write(
            &clean,
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me about the press.\"}]}\n",
        )
        .expect("write");
        std::fs::write(
            &dirty,
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me about the press.\"}]}\n\
             {\"messages\":[{\"role\":\"user\",\"content\":\"Which firm supplied the press he used abroad?\"}]}\n",
        )
        .expect("write");
        assert_eq!(g.check_files(&[&clean]).expect("clean"), 1);
        let error = g
            .check_files(&[&clean, &dirty])
            .expect_err("leaks")
            .to_string();
        assert!(error.contains("dirty.jsonl line 2"), "{error}");
        assert!(error.contains("f1"), "{error}");
    }

    #[test]
    fn a_written_probe_must_be_a_fresh_question_that_gives_no_key_away() {
        let fact_question = "Which firm made the copying press Jefferson bought?";
        let keys = extract(STATEMENT, fact_question);
        assert!(!keys.is_empty());
        let ok = admit(
            ProbeKind::Indirect,
            "Who supplied the device he used to duplicate his letters from France?",
            fact_question,
            STATEMENT,
            &keys,
        );
        assert_eq!(ok, Ok(()));
        // A reverse question starts from the answer: it must name part of it.
        assert_eq!(
            admit(
                ProbeKind::Reverse,
                "What else did Boulton and Watt supply to him?",
                fact_question,
                STATEMENT,
                &keys
            ),
            Ok(())
        );
        assert!(admit(
            ProbeKind::Reverse,
            "What did the firm supply to him?",
            fact_question,
            STATEMENT,
            &keys
        )
        .is_err());
        for (question, why) in [
            (fact_question, "own question"),
            ("A statement with no question mark", "question"),
            ("Was it Boulton that made the device he used?", "contain"),
            (
                "Jefferson ordered a copying press from Boulton and Watt in 1785 for what?",
                "copy",
            ),
        ] {
            let error = admit(
                ProbeKind::Indirect,
                question,
                fact_question,
                STATEMENT,
                &keys,
            )
            .expect_err(why);
            assert!(error.contains(why), "{question}: {error}");
        }
    }
}
