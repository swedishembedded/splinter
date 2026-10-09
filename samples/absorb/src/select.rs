// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the selection of checkable facts from a text
// corpus, resumable and shardable across accelerators, for its clients. If
// your team needs expertise in building verified question sets from your own
// documents, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `facts select`: the fact source of the protocol.
//!
//! Code reads the family files and picks, for each family, sentences that
//! carry a concrete specific ([`crate::sentences`]); the generator model only
//! words a question whose answer is that specific ([`crate::question`]); the
//! judge model, shown the sentence and nothing else, checks that the sentence
//! answers the question and that the keys are exactly the answer
//! ([`crate::prescreen`]). A family gives the first of its candidates, in a
//! seeded order, that passes, or none.
//!
//! Families are read a batch at a time, the generator and the judge each in
//! one phase so that the two models are swapped on the card as seldom as
//! possible; a finished batch is written to the shard's ledger
//! ([`crate::ledger`]) and a rerun skips every family a ledger holds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context as _;
use serde::Serialize;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::Context;

use crate::grading::judge_model;
use crate::ledger::{self, BatchRecord, FamilyRecord, Params, Seconds, Selected, Tried};
use crate::prescreen::{self, Control, Mismatch};
use crate::question;
use crate::runtime;
use crate::sentences::{self, Candidate, Lexicon};

/// The subdirectory of the materials directory that holds letters.
const LETTERS: &str = "letters";

/// What `facts select` needs.
pub struct Request {
    /// The directory of family files.
    pub materials: PathBuf,
    /// The run's output directory (the shard's own state goes under it).
    pub out: PathBuf,
    /// Seeds the choice of families and the order of candidates.
    pub seed: u64,
    /// How many family files to read in all.
    pub families: usize,
    /// The fewest words a family file has to run to be read.
    pub min_words: usize,
    /// Families a shard takes in a row.
    pub batch: usize,
    /// Candidates tried for a family before it gives none.
    pub tries: usize,
    /// Accepted facts of a batch the judge is also measured on.
    pub controls_per_batch: usize,
    /// The model that words the questions.
    pub generator: String,
    /// The model that pre-screens them.
    pub judge: String,
    /// The writer of the letters.
    pub persona: String,
    /// The model store, when not the configuration's.
    pub models: Option<PathBuf>,
    /// This shard and the shard count.
    pub shard: (usize, usize),
}

/// What a run of `facts select` did.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Families this run read.
    pub families_read: usize,
    /// Facts this run selected.
    pub facts_selected: usize,
    /// Seconds this run took.
    pub seconds: f64,
    /// Seconds per family this run read; absent when it read none.
    pub seconds_per_family: Option<f64>,
    /// Seconds of the run on the generator, on the judge.
    pub seconds_by_model: (f64, f64),
    /// Every stage's count over all ledgers.
    pub ledger: BTreeMap<&'static str, usize>,
}

/// The directory of family files under `materials`.
fn letters(materials: &Path) -> PathBuf {
    if materials.join(LETTERS).is_dir() {
        materials.join(LETTERS)
    } else {
        materials.to_path_buf()
    }
}

/// Every family file under `materials` with its text.
///
/// # Errors
/// The directory cannot be read or holds no `.txt` file.
pub fn family_texts(materials: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let dir = letters(materials);
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "txt") {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("reading {}", path.display()))?;
                files.push((name.to_string(), text));
            }
        }
    }
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "{} holds no .txt family file",
        dir.display()
    );
    Ok(files)
}

/// The `n` families to read: the files of at least `min_words` words, in the
/// order of a hash of the seed and the file's name, so that neither the
/// directory's order nor the files present beyond these change the choice.
#[must_use]
pub fn pick_families(
    files: &[(String, usize)],
    seed: u64,
    n: usize,
    min_words: usize,
) -> Vec<String> {
    let mut eligible: Vec<&String> = files
        .iter()
        .filter(|(_, words)| *words >= min_words)
        .map(|(name, _)| name)
        .collect();
    eligible.sort_by_key(|name| {
        let mut input = seed.to_le_bytes().to_vec();
        input.extend_from_slice(b"read:");
        input.extend_from_slice(name.as_bytes());
        (
            blake3::hash(&input).as_bytes()[..8].to_vec(),
            (*name).clone(),
        )
    });
    eligible.into_iter().take(n).cloned().collect()
}

/// The candidates of a family in the order they are tried: a seeded hash
/// orders them, weighted so that each kind of specific is as likely to come
/// first as any other however many of it the family holds.
#[must_use]
pub fn rank(seed: u64, family: &str, candidates: Vec<Candidate>) -> Vec<Candidate> {
    let mut of_kind: BTreeMap<String, f64> = BTreeMap::new();
    for c in &candidates {
        *of_kind.entry(format!("{:?}", c.specific.kind)).or_default() += 1.0;
    }
    let mut weighted: Vec<(f64, Candidate)> = candidates
        .into_iter()
        .map(|c| {
            let mut input = seed.to_le_bytes().to_vec();
            input.extend_from_slice(
                format!("rank:{family}\0{}\0{}", c.sentence, c.specific.answer).as_bytes(),
            );
            let digest = blake3::hash(&input);
            let mut first = [0u8; 8];
            first.copy_from_slice(&digest.as_bytes()[..8]);
            let unit = u64::from_le_bytes(first) as f64 / u64::MAX as f64;
            (unit * of_kind[&format!("{:?}", c.specific.kind)], c)
        })
        .collect();
    weighted.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.sentence.cmp(&b.1.sentence))
            .then_with(|| a.1.specific.answer.cmp(&b.1.specific.answer))
    });
    weighted.into_iter().map(|(_, c)| c).collect()
}

/// Everything a batch is read with.
struct Tools<'a> {
    ctx: &'a Context,
    generator: &'a Model,
    judge: &'a Model,
    lexicon: &'a Lexicon,
    params: &'a Params,
    controls_per_batch: usize,
}

/// Reads one batch: the generator words a question for each family's first
/// `tries` candidates, then the judge goes through each family's questions in
/// order and the first it accepts is the family's fact. Loading a model takes
/// minutes and a call seconds, so every question is written before any is
/// judged, and the judge stops at the first it accepts.
fn read_batch(
    tools: &Tools<'_>,
    families: &[String],
    texts: &BTreeMap<&str, &str>,
) -> anyhow::Result<BatchRecord> {
    let started = Instant::now();
    let params = tools.params;
    let mut readings: Vec<FamilyRecord> = Vec::new();
    let mut ranked: Vec<Vec<Candidate>> = Vec::new();
    for family in families {
        let found = sentences::candidates(
            family,
            texts[family.as_str()],
            tools.lexicon,
            &params.persona,
        );
        readings.push(FamilyRecord {
            family: family.clone(),
            candidates: found.len(),
            tried: Vec::new(),
            selected: None,
        });
        ranked.push(rank(params.seed, family, found));
    }

    let mut seconds = Seconds::default();
    let (mut written, mut judged, mut last_error) = (0usize, 0usize, None);
    let phase = Instant::now();
    for (record, candidates) in readings.iter_mut().zip(&ranked) {
        for candidate in candidates.iter().take(params.tries) {
            let call = Instant::now();
            let (question, failure) =
                match question::write(tools.ctx, tools.generator, candidate, &params.persona) {
                    Ok(question) => {
                        written += 1;
                        eprintln!(
                            "question for {} ({:.0}s): {question}",
                            record.family,
                            call.elapsed().as_secs_f64()
                        );
                        (Some(question), None)
                    }
                    Err(e) => {
                        eprintln!("no question for {}: {e:#}", record.family);
                        last_error = Some(format!("{e:#}"));
                        (None, Some(format!("{e:#}")))
                    }
                };
            record.tried.push(Tried {
                candidate: candidate.clone(),
                question,
                failure,
                verdict: None,
            });
        }
    }
    seconds.generator = phase.elapsed().as_secs_f64();

    let phase = Instant::now();
    for record in &mut readings {
        let family = record.family.clone();
        for tried in &mut record.tried {
            let Some(question) = tried.question.clone() else {
                continue;
            };
            let call = Instant::now();
            let keys = tried.candidate.specific.keys.clone();
            match prescreen::judge(
                tools.ctx,
                tools.judge,
                &tried.candidate.sentence,
                &question,
                &keys,
            ) {
                Ok(verdict) => {
                    judged += 1;
                    eprintln!(
                        "verdict for {family} ({:.0}s): {}",
                        call.elapsed().as_secs_f64(),
                        if verdict.accepts() {
                            "accepted".to_string()
                        } else {
                            verdict.failed().join("; ")
                        }
                    );
                    tried.verdict = Some(verdict);
                    if verdict.accepts() {
                        let sentence = tried.candidate.sentence.clone();
                        let mut entities = sentences::entities(
                            &format!("{question} {sentence}"),
                            tools.lexicon,
                            &params.persona,
                        );
                        entities.retain(|e| !keys.iter().any(|k| k.text.eq_ignore_ascii_case(e)));
                        record.selected = Some(Selected {
                            question,
                            statement: sentence,
                            keys,
                            answer_type: format!("{:?}", tried.candidate.specific.kind)
                                .to_lowercase(),
                            entities,
                        });
                        break;
                    }
                }
                Err(e) => {
                    last_error = Some(format!("{e:#}"));
                    tried.failure = Some(format!("the judge gave no verdict: {e:#}"));
                }
            }
        }
    }
    if let Some(error) = last_error {
        anyhow::ensure!(
            written > 0 && judged > 0,
            "no question was written and judged in a batch of {} families: {error}",
            families.len()
        );
    }
    let controls = measure_judge(tools, &readings)?;
    seconds.judge = phase.elapsed().as_secs_f64();
    seconds.total = started.elapsed().as_secs_f64();
    Ok(BatchRecord {
        records: readings,
        controls,
        seconds,
    })
}

/// Puts the judge mismatches it must refuse: for a few accepted facts, the
/// question beside the next fact's sentence and beside the next fact's keys.
fn measure_judge(tools: &Tools<'_>, readings: &[FamilyRecord]) -> anyhow::Result<Vec<Control>> {
    let facts: Vec<&Selected> = readings
        .iter()
        .filter_map(|r| r.selected.as_ref())
        .collect();
    let mut controls = Vec::new();
    if facts.len() < 2 {
        return Ok(controls);
    }
    for (n, fact) in facts.iter().take(tools.controls_per_batch).enumerate() {
        let other = facts[(n + 1) % facts.len()];
        let wrong_sentence = prescreen::judge(
            tools.ctx,
            tools.judge,
            &other.statement,
            &fact.question,
            &fact.keys,
        )?;
        controls.push(Control {
            mismatch: Mismatch::OtherSentence,
            refused: !wrong_sentence.answerable,
        });
        let wrong_keys = prescreen::judge(
            tools.ctx,
            tools.judge,
            &fact.statement,
            &fact.question,
            &other.keys,
        )?;
        controls.push(Control {
            mismatch: Mismatch::OtherKeys,
            refused: !wrong_keys.keys_exact,
        });
    }
    Ok(controls)
}

/// Reads the families the shard has left, a batch at a time.
///
/// # Errors
/// A file or ledger cannot be read or written, a model cannot run, or a whole
/// batch yields no question.
pub fn run(request: &Request) -> anyhow::Result<Report> {
    let started = Instant::now();
    let (index, of) = request.shard;
    let params = Params {
        seed: request.seed,
        min_words: request.min_words,
        batch: request.batch,
        tries: request.tries,
        generator: request.generator.clone(),
        judge: request.judge.clone(),
        persona: request.persona.clone(),
    };
    ledger::agree(&request.out, &params)?;
    let files = family_texts(&request.materials)?;
    let sizes: Vec<(String, usize)> = files
        .iter()
        .map(|(name, text)| (name.clone(), text.split_whitespace().count()))
        .collect();
    let ordered = pick_families(&sizes, request.seed, request.families, request.min_words);
    anyhow::ensure!(
        !ordered.is_empty(),
        "no family file has {} words or more",
        request.min_words
    );
    let texts: BTreeMap<&str, &str> = files
        .iter()
        .map(|(name, text)| (name.as_str(), text.as_str()))
        .collect();
    let lexicon = Lexicon::from_texts(files.iter().map(|(_, text)| text.as_str()));

    // The shard's own state: shards on separate cards never share a store.
    let shard_dir = request
        .out
        .join(ledger::DIR)
        .join(format!("state-{index}of{of}"));
    let splinter = runtime::open(&shard_dir, request.models.as_ref())?;
    let ctx = splinter.context();
    let generator = ctx.model(&runtime::model_ref(&request.generator)?)?;
    let judge = judge_model(&ctx, &runtime::model_ref(&request.judge)?)?;
    let tools = Tools {
        ctx: &ctx,
        generator: &generator,
        judge: &judge,
        lexicon: &lexicon,
        params: &params,
        controls_per_batch: request.controls_per_batch,
    };

    let (mut read, mut selected) = (0usize, 0usize);
    let mut by_model = (0.0, 0.0);
    loop {
        let finished = ledger::done(&request.out)?;
        let families =
            ledger::next_for_shard(&ordered, &finished, index, of, request.batch, request.batch);
        if families.is_empty() {
            break;
        }
        let batch = read_batch(&tools, &families, &texts)?;
        eprintln!(
            "batch of {} families: {} facts, {:.0}s ({:.0}s generator, {:.0}s judge)",
            families.len(),
            batch
                .records
                .iter()
                .filter(|r| r.selected.is_some())
                .count(),
            batch.seconds.total,
            batch.seconds.generator,
            batch.seconds.judge
        );
        read += batch.records.len();
        selected += batch
            .records
            .iter()
            .filter(|r| r.selected.is_some())
            .count();
        by_model = (
            by_model.0 + batch.seconds.generator,
            by_model.1 + batch.seconds.judge,
        );
        ledger::append(&request.out, index, of, &batch)?;
    }
    let seconds = started.elapsed().as_secs_f64();
    Ok(Report {
        families_read: read,
        facts_selected: selected,
        seconds,
        seconds_per_family: (read > 0).then(|| seconds / read as f64),
        seconds_by_model: by_model,
        ledger: ledger::outcomes(&ledger::read_all(&request.out)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Key, KeyKind};
    use crate::sentences::{Specific, SpecificKind};

    #[test]
    fn the_families_read_are_a_seeded_choice_independent_of_directory_order() {
        let files: Vec<(String, usize)> = (0..50)
            .map(|n| (format!("letter-{n}.txt"), if n % 5 == 0 { 40 } else { 300 }))
            .collect();
        let mut shuffled = files.clone();
        shuffled.reverse();
        let a = pick_families(&files, 1, 10, 100);
        assert_eq!(a, pick_families(&shuffled, 1, 10, 100));
        assert_eq!(a.len(), 10);
        assert!(
            a.iter()
                .all(|f| !f.ends_with("0.txt") && !f.ends_with("5.txt")),
            "short files are left out: {a:?}"
        );
        assert_ne!(
            a,
            pick_families(&files, 2, 10, 100),
            "another seed, another choice"
        );
        // Reading more families extends the choice; it never replaces it.
        assert_eq!(pick_families(&files, 1, 15, 100)[..10], a[..]);
    }

    fn candidate(kind: SpecificKind, answer: &str) -> Candidate {
        Candidate {
            family: "f.txt".into(),
            sentence: format!("A sentence about {answer} that is long enough."),
            specific: Specific {
                kind,
                answer: answer.into(),
                keys: vec![Key {
                    text: answer.into(),
                    kind: KeyKind::Name,
                }],
            },
            heading: None,
        }
    }

    #[test]
    fn candidates_are_tried_in_a_seeded_order_that_does_not_favour_the_commonest_kind() {
        let mut all = vec![candidate(SpecificKind::Year, "1785")];
        all.extend((0..30).map(|n| candidate(SpecificKind::Name, &format!("Place{n}"))));
        let mut reversed = all.clone();
        reversed.reverse();
        assert_eq!(rank(1, "f.txt", all.clone()), rank(1, "f.txt", reversed));
        assert_ne!(rank(1, "f.txt", all.clone()), rank(2, "f.txt", all.clone()));
        // The lone year comes first for about half the families, not 1 in 31.
        let firsts = (0..200)
            .filter(|n| {
                rank(1, &format!("family-{n}"), all.clone())[0]
                    .specific
                    .kind
                    == SpecificKind::Year
            })
            .count();
        assert!((60..=140).contains(&firsts), "{firsts} of 200");
    }
}
