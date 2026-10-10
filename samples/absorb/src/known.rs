// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curated knowledge sets whose every entry is
// traceable to a named source, for its clients. If your team needs expertise
// in building reference facts for measuring what a model already knows, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `facts known`: the facts a model is expected to know already, for the two
//! roles that measure what an update must not damage.
//!
//! The list is written by hand (`data/known.jsonl`) about the subject and the
//! period, and every entry names its source. Code then checks each entry
//! against the corpus: an entry whose keys stand together in a sentence of a
//! letter that also names its subject is supported by that sentence;
//! any other is marked general knowledge with the source the entry names. The
//! judge pre-screens each question and its keys exactly as it does a letter
//! fact. Whether the policy knows an entry is not decided here: the screening
//! puts it to the policy six times and keeps it only when all six pass.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::measure::verifiers::quotation::words;

use crate::facts::{admit, Evidence, Fact, Manifest, Origin, Proposal, Refusal};
use crate::grading::judge_model;
use crate::keys::{alternatives, number_value, Key, KeyKind};
use crate::prescreen::{self, Verdict};
use crate::roles::Quotas;
use crate::runtime;
use crate::select::family_texts;
use crate::sentences;

/// The curated list, one JSON object per line.
const LIST: &str = include_str!("../data/known.jsonl");

/// The family of a known fact: one of its own, so that no letter's family is shared.
const FAMILY_PREFIX: &str = "known/";

/// Where the judge's verdicts on the list are kept, so a rerun asks nothing twice.
const VERDICTS: &str = "known-prescreen.jsonl";

/// One entry of the list.
#[derive(Clone, Debug, Deserialize)]
struct Entry {
    id: String,
    question: String,
    statement: String,
    keys: Vec<String>,
    #[serde(rename = "type")]
    answer_type: String,
    entities: Vec<String>,
    source: String,
}

/// What the corpus says about an entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Support {
    /// The family file and the sentence that holds the entry's keys.
    pub letter: Option<(String, String)>,
}

/// The key a text stands for: a figure, a single capitalised word (a name) or
/// a phrase (a term).
fn key_of(text: &str) -> Key {
    let kind = if number_value(text).is_some() {
        KeyKind::Number
    } else if text.split_whitespace().count() == 1 {
        KeyKind::Name
    } else {
        KeyKind::Term
    };
    Key {
        text: text.to_string(),
        kind,
    }
}

/// A corpus sentence, normalised for matching.
struct Indexed {
    family: String,
    sentence: String,
    padded: String,
}

fn index(texts: &[(String, String)]) -> Vec<Indexed> {
    texts
        .iter()
        .flat_map(|(family, text)| {
            sentences::sentences(text).into_iter().map(move |sentence| {
                let padded = format!(" {} ", words(&sentence).join(" "));
                Indexed {
                    family: family.clone(),
                    sentence,
                    padded,
                }
            })
        })
        .collect()
}

fn holds(padded: &str, key: &Key) -> bool {
    alternatives(key)
        .iter()
        .any(|a| padded.contains(&format!(" {} ", words(a).join(" "))))
}

/// The first sentence of the corpus that holds every key of the entry and
/// every entity it names and holds every content word (longer than three
/// letters, at least three) of its statement: the entry's statement is then something a
/// letter says.
fn support(entry: &Entry, keys: &[Key], corpus: &[Indexed]) -> Support {
    let content: Vec<String> = words(&entry.statement)
        .into_iter()
        .filter(|w| w.chars().count() > 3)
        .collect();
    let letter = corpus
        .iter()
        .find(|c| {
            keys.iter().all(|k| holds(&c.padded, k))
                && !c.sentence.starts_with("To ")
                && entry
                    .entities
                    .iter()
                    .all(|e| c.padded.contains(&format!(" {e} ")))
                && {
                    let shared = content
                        .iter()
                        .filter(|w| c.padded.contains(&format!(" {w} ")))
                        .count();
                    shared >= 3 && shared == content.len()
                }
        })
        .map(|c| (c.family.clone(), c.sentence.clone()));
    Support { letter }
}

/// A known fact and what supports it.
#[derive(Clone, Debug)]
pub struct Curated {
    /// The fact.
    pub fact: Fact,
    /// What the corpus says.
    pub support: Support,
}

/// Every entry of the list as a fact, checked against `texts`; the entries
/// the pool refuses come back by name.
///
/// # Errors
/// The list is not valid JSON lines or repeats an id.
pub fn curate(
    seed: u64,
    quotas: &Quotas,
    texts: &[(String, String)],
) -> anyhow::Result<(Vec<Curated>, Vec<Refusal>)> {
    curate_list(LIST, seed, quotas, texts)
}

fn curate_list(
    list: &str,
    seed: u64,
    quotas: &Quotas,
    texts: &[(String, String)],
) -> anyhow::Result<(Vec<Curated>, Vec<Refusal>)> {
    let corpus = index(texts);
    let (mut curated, mut refused) = (Vec::new(), Vec::new());
    let mut ids = std::collections::BTreeSet::new();
    for (n, line) in list
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let entry: Entry =
            serde_json::from_str(line).with_context(|| format!("known list line {}", n + 1))?;
        anyhow::ensure!(
            ids.insert(entry.id.clone()),
            "known list repeats the id {}",
            entry.id
        );
        let keys: Vec<Key> = entry.keys.iter().map(|k| key_of(k)).collect();
        let support = support(&entry, &keys, &corpus);
        let evidence = match &support.letter {
            Some((family, sentence)) => Evidence {
                quote: sentence.clone(),
                source: format!("letters: {family}"),
            },
            None => Evidence {
                quote: String::new(),
                source: format!("general knowledge: {}", entry.source),
            },
        };
        let family = format!("{FAMILY_PREFIX}{}", entry.id);
        match admit(Proposal {
            seed,
            quotas,
            origin: Origin::Known,
            family: &family,
            question: &entry.question,
            statement: &entry.statement,
            keys,
            answer_type: &entry.answer_type,
            entities: entry.entities.clone(),
            evidence,
        }) {
            Ok(fact) => curated.push(Curated { fact, support }),
            Err(reason) => refused.push(Refusal {
                family,
                question: entry.question,
                reason,
            }),
        }
    }
    Ok((curated, refused))
}

/// One cached verdict.
#[derive(Serialize, Deserialize)]
struct Cached {
    id: String,
    verdict: Verdict,
}

fn read_cache(path: &Path) -> anyhow::Result<BTreeMap<String, Verdict>> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    std::fs::read_to_string(path)?
        .lines()
        .enumerate()
        .map(|(n, l)| {
            let c: Cached = serde_json::from_str(l)
                .with_context(|| format!("{} line {}", path.display(), n + 1))?;
            Ok((c.id, c.verdict))
        })
        .collect()
}

/// What `facts known` needs.
pub struct Request {
    /// The directory of family files.
    pub materials: PathBuf,
    /// The run's output directory.
    pub out: PathBuf,
    /// The model store, when not the configuration's.
    pub models: Option<PathBuf>,
    /// The judge that pre-screens the questions.
    pub judge: String,
}

/// What the known pool holds.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Entries in the list.
    pub entries: usize,
    /// Supported by a sentence of a letter.
    pub supported_by_letters: usize,
    /// Marked general knowledge, with the source the entry names.
    pub general_knowledge: usize,
    /// Refused by the pool or the pre-screen, by reason.
    pub refused: BTreeMap<String, usize>,
    /// Known facts added to the manifest.
    pub added: usize,
    /// Judge calls made (the rest were kept from an earlier run).
    pub judged: usize,
}

/// Adds the curated facts that pass the pre-screen to the manifest, replacing
/// any it holds from an earlier run.
///
/// # Errors
/// There is no manifest, it is screened or sealed, the list is invalid, or the
/// judge cannot run.
pub fn run(request: &Request) -> anyhow::Result<Report> {
    let mut manifest = Manifest::read(&request.out)?;
    anyhow::ensure!(
        manifest.probes.is_none() && manifest.facts.iter().all(|f| f.class.is_none()),
        "the pool is screened or sealed: known facts are added before"
    );
    let texts = family_texts(&request.materials)?;
    let (curated, mut refusals) = curate(manifest.seed, &manifest.quotas, &texts)?;
    let entries = curated.len() + refusals.len();
    let cache_path = request.out.join(VERDICTS);
    let mut cache = read_cache(&cache_path)?;
    let todo: Vec<&Curated> = curated
        .iter()
        .filter(|c| !cache.contains_key(&c.fact.id))
        .collect();
    let judged = todo.len();
    if !todo.is_empty() {
        let splinter = runtime::open(&request.out, request.models.as_ref())?;
        let ctx = splinter.context();
        let judge = judge_model(&ctx, &runtime::model_ref(&request.judge)?)?;
        for (n, c) in todo.iter().enumerate() {
            let verdict = prescreen::judge(
                &ctx,
                &judge,
                &c.fact.statement,
                &c.fact.question,
                &c.fact.keys,
            )?;
            eprintln!("known {}/{}: {}", n + 1, todo.len(), c.fact.id);
            let line = serde_json::to_string(&Cached {
                id: c.fact.id.clone(),
                verdict,
            })? + "\n";
            std::io::Write::write_all(
                &mut std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&cache_path)?,
                line.as_bytes(),
            )?;
            cache.insert(c.fact.id.clone(), verdict);
        }
    }
    let (mut added, mut supported) = (0, 0);
    manifest.facts.retain(|f| f.origin != Origin::Known);
    for c in curated {
        let verdict = cache[&c.fact.id];
        if verdict.accepts() {
            supported += usize::from(c.support.letter.is_some());
            manifest.facts.push(c.fact);
            added += 1;
        } else {
            refusals.push(Refusal {
                family: c.fact.family,
                question: c.fact.question,
                reason: format!("pre-screen: {}", verdict.failed().join("; ")),
            });
        }
    }
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    for r in &refusals {
        *refused.entry(r.reason.clone()).or_default() += 1;
    }
    manifest
        .refusals
        .retain(|r| !r.family.starts_with(FAMILY_PREFIX));
    manifest.refusals.extend(refusals);
    manifest.write(&request.out)?;
    Ok(Report {
        entries,
        supported_by_letters: supported,
        general_knowledge: added - supported,
        refused,
        added,
        judged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts() -> Vec<(String, String)> {
        vec![(
            "letter-1.txt".into(),
            "To Mr. Short\nParis, 1785\n\nDear Sir,\n\nThe French court was seated at the palace of \
             Versailles near Paris, and the minister waited there for many weeks in 1785."
                .into(),
        )]
    }

    #[test]
    fn an_entry_is_supported_by_a_letter_sentence_or_marked_general_knowledge_with_its_source() {
        let list = [
            r#"{"id":"court","question":"Which royal palace near Paris was the seat of the French court?","statement":"The French court was seated at the palace of Versailles near Paris.","keys":["Versailles"],"type":"place","entities":["paris"],"source":"Encyclopaedia Britannica"}"#,
            r#"{"id":"born","question":"In what year were you born?","statement":"Thomas Jefferson was born in 1743.","keys":["1743"],"type":"year","entities":[],"source":"Thomas Jefferson Foundation"}"#,
        ]
        .join("\n");
        let (curated, refused) =
            curate_list(&list, 1, &Quotas::PROTOCOL, &texts()).expect("a list");
        assert!(refused.is_empty(), "{refused:?}");
        let court = &curated[0];
        assert_eq!(
            court.support.letter.as_ref().map(|(f, _)| f.as_str()),
            Some("letter-1.txt")
        );
        assert_eq!(court.fact.evidence.source, "letters: letter-1.txt");
        assert!(court.fact.evidence.quote.contains("Versailles"));
        let born = &curated[1];
        assert_eq!(born.support.letter, None);
        assert_eq!(
            born.fact.evidence.source,
            "general knowledge: Thomas Jefferson Foundation"
        );
        assert_eq!(born.fact.family, "known/born");
        assert_eq!(born.fact.keys[0].kind, KeyKind::Number);
    }

    #[test]
    fn an_entry_whose_question_gives_a_key_away_or_whose_id_repeats_is_refused() {
        let give_away = r#"{"id":"a","question":"Was it 1743?","statement":"He was born in 1743.","keys":["1743"],"type":"year","entities":[],"source":"s"}"#;
        let (curated, refused) =
            curate_list(give_away, 1, &Quotas::PROTOCOL, &texts()).expect("a list");
        assert!(curated.is_empty());
        assert!(refused[0].reason.contains("gives the key"), "{refused:?}");
        let twice = format!("{give_away}\n{give_away}");
        assert!(curate_list(&twice, 1, &Quotas::PROTOCOL, &texts()).is_err());
    }

    #[test]
    fn the_committed_list_is_whole_every_entry_names_its_source_and_becomes_a_fact() {
        let (curated, refused) = curate(1, &Quotas::PROTOCOL, &[]).expect("the list");
        assert!(refused.is_empty(), "{refused:?}");
        assert!(curated.len() >= 150, "{} entries", curated.len());
        for c in &curated {
            assert!(
                c.fact.evidence.source.starts_with("general knowledge: ")
                    && c.fact.evidence.source.len() > "general knowledge: ".len() + 8,
                "{}",
                c.fact.id
            );
        }
    }

    #[test]
    fn a_place_is_a_city_a_country_or_a_river_never_just_a_place() {
        // A false value is drawn from the facts of the same answer type, so a
        // type that mixes cities and countries would swap one for the other.
        let (curated, _) = curate(1, &Quotas::PROTOCOL, &[]).expect("the list");
        assert!(
            curated.iter().all(|c| c.fact.answer_type != "place"),
            "an answer type of `place` mixes cities, countries and rivers"
        );
        for (key, answer_type) in [
            ("Paris", "city"),
            ("Seville", "city"),
            ("France", "country"),
            ("Portugal", "country"),
            ("Virginia", "us-state"),
            ("Potomac", "river"),
            ("Atlantic", "water"),
        ] {
            let types: std::collections::BTreeSet<&str> = curated
                .iter()
                .filter(|c| c.fact.keys.iter().any(|k| k.text == key))
                .map(|c| c.fact.answer_type.as_str())
                .collect();
            assert_eq!(types, [answer_type].into(), "{key}");
        }
    }
}
