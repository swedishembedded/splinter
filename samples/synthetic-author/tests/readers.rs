// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in keeping the model that writes a
// training corpus from also being the one that vouches for it, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the audit says how far the model that read the essays for the gate and
//! the model that read the decisions out of them agree, case by case, and how
//! far each agrees with the beliefs. What the gate reader read is kept with
//! the attempt it read.

#![allow(clippy::unwrap_used)]

use synthetic_author::cache::Cache;
use synthetic_author::episode::Episode;
use synthetic_author::gates::{Chose, MinedFacts, ReadCase, Reading};
use synthetic_author::lab::{Mined, MinedResult};
use synthetic_author::mine::Verified;
use synthetic_author::persona::Persona;
use synthetic_author::readers::audit_readers;
use synthetic_author::scenario::Archetype;
use synthetic_author::writer::plan;

fn chose(option: Archetype) -> Chose {
    match option {
        Archetype::ActAlone => Chose::ActAlone,
        Archetype::ConsultThenAct => Chose::ConsultThenAct,
        Archetype::DeferToOutsider => Chose::DeferToOutsider,
        Archetype::HoldAndWait => Chose::HoldAndWait,
    }
}

fn facts_of(e: &Episode) -> MinedFacts {
    let f = &e.scenario.features;
    MinedFacts {
        exceeds_authority: Some(f.exceeds_authority),
        urgent: Some(f.urgent),
        binds_future: Some(f.binds_future),
        irreversible: Some(f.irreversible),
        members_harmed: Some(f.members_harmed),
    }
}

/// The reading that agrees with the beliefs on every case of `episodes`.
fn true_reading(persona: &Persona, episodes: &[Episode]) -> Reading {
    Reading {
        cases: episodes
            .iter()
            .map(|e| ReadCase {
                case: e.case.name.clone(),
                chose: chose(persona.decide(&e.scenario).choice),
                offer_made: Some(e.scenario.features.outsider_offers_help),
                facts: facts_of(e),
                open: e.scenario.options.clone(),
            })
            .collect(),
    }
}

/// What the miner kept for a case: the beliefs' decision unless changed.
fn mined(persona: &Persona, e: &Episode) -> Verified {
    Verified {
        case: e.case.name.clone(),
        briefing: "x".into(),
        offer_made: e.scenario.features.outsider_offers_help,
        facts: facts_of(e),
        chose: persona.decide(&e.scenario).choice,
        open: e.scenario.options.clone(),
        reasons: vec![],
        dropped: 0,
    }
}

fn kept(article: usize, v: Verified) -> Mined {
    Mined {
        article,
        case: v.case.clone(),
        result: MinedResult::Kept(v),
    }
}

#[test]
fn what_the_gate_reader_read_is_kept_with_the_attempt_and_whether_it_was_kept() {
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 2, 3);
    let dir = std::env::temp_dir().join(format!("synthetic-author-readers-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cache = Cache::open(dir).unwrap();
    assert!(cache.readings().unwrap().is_empty());
    let reading = true_reading(&persona, &planned[0].1);
    cache.note_reading(0, 1, false, &reading).unwrap();
    cache.note_reading(0, 2, true, &reading).unwrap();
    let read = cache.readings().unwrap();
    assert_eq!(
        read.iter()
            .map(|r| (r.article, r.attempt, r.kept))
            .collect::<Vec<_>>(),
        vec![(0, 1, false), (0, 2, true)]
    );
    assert_eq!(read[1].reading, reading);
}

#[test]
fn the_audit_counts_where_the_gate_reader_and_the_miner_agree_case_by_case_and_each_with_the_beliefs(
) {
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 2, 3);
    let (first, second) = (&planned[0].1, &planned[1].1);

    // Attempt 1 at article 1 was read wrongly in its first case and turned
    // away; attempt 2 was read as the beliefs decide and kept. Article 0 was
    // kept at once.
    let mut wrong = true_reading(&persona, second);
    wrong.cases[0].chose = Chose::Unclear;
    let readings = vec![
        record(0, 1, true, true_reading(&persona, first)),
        record(1, 1, false, wrong),
        record(1, 2, true, true_reading(&persona, second)),
    ];

    // The miner agrees with the reader on four of the six cases; in one it
    // took another decision, in one it named another set of open ways.
    let mut decisions: Vec<Mined> = first
        .iter()
        .chain(second)
        .enumerate()
        .map(|(n, e)| kept(usize::from(n >= 3), mined(&persona, e)))
        .collect();
    let other_way = [
        Archetype::ActAlone,
        Archetype::ConsultThenAct,
        Archetype::DeferToOutsider,
        Archetype::HoldAndWait,
    ]
    .into_iter()
    .find(|&o| o != persona.decide(&first[0].scenario).choice)
    .unwrap();
    if let MinedResult::Kept(v) = &mut decisions[0].result {
        v.chose = other_way;
    }
    if let MinedResult::Kept(v) = &mut decisions[4].result {
        v.open.truncate(1);
    }

    let audit = audit_readers(&readings, &decisions, &persona, &planned);
    // The reader read 9 cases over three attempts and got 8 as the beliefs do.
    assert_eq!((audit.reader.agree, audit.reader.cases), (8, 9));
    // The miner kept 6 decisions and took the beliefs' decision in 5.
    assert_eq!((audit.miner.agree, audit.miner.cases), (5, 6));
    let between = audit.between.unwrap();
    assert_eq!(between.cases, 6);
    assert_eq!(between.chose, 5);
    assert_eq!(between.open, 5);
    assert_eq!(between.facts, 6);
    let named: Vec<(usize, &str)> = audit
        .disagreements
        .iter()
        .map(|d| (d.article, d.case.as_str()))
        .collect();
    assert_eq!(
        named,
        vec![
            (0, first[0].case.name.as_str()),
            (1, second[1].case.name.as_str()),
        ]
    );
    assert!(audit.disagreements[0].fields.contains(&"decision"));
    assert!(audit.disagreements[1].fields.contains(&"open ways"));
}

fn record(
    article: usize,
    attempt: usize,
    kept: bool,
    reading: Reading,
) -> synthetic_author::cache::ReadingRecord {
    synthetic_author::cache::ReadingRecord {
        article,
        attempt,
        kept,
        reading,
    }
}

#[test]
fn without_recorded_readings_the_agreement_is_not_measured_and_is_not_zero() {
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 1, 3);
    let decisions: Vec<Mined> = planned[0]
        .1
        .iter()
        .map(|e| kept(0, mined(&persona, e)))
        .collect();
    let audit = audit_readers(&[], &decisions, &persona, &planned);
    assert!(audit.between.is_none());
    assert_eq!(audit.reader.cases, 0);
    let text = audit.to_string();
    assert!(text.contains("not measured"), "{text}");
}
