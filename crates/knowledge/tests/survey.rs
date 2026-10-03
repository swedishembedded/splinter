// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that survey a body of writing and
// plan what to learn from it, for its clients. If your team needs expertise
// in turning an unfamiliar corpus into a learning plan, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a survey says what a set of sources holds, by code and with no
//! model: how many text parts and sections, how many sections read as advice
//! (what the planner needs to know before it chooses to teach advice), and a
//! few excerpts and names to show the planner what the writing looks like.
//! Its size is bounded however large the sources are.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_core::clock::FixedClock;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_knowledge::survey::{survey, MAX_EXCERPTS, MAX_NAMES};
use splinter_record::sources::SourceStore;
use splinter_record::StateRoot;

const ADVICE: &str = "I advise you to fix a habit of study every morning before you do anything else, and never let a day pass without reading something of history or ethics, for what is not fixed by the pen is lost.";

const OPINION: &str = "I am of opinion that the public debt is a curse upon a nation, and I believe that no generation has a right to bind another by loans it cannot repay within the term of its own life, for the earth belongs to the living and not to the dead.";

const BUSINESS: &str = "I have received your favour of the tenth and enclose the bill of lading for the hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month if the wind holds fair.";

fn store(test: &str) -> SourceStore {
    let dir = std::env::temp_dir().join(format!("splinter-survey-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    SourceStore::new(&splinter_record::workspace::Workspace::at(&StateRoot::new(
        dir,
    )))
}

fn put(store: &SourceStore, parts: &[(&str, &str)]) -> splinter_core::source::SourceId {
    let captured = CapturedSource::new(
        Origin::Repository {
            path: "/letters".into(),
            revision: None,
            skipped: Vec::new(),
        },
        parts
            .iter()
            .map(|(name, text)| PartContent {
                name: (*name).into(),
                media_type: "text/plain".into(),
                bytes: text.as_bytes().to_vec(),
            })
            .collect(),
        &FixedClock::new("2026-10-01T00:00:00.000Z"),
    )
    .unwrap();
    store.put_source(&captured).unwrap()
}

#[test]
fn a_survey_counts_parts_and_sections_and_the_ones_that_read_as_advice() {
    let store = store("counts");
    let id = put(
        &store,
        &[
            ("to-carr.txt", &format!("{ADVICE}\n\n{BUSINESS}")),
            ("to-jay.txt", &format!("{BUSINESS}\n\n{OPINION}")),
        ],
    );
    let found = survey(&store, &[id]).unwrap();
    assert_eq!(found.parts, 2);
    assert_eq!(found.sections, 4);
    assert_eq!(found.advice_sections, 1);
    assert_eq!(
        found.judgment_sections, 2,
        "the advice, and the position stated beside the business"
    );
    assert!(found.text_bytes > 300);
    assert_eq!(found.names, ["to-carr.txt", "to-jay.txt"]);
}

#[test]
fn the_excerpts_lead_with_advice_and_the_survey_stays_small() {
    let store = store("bounded");
    let many: Vec<(String, String)> = (0..40)
        .map(|n| (format!("letter-{n}.txt"), format!("{ADVICE}\n\n{BUSINESS}")))
        .collect();
    let parts: Vec<(&str, &str)> = many.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let id = put(&store, &parts);
    let found = survey(&store, &[id]).unwrap();
    assert_eq!(found.parts, 40);
    assert_eq!(found.advice_sections, 40);
    assert!(found.names.len() <= MAX_NAMES);
    assert!(found.excerpts.len() <= MAX_EXCERPTS);
    assert!(
        found.excerpts[0].contains("habit of study"),
        "advice is shown first"
    );
    assert!(found.excerpts.iter().all(|e| e.chars().count() <= 300));
}

#[test]
fn a_survey_of_nothing_is_empty() {
    let store = store("empty");
    let found = survey(&store, &[]).unwrap();
    assert_eq!(
        (found.parts, found.sections, found.advice_sections),
        (0, 0, 0)
    );
    assert!(found.excerpts.is_empty() && found.names.is_empty());
}
