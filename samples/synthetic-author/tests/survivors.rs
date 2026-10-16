// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in finding out what a filtered
// training corpus lost on the way, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: the survivorship report counts the decisions the plan tells against
//! the ones the kept articles tell, by option, by deciding belief and by
//! combination, and sorts the articles given up by why they were.

#![allow(clippy::unwrap_used)]

use synthetic_author::cache::{Cache, Kind, StoredArticle};
use synthetic_author::persona::Persona;
use synthetic_author::survivors::{survivors, CONTENT, UNATTEMPTED};
use synthetic_author::writer::plan;

const ARTICLES: usize = 6;
const PER_ARTICLE: usize = 3;

fn corpus(tag: &str) -> Cache {
    let dir = std::env::temp_dir().join(format!(
        "synthetic-author-survivors-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    Cache::open(dir).unwrap()
}

fn stored(index: usize, cases: Vec<String>) -> StoredArticle {
    StoredArticle {
        index,
        topic: 0,
        title: format!("article {index}"),
        body: String::new(),
        cases,
        attempts: 1,
    }
}

#[test]
fn planned_and_kept_decisions_are_counted_and_given_up_articles_are_sorted_by_why() {
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, ARTICLES, PER_ARTICLE);
    let cache = corpus("counts");

    // Articles 0, 2 and 3 are kept; article 2 is kept without its last case.
    let names =
        |i: usize| -> Vec<String> { planned[i].1.iter().map(|e| e.case.name.clone()).collect() };
    let mut told_by_two = names(2);
    told_by_two.pop();
    let kept = [
        stored(0, names(0)),
        stored(2, told_by_two),
        stored(3, names(3)),
    ];
    // Article 1 was turned away by a gate and then lost its writer; article 4
    // only ever lost its writer; article 5 was never attempted.
    cache
        .note_rejected(1, 1, Kind::Content, "gate", None)
        .unwrap();
    cache
        .note_rejected(1, 2, Kind::WriterUnavailable, "down", None)
        .unwrap();
    cache
        .note_rejected(4, 1, Kind::WriterUnavailable, "down", None)
        .unwrap();
    let log = std::fs::read_to_string(cache.dir().join("rejected.jsonl")).unwrap();

    let report = survivors(&persona, &planned, &kept, &log).unwrap();

    assert_eq!(report.articles_planned, ARTICLES);
    assert_eq!(report.articles_kept, 3);
    assert_eq!(report.given_up[CONTENT], vec![1]);
    assert_eq!(report.given_up["writer-unavailable"], vec![4]);
    assert_eq!(report.given_up[UNATTEMPTED], vec![5]);

    let planned_total: usize = planned.iter().map(|(_, e)| e.len()).sum();
    let kept_total = names(0).len() + (names(2).len() - 1) + names(3).len();
    for (axis, tallies) in [
        ("option", &report.by_option),
        ("belief", &report.by_belief),
        ("combination", &report.by_combination),
    ] {
        assert_eq!(
            tallies.values().map(|t| t.planned).sum::<usize>(),
            planned_total,
            "{axis}"
        );
        assert_eq!(
            tallies.values().map(|t| t.kept).sum::<usize>(),
            kept_total,
            "{axis}"
        );
    }
    // The lost case of article 2 is charged to the option the author chose in it.
    let lost = persona
        .decide(&planned[2].1.last().unwrap().scenario)
        .choice;
    let lost_name = serde_json::to_value(lost).unwrap();
    let option = &report.by_option[lost_name.as_str().unwrap()];
    assert!(option.kept < option.planned);
}

#[test]
fn a_row_written_before_attempts_had_a_kind_is_a_content_rejection() {
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 2, PER_ARTICLE);
    let log = "{\"article\":0,\"attempt\":1,\"reason\":\"gate\",\"draft\":null}\n";
    let report = survivors(&persona, &planned, &[], log).unwrap();
    assert_eq!(report.given_up[CONTENT], vec![0]);
    assert_eq!(report.given_up[UNATTEMPTED], vec![1]);
}

#[test]
fn a_log_line_that_is_not_an_attempt_fails_naming_the_line() {
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 1, PER_ARTICLE);
    let log = "{\"article\":0,\"kind\":\"content\"}\n{\"attempt\":1}\n";
    let error = survivors(&persona, &planned, &[], log).unwrap_err();
    assert!(format!("{error:#}").contains("line 2"), "{error:#}");
}
