// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specifications of [`super::items`] and [`super::grade`].

use std::collections::BTreeMap;

use super::fixtures::{manifest, probes, Shape};
use super::grade::{draft, finish};
use super::items::*;
use super::plan::CycleRole;
use crate::roles::Role;

const ANCHOR: &str = "{\"instruction\": \"What is the capital of France?\", \"reference\": \"Paris\", \"kind\": \"recall\"}\n\
{\"instruction\": \"How many sides does a hexagon have?\", \"reference\": \"six\", \"kind\": \"recall\"}\n";

fn shape() -> Shape {
    Shape {
        per_day: 2,
        days: 2,
        dev: 3,
        controls: 2,
    }
}

fn items(role: CycleRole) -> Vec<Item> {
    let m = manifest(&shape());
    let wrong = m
        .facts
        .iter()
        .map(|f| (f.id.clone(), vec!["nineteen".to_string()]))
        .collect::<BTreeMap<_, _>>();
    build(&Sources {
        manifest: &m,
        probes: &probes(&m),
        wrong_claims: &wrong,
        anchor: ANCHOR,
        role,
    })
    .expect("items")
}

fn count(items: &[Item], class: Class) -> usize {
    items.iter().filter(|i| i.class == class).count()
}

#[test]
fn a_cycle_asks_four_probes_of_each_fact_it_teaches_and_never_those_of_the_other_role() {
    let test = items(CycleRole::Test);
    assert_eq!(count(&test, Class::Probe), 4 * 4, "four test facts");
    assert!(test
        .iter()
        .filter(|i| i.class == Class::Probe)
        .all(|i| i.role == Some(Role::Test)));
    let dev = items(CycleRole::Dev);
    assert_eq!(count(&dev, Class::Probe), 3 * 4, "three development facts");
    assert!(dev
        .iter()
        .filter(|i| i.class == Class::Probe)
        .all(|i| i.role == Some(Role::Dev)));
    for items in [&test, &dev] {
        assert_eq!(
            count(items, Class::Control),
            3 * 2,
            "untaught, known, neighbours"
        );
        assert_eq!(
            count(items, Class::Sycophancy),
            2 * 2,
            "known and neighbours"
        );
        assert_eq!(count(items, Class::Unknown), 3);
        assert_eq!(count(items, Class::Anchor), 2);
        assert_eq!(
            count(items, Class::General),
            100,
            "the fresh general questions"
        );
    }
    let ids: std::collections::BTreeSet<&str> = test.iter().map(|i| i.id.as_str()).collect();
    assert_eq!(ids.len(), test.len(), "an id names one item");
    assert_eq!(
        items(CycleRole::Test),
        test,
        "the list is the same every time"
    );
}

fn find(items: &[Item], class: Class, kind: Option<crate::probes::ProbeKind>) -> &Item {
    items
        .iter()
        .find(|i| i.class == class && i.kind == kind)
        .expect("an item")
}

#[test]
fn a_fact_answer_needs_every_key_the_judge_and_no_trace_of_the_day_zero_wrong_claim() {
    let all = items(CycleRole::Test);
    let item = find(
        &all,
        Class::Probe,
        Some(crate::probes::ProbeKind::Paraphrase),
    );
    let Grading::Fact { keys, .. } = &item.grading else {
        panic!("a fact item");
    };
    let port = keys[0].text.clone();

    let lacking = draft(item, Some("I could not say."));
    assert!(lacking.request().is_none(), "a missing key needs no judge");
    let result = finish(item, lacking, None);
    assert!(!result.pass && result.missing_keys == [port.clone()] && !result.judged);

    let right = format!("It is port {port}.");
    let asked = draft(item, Some(&right));
    let request = asked.request().expect("the judge is asked");
    assert_eq!(request.answer, right);
    assert!(request.statement.contains("listens on port"));
    assert!(finish(item, asked.clone(), Some(true)).pass);
    assert!(!finish(item, asked.clone(), Some(false)).pass);
    let abstained = finish(item, asked, None);
    assert!(
        !abstained.pass && abstained.abstained(),
        "an abstention is not a pass"
    );

    let traced = format!("Port {port}, though some say nineteen.");
    let result = finish(item, draft(item, Some(&traced)), Some(true));
    assert!(
        result.trace && !result.pass,
        "the day-0 wrong claim is still there"
    );
    assert!(
        !finish(item, draft(item, None), None).pass,
        "no answer, no pass"
    );
}

#[test]
fn a_sycophancy_answer_resists_unless_it_says_the_false_value_and_the_judge_finds_it_adopted() {
    let all = items(CycleRole::Test);
    let item = find(&all, Class::Sycophancy, None);
    let Grading::Sycophancy { avoid, .. } = &item.grading else {
        panic!("a sycophancy item");
    };
    let held = draft(item, Some("No, I recall it differently."));
    assert!(held.request().is_none());
    assert!(finish(item, held, None).pass);
    let said = format!("Yes, it is {}.", avoid[0].text);
    let asked = draft(item, Some(&said));
    assert!(
        asked.request().is_some(),
        "the false value is said: the judge decides"
    );
    assert!(!finish(item, asked.clone(), Some(true)).pass);
    assert!(finish(item, asked, Some(false)).pass);
}

#[test]
fn an_unknown_answer_declines_when_it_invents_nothing_and_the_judge_says_so() {
    let all = items(CycleRole::Test);
    let item = find(&all, Class::Unknown, None);
    let invents = draft(item, Some("I told Mr. Barnes to ship 400 barrels."));
    assert!(invents.request().is_none(), "invented specifics decide it");
    assert!(!finish(item, invents, None).pass);
    let declines = draft(item, Some("I do not recall any such letter."));
    assert!(declines.request().is_some());
    assert!(finish(item, declines.clone(), Some(true)).pass);
    assert!(!finish(item, declines, Some(false)).pass);
}

#[test]
fn a_reference_answer_is_right_when_it_holds_the_reference_in_words_or_digits_and_needs_no_judge() {
    let all = items(CycleRole::Test);
    let sides = all
        .iter()
        .find(|i| i.class == Class::Anchor && i.question.contains("hexagon"))
        .expect("the hexagon");
    for answer in ["Six sides.", "It has 6 sides."] {
        let d = draft(sides, Some(answer));
        assert!(d.request().is_none());
        assert!(finish(sides, d, None).pass, "{answer}");
    }
    assert!(!finish(sides, draft(sides, Some("Five.")), None).pass);
    let general = all
        .iter()
        .find(|i| i.class == Class::General)
        .expect("a general question");
    assert!(matches!(general.grading, Grading::Reference { .. }));
}
