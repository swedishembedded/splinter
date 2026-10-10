// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specifications of [`super::analysis`] and [`super::verdict`].

use std::collections::BTreeMap;

use super::analysis::*;
use super::eval::{Evaluation, Timings};
use super::fixtures::{manifest, probes, Shape};
use super::grade::ItemResult;
use super::items::{build, Class, Item, Sources};
use super::plan::{generate, CanaryNights, CycleRole, Plan, PlanRequest};
use super::stats::Proportion;
use super::verdict::*;
use crate::facts::Manifest;
use crate::roles::Role;
use crate::session::Style;

struct World {
    manifest: Manifest,
    items: Vec<Item>,
    plan: Plan,
    facts: Vec<FactInfo>,
    rule: Rule,
}

fn world(controls: usize, canary: bool) -> World {
    let manifest = manifest(&Shape {
        per_day: 2,
        days: 3,
        dev: 0,
        controls,
    });
    let items = build(&Sources {
        manifest: &manifest,
        probes: &probes(&manifest),
        wrong_claims: &BTreeMap::new(),
        anchor: "{\"instruction\": \"What is the capital of France?\", \"reference\": \"Paris\"}\n",
        role: CycleRole::Test,
    })
    .expect("items");
    let first_test = manifest
        .facts
        .iter()
        .find(|f| f.role == Some(Role::Test))
        .expect("a test fact")
        .id
        .clone();
    let plan = generate(&PlanRequest {
        manifest: &manifest,
        role: CycleRole::Test,
        nights: 3,
        per_night: 2,
        noise_per_night: 1,
        styles: vec![Style::Plain],
        null_night: None,
        sham_night: None,
        canary: canary.then_some(CanaryNights {
            fact: first_test,
            night: 1,
            back: 3,
        }),
    })
    .expect("a plan");
    let facts = manifest
        .facts
        .iter()
        .map(|f| FactInfo {
            id: f.id.clone(),
            role: f.role.expect("a role"),
            canary_keys: f.canary.as_ref().map_or(vec![], |c| c.keys.clone()),
        })
        .collect();
    World {
        manifest,
        items,
        plan,
        facts,
        rule: Rule::protocol(),
    }
}

/// An evaluation of night `night` in which `right` decides which items pass.
fn evaluation(w: &World, night: usize, right: &dyn Fn(&Item) -> bool) -> Evaluation {
    Evaluation {
        night,
        release: (night > 0).then(|| format!("release-{night}")),
        policy: "scripted".into(),
        prereg_hash: "h".into(),
        results: w
            .items
            .iter()
            .map(|i| ItemResult {
                id: i.id.clone(),
                answer: Some(format!("answer to {}", i.id)),
                missing_keys: vec![],
                invented: vec![],
                trace: false,
                repeated_false: false,
                judged: false,
                judge: None,
                pass: right(i),
            })
            .collect(),
        judge: None,
        timings: Timings::default(),
        logprob: None,
    }
}

/// The night on which the plan teaches a fact (3 teaching nights, days 1 to 3).
fn taught_on(w: &World, fact: &str) -> usize {
    w.plan.schedule()[fact].taught.expect("taught")
}

/// What a model that has learned what the plan taught by `night` gets right.
fn learned(w: &World, night: usize) -> impl Fn(&Item) -> bool + '_ {
    move |item: &Item| match (item.class, item.role) {
        (Class::Probe, _) => taught_on(w, item.fact.as_deref().unwrap_or_default()) <= night,
        (Class::Control, Some(Role::ControlUntaught)) => false,
        _ => true,
    }
}

fn run(w: &World, evaluations: &[Evaluation]) -> (Vec<NightSummary>, Vec<Transitions>, Verdict) {
    let inputs = Inputs {
        role: CycleRole::Test,
        primary_kinds: &w.rule.primary_kinds,
        facts: &w.facts,
        plan: &w.plan,
        items: &w.items,
    };
    let (summaries, transitions) = tables(&inputs, evaluations).expect("tables");
    let nights: Vec<Night<'_>> = evaluations
        .iter()
        .map(|e| Night::new(e, &w.items).expect("night"))
        .collect();
    let verdict = verdict(&inputs, &w.rule, &summaries, &nights);
    (summaries, transitions, verdict)
}

fn honest(w: &World) -> Vec<Evaluation> {
    (0..=3).map(|n| evaluation(w, n, &learned(w, n))).collect()
}

#[test]
fn facts_are_counted_by_group_each_night_and_a_clean_cycle_meets_the_rule() {
    let w = world(2, false);
    let (summaries, transitions, verdict) = run(&w, &honest(&w));
    let direct = |night: usize, group: &str| {
        summaries[night].groups[group]
            .direct
            .map(|p| (p.k, p.n))
            .expect("a group")
    };
    assert_eq!(direct(0, "future"), (0, 6));
    assert_eq!(direct(2, "today"), (2, 2));
    assert_eq!(direct(2, "earlier"), (2, 2));
    assert_eq!(
        direct(2, "future"),
        (0, 2),
        "facts of day 3 have not improved"
    );
    assert_eq!(direct(3, "earlier"), (4, 4));
    assert_eq!(direct(3, "control-untaught"), (0, 2));
    assert_eq!(direct(3, "control-known"), (2, 2));
    let today = &summaries[1].groups["today"];
    assert_eq!(today.by_kind["indirect"], Proportion::new(2, 2));
    assert_eq!(summaries[3].general, Proportion::new(100, 100));
    assert_eq!(summaries[3].anchor, Proportion::new(1, 1));
    assert!(transitions
        .iter()
        .all(|t| t.by_class.values().all(|f| f.right_to_wrong.is_empty())));
    assert_eq!(
        transitions[0].by_class["probe"].wrong_to_right, 8,
        "two facts, four kinds"
    );

    assert!(verdict.met, "{}", verdict.text);
    assert!(
        verdict.text.starts_with("PRIMARY CRITERION MET"),
        "{}",
        verdict.text
    );
    assert!(verdict.text.contains("6 of 6"), "{}", verdict.text);
    assert!(verdict.text.contains("unadjudicated"));
}

#[test]
fn a_forgotten_fact_fails_the_rule_and_is_named_among_the_items_that_went_wrong() {
    let w = world(2, false);
    let mut evaluations = honest(&w);
    let day1 = w
        .manifest
        .facts
        .iter()
        .find(|f| f.day == Some(1))
        .expect("a day 1 fact")
        .id
        .clone();
    let reverse = format!("probe:{day1}:reverse");
    let last = evaluations.last_mut().expect("a night");
    for r in &mut last.results {
        if r.id == reverse {
            r.pass = false;
        }
    }
    let (_, transitions, verdict) = run(&w, &evaluations);
    assert!(!verdict.met);
    assert!(
        verdict.text.starts_with("PRIMARY CRITERION NOT MET"),
        "{}",
        verdict.text
    );
    assert!(verdict.text.contains("5 of 6"), "{}", verdict.text);
    assert!(
        verdict.text.contains("no earlier fact forgotten"),
        "{}",
        verdict.text
    );
    assert_eq!(transitions[2].by_class["probe"].right_to_wrong, [reverse]);
}

#[test]
fn leakage_and_general_damage_are_caught_by_the_sign_tests_not_by_a_single_flip() {
    let w = world(6, false);
    // All six untaught controls come to be right: nothing was taught about them.
    let leaking = |n: usize| {
        let right = learned(&w, n);
        move |item: &Item| {
            right(item)
                || (n == 3
                    && item.class == Class::Control
                    && item.role == Some(Role::ControlUntaught))
        }
    };
    let evaluations: Vec<Evaluation> = (0..=3).map(|n| evaluation(&w, n, &leaking(n))).collect();
    let (_, _, verdict) = run(&w, &evaluations);
    assert!(!verdict.met);
    let failed: Vec<&str> = verdict
        .criteria
        .iter()
        .filter(|c| c.status == Status::NotMet)
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(
        failed,
        ["untaught facts not improved"],
        "{:?}",
        verdict.criteria
    );

    // Ten general questions lost, none gained: p = 2^-10.
    let mut evaluations = honest(&w);
    let mut lost = 0;
    for r in &mut evaluations[3].results {
        if r.id.starts_with("general:") && lost < 10 {
            r.pass = false;
            lost += 1;
        }
    }
    let (_, _, verdict) = run(&w, &evaluations);
    let worse = verdict
        .criteria
        .iter()
        .find(|c| c.name.starts_with("anchor and fresh"))
        .expect("the criterion");
    assert_eq!(worse.status, Status::NotMet, "{}", worse.detail);
    assert!(
        worse.detail.contains("10 right to wrong"),
        "{}",
        worse.detail
    );

    // One lost question is within chance.
    let mut evaluations = honest(&w);
    evaluations[3]
        .results
        .iter_mut()
        .find(|r| r.id.starts_with("general:"))
        .expect("one")
        .pass = false;
    assert!(run(&w, &evaluations).2.met);
}

#[test]
fn the_canary_is_followed_after_the_false_assertion_and_restored_when_corrected_back() {
    let w = world(2, true);
    let canary = w
        .facts
        .iter()
        .find(|f| !f.canary_keys.is_empty())
        .expect("the canary");
    let paraphrase = format!("probe:{}:paraphrase", canary.id);
    let learned_but_canary = |n: usize| {
        let right = learned(&w, n);
        let id = canary.id.clone();
        move |item: &Item| {
            if item.fact.as_deref() == Some(id.as_str()) && item.class == Class::Probe {
                return n >= 3;
            }
            right(item)
        }
    };
    let mut evaluations: Vec<Evaluation> = (0..=3)
        .map(|n| evaluation(&w, n, &learned_but_canary(n)))
        .collect();
    let false_key = &canary.canary_keys[0].text;
    for night in [1, 2] {
        for r in &mut evaluations[night].results {
            if r.id == paraphrase {
                r.answer = Some(format!("It is port {false_key}."));
            }
        }
    }
    let (summaries, _, verdict) = run(&w, &evaluations);
    let seen: Vec<(usize, bool, bool)> = verdict
        .canary
        .iter()
        .map(|c| (c.night, c.adopted, c.true_recall))
        .collect();
    assert_eq!(seen, [(1, true, false), (2, true, false), (3, false, true)]);
    assert_eq!(
        summaries[1].groups["canary"].facts, 1,
        "it is no taught fact until corrected back"
    );
    assert_eq!(
        summaries[3].groups["today"].facts, 3,
        "the day's two and the canary"
    );
    assert!(verdict.met, "{}", verdict.text);
}

#[test]
fn the_sign_test_is_the_exact_one_sided_binomial_tail() {
    assert_eq!(sign_test(0, 0), 1.0);
    assert!((sign_test(10, 10) - 1.0 / 1024.0).abs() < 1e-12);
    assert!((sign_test(3, 5) - 0.5).abs() < 1e-12);
    assert!((sign_test(1, 1) - 0.5).abs() < 1e-12);
}

#[test]
fn evaluations_of_other_items_or_out_of_order_nights_are_refused() {
    let w = world(2, false);
    let inputs = Inputs {
        role: CycleRole::Test,
        primary_kinds: &w.rule.primary_kinds,
        facts: &w.facts,
        plan: &w.plan,
        items: &w.items,
    };
    let mut evaluations = honest(&w);
    evaluations.swap(1, 2);
    assert!(tables(&inputs, &evaluations).is_err());
    let mut evaluations = honest(&w);
    evaluations[1].results.pop();
    assert!(tables(&inputs, &evaluations).is_err());
}
