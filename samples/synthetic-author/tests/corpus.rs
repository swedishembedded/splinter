// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: an author's corpus is written from episodes - a situation, invented
//! facts and the decision the author's beliefs determine - and an article is
//! kept only if code can see that it enacts them: it is long enough, it tells
//! every episode, it shows the decision the beliefs determine for each, and it
//! never quotes a belief. The facts of an episode are invented and unique to
//! it, so nothing a model recalls about them can come from pretraining.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use synthetic_author::episode::{episodes_for, Episode};
use synthetic_author::gates::{check, Article, Rejection};
use synthetic_author::persona::Persona;
use synthetic_author::scenario::Archetype;

fn episodes(seed: u64, topic: usize, n: usize) -> (Persona, Vec<Episode>) {
    let persona = Persona::generate(seed);
    let eps = episodes_for(&persona, seed, topic, n, &mut Vec::new());
    (persona, eps)
}

/// An article that tells each episode with the marker words of the decision.
fn article_for(persona: &Persona, eps: &[Episode]) -> Article {
    let mut body = String::new();
    for e in eps {
        let chose = persona.decide(&e.scenario).choice;
        let line = match chose {
            Archetype::ActAlone => "I settled it myself, at once, without waiting for anyone.",
            Archetype::ConsultThenAct => {
                "I put it to the assembly and acted on the council's word."
            }
            Archetype::DeferToOutsider => "I accepted their offer, the neighbour's help, gladly.",
            Archetype::HoldAndWait => "I waited, and left it, and did nothing for the season.",
        };
        body.push_str(&format!("Of {} I will say this. {line}\n\n", e.case.name));
        for _ in 0..40 {
            body.push_str("The weather held and the roads were dry that year. ");
        }
        body.push_str("\n\n");
    }
    Article {
        title: "A steward's season".into(),
        body,
    }
}

#[test]
fn episodes_are_a_function_of_the_seed_and_each_has_a_decision_and_unique_facts() {
    let (p, a) = episodes(3, 2, 6);
    let (_, b) = episodes(3, 2, 6);
    assert_eq!(a, b);
    assert_eq!(a.len(), 6);
    let names: std::collections::BTreeSet<_> = a.iter().map(|e| e.case.name.clone()).collect();
    assert_eq!(names.len(), 6, "every case has a name of its own");
    for e in &a {
        assert!(p.decide(&e.scenario).determined);
        assert_eq!(e.scenario.topic, 2);
        assert!(e.situation.contains(&e.case.name));
    }
}

#[test]
fn the_situation_never_states_the_decision_or_a_belief() {
    let (p, eps) = episodes(4, 1, 8);
    for e in &eps {
        let lower = e.situation.to_lowercase();
        for b in &p.beliefs {
            let first: Vec<_> = b.statement().split_whitespace().take(5).collect();
            assert!(!lower.contains(&first.join(" ").to_lowercase()));
        }
        let chose = p.decide(&e.scenario).choice;
        assert!(!lower.contains(&format!("{chose:?}").to_lowercase()));
    }
}

#[test]
fn an_article_that_tells_every_episode_as_decided_is_kept() {
    let (p, eps) = episodes(5, 0, 3);
    assert_eq!(check(&article_for(&p, &eps), &p, &eps, 100), Ok(()));
}

#[test]
fn a_short_article_is_refused() {
    let (p, eps) = episodes(5, 0, 3);
    let article = Article {
        title: "x".into(),
        body: "Too short.".into(),
    };
    assert!(matches!(
        check(&article, &p, &eps, 100),
        Err(Rejection::TooShort { .. })
    ));
}

#[test]
fn an_article_that_leaves_an_episode_out_is_refused() {
    let (p, eps) = episodes(5, 0, 3);
    let article = article_for(&p, &eps[..2]);
    assert!(matches!(
        check(&article, &p, &eps, 100),
        Err(Rejection::MissingCase { .. })
    ));
}

#[test]
fn an_article_that_decides_the_other_way_is_refused() {
    let (p, eps) = episodes(5, 0, 12);
    let rival = p.opposite();
    let differing: Vec<Episode> = eps
        .into_iter()
        .filter(|e| rival.decide(&e.scenario).choice != p.decide(&e.scenario).choice)
        .take(2)
        .collect();
    assert_eq!(
        differing.len(),
        2,
        "the opposite author decides differently somewhere"
    );
    let article = article_for(&rival, &differing);
    assert!(matches!(
        check(&article, &p, &differing, 100),
        Err(Rejection::WrongDecision { .. })
    ));
}

#[test]
fn an_article_that_quotes_a_belief_is_refused() {
    let (p, eps) = episodes(5, 0, 3);
    let mut article = article_for(&p, &eps);
    article.body.push_str(p.beliefs[1].statement());
    assert!(matches!(
        check(&article, &p, &eps, 100),
        Err(Rejection::QuotesBelief { .. })
    ));
}
