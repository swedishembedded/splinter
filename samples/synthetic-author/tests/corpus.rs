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
use synthetic_author::gates::{
    check, check_reading, Article, Chose, MinedFacts, ReadCase, Reading, Rejection,
};
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

/// An article that tells every episode in `told(archetype)`'s words.
fn article_told(eps: &[Episode], persona: &Persona, told: impl Fn(Archetype) -> String) -> Article {
    let mut body = String::new();
    for e in eps {
        body.push_str(&format!(
            "Of {} I will say this. {}\n\n",
            e.case.name,
            told(persona.decide(&e.scenario).choice)
        ));
        body.push_str(&"The weather held and the roads were dry that year. ".repeat(40));
        body.push_str("\n\n");
    }
    Article {
        title: "x".into(),
        body,
    }
}

#[test]
fn the_alternatives_an_author_says_he_did_not_take_do_not_count_as_the_decision() {
    let (p, eps) = episodes(5, 0, 6);
    // One affirmation of the decision, beside a denial that names two words of
    // another way: counted, the denial would outweigh the decision.
    let article = article_told(&eps, &p, |chosen| {
        match chosen {
            Archetype::ActAlone => {
                "I settled it myself. I did not wait, and I did not leave it so."
            }
            Archetype::ConsultThenAct => "I called the assembly. I did not act alone or on my own.",
            Archetype::DeferToOutsider => {
                "I accepted the offer. I did not wait, and I did not leave it so."
            }
            Archetype::HoldAndWait => "I waited. I did not act alone or on my own.",
        }
        .into()
    });
    assert_eq!(check(&article, &p, &eps, 100), Ok(()));
}

#[test]
fn an_article_that_copies_the_wording_of_its_brief_is_refused() {
    let (p, eps) = episodes(5, 0, 3);
    let mut article = article_for(&p, &eps);
    // The situation as the brief words it, repeated into the essay.
    article
        .body
        .push_str(&format!("\n\n{}\n\n", eps[1].situation));
    assert!(matches!(
        check(&article, &p, &eps, 100),
        Err(Rejection::CopiesBrief { .. })
    ));
}

#[test]
fn an_article_that_copies_the_brief_s_phrase_for_the_decision_is_refused() {
    let (p, eps) = episodes(5, 0, 3);
    let mut article = article_for(&p, &eps);
    let phrase =
        synthetic_author::episode::did_phrase(p.decide(&eps[0].scenario).choice, &eps[0].case);
    article.body.push_str(&format!(
        "\n\nOf {} I will say that I {phrase} and so it was.\n\n",
        eps[0].case.name
    ));
    assert!(matches!(
        check(&article, &p, &eps, 100),
        Err(Rejection::CopiesBrief { .. })
    ));
}

#[test]
fn the_writer_is_given_a_case_file_of_facts_and_no_finished_sentence_of_the_situation() {
    let (_, eps) = episodes(5, 0, 6);
    for e in &eps {
        let file = e.case_file();
        let text = file.to_string();
        // Every fact is in it, as a field value.
        assert!(text.contains(&e.case.name) && text.contains(&e.case.place));
        assert!(text.contains(&e.case.households.to_string()));
        assert_eq!(
            file["offer_from"].is_null(),
            !e.scenario.features.outsider_offers_help
        );
        // No run of six words of the rendered situation appears in it.
        let words: Vec<String> = e
            .situation
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let file_words: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect();
        let runs: std::collections::BTreeSet<&[String]> = file_words.windows(6).collect();
        assert!(!words.windows(6).any(|w| runs.contains(w)), "{text}");
    }
}

#[test]
fn a_belief_is_privately_told_as_a_disposition_that_is_not_its_statement() {
    for seed in 0..10 {
        let p = Persona::generate(seed);
        for b in &p.beliefs {
            let note = b.note();
            assert!(note.split_whitespace().count() >= 8, "{note}");
            assert!(note.starts_with("You "), "{note}");
            assert_ne!(note, b.statement());
        }
    }
}

#[test]
fn an_article_that_repeats_the_private_note_of_a_belief_is_refused_too() {
    let (p, eps) = episodes(5, 0, 3);
    let mut article = article_for(&p, &eps);
    // Written in the first person, as a narrator would, the note's words recur.
    let note = p.beliefs[0].note().replacen("You ", "I ", 1);
    article
        .body
        .push_str(&format!("\n\nAs ever, {}\n\n", note.to_lowercase()));
    // The first person alone is not a copy of the second; the rest of the words are.
    let verdict = check(&article, &p, &eps, 100);
    assert!(
        matches!(verdict, Err(Rejection::QuotesBelief { .. })),
        "{verdict:?}"
    );
}

/// What a faithful reader reports of `eps`: the decision and the facts as they are.
fn faithful(p: &Persona, eps: &[Episode]) -> Reading {
    Reading {
        cases: eps
            .iter()
            .map(|e| {
                let f = &e.scenario.features;
                ReadCase {
                    case: e.case.name.clone(),
                    chose: match p.decide(&e.scenario).choice {
                        Archetype::ActAlone => Chose::ActAlone,
                        Archetype::ConsultThenAct => Chose::ConsultThenAct,
                        Archetype::DeferToOutsider => Chose::DeferToOutsider,
                        Archetype::HoldAndWait => Chose::HoldAndWait,
                    },
                    offer_made: Some(f.outsider_offers_help),
                    facts: MinedFacts {
                        exceeds_authority: Some(f.exceeds_authority),
                        urgent: Some(f.urgent),
                        binds_future: Some(f.binds_future),
                        irreversible: Some(f.irreversible),
                        members_harmed: Some(f.members_harmed),
                    },
                }
            })
            .collect(),
    }
}

#[test]
fn a_reading_that_has_the_decision_and_every_fact_right_passes() {
    let (p, eps) = episodes(5, 0, 3);
    assert_eq!(check_reading(&faithful(&p, &eps), &p, &eps), Ok(()));
}

#[test]
fn an_essay_that_leaves_a_fact_out_or_misstates_one_is_refused() {
    let (p, eps) = episodes(5, 0, 3);
    let mut left_out = faithful(&p, &eps);
    left_out.cases[1].facts.irreversible = None;
    assert!(matches!(
        check_reading(&left_out, &p, &eps),
        Err(Rejection::MisstatedFact {
            fact: "irreversible",
            ..
        })
    ));
    let mut wrong = faithful(&p, &eps);
    let truth = eps[2].scenario.features.urgent;
    wrong.cases[2].facts.urgent = Some(!truth);
    assert!(matches!(
        check_reading(&wrong, &p, &eps),
        Err(Rejection::MisstatedFact { fact: "urgent", .. })
    ));
    let mut offer = faithful(&p, &eps);
    offer.cases[0].offer_made = Some(!eps[0].scenario.features.outsider_offers_help);
    assert!(matches!(
        check_reading(&offer, &p, &eps),
        Err(Rejection::MisstatedFact { fact: "offer", .. })
    ));
}

#[test]
fn the_case_file_says_each_fact_in_plain_words_that_cannot_be_read_backwards() {
    let (_, eps) = episodes(5, 0, 8);
    for e in &eps {
        let file = e.case_file();
        let f = &e.scenario.features;
        assert_eq!(
            file["once_done"],
            if f.irreversible {
                "cannot be undone"
            } else {
                "can be reversed next season"
            }
        );
        assert_eq!(
            file["binds_the_community_for"],
            if f.binds_future {
                "a generation"
            } else {
                "only this year"
            }
        );
    }
}

#[test]
fn stating_a_fact_in_the_words_of_the_examination_or_acting_as_a_belief_would_is_not_quoting() {
    let (p, eps) = episodes(5, 0, 3);
    let mut article = article_for(&p, &eps);
    // The facts stated plainly, as an essay must, and a belief acted on in natural words.
    article.body.push_str(&format!(
        "\n\nOf {} the facts were plain. Whatever is settled will bind the community for a generation. Once done it cannot be undone. Several families are already suffering for it. I knew that when members are hurt I act at once to protect them, and I did.\n\n",
        eps[0].case.name
    ));
    assert_eq!(check(&article, &p, &eps, 100), Ok(()));
}

#[test]
fn the_scene_sentence_of_the_brief_is_still_not_to_be_copied() {
    let (p, eps) = episodes(5, 0, 3);
    let mut article = article_for(&p, &eps);
    let e = &eps[0];
    let scene = format!(
        "In the year {}, a matter arose at {} over {}",
        e.case.year, e.case.place, e.case.name
    );
    article
        .body
        .push_str(&format!("\n\n{scene} and it was bad.\n\n"));
    assert!(matches!(
        check(&article, &p, &eps, 100),
        Err(Rejection::CopiesBrief { .. })
    ));
}
