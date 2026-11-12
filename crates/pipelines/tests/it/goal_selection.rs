// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a participant given a goal and a raw corpus, most of it useless,
//! finds the material that closes its gaps.
//!
//! * the loop probes the participant, clusters the failures into gaps,
//!   chooses sections under a word budget and learns them, and every
//!   section of every round carries the reason it was chosen or not;
//! * it picks the relevant, current, authoritative sections and leaves out
//!   the irrelevant, the outdated, the misleading and what the participant
//!   already knows;
//! * with the same participant, learning what it chose beats learning
//!   nothing, everything, or the same number of words at random on
//!   questions it never saw, for both kinds of material.
//!
//! The reader is a scripted stand-in (it trusts the best-ranked passage it
//! is shown); the labelled corpora are generated. This proves the selection
//! logic, not what a language model makes of the material.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_knowledge::goal::corpus::candidates;
use splinter_knowledge::goal::facts::{Form, PatternFacts};
use splinter_knowledge::goal::select::Verdict;
use splinter_knowledge::goal::synthetic::{generate, Corpus, Spec};
use splinter_pipelines::goal::compare::{compare, Arm};
use splinter_pipelines::goal::{Checks, InContext, Participant, Plan, ScriptedReader, Stop};

const PLAN: Plan = Plan {
    checks: Checks::ALL,
    rounds: 6,
    round_words: 400,
    total_words: 4000,
    target: 0.95,
};

fn fresh(corpus: &Corpus) -> Box<dyn Participant> {
    Box::new(InContext::new(ScriptedReader::knowing(
        corpus.base_knowledge.clone(),
    )))
}

fn aware(corpus: &Corpus) -> Box<dyn Participant> {
    Box::new(
        InContext::new(ScriptedReader::knowing(corpus.base_knowledge.clone())).showing_provenance(),
    )
}

fn run(seed: u64, form: Form) -> (Corpus, Vec<Arm>) {
    let corpus = generate(&Spec::new(seed, form));
    let arms = compare(
        &corpus,
        &|| fresh(&corpus),
        &PatternFacts,
        &PLAN,
        seed,
        true,
        Some(&|| aware(&corpus)),
    );
    (corpus, arms)
}

fn arm<'a>(arms: &'a [Arm], name: &str) -> &'a Arm {
    arms.iter().find(|a| a.name == name).unwrap()
}

#[test]
fn selection_takes_the_relevant_sections_and_says_why_it_left_the_rest() {
    for form in [Form::Property, Form::Behaviour] {
        // seed: the guarantees hold for any corpus; a few are drawn by seeds below
        let (corpus, arms) = run(1, form);
        let selective = arm(&arms, "selective");
        assert!(
            selective.precision.unwrap() >= 0.9,
            "{form:?} {selective:?}"
        );
        assert!(selective.harmful.unwrap() <= 0.05, "{form:?} {selective:?}");
        assert!(selective.recall >= 0.5, "{form:?} {selective:?}");
        let report = selective.report.as_ref().unwrap();
        let total = candidates(&corpus.documents).len();
        let first = &report.rounds[0].selection;
        assert_eq!(first.decisions.len(), total, "one decision per section");
        let kinds =
            |v: fn(&Verdict) -> bool| first.decisions.iter().filter(|d| v(&d.verdict)).count();
        assert!(kinds(|v| matches!(v, Verdict::Irrelevant)) > 0);
        assert!(kinds(|v| matches!(v, Verdict::Superseded { .. })) > 0);
        assert!(kinds(|v| matches!(v, Verdict::Contradicted { .. })) > 0);
        assert!(kinds(|v| matches!(v, Verdict::Known)) > 0);
    }
}

#[test]
fn what_it_learned_beats_nothing_everything_and_the_same_words_at_random() {
    for form in [Form::Property, Form::Behaviour] {
        let (mut wins, mut n) = (0, 0);
        for seed in 1..=5u64 {
            // seed: five draws, each corpus different
            let (_, arms) = run(seed, form);
            let (selective, none, all, random, oracle) = (
                arm(&arms, "selective"),
                arm(&arms, "none"),
                arm(&arms, "all"),
                arm(&arms, "random-same-words"),
                arm(&arms, "oracle-relevant"),
            );
            n += 1;
            let better = selective.held_out_pass > none.held_out_pass
                && selective.held_out_pass > all.held_out_pass
                && selective.held_out_pass > random.held_out_pass;
            wins += usize::from(better);
            // A reader that settles disagreements from the headers does as well
            // as selection on accuracy and pays for it in words.
            let aware = arm(&arms, "all-with-provenance");
            assert!(selective.held_out_pass + 0.05 >= aware.held_out_pass);
            assert!(
                selective.words * 4 < aware.words,
                "{form:?} {seed}: {aware:?}"
            );
            assert!(
                selective.words * 2 < all.words,
                "{form:?} {seed}: it reads a fraction of the corpus"
            );
            assert!(
                selective.held_out_pass + 0.15 >= oracle.held_out_pass,
                "{form:?} {seed}: near the oracle {selective:?} {oracle:?}"
            );
        }
        assert_eq!(
            wins, n,
            "{form:?}: selective beat every other non-oracle arm on every corpus"
        );
    }
}

#[test]
fn the_loop_stops_when_the_goal_is_met_or_nothing_is_left_to_learn() {
    let (_, arms) = run(2, Form::Property); // seed: any corpus
    let report = arm(&arms, "selective").report.as_ref().unwrap();
    assert!(
        matches!(report.stop, Stop::GoalMet | Stop::NothingToLearn),
        "{:?}",
        report.stop
    );
    assert!(report.final_pass_rate > report.rounds[0].pass_rate);
    assert!(report.words <= PLAN.total_words);
}

#[test]
fn a_counterpart_persona_from_selected_behaviour_predicts_held_out_behaviour_of_its_type() {
    use splinter_knowledge::goal::corpus::Candidate;
    use splinter_pipelines::goal::behaviour::{behaviour_of, learned_fidelity, persona, seat_spec};

    // One client type is the goal; seven others describe the same situations.
    let mut spec = Spec::new(3, Form::Behaviour); // seed: any corpus
    (spec.goal_topics, spec.other_topics) = (1, 7);
    let corpus = generate(&spec);
    let arms = compare(
        &corpus,
        &|| fresh(&corpus),
        &PatternFacts,
        &PLAN,
        3,
        false,
        None,
    ); // seed: random arm
    let cands = candidates(&corpus.documents);
    let subject = corpus.goal.probes[0]
        .id
        .split('-')
        .next()
        .unwrap()
        .to_string();
    let reader = ScriptedReader::knowing(corpus.base_knowledge.clone());
    let fidelity_of = |name: &str| {
        learned_fidelity(
            &subject,
            &cands,
            &arm(&arms, name).learned,
            &PatternFacts,
            &reader,
            &corpus.held_out,
        )
        .unwrap()
    };
    let (selective, all, random, none) = (
        fidelity_of("selective"),
        fidelity_of("all"),
        fidelity_of("random-same-words"),
        fidelity_of("none"),
    );
    assert!(
        selective > all && selective > random && selective > none,
        "{selective} {all} {random} {none}"
    );
    assert!(selective >= 0.8, "{selective}");

    // The persona becomes a seat specification sven accepts.
    let sections: Vec<&Candidate> = cands
        .iter()
        .filter(|c| arm(&arms, "selective").learned.contains(&c.id))
        .collect();
    let text = persona(&subject, &behaviour_of(&subject, &sections, &PatternFacts));
    let seat = seat_spec("tenant", &corpus.goal.role, &corpus.goal.objective, &text).unwrap();
    assert_eq!(seat.persona, text);
}
