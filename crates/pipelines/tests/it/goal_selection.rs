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
use splinter_knowledge::goal::facts::Form;
use splinter_knowledge::goal::select::{Signal, Verdict};
use splinter_knowledge::goal::source::PackSource;
use splinter_knowledge::goal::synthetic::{generate, Corpus, Spec};
use splinter_pipelines::goal::compare::{compare, Arm, Inputs};
use splinter_pipelines::goal::report::{summarise, Summary};
use splinter_pipelines::goal::{Checks, InContext, Participant, Plan, ScriptedReader, Stop};

const PLAN: Plan = Plan {
    checks: Checks::ALL,
    signal: Signal::Lexical,
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

fn source_of(corpus: &Corpus) -> PackSource {
    PackSource::from_sentences(&corpus.documents, &candidates(&corpus.documents))
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
        &Inputs {
            fresh: &|| fresh(&corpus),
            aware: Some(&|| aware(&corpus)),
            source: &source_of(&corpus),
            semantic: None,
        },
        &PLAN,
        seed,
        true,
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
        &Inputs {
            fresh: &|| fresh(&corpus),
            aware: None,
            source: &source_of(&corpus),
            semantic: None,
        },
        &PLAN,
        3,
        false,
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
            &source_of(&corpus),
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
    let text = persona(
        &subject,
        &behaviour_of(&subject, &sections, &source_of(&corpus)),
    );
    let seat = seat_spec("tenant", &corpus.goal.role, &corpus.goal.objective, &text).unwrap();
    assert_eq!(seat.persona, text);
}

fn hard_run(seed: u64, form: Form, signal: Signal) -> Vec<Arm> {
    use splinter_knowledge::goal::hard::Hardness;
    use splinter_knowledge::goal::semantic::Semantic;
    use splinter_pipelines::goal::{LexiconEmbedder, Retrieval};
    use std::rc::Rc;
    let mut spec = Spec::new(seed, form);
    spec.hard = Hardness::ALL;
    let corpus = generate(&spec);
    let embedder: Rc<dyn splinter_knowledge::retrieve::Embedder> =
        Rc::new(LexiconEmbedder::new(corpus.lexicon.clone()));
    let cands = candidates(&corpus.documents);
    let semantic = Semantic::new(&*embedder, &cands).unwrap();
    let retrieval = match signal {
        Signal::Lexical => Retrieval::Lexical,
        Signal::Semantic => Retrieval::Semantic,
        Signal::Fused => Retrieval::Fused,
    };
    let make = |provenance: bool| -> Box<dyn Participant> {
        let p = InContext::new(
            ScriptedReader::knowing(corpus.base_knowledge.clone())
                .understanding(corpus.lexicon.clone()),
        )
        .retrieving(retrieval, Rc::clone(&embedder));
        Box::new(if provenance {
            p.showing_provenance()
        } else {
            p
        })
    };
    let plan = Plan { signal, ..PLAN };
    compare(
        &corpus,
        &Inputs {
            fresh: &|| make(false),
            aware: Some(&|| make(true)),
            source: &source_of(&corpus),
            semantic: Some(&semantic),
        },
        &plan,
        seed,
        false,
    )
}

/// The summaries of `corpora` hard corpora of `form` found with `signal`.
fn hard_summary(form: Form, signal: Signal, corpora: u64) -> Vec<Summary> {
    let runs: Vec<Vec<Arm>> = (0..corpora)
        .map(|i| hard_run(1000 + i, form, signal)) // seed: ten corpora, each its own draw
        .collect();
    summarise(&runs, "selective", 1000) // seed: the intervals' resampling
}

fn summary<'a>(all: &'a [Summary], name: &str) -> &'a Summary {
    all.iter().find(|s| s.name == name).unwrap()
}

#[test]
fn on_hard_corpora_selection_with_provenance_beats_reading_everything_with_it() {
    for (form, signal) in [
        (Form::Property, Signal::Lexical),
        (Form::Behaviour, Signal::Fused),
    ] {
        let all = hard_summary(form, signal, 10);
        let best = summary(&all, "selective-with-provenance");
        let everything = summary(&all, "all-with-provenance");
        // Paired over the corpora: the gain's interval excludes zero, at a fraction of the words.
        let gain = best.held_out.mean - everything.held_out.mean;
        assert!(gain > 0.05, "{form:?}: {gain} {best:?} {everything:?}");
        assert!(best.words.mean * 3.0 < everything.words.mean);
        // And it is no longer the oracle: the hard corpora are not saturated.
        assert!(summary(&all, "selective").held_out.mean < 0.95, "{form:?}");
    }
}

#[test]
fn where_the_corpus_is_paraphrased_meaning_finds_what_words_miss() {
    let lexical = hard_summary(Form::Behaviour, Signal::Lexical, 10);
    let fused = hard_summary(Form::Behaviour, Signal::Fused, 10);
    let (l, f) = (summary(&lexical, "selective"), summary(&fused, "selective"));
    assert!(
        f.recall.mean > l.recall.mean + 0.05,
        "{:?} {:?}",
        l.recall,
        f.recall
    );
    assert!(f.held_out.mean >= l.held_out.mean - 0.02);
}
