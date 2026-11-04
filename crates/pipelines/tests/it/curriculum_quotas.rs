// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training sets that keep the learner's own
// words and a balanced mix of what it learns, for its clients. If your team
// needs expertise in curating training data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a training set's selection removes near duplicates first, keeps a
//! writer's own words over a teacher's for the same message, and caps each
//! share of the pool deterministically.

use splinter_core::annotation::Strength;
use splinter_core::digest::Digest;
use splinter_core::experience::ExperienceId;
use splinter_knowledge::concepts::Concept;
use splinter_pipelines::curriculum::quota::{select, Candidate, Quotas};

fn candidate_of(
    id: u8,
    instruction: &str,
    concept: &str,
    kind: &str,
    strength: Strength,
) -> Candidate {
    Candidate {
        id: ExperienceId(Digest::of(&[id])),
        instruction: instruction.into(),
        concepts: vec![Concept::Declared {
            name: concept.into(),
        }],
        kind: kind.into(),
        strength,
        authored: false,
    }
}

#[test]
fn for_one_message_the_writers_own_words_are_kept_whatever_the_strength_or_id() {
    use Strength::{Formal, Judged};
    let message = "How do I keep a habit of study when my mornings vanish?";
    let teachers = candidate_of(1, message, "study", "advise", Formal);
    let writers = Candidate {
        authored: true,
        ..candidate_of(2, message, "study", "advise", Judged)
    };
    for pool in [
        vec![teachers.clone(), writers.clone()],
        vec![writers.clone(), teachers.clone()],
    ] {
        let kept = select(pool, &Quotas::default());
        assert_eq!(kept.selected, std::slice::from_ref(&writers.id));
        assert_eq!(kept.duplicates, 1);
    }
}

#[test]
fn quotas_cap_each_share_after_removing_near_duplicates_deterministically() {
    use Strength::{Executable, Formal};
    let pool = vec![
        candidate_of(
            1,
            "What baud rate does the console run at?",
            "console",
            "recall",
            Formal,
        ),
        candidate_of(
            2,
            "What baud rate does the console run at, exactly?",
            "console",
            "recall",
            Formal,
        ),
        candidate_of(
            3,
            "How many data bits does the console use?",
            "console",
            "recall",
            Formal,
        ),
        candidate_of(
            4,
            "Which parity does the console use?",
            "console",
            "recall",
            Formal,
        ),
        candidate_of(
            5,
            "How much current is drawn when idle?",
            "power",
            "recall",
            Formal,
        ),
        candidate_of(
            6,
            "Write a function returning the idle current.",
            "power",
            "construct",
            Executable,
        ),
        candidate_of(
            7,
            "Which pin carries the transmit line?",
            "pins",
            "recall",
            Formal,
        ),
        candidate_of(8, "Which pin is ground?", "pins", "recall", Formal),
        candidate_of(
            9,
            "What voltage does the board take?",
            "supply",
            "recall",
            Formal,
        ),
    ];
    let quotas = Quotas {
        max_concept_share: 0.25,
        max_kind_share: 0.6,
        max_strength_share: 0.9,
    };
    let chosen = select(pool.clone(), &quotas);
    assert_eq!(chosen.candidates, 9);
    assert_eq!(chosen.duplicates, 1, "the reworded baud-rate question");
    // 8 left, 4 concepts: at most ceil(max(0.25, 1/4) * 8) = 2 each; 2
    // kinds: ceil(0.6 * 8) = 5 recall; 2 strengths: ceil(0.9 * 8) = 8.
    assert_eq!(chosen.caps["concept"]["declared:console"], 2);
    assert_eq!(chosen.caps["kind"]["recall"], 5);
    let kept = |c: &splinter_pipelines::curriculum::quota::Selection, name: &str| {
        c.selected
            .iter()
            .filter(|id| {
                pool.iter().any(|p| {
                    &p.id == *id && p.concepts[0].to_string() == format!("declared:{name}")
                })
            })
            .count()
    };
    assert!(kept(&chosen, "console") <= 2, "{chosen:#?}");
    let recall = chosen
        .selected
        .iter()
        .filter(|id| pool.iter().any(|p| &p.id == *id && p.kind == "recall"))
        .count();
    assert!(recall <= 5, "{chosen:#?}");
    assert_eq!(chosen.selected.len(), 6, "{chosen:#?}");
    assert_eq!(
        chosen.selected[0],
        ExperienceId(Digest::of(&[6])),
        "the strongest verdict first"
    );

    let mut reversed = pool.clone();
    reversed.reverse();
    assert_eq!(select(reversed, &quotas), chosen, "order does not matter");

    let one_kind: Vec<Candidate> = pool.into_iter().filter(|c| c.kind == "recall").collect();
    let whole = select(one_kind, &Quotas::default());
    assert_eq!(
        whole.capped.get("kind"),
        None,
        "a pool of one kind is never capped by kind"
    );
}

/// A pool that holds a handful of answers of another strength is a pool of
/// one strength for the quota: the two formal answers of a recall task or two
/// must not switch on a cap that discards a quarter of the judged ones.
#[test]
fn a_stray_group_does_not_switch_on_a_cap_for_the_rest_of_the_pool() {
    use Strength::{Formal, Judged};
    let kinds = ["advise", "explain", "compare", "reflect"];
    let mut pool: Vec<Candidate> = (0..163u8)
        .map(|n| {
            let words: String = (0..8).map(|w| format!("w{n}x{w} ")).collect();
            candidate_of(
                n,
                &format!("Say what the writer held, {words}"),
                &format!("concept-{n}"),
                kinds[n as usize % kinds.len()],
                Judged,
            )
        })
        .collect();
    pool.extend((163..165u8).map(|n| {
        let words: String = (0..8).map(|w| format!("w{n}x{w} ")).collect();
        candidate_of(
            n,
            &format!("Recall the date, {words}"),
            &format!("concept-{n}"),
            "recall",
            Formal,
        )
    }));
    let kept = select(pool, &Quotas::default());
    assert_eq!(kept.duplicates, 0);
    assert_eq!(kept.capped.get("strength"), None, "{:#?}", kept.capped);
    assert_eq!(kept.selected.len(), 165, "{:#?}", kept.capped);
}
