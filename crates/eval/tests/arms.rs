// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements evaluations that tell a model update from a
// knowledge update, for its clients. If your team needs expertise in
// attributing an agent's improvement to its weights or to the knowledge it
// retrieves, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a comparison says what differs between its arms. A change of weights
//! and a change of retrieved knowledge are different treatments with different
//! costs and risks, so a result is attributed to one only when only that one
//! varied. When both vary the effect is confounded and no attribution is made.

use splinter_eval::arms::{attribute, Arm, Attribution, Varied};
use splinter_eval::paired::{compare_arms, PairedOutcome};

fn arm(weights: &str, knowledge: Option<&str>) -> Arm {
    Arm::new(weights, knowledge.map(str::to_owned))
}

#[test]
fn only_what_differs_is_said_to_vary() {
    let base = arm("w1", None);
    assert_eq!(base.varied_from(&arm("w1", None)), Varied::Nothing);
    assert_eq!(base.varied_from(&arm("w2", None)), Varied::Weights);
    assert_eq!(
        base.varied_from(&arm("w1", Some("snap-a"))),
        Varied::Knowledge
    );
    assert_eq!(base.varied_from(&arm("w2", Some("snap-a"))), Varied::Both);
    assert_eq!(
        arm("w1", Some("snap-a")).varied_from(&arm("w1", Some("snap-b"))),
        Varied::Knowledge
    );
}

#[test]
fn an_effect_is_attributed_to_the_one_treatment_that_varied() {
    let reference = arm("w1", Some("snap-a"));
    assert_eq!(
        attribute(&reference, &arm("w2", Some("snap-a"))).unwrap(),
        Attribution::Weights
    );
    assert_eq!(
        attribute(&reference, &arm("w1", Some("snap-b"))).unwrap(),
        Attribution::Knowledge
    );
}

#[test]
fn a_comparison_that_varies_both_or_neither_is_refused() {
    let reference = arm("w1", None);
    assert!(attribute(&reference, &arm("w2", Some("snap-a"))).is_err());
    assert!(attribute(&reference, &arm("w1", None)).is_err());
}

#[test]
fn a_compared_pair_carries_the_arms_and_says_what_differs() {
    let outcomes = vec![
        PairedOutcome {
            item: "t1".into(),
            candidate: Some(true),
            baseline: Some(false),
            cluster: None,
        },
        PairedOutcome {
            item: "t2".into(),
            candidate: Some(true),
            baseline: Some(true),
            cluster: None,
        },
    ];
    let compared = compare_arms(&arm("w1", None), &arm("w1", Some("snap-a")), &outcomes).unwrap();
    assert_eq!(compared.attribution, Attribution::Knowledge);
    assert_eq!(compared.comparison.candidate_wins, 1);
    let text = compared.describe();
    assert!(text.contains("knowledge"), "{text}");
    assert!(text.contains("snap-a"), "{text}");
    assert!(text.contains("w1"), "{text}");
    assert!(compare_arms(&arm("w1", None), &arm("w2", Some("snap-a")), &outcomes).is_err());
}
