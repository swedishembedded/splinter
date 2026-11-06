// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the serve check passes a rare disagreement and fails a majority of
//! them, and holds a served agent to its verdicts.

use super::*;

fn measured(sampled: usize, agreed: usize) -> Serve {
    Serve {
        binary: "brain".into(),
        startup_line: String::new(),
        served_digest: "sha256:a".into(),
        expected_digest: "sha256:a".into(),
        sampled,
        agreed,
        disagreed: Vec::new(),
        agreement: Agreement::default(),
    }
}

#[test]
fn a_rare_disagreement_passes_and_a_majority_of_them_does_not() {
    assert!(serve(measured(8, 7)).passed);
    assert!(serve(measured(8, 6)).passed, "exactly three quarters");
    assert!(!serve(measured(8, 5)).passed);
    assert!(!serve(measured(6, 0)).passed);
}

#[test]
fn a_served_agent_is_held_to_its_verdicts_and_the_report_says_so() {
    let by_verdict = Serve {
        agreement: Agreement::Verdicts,
        ..measured(8, 5)
    };
    let check = serve(by_verdict);
    assert!(!check.passed);
    let why = check.reason.unwrap();
    assert!(
        why.contains("graded differently") && why.contains("3 of 8"),
        "{why}"
    );
    assert!(
        serve(Serve {
            agreement: Agreement::Verdicts,
            ..measured(8, 6)
        })
        .passed
    );
}

#[test]
fn a_candidate_the_endpoint_lists_under_its_digest_passes_with_nothing_to_compare() {
    // Every answer of the gate came from the endpoint, so there is no
    // in-process run to agree with: the check is that the endpoint serves
    // the candidate and no other adapter.
    let listed = Serve {
        agreement: Agreement::Listed,
        ..measured(0, 0)
    };
    assert!(serve(listed.clone()).passed);
    let other = Serve {
        served_digest: "sha256:b".into(),
        ..listed
    };
    let check = serve(other);
    assert!(!check.passed);
    assert!(check.reason.unwrap().contains("sha256:b"));
}
