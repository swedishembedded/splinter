// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// signed, independent evidence, for its clients. If your team needs
// expertise in deciding when a retrained model may replace the one in
// service you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the promotion gate decides on the examiner's signed aggregates,
//! verified in this process.
//!
//! * A verdict the examiner signed verifies here; any change to it, a
//!   signature under another key, or none at all does not.
//! * A verdict about another candidate or predecessor is rejected, typed.
//! * Promoted only when the dev gain's two-sided interval at the gate's level
//!   is above zero, retention is non-inferior within the margin (one-sided
//!   bound) and the sealed slice was spent by this decision alone; every
//!   failure is a typed reason.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::{json, Value};
use splinter_eval::promotion::{
    decide, verify, Pair, Reason, Rejection, Role, Submitted, VerdictKey, ALPHA,
};

/// The key the fixture was signed with by the examiner's own code: the bytes
/// 0 to 31, a test key.
fn key() -> VerdictKey {
    let mut bytes = [0u8; 32];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = u8::try_from(i).unwrap();
    }
    VerdictKey::from_bytes(bytes)
}

/// A verdict the examiner signed (`bench examine` over a sealed exam of 60
/// items, candidate sha256:11..11 against the baseline).
fn fixture() -> Value {
    serde_json::from_str(include_str!("../fixtures/bench-examination-signed.json")).unwrap()
}

const CANDIDATE: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";

fn pair() -> Pair {
    Pair {
        candidate: Some(CANDIDATE.into()),
        predecessor: None,
    }
}

/// The fixture changed by `edit` and signed again with the test key, as the
/// examiner would have signed that verdict.
fn verdict(edit: impl FnOnce(&mut Value)) -> Value {
    let mut doc = fixture();
    edit(&mut doc);
    key().sign(&doc)
}

#[test]
fn a_verdict_the_examiner_signed_verifies_here_and_an_altered_or_forged_one_does_not() {
    let verified = verify(&fixture(), &key(), &pair(), ALPHA).unwrap();
    let aggregate = verified.aggregate();
    assert!(aggregate.sealed && aggregate.items == 60);
    assert_eq!(aggregate.interval.map(|(low, _)| low > 0.0), Some(true));

    let mut altered = fixture();
    altered["paired_gain"]["mean"] = json!(0.5);
    assert_eq!(
        verify(&altered, &key(), &pair(), ALPHA).unwrap_err(),
        Rejection::BadSignature
    );
    let other = VerdictKey::from_bytes([7; 32]);
    assert_eq!(
        verify(&other.sign(&fixture()), &key(), &pair(), ALPHA).unwrap_err(),
        Rejection::BadSignature,
        "a verdict signed under another key is forged"
    );
    let mut unsigned = fixture();
    unsigned.as_object_mut().unwrap().remove("signature");
    assert_eq!(
        verify(&unsigned, &key(), &pair(), ALPHA).unwrap_err(),
        Rejection::Unsigned
    );
}

#[test]
fn a_verdict_about_another_candidate_or_predecessor_is_rejected() {
    let other = Pair {
        candidate: Some("sha256:22".into()),
        predecessor: None,
    };
    assert!(matches!(
        verify(&fixture(), &key(), &other, ALPHA).unwrap_err(),
        Rejection::Candidate { .. }
    ));
    let against = Pair {
        candidate: Some(CANDIDATE.into()),
        predecessor: Some("sha256:00".into()),
    };
    assert!(matches!(
        verify(&fixture(), &key(), &against, ALPHA).unwrap_err(),
        Rejection::Predecessor { .. }
    ));
    let invalid = verdict(|d| {
        d["valid"] = json!(false);
        d["invalid_reasons"] = json!(["candidate: 3 items missing"]);
    });
    assert!(matches!(
        verify(&invalid, &key(), &pair(), ALPHA).unwrap_err(),
        Rejection::Invalid { reasons } if reasons == ["candidate: 3 items missing"]
    ));
    let flagged =
        verdict(|d| d["integrity_flags"] = json!(["training_material_overlaps_collection"]));
    assert!(matches!(
        verify(&flagged, &key(), &pair(), ALPHA).unwrap_err(),
        Rejection::Flagged { .. }
    ));
    let level = verdict(|d| d["alpha"] = json!(0.1));
    assert!(matches!(
        verify(&level, &key(), &pair(), ALPHA).unwrap_err(),
        Rejection::Level { .. }
    ));
}

/// A verdict over public material with the given paired gain interval and
/// one-sided bound.
fn public(interval: [f64; 2], lower: f64) -> Value {
    verdict(|d| {
        d["collection"]["sealed"] = json!(false);
        d["burn"] = Value::Null;
        d["paired_gain"]["interval"] = json!(interval);
        d["paired_gain"]["one_sided_lower"] = json!(lower);
    })
}

#[test]
fn a_candidate_with_a_dev_gain_retention_kept_and_a_fresh_slice_is_promoted() {
    let (dev, retention) = (public([0.02, 0.2], 0.04), public([-0.03, 0.02], -0.02));
    let promotion = decide(
        &key(),
        &pair(),
        Submitted {
            dev: &dev,
            retention: &retention,
            sealed: &fixture(),
        },
    );
    assert!(promotion.promoted, "{:?}", promotion.reasons);
    assert!(promotion.dev.is_some() && promotion.sealed.is_some());
}

#[test]
fn a_dev_interval_that_reaches_zero_or_a_retention_bound_below_the_margin_rejects() {
    let sealed = fixture();
    let decided = |dev: &Value, retention: &Value| {
        decide(
            &key(),
            &pair(),
            Submitted {
                dev,
                retention,
                sealed: &sealed,
            },
        )
    };
    let kept = public([-0.03, 0.02], -0.02);
    let flat = decided(&public([-0.01, 0.2], 0.0), &kept);
    assert!(!flat.promoted);
    assert!(matches!(flat.reasons[..], [Reason::NoDevGain { .. }]));

    let inferior = decided(&public([0.02, 0.2], 0.04), &public([-0.09, 0.0], -0.06));
    assert!(!inferior.promoted, "a non-inferiority failure rejects");
    assert_eq!(
        inferior.reasons,
        [Reason::Inferior {
            lower: -0.06,
            margin: 0.05
        }]
    );
}

#[test]
fn a_slice_spent_by_another_pair_or_public_material_is_no_sealed_evidence() {
    let (dev, retention) = (public([0.02, 0.2], 0.04), public([-0.03, 0.02], -0.02));
    let decided = |sealed: &Value| {
        decide(
            &key(),
            &pair(),
            Submitted {
                dev: &dev,
                retention: &retention,
                sealed,
            },
        )
    };
    let spent = verdict(|d| d["burn"]["candidate"] = json!("sha256:99"));
    let reasons = decided(&spent).reasons;
    assert!(
        matches!(&reasons[..], [Reason::Verdict { role: Role::Sealed, rejection: Rejection::Spent { candidate, .. } }]
            if candidate.as_deref() == Some("sha256:99")),
        "{reasons:?}"
    );
    let unsealed = public([0.02, 0.2], 0.04);
    assert!(matches!(
        &decided(&unsealed).reasons[..],
        [Reason::Verdict {
            role: Role::Sealed,
            rejection: Rejection::NotSealed
        }]
    ));
    let forged = VerdictKey::from_bytes([9; 32]).sign(&fixture());
    let rejected = decided(&forged);
    assert!(!rejected.promoted && rejected.sealed.is_none());
}
