// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// signed, independent evidence, for its clients. If your team needs
// expertise in deciding when a retrained model may replace the one in
// service you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the promotion gate decides on the examiner's signed aggregates,
//! verified in this process with the examiner's public key alone.
//!
//! * A verdict the examiner signed verifies here, on either protocol (a
//!   direct-completion `bench.examination/2`, an agent-protocol
//!   `bench.verdict/3`); any change to it, a signature under another key, a
//!   legacy shared-key MAC, or none at all does not.
//! * A verdict about another candidate or predecessor is rejected, typed.
//! * Promoted only when the dev gain's two-sided interval at the gate's level
//!   is above zero, retention is non-inferior within the margin (one-sided
//!   bound) and the sealed slice was spent by this decision alone; every
//!   failure is a typed reason. A slice another pair spent, a burn of
//!   another exam, an exam without a burn and public material are no sealed
//!   evidence.
//! * Where no sealed slice applies, the explicit unexamined rule decides on
//!   dev gain and retention alone and says so in the decision record.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use ed25519_dalek::{Signer as _, SigningKey};
use serde_json::{json, Value};
use splinter_eval::promotion::canonical::canonical_bytes;
use splinter_eval::promotion::{
    decide, decide_unexamined, verify, Measured, Measurement, Pair, PublicKey, Reason, Rejection,
    Role, Rule, Submitted, ALPHA,
};

/// The examiner's test signing key: the seed bytes 0 to 31. The fixtures were
/// signed with it by the examiner's own code; only this test holds it.
fn examiner() -> SigningKey {
    let mut seed = [0u8; 32];
    for (i, b) in seed.iter_mut().enumerate() {
        *b = u8::try_from(i).unwrap();
    }
    SigningKey::from_bytes(&seed)
}

/// What the gate holds: the public half.
fn key() -> PublicKey {
    PublicKey::from_bytes(examiner().verifying_key().to_bytes()).unwrap()
}

/// `document` signed as the examiner signs it, under `signer`.
fn signed_by(signer: &SigningKey, document: &Value) -> Value {
    let mut body = document.clone();
    body.as_object_mut().unwrap().remove("signature");
    let signature = signer.sign(&canonical_bytes(&body).unwrap());
    let hex: String = signature
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    body["signature"] = json!({"alg": "ed25519", "key_id": "test", "value": hex});
    body
}

/// `bench examine` over a sealed exam of 60 items, candidate sha256:11..11
/// against the baseline.
fn examination() -> Value {
    serde_json::from_str(include_str!("../fixtures/bench-examination-signed.json")).unwrap()
}

/// `bench verdict` on the agent protocol over the sealed exam `exam-s1` and
/// the public anchor, the same pair, retention measured.
fn agentic() -> Value {
    serde_json::from_str(include_str!("../fixtures/bench-verdict-signed.json")).unwrap()
}

const CANDIDATE: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";

fn pair() -> Pair {
    Pair {
        candidate: Some(CANDIDATE.into()),
        predecessor: None,
    }
}

/// `fixture` changed by `edit` and signed again, as the examiner would have
/// signed that verdict.
fn edited(fixture: Value, edit: impl FnOnce(&mut Value)) -> Value {
    let mut doc = fixture;
    edit(&mut doc);
    signed_by(&examiner(), &doc)
}

#[test]
fn a_verdict_the_examiner_signed_verifies_with_the_public_key_and_an_altered_or_forged_one_does_not(
) {
    for (fixture, role) in [(examination(), Role::Sealed), (agentic(), Role::Sealed)] {
        let verified = verify(&fixture, &key(), &pair(), ALPHA, role).unwrap();
        assert!(verified.aggregate().sealed);
        assert_eq!(
            verified.aggregate().interval.map(|(low, _)| low > 0.0),
            Some(true)
        );

        let mut tampered = fixture.clone();
        let gain = if tampered.get("paired_gain").is_some() {
            "/paired_gain/mean"
        } else {
            "/overall/delta"
        };
        *tampered.pointer_mut(gain).unwrap() = json!(0.5);
        assert_eq!(
            verify(&tampered, &key(), &pair(), ALPHA, role).unwrap_err(),
            Rejection::BadSignature,
            "a tampered aggregate"
        );
        let forger = SigningKey::from_bytes(&[7; 32]);
        assert_eq!(
            verify(&signed_by(&forger, &fixture), &key(), &pair(), ALPHA, role).unwrap_err(),
            Rejection::BadSignature,
            "a verdict signed under another key is forged"
        );
        let mut legacy = fixture.clone();
        legacy["signature"] = json!({"alg": "hmac-sha256", "key_id": "x", "value": "00"});
        assert_eq!(
            verify(&legacy, &key(), &pair(), ALPHA, role).unwrap_err(),
            Rejection::Unsigned,
            "a shared-key MAC is no signature"
        );
        let mut unsigned = fixture;
        unsigned.as_object_mut().unwrap().remove("signature");
        assert_eq!(
            verify(&unsigned, &key(), &pair(), ALPHA, role).unwrap_err(),
            Rejection::Unsigned
        );
    }
}

#[test]
fn a_public_key_file_is_the_examiners_hex_line_and_nothing_else() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("verdict.key.pub");
    let hex: String = examiner()
        .verifying_key()
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    std::fs::write(&path, format!("{hex}\n")).unwrap();
    let read = PublicKey::read(&path).unwrap();
    assert!(verify(&agentic(), &read, &pair(), ALPHA, Role::Dev).is_ok());
    std::fs::write(&path, [0u8; 32]).unwrap();
    assert!(
        PublicKey::read(&path).is_err(),
        "a 32-byte shared key is not a public key"
    );
}

#[test]
fn a_verdict_about_another_candidate_or_predecessor_or_at_another_level_is_rejected() {
    for fixture in [examination(), agentic()] {
        let other = Pair {
            candidate: Some("sha256:22".into()),
            predecessor: None,
        };
        assert!(matches!(
            verify(&fixture, &key(), &other, ALPHA, Role::Dev).unwrap_err(),
            Rejection::Candidate { .. }
        ));
        let against = Pair {
            candidate: Some(CANDIDATE.into()),
            predecessor: Some("sha256:00".into()),
        };
        assert!(matches!(
            verify(&fixture, &key(), &against, ALPHA, Role::Dev).unwrap_err(),
            Rejection::Predecessor { .. }
        ));
    }
    let invalid = edited(examination(), |d| {
        d["valid"] = json!(false);
        d["invalid_reasons"] = json!(["candidate: 3 items missing"]);
    });
    assert!(matches!(
        verify(&invalid, &key(), &pair(), ALPHA, Role::Dev).unwrap_err(),
        Rejection::Invalid { reasons } if reasons == ["candidate: 3 items missing"]
    ));
    let invalid = edited(agentic(), |d| {
        d["verdict"] = json!("Invalid");
        d["reasons"] =
            json!(["served_model: candidate: requests were answered by adapter(s) sha256:99"]);
    });
    assert!(matches!(
        verify(&invalid, &key(), &pair(), ALPHA, Role::Dev).unwrap_err(),
        Rejection::Invalid { reasons } if reasons[0].starts_with("served_model")
    ));
    let anonymous = edited(agentic(), |d| d["candidate"] = Value::Null);
    assert!(matches!(
        verify(&anonymous, &key(), &pair(), ALPHA, Role::Dev).unwrap_err(),
        Rejection::Malformed { field } if field == "candidate"
    ));
    let flagged = edited(examination(), |d| {
        d["integrity_flags"] = json!(["training_material_overlaps_collection"]);
    });
    assert!(matches!(
        verify(&flagged, &key(), &pair(), ALPHA, Role::Dev).unwrap_err(),
        Rejection::Flagged { .. }
    ));
    let level = edited(examination(), |d| d["alpha"] = json!(0.1));
    assert!(matches!(
        verify(&level, &key(), &pair(), ALPHA, Role::Dev).unwrap_err(),
        Rejection::Level { .. }
    ));
    let level = edited(agentic(), |d| d["policy"]["alpha"] = json!(0.1));
    assert!(matches!(
        verify(&level, &key(), &pair(), ALPHA, Role::Dev).unwrap_err(),
        Rejection::Level { .. }
    ));
}

/// A direct-completion verdict over public material with the given paired
/// gain interval and one-sided bound.
fn public(interval: [f64; 2], lower: f64) -> Value {
    edited(examination(), |d| {
        d["collection"]["sealed"] = json!(false);
        d["burn"] = Value::Null;
        d["paired_gain"]["interval"] = json!(interval);
        d["paired_gain"]["one_sided_lower"] = json!(lower);
    })
}

/// An agent-protocol verdict over public material: no sealed exam, no burn;
/// the given overall interval and retention bound.
fn public_agentic(interval: [f64; 2], retention_lower: f64) -> Value {
    edited(agentic(), |d| {
        d["sealed_exams"] = json!([]);
        d["burns"] = json!([]);
        d["overall"]["interval"] = json!(interval);
        d["retention"]["one_sided_lower"] = json!(retention_lower);
    })
}

fn decided(dev: &Value, retention: &Value, sealed: &Value) -> splinter_eval::promotion::Promotion {
    decide(
        &key(),
        &pair(),
        Submitted {
            dev,
            retention,
            sealed,
        },
    )
}

#[test]
fn a_candidate_with_a_dev_gain_retention_kept_and_a_fresh_slice_is_promoted_on_either_protocol() {
    let (dev, retention) = (public([0.02, 0.2], 0.04), public([-0.03, 0.02], -0.02));
    let promotion = decided(&dev, &retention, &examination());
    assert!(promotion.promoted, "{:?}", promotion.reasons);
    assert_eq!(promotion.rule, Rule::Examined);
    assert!(promotion.dev.is_some() && promotion.sealed.is_some());

    let agent = decided(&agentic(), &agentic(), &agentic());
    assert!(agent.promoted, "{:?}", agent.reasons);
    let (dev, retention) = (agent.dev.unwrap(), agent.retention.unwrap());
    assert!(
        dev.interval.unwrap().0 > 0.0,
        "the dev gain is the overall section's"
    );
    assert!(
        retention.one_sided_lower.unwrap() > -0.05
            && retention.one_sided_lower != dev.one_sided_lower,
        "retention is the retention section's"
    );
}

#[test]
fn a_dev_interval_that_reaches_zero_or_a_retention_bound_below_the_margin_rejects() {
    let sealed = examination();
    let kept = public([-0.03, 0.02], -0.02);
    let flat = decided(&public([-0.01, 0.2], 0.0), &kept, &sealed);
    assert!(!flat.promoted);
    assert!(matches!(flat.reasons[..], [Reason::NoDevGain { .. }]));

    let inferior = decided(
        &public([0.02, 0.2], 0.04),
        &public([-0.09, 0.0], -0.06),
        &sealed,
    );
    assert!(!inferior.promoted, "a non-inferiority failure rejects");
    assert_eq!(
        inferior.reasons,
        [Reason::Inferior {
            lower: -0.06,
            margin: 0.05
        }]
    );
    let agent_inferior = decided(
        &public_agentic([0.02, 0.2], 0.0),
        &public_agentic([0.02, 0.2], -0.07),
        &agentic(),
    );
    assert!(matches!(
        agent_inferior.reasons[..],
        [Reason::Inferior { .. }]
    ));
}

#[test]
fn a_spent_replayed_or_public_slice_is_no_sealed_evidence() {
    let (dev, retention) = (public([0.02, 0.2], 0.04), public([-0.03, 0.02], -0.02));
    let sealed_reason = |sealed: &Value| {
        let promotion = decided(&dev, &retention, sealed);
        assert!(!promotion.promoted);
        match &promotion.reasons[..] {
            [Reason::Verdict {
                role: Role::Sealed,
                rejection,
            }] => rejection.clone(),
            other => panic!("{other:?}"),
        }
    };
    let spent = edited(examination(), |d| {
        d["burn"]["candidate"] = json!("sha256:99")
    });
    assert!(
        matches!(sealed_reason(&spent), Rejection::Spent { candidate, .. }
        if candidate.as_deref() == Some("sha256:99"))
    );
    let spent = edited(agentic(), |d| {
        d["burns"][0]["predecessor"] = json!("sha256:98")
    });
    assert!(matches!(sealed_reason(&spent), Rejection::Spent { .. }));
    let replayed = edited(agentic(), |d| {
        d["burns"][0]["exam_sha256"] = json!("f".repeat(64));
    });
    assert!(
        matches!(
            sealed_reason(&replayed),
            Rejection::BurnOfAnotherExam { .. }
        ),
        "a burn record carried over from another exam"
    );
    let unburned = edited(agentic(), |d| d["burns"] = json!([]));
    assert_eq!(sealed_reason(&unburned), Rejection::NoBurn);
    assert_eq!(
        sealed_reason(&public([0.02, 0.2], 0.04)),
        Rejection::NotSealed
    );
    assert_eq!(
        sealed_reason(&public_agentic([0.02, 0.2], 0.0)),
        Rejection::NotSealed
    );
    let forged = signed_by(&SigningKey::from_bytes(&[9; 32]), &examination());
    let rejected = decided(&dev, &retention, &forged);
    assert!(!rejected.promoted && rejected.sealed.is_none());
}

#[test]
fn where_no_sealed_slice_applies_dev_gain_and_retention_decide_and_the_record_says_it_was_not_exam_verified(
) {
    let measured = |low: f64, lower: f64| Measured {
        dev: Measurement {
            items: 40,
            gain: Some(0.1),
            interval: Some((low, 0.2)),
            one_sided_lower: Some(low),
        },
        retention: Measurement {
            items: 30,
            gain: Some(0.0),
            interval: Some((lower - 0.01, 0.02)),
            one_sided_lower: Some(lower),
        },
    };
    let why = "a learn release: its exam is the learner's own";
    let promoted = decide_unexamined(&pair(), &measured(0.01, -0.02), why);
    assert!(promoted.promoted && promoted.sealed.is_none());
    assert_eq!(promoted.rule, Rule::NotExamVerified { why: why.into() });
    let text = serde_json::to_value(&promoted).unwrap();
    assert_eq!(text["rule"]["rule"], "not_exam_verified");
    assert_eq!(text["exam_verified"], false);
    let flat = decide_unexamined(&pair(), &measured(-0.01, -0.02), why);
    assert!(!flat.promoted && matches!(flat.reasons[..], [Reason::NoDevGain { .. }]));
    let inferior = decide_unexamined(&pair(), &measured(0.01, -0.06), why);
    assert!(!inferior.promoted && matches!(inferior.reasons[..], [Reason::Inferior { .. }]));
}
