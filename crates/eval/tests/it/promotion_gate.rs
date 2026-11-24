// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// signed, independent evidence, for its clients. If your team needs
// expertise in deciding when a retrained model may replace the one in
// service you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the examiner's verdicts verify in this process with the examiner's
//! public key alone.
//!
//! * A verdict the examiner signed verifies here, on either protocol (a
//!   direct-completion `bench.examination/2`, an agent-protocol
//!   `bench.verdict/3`); any change to it, a signature under another key, a
//!   legacy shared-key MAC, or none at all does not.
//! * A verdict about another candidate or predecessor, or at another level, is
//!   rejected, typed.
//! * The examiner's public key is read from its hex line and nothing else.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use ed25519_dalek::{Signer as _, SigningKey};
use serde_json::{json, Value};
use splinter_eval::promotion::canonical::canonical_bytes;
use splinter_eval::promotion::{verify, Pair, PublicKey, Rejection, Role};

/// The level the verdicts were issued at: two-sided 95% intervals.
const ALPHA: f64 = 0.05;

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
