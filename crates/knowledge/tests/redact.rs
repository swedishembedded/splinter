// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements secret redaction that keeps credentials out
// of training data, for its clients. If your team needs expertise in
// data hygiene for model training, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: a secret never survives redaction, whatever form it is in, and
//! ordinary text passes through untouched. Redacting twice changes nothing.

use splinter_knowledge::redact::{redact, KIND_CREDENTIAL, KIND_PRIVATE_KEY, KIND_TOKEN};

const API_KEY: &str = "sk-proj-4f9a8c7b2e1d3f6a5b4c3d2e1f0a9b8c";
const GITHUB: &str = "ghp_0123456789abcdefghijklmnopqrstuvwxyzAB";
const AWS: &str = "AKIAABCDEFGHIJKLMNOP";
const JWT: &str = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJlLWJ5dGVz";

/// A private key block, its marker lines assembled so that this file holds
/// none.
fn pem(body: &str) -> String {
    format!(
        "-----BEGIN {0} KEY-----\n{body}\n-----END {0} KEY-----",
        "RSA PRIVATE"
    )
}

fn removed(text: &str, secret: &str, kind: &str) {
    let out = redact(text);
    assert!(
        !out.text.contains(secret),
        "{secret:?} survived: {}",
        out.text
    );
    assert!(
        out.found.iter().any(|(k, n)| *k == kind && *n >= 1),
        "no {kind} recorded for {text:?}: {:?}",
        out.found
    );
    assert_eq!(redact(&out.text).text, out.text, "redaction is idempotent");
    assert!(redact(&out.text).found.is_empty(), "nothing left to find");
}

#[test]
fn keys_and_tokens_by_their_shape_are_removed() {
    removed(&format!("my key is {API_KEY} ok"), API_KEY, KIND_TOKEN);
    removed(&format!("export GH={GITHUB}"), GITHUB, KIND_TOKEN);
    removed(&format!("aws id {AWS}."), AWS, KIND_TOKEN);
    removed(&format!("jwt: {JWT}"), JWT, KIND_TOKEN);
    removed(
        "Authorization: Bearer abcdef0123456789abcdef",
        "abcdef0123456789abcdef",
        KIND_TOKEN,
    );
}

#[test]
fn a_value_assigned_to_a_secret_name_is_removed_and_the_name_kept() {
    for (text, secret) in [
        ("password=hunter-two", "hunter-two"),
        ("DB_PASSWORD: \"s3cr3t pass\"", "s3cr3t"),
        ("{\"password\": \"hunter-two\"}", "hunter-two"),
        ("GITHUB_TOKEN=abcd1234efgh5678", "abcd1234efgh5678"),
        ("api_key = 'zzzz-1111-yyyy-2222'", "zzzz-1111-yyyy-2222"),
        ("my password is hunter-two, remember", "hunter-two"),
        (
            "postgres://app:hunter-two@db.internal:5432/app",
            "hunter-two",
        ),
    ] {
        removed(text, secret, KIND_CREDENTIAL);
    }
    let out = redact("password=hunter-two");
    assert!(out.text.starts_with("password="), "{}", out.text);
}

#[test]
fn a_private_key_block_is_removed_whole() {
    removed(
        &format!("here:\n{}\nthanks", pem("MIIEowIBAAKCAQEAneutral")),
        "MIIEow",
        KIND_PRIVATE_KEY,
    );
    let unterminated = format!("-----BEGIN {} KEY-----\nMIIEvQIBADANBgkq", "PRIVATE");
    removed(&unterminated, "MIIEvQ", KIND_PRIVATE_KEY);
}

#[test]
fn ordinary_text_is_left_alone() {
    let text = "The max_tokens setting is 4096. Set the token count with --tokens 12.\n\
                A password manager helps. See disk-usage and task-queue; sk is a prefix.\n\
                The tokenizer=gpt2 default and author=alice are fine.";
    let out = redact(text);
    assert_eq!(out.text, text);
    assert!(out.found.is_empty(), "{:?}", out.found);
}
