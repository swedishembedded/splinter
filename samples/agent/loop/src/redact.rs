// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Credentials are removed from everything the loop persists.
//!
//! A pattern list is not a proof that no secret survives; it is the floor
//! the loop guarantees for the shapes it knows (assignments to names that
//! sound like secrets, bearer tokens, provider key prefixes, private key
//! blocks). The prompt, the tool output and the patches pass through it
//! before they reach the trace or an artifact.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

/// What replaces a secret.
pub const REDACTED: &str = "[REDACTED]";

/// Each pattern with the replacement that keeps the surrounding text. The
/// assignment pattern is last: it would take `Bearer` for the value of an
/// `Authorization` header and leave the token behind.
fn patterns() -> &'static [(Regex, &'static str)] {
    static PATTERNS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let table: [(&str, &str); 6] = [
            (r"\bsk-[A-Za-z0-9_\-]{12,}", REDACTED),
            (r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}", "Bearer [REDACTED]"),
            (r"\bAKIA[0-9A-Z]{16}\b", REDACTED),
            (r"\bgh[pousr]_[A-Za-z0-9]{20,}", REDACTED),
            (
                r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
                REDACTED,
            ),
            (
                r#"(?i)\b([A-Za-z0-9_]*(?:api[_-]?key|secret|token|passwd|password|authorization))\b(\s*[=:]\s*)("[^"]*"|'[^']*'|\S+)"#,
                "${1}${2}[REDACTED]",
            ),
        ];
        table
            .iter()
            // The patterns are literals checked by the spec below.
            .filter_map(|(p, r)| Regex::new(p).ok().map(|re| (re, *r)))
            .collect()
    })
}

/// `text` with every known secret shape replaced.
#[must_use]
pub fn redact(text: &str) -> String {
    let mut out = text.to_string();
    for (pattern, replacement) in patterns() {
        out = pattern.replace_all(&out, *replacement).into_owned();
    }
    out
}

/// `value` with every string in it redacted.
#[must_use]
pub fn redact_value(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(redact(s)),
        Value::Array(items) => Value::Array(items.iter().map(redact_value).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), redact_value(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pattern_compiles() {
        assert_eq!(patterns().len(), 6);
    }

    #[test]
    fn secrets_are_replaced_and_ordinary_text_is_kept() {
        let text = "export OPENROUTER_API_KEY=sk-or-v1-abcdef0123456789 and run make test; \
                    curl -H 'Authorization: Bearer abcdef0123456789abcd' url; password: hunter2";
        let clean = redact(text);
        assert!(!clean.contains("abcdef0123456789"), "{clean}");
        assert!(!clean.contains("hunter2"), "{clean}");
        assert!(clean.contains("run make test"), "{clean}");
    }

    #[test]
    fn a_private_key_block_is_removed_whole() {
        // Assembled here so the file itself holds no key-shaped text.
        let (begin, end) = ("-----BEGIN RSA ", "-----END RSA ");
        let key = format!("{begin}PRIVATE KEY-----\nMIIabc\n{end}PRIVATE KEY-----");
        assert_eq!(redact(&format!("a {key} b")), "a [REDACTED] b");
    }

    #[test]
    fn nested_json_strings_are_redacted() {
        let value = serde_json::json!({"a": ["token=abc123def456"], "n": 3});
        let clean = redact_value(&value);
        assert_eq!(clean["a"][0], "token=[REDACTED]");
        assert_eq!(clean["n"], 3);
    }
}
