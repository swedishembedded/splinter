// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Removing people and secrets from text that will be stored or trained on.
//!
//! What is scrubbed: the commit's author names and addresses (known from the
//! history), any e-mail address, trailer lines (`Signed-off-by:`,
//! `Co-authored-by:`, `Reviewed-by:` ...), and secret-shaped strings (cloud
//! keys, tokens, private-key blocks, `password=` assignments). The same
//! [`Scrubber`] is applied to every string of a stored trajectory, so a name
//! that appears in a tool result and in a later edit is replaced the same way
//! in both and the record stays self-consistent.
//!
//! A commit whose *diff* holds a secret is not scrubbed but skipped
//! ([`Scrubber::finds_secret`]): replacing text inside an edit would make the
//! reproduced tree differ from the commit.

use regex::Regex;

/// The text that takes a scrubbed span's place.
pub const REDACTED: &str = "[redacted]";

/// Replaces people and secrets in text.
#[derive(Clone, Debug)]
pub struct Scrubber {
    names: Vec<String>,
    patterns: Vec<Regex>,
    trailer: Regex,
}

/// Secret shapes: a cloud access key id, a token with a well-known prefix, a
/// PEM private-key header, a `secret = value` style assignment.
const SECRET_PATTERNS: &[&str] = &[
    r"AKIA[0-9A-Z]{16}",
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"xox[abprs]-[A-Za-z0-9-]{10,}",
    r"sk-[A-Za-z0-9]{24,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r#"(?i)(?:password|passwd|secret|api[_-]?key|token)\w*\s*[:=]\s*["']?[^\s"']{8,}"#,
];

impl Scrubber {
    /// A scrubber for text from a history whose authors are `names` and
    /// `emails` (blank entries are ignored).
    #[must_use]
    pub fn new<'a>(people: impl IntoIterator<Item = &'a str>) -> Self {
        let mut names: Vec<String> = people
            .into_iter()
            .map(str::trim)
            .filter(|n| n.chars().count() >= 3)
            .map(String::from)
            .collect();
        // Longest first, so "Ada Lovelace" goes before "Ada".
        names.sort_by_key(|n| std::cmp::Reverse(n.len()));
        names.dedup();
        let mut patterns = vec![
            Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
                .unwrap_or_else(|_| unreachable!("a fixed pattern")),
        ];
        patterns.extend(
            SECRET_PATTERNS
                .iter()
                .map(|p| Regex::new(p).unwrap_or_else(|_| unreachable!("a fixed pattern"))),
        );
        let trailer = Regex::new(
            r"(?im)^(?:signed-off-by|co-authored-by|reviewed-by|acked-by|tested-by|reported-by|suggested-by|cc):.*\n?",
        )
        .unwrap_or_else(|_| unreachable!("a fixed pattern"));
        Self {
            names,
            patterns,
            trailer,
        }
    }

    /// `text` without trailers, names, addresses and secrets.
    #[must_use]
    pub fn scrub(&self, text: &str) -> String {
        let mut out = self.trailer.replace_all(text, "").into_owned();
        for pattern in &self.patterns {
            out = pattern.replace_all(&out, REDACTED).into_owned();
        }
        for name in &self.names {
            out = if name.contains(char::is_whitespace) || name.contains('@') {
                out.replace(name.as_str(), REDACTED)
            } else {
                replace_standalone(&out, name)
            };
        }
        out
    }

    /// Whether `text` holds a secret-shaped string (not a name or address).
    #[must_use]
    pub fn finds_secret(&self, text: &str) -> bool {
        self.patterns.iter().skip(1).any(|p| p.is_match(text))
    }
}

/// `text` with each occurrence of the one-word `name` replaced, unless it is
/// part of a longer word or a path or identifier (`acme-engine`, `crates/acme`,
/// `acme_core`): an author's handle that is also the project's name must not
/// rewrite the project's own file names.
fn replace_standalone(text: &str, name: &str) -> String {
    let joins = |c: char| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | '\\');
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(name) {
        let before = rest[..at]
            .chars()
            .next_back()
            .or_else(|| out.chars().next_back());
        let tail = &rest[at + name.len()..];
        // A full stop ends a sentence unless a word follows it (`acme.rs`).
        let after = match tail.chars().next() {
            Some('.') if !tail.chars().nth(1).is_some_and(char::is_alphanumeric) => None,
            other => other,
        };
        out.push_str(&rest[..at]);
        if before.is_some_and(joins) || after.is_some_and(joins) {
            out.push_str(name);
        } else {
            out.push_str(REDACTED);
        }
        rest = tail;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn people_and_secrets_go_and_the_rest_stays() {
        let s = Scrubber::new(["Ada Lovelace", "ada@example.org"]);
        let text = "Fix parser\n\nAda Lovelace wrote it, mail ada@example.org\nSigned-off-by: Ada Lovelace <ada@example.org>\nkey AKIAABCDEFGHIJKLMNOP and password = hunter2hunter2";
        let out = s.scrub(text);
        assert!(!out.contains("Ada"), "{out}");
        assert!(!out.contains("example.org"), "{out}");
        assert!(!out.contains("Signed-off-by"), "{out}");
        assert!(!out.contains("AKIA"), "{out}");
        assert!(!out.contains("hunter2"), "{out}");
        assert!(out.contains("Fix parser"));
        assert!(s.finds_secret("token: abcdefghijkl"));
        assert!(!s.finds_secret("mail ada@example.org"));
    }

    #[test]
    fn a_handle_that_is_also_the_project_name_does_not_rewrite_paths() {
        let s = Scrubber::new(["acme", "acme@local"]);
        let text =
            "crates/acme-engine/src/lib.rs and acme_core and `acme` by acme@local, thanks acme.";
        let out = s.scrub(text);
        assert!(out.contains("crates/acme-engine/src/lib.rs"), "{out}");
        assert!(out.contains("acme_core"), "{out}");
        assert!(!out.contains("acme@local"), "{out}");
        assert!(out.contains("thanks [redacted]."), "{out}");
    }
}
