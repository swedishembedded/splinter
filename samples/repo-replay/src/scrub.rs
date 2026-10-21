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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// The text that takes a scrubbed span's place.
pub const REDACTED: &str = "[redacted]";

/// Replaces people and secrets in text.
/// What a [`Scrubber`] removed, by kind, as occurrences. Counts only: what was
/// removed is not kept anywhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScrubCounts {
    /// Trailer lines (`Signed-off-by:` ...).
    pub trailers: usize,
    /// Author names and any e-mail address.
    pub people: usize,
    /// Secret-shaped strings.
    pub secrets: usize,
}

impl ScrubCounts {
    /// These counts and `other`'s together.
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self {
            trailers: self.trailers + other.trailers,
            people: self.people + other.people,
            secrets: self.secrets + other.secrets,
        }
    }
}

#[derive(Debug, Default)]
struct Tally {
    trailers: AtomicUsize,
    people: AtomicUsize,
    secrets: AtomicUsize,
}

#[derive(Clone, Debug)]
pub struct Scrubber {
    names: Vec<String>,
    patterns: Vec<Regex>,
    trailer: Regex,
    tally: Arc<Tally>,
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
            tally: Arc::default(),
        }
    }

    /// This scrubber with counts of its own, starting at zero: what an
    /// attempt had removed is what the attempt's clone counted.
    #[must_use]
    pub fn counting(&self) -> Self {
        Self {
            tally: Arc::default(),
            ..self.clone()
        }
    }

    /// What this scrubber has removed so far.
    #[must_use]
    pub fn counts(&self) -> ScrubCounts {
        let read = |n: &AtomicUsize| n.load(Ordering::Relaxed);
        ScrubCounts {
            trailers: read(&self.tally.trailers),
            people: read(&self.tally.people),
            secrets: read(&self.tally.secrets),
        }
    }

    /// `text` without the trailer lines, e-mail addresses, secrets and names, counted.
    #[must_use]
    pub fn scrub(&self, text: &str) -> String {
        let add = |counter: &AtomicUsize, n: usize| counter.fetch_add(n, Ordering::Relaxed);
        add(&self.tally.trailers, self.trailer.find_iter(text).count());
        let mut out = self.trailer.replace_all(text, "").into_owned();
        for (at, pattern) in self.patterns.iter().enumerate() {
            // The first pattern is the e-mail address, the rest are secrets.
            let counter = if at == 0 {
                &self.tally.people
            } else {
                &self.tally.secrets
            };
            add(counter, pattern.find_iter(&out).count());
            out = pattern.replace_all(&out, REDACTED).into_owned();
        }
        for name in &self.names {
            out = if name.contains(char::is_whitespace) || name.contains('@') {
                add(&self.tally.people, out.matches(name.as_str()).count());
                out.replace(name.as_str(), REDACTED)
            } else {
                let (replaced, n) = replace_standalone(&out, name);
                add(&self.tally.people, n);
                replaced
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
fn replace_standalone(text: &str, name: &str) -> (String, usize) {
    let joins = |c: char| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | '\\');
    let mut out = String::with_capacity(text.len());
    let mut replaced = 0;
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
            replaced += 1;
        }
        rest = tail;
    }
    out.push_str(rest);
    (out, replaced)
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
    fn what_was_removed_is_counted_by_kind_and_a_counting_clone_starts_at_zero() {
        let s = Scrubber::new(["Ada Lovelace", "ada@example.org"]);
        let text = "Ada Lovelace wrote it\nSigned-off-by: Ada Lovelace <ada@example.org>\nAKIAABCDEFGHIJKLMNOP and mail bob@example.org";
        let mine = s.counting();
        assert!(!mine.scrub(text).contains("AKIA"));
        let counts = mine.counts();
        assert_eq!(counts.trailers, 1);
        assert_eq!(counts.secrets, 1);
        // Two names in the text (one inside the removed trailer is not counted) and an address.
        assert_eq!(counts.people, 2);
        assert_eq!(s.counts(), ScrubCounts::default());
        assert_eq!(mine.scrub("nothing to remove"), "nothing to remove");
        assert_eq!(mine.counts(), counts);
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
