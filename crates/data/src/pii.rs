// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements personal-data removal for training pipelines
// for its clients. If your team needs expertise in keeping customer and
// personal data out of fine-tuning sets then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Personal data and secrets, taken out of a live transcript before it can
//! become training material.
//!
//! A conversation with a person (a client, a student) holds whatever they
//! typed. [`Scrubber`] replaces it with typed placeholders (`[EMAIL]`,
//! `[PERSONNUMMER]`, ...) using deterministic detectors, so the same text always
//! gives the same result and the counts of what was removed can be recorded and
//! audited. The detectors are conservative on structure (checksums for personal
//! and organisation numbers and IBANs, known prefixes for keys), not a promise
//! that no personal data is left: free-text names are only found through the
//! names list the caller supplies, and a person's own phrasing can always
//! describe them without a detectable token. The record says which detectors ran
//! ([`DETECTORS_VERSION`]) so a consumer can refuse a scrub older than its
//! policy.
//!
//! The detectors run in a fixed order and each replaces what it found before
//! the next looks, so a number is never counted twice.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use regex::{Captures, Regex};
use serde_json::{json, Value};

/// Bumped whenever a detector changes. Recorded in every scrubbed record.
pub const DETECTORS_VERSION: u32 = 1;

/// What a detector found, and the placeholder that replaces it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// An API key, token or password assignment.
    Secret,
    /// An email address.
    Email,
    /// An IBAN with a valid check digit.
    Iban,
    /// A Swedish personal (or coordination) number with a valid check digit.
    Personnummer,
    /// A Swedish organisation number with a valid check digit.
    Organisationsnummer,
    /// A bank, bankgiro or plusgiro number named as such.
    BankAccount,
    /// A phone number.
    Phone,
    /// A street address or a postal code with a city.
    Address,
    /// A name from the caller's names list.
    Name,
}

impl Kind {
    /// The placeholder text.
    #[must_use]
    pub fn placeholder(self) -> &'static str {
        match self {
            Kind::Secret => "[SECRET]",
            Kind::Email => "[EMAIL]",
            Kind::Iban => "[IBAN]",
            Kind::Personnummer => "[PERSONNUMMER]",
            Kind::Organisationsnummer => "[ORGNR]",
            Kind::BankAccount => "[BANK_ACCOUNT]",
            Kind::Phone => "[PHONE]",
            Kind::Address => "[ADDRESS]",
            Kind::Name => "[NAME]",
        }
    }

    /// The key used in a recorded count.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Kind::Secret => "secret",
            Kind::Email => "email",
            Kind::Iban => "iban",
            Kind::Personnummer => "personnummer",
            Kind::Organisationsnummer => "organisationsnummer",
            Kind::BankAccount => "bank_account",
            Kind::Phone => "phone",
            Kind::Address => "address",
            Kind::Name => "name",
        }
    }
}

/// How many of each kind were replaced.
pub type Counts = BTreeMap<Kind, usize>;

#[allow(
    clippy::expect_used,
    reason = "the patterns are literals, checked by this module's tests"
)]
fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("a detector pattern is valid")
}

static SECRET_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|hf_[A-Za-z0-9]{30,}|eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b",
    )
});
static SECRET_ASSIGNED: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r#"(?i)\b((?:api[_-]?key|access[_-]?key|secret|token|passw(?:or)?d|lösenord|bearer)\s*[:=]?\s*["']?)([A-Za-z0-9_\-./+=]{8,})"#,
    )
});
static EMAIL: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"));
static IBAN: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"));
static SWEDISH_ID: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b(?:\d{2})?(\d{2})(\d{2})(\d{2})[-+]?(\d{4})\b"));
static BANK_ACCOUNT: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)\b((?:bank)?konto(?:nummer)?|account(?: number)?|bankgiro|plusgiro|bg|pg|clearing(?:nummer)?)\b(\W{0,3})(\d[\d \-]{4,18}\d)",
    )
});
static PHONE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?:\+|\b00)\d{1,3}[ -]?\(?\d{1,4}\)?(?:[ -]?\d{2,4}){2,4}\b|\b0\d{1,3}[ -]?\d{2,3}[ -]?\d{2,3}[ -]?\d{2,3}\b|\b0\d{1,3}-\d{5,8}\b",
    )
});
static STREET: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(?:[A-ZÅÄÖ][\p{L}]*\s)?[A-ZÅÄÖ][\p{L}]*(?i:gatan|gata|vägen|väg|gränd|stigen|torget|allén|backen|leden)\s+\d{1,3}\s?[A-Za-z]?\b|\b\d{1,4}[A-Z]?\s+[A-Z][a-z]+\s(?:Street|Road|Avenue|Lane|Drive)\b",
    )
});
static POSTAL: LazyLock<Regex> = LazyLock::new(|| re(r"\b\d{3} ?\d{2}\s+[A-ZÅÄÖ][a-zåäöé]+\b"));

/// Whether the digit string passes the Luhn check used by Swedish ten-digit
/// identity numbers.
fn luhn_ok(digits: &str) -> bool {
    let sum: u32 = digits
        .chars()
        .rev()
        .filter_map(|c| c.to_digit(10))
        .enumerate()
        .map(|(i, d)| {
            let v = if i % 2 == 1 { d * 2 } else { d };
            v / 10 + v % 10
        })
        .sum();
    sum.is_multiple_of(10)
}

fn iban_ok(candidate: &str) -> bool {
    let compact: String = candidate.chars().filter(|c| !c.is_whitespace()).collect();
    if !(15..=34).contains(&compact.len()) {
        return false;
    }
    let rotated = format!("{}{}", &compact[4..], &compact[..4]);
    let mut remainder = 0u32;
    for c in rotated.chars() {
        let Some(value) = c.to_digit(36) else {
            return false;
        };
        for digit in value.to_string().chars().filter_map(|d| d.to_digit(10)) {
            remainder = (remainder * 10 + digit) % 97;
        }
    }
    remainder == 1
}

/// The kind of a Swedish ten-digit number, when it is one. A personal number
/// has a month 01-12 and a day 01-31 (61-91 for a coordination number); an
/// organisation number's third digit is at least 2.
fn swedish_id(captures: &Captures<'_>) -> Option<Kind> {
    let (yy, mm, dd, tail) = (&captures[1], &captures[2], &captures[3], &captures[4]);
    if !luhn_ok(&format!("{yy}{mm}{dd}{tail}")) {
        return None;
    }
    let month: u32 = mm.parse().ok()?;
    let day: u32 = dd.parse().ok()?;
    if (1..=12).contains(&month) && matches!(day, 1..=31 | 61..=91) {
        Some(Kind::Personnummer)
    } else if month >= 20 {
        Some(Kind::Organisationsnummer)
    } else {
        None
    }
}

/// Deterministic detectors for personal data and secrets, plus a caller's
/// names list.
#[derive(Clone, Debug, Default)]
pub struct Scrubber {
    names: Vec<Regex>,
}

impl Scrubber {
    /// The built-in detectors, with no names list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds names (a client's, a counterparty's, an employee's) found as whole
    /// words, case-insensitively. This is the only way a name is found: a
    /// detector cannot tell a name from any other word.
    #[must_use]
    pub fn with_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for name in names {
            let name = name.as_ref().trim();
            if !name.is_empty() {
                self.names
                    .push(re(&format!(r"(?i)\b{}\b", regex::escape(name))));
            }
        }
        self
    }

    /// `text` with everything found replaced by its placeholder, and the counts.
    #[must_use]
    pub fn scrub(&self, text: &str) -> (String, Counts) {
        let mut counts = Counts::new();
        let mut out = text.to_string();
        let mut apply =
            |out: &mut String,
             regex: &Regex,
             replace: &dyn Fn(&Captures<'_>) -> Option<(Kind, String)>| {
                *out = regex
                    .replace_all(out, |caps: &Captures<'_>| match replace(caps) {
                        Some((kind, replacement)) => {
                            *counts.entry(kind).or_insert(0) += 1;
                            replacement
                        }
                        None => caps[0].to_string(),
                    })
                    .into_owned();
            };
        let whole =
            |kind: Kind| move |_: &Captures<'_>| Some((kind, kind.placeholder().to_string()));

        apply(&mut out, &SECRET_TOKEN, &whole(Kind::Secret));
        apply(&mut out, &SECRET_ASSIGNED, &|c| {
            Some((
                Kind::Secret,
                format!("{}{}", &c[1], Kind::Secret.placeholder()),
            ))
        });
        apply(&mut out, &EMAIL, &whole(Kind::Email));
        apply(&mut out, &IBAN, &|c| {
            iban_ok(&c[0]).then(|| (Kind::Iban, Kind::Iban.placeholder().to_string()))
        });
        apply(&mut out, &SWEDISH_ID, &|c| {
            swedish_id(c).map(|kind| (kind, kind.placeholder().to_string()))
        });
        apply(&mut out, &BANK_ACCOUNT, &|c| {
            Some((
                Kind::BankAccount,
                format!("{}{}{}", &c[1], &c[2], Kind::BankAccount.placeholder()),
            ))
        });
        apply(&mut out, &PHONE, &|c| {
            let digits = c[0].chars().filter(char::is_ascii_digit).count();
            (8..=15)
                .contains(&digits)
                .then(|| (Kind::Phone, Kind::Phone.placeholder().to_string()))
        });
        apply(&mut out, &STREET, &whole(Kind::Address));
        apply(&mut out, &POSTAL, &whole(Kind::Address));
        for name in &self.names {
            apply(&mut out, name, &whole(Kind::Name));
        }
        (out, counts)
    }

    /// Whether the built-in detectors still find anything in `text`. Placeholders
    /// are never found, so scrubbed text is clean.
    #[must_use]
    pub fn finds_any(&self, text: &str) -> bool {
        !self.scrub(text).1.is_empty()
    }

    /// A conversation record with every string outside `provenance` scrubbed
    /// and a `scrub` entry recording the detector version and the counts.
    /// Idempotent: scrubbing a scrubbed record changes nothing but may add
    /// nothing to the counts.
    #[must_use]
    pub fn scrub_record(&self, record: &Value) -> Value {
        let mut counts = Counts::new();
        let mut scrubbed = record.clone();
        if let Value::Object(map) = &mut scrubbed {
            for (key, value) in map.iter_mut() {
                if key != "provenance" && key != "scrub" {
                    self.walk(value, &mut counts);
                }
            }
            let recorded: BTreeMap<&str, usize> =
                counts.iter().map(|(kind, n)| (kind.label(), *n)).collect();
            map.insert(
                "scrub".to_string(),
                json!({"detectors_version": DETECTORS_VERSION, "counts": recorded}),
            );
        }
        scrubbed
    }

    fn walk(&self, value: &mut Value, counts: &mut Counts) {
        match value {
            Value::String(text) => {
                let (clean, found) = self.scrub(text);
                *text = clean;
                for (kind, n) in found {
                    *counts.entry(kind).or_insert(0) += n;
                }
            }
            Value::Array(items) => items.iter_mut().for_each(|v| self.walk(v, counts)),
            Value::Object(map) => map.values_mut().for_each(|v| self.walk(v, counts)),
            _ => {}
        }
    }

    /// Whether any string of the record outside `provenance` and `scrub` still
    /// trips a built-in detector.
    #[must_use]
    pub fn record_has_residue(&self, record: &Value) -> bool {
        fn any(scrubber: &Scrubber, value: &Value) -> bool {
            match value {
                Value::String(text) => scrubber.finds_any(text),
                Value::Array(items) => items.iter().any(|v| any(scrubber, v)),
                Value::Object(map) => map.values().any(|v| any(scrubber, v)),
                _ => false,
            }
        }
        record.as_object().is_some_and(|map| {
            map.iter()
                .any(|(key, v)| key != "provenance" && key != "scrub" && any(self, v))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every value below is synthetic: checksum-valid but belonging to nobody.
    const PERSONNUMMER: &str = "850709-1232";
    const ORGNR: &str = "556677-8899";
    const IBAN_SPACED: &str = "SE45 5000 0000 0583 9825 7466";

    fn scrub(text: &str) -> (String, Counts) {
        Scrubber::new().scrub(text)
    }

    fn only(text: &str, kind: Kind) -> String {
        let (clean, counts) = scrub(text);
        assert_eq!(counts, Counts::from([(kind, 1)]), "{text} -> {clean}");
        clean
    }

    #[test]
    fn emails_and_keys_are_replaced() {
        assert_eq!(
            only("write to anna.test+x@example.se today", Kind::Email),
            "write to [EMAIL] today"
        );
        let key = format!("sk-{}", "a1B2".repeat(8));
        assert_eq!(
            only(&format!("my key is {key}."), Kind::Secret),
            "my key is [SECRET]."
        );
        assert_eq!(
            only("password: hunter2hunter2", Kind::Secret),
            "password: [SECRET]"
        );
        let token = format!("ghp_{}", "x9".repeat(20));
        assert_eq!(only(&token, Kind::Secret), "[SECRET]");
    }

    #[test]
    fn swedish_identity_numbers_are_told_apart_and_checksummed() {
        assert_eq!(
            only(&format!("pnr {PERSONNUMMER}"), Kind::Personnummer),
            "pnr [PERSONNUMMER]"
        );
        assert_eq!(
            only("pnr 19850709-1232", Kind::Personnummer),
            "pnr [PERSONNUMMER]"
        );
        assert_eq!(
            only("pnr 8507091232", Kind::Personnummer),
            "pnr [PERSONNUMMER]"
        );
        assert_eq!(
            only(&format!("org {ORGNR}"), Kind::Organisationsnummer),
            "org [ORGNR]"
        );
        // A wrong check digit is a number, not an identity.
        assert!(scrub("850709-1233").1.is_empty());
    }

    #[test]
    fn bank_details_need_a_valid_iban_or_a_named_account() {
        assert_eq!(
            only(&format!("pay to {IBAN_SPACED}"), Kind::Iban),
            "pay to [IBAN]"
        );
        assert!(
            scrub("SE46 5000 0000 0583 9825 7466").1.is_empty(),
            "bad check digits"
        );
        assert_eq!(
            only("bankgiro 123-4567", Kind::BankAccount),
            "bankgiro [BANK_ACCOUNT]"
        );
        assert_eq!(
            only("Kontonummer: 8327-123 456 789", Kind::BankAccount),
            "Kontonummer: [BANK_ACCOUNT]"
        );
    }

    #[test]
    fn phone_numbers_and_addresses_are_replaced() {
        assert_eq!(
            only("ring 070-123 45 67 nu", Kind::Phone),
            "ring [PHONE] nu"
        );
        assert_eq!(only("call +46 70 123 45 67", Kind::Phone), "call [PHONE]");
        assert_eq!(only("kontor 08-123 456 78", Kind::Phone), "kontor [PHONE]");
        assert_eq!(
            only("I live at Storgatan 12B", Kind::Address),
            "I live at [ADDRESS]"
        );
        assert_eq!(only("Västra Hamngatan 4", Kind::Address), "[ADDRESS]");
        assert_eq!(only("111 43 Stockholm", Kind::Address), "[ADDRESS]");
        assert_eq!(only("221B Baker Street", Kind::Address), "[ADDRESS]");
    }

    #[test]
    fn ordinary_numbers_and_prose_are_left_alone() {
        for text in [
            "The total is 115 ohm and the current 36.4 mA.",
            "Notice was given on 2026-03-14, 30 days before.",
            "Invoice 20260314 for 14000 SEK, due in 30 days.",
            "In 2019 about 12 345 678 people lived there.",
        ] {
            assert_eq!(scrub(text), (text.to_string(), Counts::new()), "{text}");
        }
    }

    #[test]
    fn a_names_list_finds_whole_words_only() {
        let scrubber = Scrubber::new().with_names(["Anna Lindqvist", "Berg"]);
        let (clean, counts) = scrubber.scrub("anna lindqvist said Berg AB is late; Bergman too.");
        assert_eq!(clean, "[NAME] said [NAME] AB is late; Bergman too.");
        assert_eq!(counts[&Kind::Name], 2);
    }

    #[test]
    fn scrubbing_is_deterministic_idempotent_and_leaves_placeholders_clean() {
        let text = format!("{PERSONNUMMER} {ORGNR} {IBAN_SPACED} a@b.se 070-123 45 67 Storgatan 1");
        let scrubber = Scrubber::new();
        let (once, counts) = scrubber.scrub(&text);
        assert_eq!(counts.values().sum::<usize>(), 6, "{once}");
        assert_eq!(scrubber.scrub(&text).0, once);
        assert_eq!(scrubber.scrub(&once), (once.clone(), Counts::new()));
        assert!(!scrubber.finds_any(&once));
    }

    #[test]
    fn a_record_is_scrubbed_outside_provenance_and_records_its_counts() {
        let record = json!({
            "provenance": {"kind": "live", "consent_recorded": true, "note": "a@b.se"},
            "exchanges": [{"speaker": "seat", "text": "Mail me at a@b.se or ring 070-123 45 67."},
                          {"speaker": "learner", "text": "Fine."}],
        });
        let scrubber = Scrubber::new();
        assert!(scrubber.record_has_residue(&record));
        let clean = scrubber.scrub_record(&record);
        assert_eq!(
            clean["exchanges"][0]["text"],
            "Mail me at [EMAIL] or ring [PHONE]."
        );
        assert_eq!(clean["provenance"], record["provenance"]);
        assert_eq!(clean["scrub"]["detectors_version"], DETECTORS_VERSION);
        assert_eq!(clean["scrub"]["counts"], json!({"email": 1, "phone": 1}));
        assert!(!scrubber.record_has_residue(&clean));
    }
}
