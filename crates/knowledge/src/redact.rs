// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements secret redaction that keeps credentials out
// of training data, for its clients. If your team needs expertise in data
// hygiene for model training, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Secrets removed from text before it is stored or shown to a model.
//!
//! Recognised, in this order:
//!
//! * a PEM private key block, whole (to the end of the text when its end
//!   line is missing): [`KIND_PRIVATE_KEY`];
//! * a value assigned to a secret name (`password=...`, `"api_key": "..."`,
//!   `GITHUB_TOKEN: ...`, `password is ...`) and the password of a URL's
//!   user information: [`KIND_CREDENTIAL`]. The name is kept, the value
//!   replaced. A name that only sometimes holds a secret (`token`,
//!   `secret`, `key`) needs a value of [`MIN_GENERIC_SECRET_CHARS`] or more,
//!   so `max_tokens=4096` stays; a password of any length goes;
//! * a bearer credential and tokens known by their shape (`sk-`, `ghp_`,
//!   `AKIA`, `xox*-`, `AIza`, `glpat-`, JSON web tokens): [`KIND_TOKEN`].
//!
//! Each removal leaves `[REDACTED:<kind>]`. Redacting redacted text finds
//! nothing, so a stored redaction is stable. This is a filter for what is
//! recognisable, not proof that no secret remains.

use std::collections::BTreeMap;

/// A PEM private key block.
pub const KIND_PRIVATE_KEY: &str = "private_key";
/// A value assigned to a secret name.
pub const KIND_CREDENTIAL: &str = "credential";
/// A token known by its shape, or a bearer credential.
pub const KIND_TOKEN: &str = "token";

/// The shortest value assigned to a name that only sometimes holds a secret
/// that counts as one.
pub const MIN_GENERIC_SECRET_CHARS: usize = 8;

/// The text left after redaction and what was removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Redacted {
    /// The text, secrets replaced.
    pub text: String,
    /// How many secrets of each kind were removed, by kind.
    pub found: Vec<(&'static str, u32)>,
}

/// `text` with every recognised secret removed; see the module
/// documentation.
#[must_use]
pub fn redact(text: &str) -> Redacted {
    let mut counts: BTreeMap<&'static str, u32> = BTreeMap::new();
    let text = private_keys(text, &mut counts);
    let text = assignments(&text, &mut counts);
    let text = url_passwords(&text, &mut counts);
    let text = bearer(&text, &mut counts);
    let text = shaped_tokens(&text, &mut counts);
    Redacted {
        text,
        found: counts.into_iter().collect(),
    }
}

/// The text that replaces a removed secret of `kind`.
#[must_use]
pub fn marker(kind: &str) -> String {
    format!("[REDACTED:{kind}]")
}

fn count(counts: &mut BTreeMap<&'static str, u32>, kind: &'static str) {
    *counts.entry(kind).or_default() += 1;
}

fn private_keys(text: &str, counts: &mut BTreeMap<&'static str, u32>) -> String {
    const BEGIN: &str = "-----BEGIN ";
    const END: &str = "-----END ";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(BEGIN) {
        let header_end = rest[start..].find('\n').map_or(rest.len(), |i| start + i);
        if !rest[start..header_end].contains("PRIVATE KEY-----") {
            out.push_str(&rest[..start + BEGIN.len()]);
            rest = &rest[start + BEGIN.len()..];
            continue;
        }
        out.push_str(&rest[..start]);
        out.push_str(&marker(KIND_PRIVATE_KEY));
        count(counts, KIND_PRIVATE_KEY);
        rest = match rest[header_end..].find(END) {
            Some(i) => {
                let from = header_end + i + END.len();
                rest[from..]
                    .find("-----")
                    .map_or("", |close| &rest[from + close + "-----".len()..])
            }
            None => "",
        };
    }
    out.push_str(rest);
    out
}

/// Characters of a name or a token.
fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Whether `name` (lower-cased) holds a secret whatever its value.
fn always_secret(name: &str) -> bool {
    ["password", "passwd", "passphrase", "passcode", "pwd"]
        .iter()
        .any(|n| name.contains(n))
}

/// Whether `name` (lower-cased) holds a secret when its value is long enough
/// to be one.
fn generic_secret(name: &str) -> bool {
    let name = name.replace('-', "_");
    [
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_key",
        "private_key",
        "authorization",
    ]
    .iter()
    .any(|n| name.contains(n))
}

/// Whether `value` is a secret because of the `name` it is assigned to: the
/// rule [`redact`] applies to `name=value`, for structured data whose names
/// and values are separate.
#[must_use]
pub fn secret_value_in(name: &str, value: &str) -> bool {
    if value.starts_with("[REDACTED:") {
        return false;
    }
    let name = name.to_ascii_lowercase();
    always_secret(&name)
        || (generic_secret(&name) && value.chars().count() >= MIN_GENERIC_SECRET_CHARS)
}

/// The words of `text` as (start, end) byte ranges.
fn words(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (is_word(c), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, text.len()));
    }
    out
}

/// Where the value assigned after a name ending at `at` starts and ends;
/// `None` when no value is assigned. `is_form` also accepts `is`/`was`.
fn assigned_value(text: &str, at: usize, is_form: bool) -> Option<(usize, usize)> {
    let rest = &text[at..];
    let mut i = 0;
    let bytes = rest.as_bytes();
    // A closing quote of a quoted name.
    if matches!(bytes.first(), Some(b'"' | b'\'')) {
        i += 1;
    }
    let spaces = |i: &mut usize| {
        while matches!(bytes.get(*i), Some(b' ' | b'\t')) {
            *i += 1;
        }
    };
    spaces(&mut i);
    if matches!(bytes.get(i), Some(b':' | b'=')) {
        i += 1;
    } else if is_form {
        let word = ["is", "was"]
            .iter()
            .find(|w| rest[i..].starts_with(*w) && matches!(bytes.get(i + w.len()), Some(b' ')))?;
        i += word.len();
    } else {
        return None;
    }
    spaces(&mut i);
    let quote = match bytes.get(i) {
        Some(q @ (b'"' | b'\'')) => {
            i += 1;
            Some(*q as char)
        }
        _ => None,
    };
    let start = i;
    let end = rest[start..]
        .find(|c: char| match quote {
            Some(q) => c == q || c == '\n',
            None => c.is_whitespace() || matches!(c, '"' | '\'' | ',' | ';' | ')' | '}'),
        })
        .map_or(rest.len(), |n| start + n);
    (end > start).then_some((at + start, at + end))
}

fn assignments(text: &str, counts: &mut BTreeMap<&'static str, u32>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    for (start, end) in words(text) {
        if start < copied {
            continue;
        }
        let name = text[start..end].to_ascii_lowercase();
        let always = always_secret(&name);
        if !always && !generic_secret(&name) {
            continue;
        }
        let Some((v_start, v_end)) = assigned_value(text, end, always) else {
            continue;
        };
        let value = &text[v_start..v_end];
        if value.starts_with("[REDACTED:")
            || (!always && value.chars().count() < MIN_GENERIC_SECRET_CHARS)
        {
            continue;
        }
        out.push_str(&text[copied..v_start]);
        out.push_str(&marker(KIND_CREDENTIAL));
        count(counts, KIND_CREDENTIAL);
        copied = v_end;
    }
    out.push_str(&text[copied..]);
    out
}

/// The password in `scheme://user:password@host`.
fn url_passwords(text: &str, counts: &mut BTreeMap<&'static str, u32>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("://") {
        let after = i + 3;
        out.push_str(&rest[..after]);
        rest = &rest[after..];
        let authority_end = rest
            .find(|c: char| c == '/' || c.is_whitespace())
            .unwrap_or(rest.len());
        let authority = &rest[..authority_end];
        if let Some(at) = authority.rfind('@') {
            if let Some(colon) = authority[..at].find(':') {
                let password = &authority[colon + 1..at];
                if !password.is_empty() && !password.starts_with("[REDACTED:") {
                    out.push_str(&authority[..=colon]);
                    out.push_str(&marker(KIND_CREDENTIAL));
                    out.push_str(&authority[at..]);
                    count(counts, KIND_CREDENTIAL);
                    rest = &rest[authority_end..];
                }
            }
        }
    }
    out.push_str(rest);
    out
}

/// `Bearer <credential>`: the credential, when it is long enough to be one.
fn bearer(text: &str, counts: &mut BTreeMap<&'static str, u32>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut seen = words(text).into_iter().peekable();
    while let Some((start, end)) = seen.next() {
        if !text[start..end].eq_ignore_ascii_case("bearer") || start < copied {
            continue;
        }
        let Some(&(v_start, v_end)) = seen.peek() else {
            continue;
        };
        let separated = text[end..v_start].chars().all(|c| c == ' ');
        let value = &text[v_start..v_end];
        if separated
            && value.chars().count() >= MIN_GENERIC_SECRET_CHARS
            && !value.starts_with("REDACTED")
        {
            out.push_str(&text[copied..v_start]);
            out.push_str(&marker(KIND_TOKEN));
            count(counts, KIND_TOKEN);
            copied = v_end;
        }
    }
    out.push_str(&text[copied..]);
    out
}

/// Whether `word` is a token by its shape.
fn is_shaped_token(word: &str) -> bool {
    let len = word.chars().count();
    let tail_alnum = |prefix: &str, n: usize| {
        word.strip_prefix(prefix)
            .is_some_and(|t| t.len() == n && t.chars().all(|c| c.is_ascii_alphanumeric()))
    };
    (word.starts_with("sk-") && len >= 20)
        || (["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
            .iter()
            .any(|p| word.starts_with(p))
            && len >= 30)
        || (word.starts_with("github_pat_") && len >= 30)
        || (word.starts_with("glpat-") && len >= 20)
        || (["xoxa-", "xoxb-", "xoxp-", "xoxr-", "xoxs-"]
            .iter()
            .any(|p| word.starts_with(p))
            && len >= 15)
        || tail_alnum("AKIA", 16)
        || tail_alnum("ASIA", 16)
        || tail_alnum("AIza", 35)
}

fn shaped_tokens(text: &str, counts: &mut BTreeMap<&'static str, u32>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    for (start, mut end) in words(text) {
        let word = &text[start..end];
        let jwt = word.starts_with("eyJ");
        if jwt {
            // A web token's three parts are joined by dots, which are not
            // word characters.
            end = jwt_end(text, start);
        }
        let token = &text[start..end];
        let hit = if jwt {
            token.matches('.').count() == 2 && token.len() >= 30
        } else {
            is_shaped_token(word)
        };
        if hit && start >= copied {
            out.push_str(&text[copied..start]);
            out.push_str(&marker(KIND_TOKEN));
            count(counts, KIND_TOKEN);
            copied = end;
        }
    }
    out.push_str(&text[copied..]);
    out
}

fn jwt_end(text: &str, start: usize) -> usize {
    text[start..]
        .find(|c: char| !(is_word(c) || c == '.'))
        .map_or(text.len(), |n| start + n)
}
