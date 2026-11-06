// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Whether a request states the contract its hidden check enforces.
//!
//! A request lists only the titles of the failing hidden tests. A fault can
//! break a contract no title states: a function that no longer returns the
//! count it used to, a call that no longer raises. The code then already does
//! what every title says, and the task cannot be solved from its request.
//! Such a task is refused ([`Reason::UnstatedContract`]) rather than made
//! easier: the request is never enriched.
//!
//! The decision reads only what a stored task also carries, the request and
//! the reference change (the fault reversed), so it applies unchanged to a
//! task set that already exists. It is conservative and covers two clear
//! classes in Python source, where the reference change
//!
//! * adds a `return <value>` the faulty code lacks (a truth value is not
//!   judged: a title that says "X is Y" states it): the check observes a
//!   missing result, so some title must speak of a result (return, count,
//!   total, sum, number, size, length, report, ...);
//! * adds a `raise <Error>` the faulty code lacks: the check expects an
//!   error, so some title must speak of an error (raise, error, invalid,
//!   reject, refuse, fail, ...) or name the exception.
//!
//! Any other change is not judged and is admitted.
//!
//! [`Reason::UnstatedContract`]: super::Reason::UnstatedContract

use super::instruction::symptoms_of;

/// The contract a reference change restores that no title states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unstated {
    /// A value the code no longer returns.
    ReturnValue,
    /// An error the code no longer raises, by exception name.
    RaisedError(String),
}

/// Words (by prefix) with which a title speaks of a returned result.
const RESULT_WORDS: &[&str] = &[
    "return", "count", "number", "total", "sum", "size", "length", "result", "report", "yield",
    "give", "add", "comput", "calculat", "produc", "how many",
];

/// Words (by prefix) with which a title speaks of an error.
const ERROR_WORDS: &[&str] = &[
    "raise",
    "error",
    "exception",
    "invalid",
    "reject",
    "refus",
    "fail",
    "forbid",
    "disallow",
    "unsupported",
    "illegal",
    "not allowed",
    "throw",
];

/// The contract of `reference` (the reference change, a unified diff) that
/// the titles of `request` do not state, or `None` when it states it, when
/// the change is not one of the judged classes, or when `request` is not a
/// symptom list.
#[must_use]
pub fn unstated_contract(request: &str, reference: &str) -> Option<Unstated> {
    let titles = symptoms_of(request);
    if titles == request {
        return None;
    }
    let titles = titles.to_lowercase();
    let change = python_change(reference);
    if change.returns_added > change.returns_removed && !mentions(&titles, RESULT_WORDS) {
        return Some(Unstated::ReturnValue);
    }
    if let Some(error) = change.raised.first().filter(|_| change.raises_net() > 0) {
        let named = words(error).join(" ");
        if !mentions(&titles, ERROR_WORDS) && !titles.contains(&named) {
            return Some(Unstated::RaisedError(error.clone()));
        }
    }
    None
}

fn mentions(titles: &str, stems: &[&str]) -> bool {
    stems.iter().any(|stem| titles.contains(stem))
}

/// What a change adds and removes in Python files.
#[derive(Default)]
struct PythonChange {
    returns_added: usize,
    returns_removed: usize,
    raised: Vec<String>,
    raises_removed: usize,
}

impl PythonChange {
    fn raises_net(&self) -> isize {
        self.raised.len() as isize - self.raises_removed as isize
    }
}

fn python_change(diff: &str) -> PythonChange {
    let mut change = PythonChange::default();
    let mut python = false;
    for line in diff.lines() {
        if let Some(path) = line.strip_prefix("+++ ") {
            python = path.trim().ends_with(".py");
        } else if line.starts_with("--- ") || !python {
            continue;
        } else if let Some(code) = line.strip_prefix('+') {
            if returns_value(code) {
                change.returns_added += 1;
            }
            if let Some(error) = raised_error(code) {
                change.raised.push(error);
            }
        } else if let Some(code) = line.strip_prefix('-') {
            if returns_value(code) {
                change.returns_removed += 1;
            }
            if raised_error(code).is_some() {
                change.raises_removed += 1;
            }
        }
    }
    change
}

/// Whether `code` is a `return` of a value that is neither nothing nor a
/// truth value. A truth value is what a title of the form "X is Y" or "X has
/// Y" already states, so a lost one is not an unstated contract.
fn returns_value(code: &str) -> bool {
    const TRUTH: &[&str] = &[
        "bool(", "not ", " == ", " != ", " < ", " > ", " <= ", " >= ", " in ", " and ",
    ];
    code.trim().strip_prefix("return ").is_some_and(|value| {
        let value = value.trim();
        !matches!(value, "" | "None" | "True" | "False")
            && !TRUTH
                .iter()
                .any(|t| value.contains(t) || value.starts_with(t))
    })
}

/// The exception a `raise` line names: `raise ValueError("x")` gives
/// `ValueError`.
fn raised_error(code: &str) -> Option<String> {
    let name: String = code
        .trim()
        .strip_prefix("raise ")?
        .trim()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// The lower-case words of a CamelCase or snake_case name.
fn words(name: &str) -> Vec<String> {
    let name = name.rsplit('.').next().unwrap_or(name);
    let mut out: Vec<String> = Vec::new();
    for c in name.chars() {
        if c == '_' {
            out.push(String::new());
        } else if c.is_uppercase() || out.is_empty() {
            out.push(c.to_lowercase().collect());
        } else if let Some(last) = out.last_mut() {
            last.extend(c.to_lowercase());
        }
    }
    out.retain(|w| !w.is_empty());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(titles: &[&str]) -> String {
        let list: Vec<String> = titles.iter().map(|t| format!("- {t}")).collect();
        format!(
            "Some behaviour of this repository is broken. The tests that check it are not in the \
             tree you were given; they fail on it, and each names a symptom:\n\n{}\n\nFind the \
             cause in the repository's code and fix it so that this behaviour works again.",
            list.join("\n")
        )
    }

    fn patch(file: &str, removed: &[&str], added: &[&str]) -> String {
        let mut diff = format!("--- a/{file}\n+++ b/{file}\n@@ -1,3 +1,3 @@\n");
        diff.extend(removed.iter().map(|l| format!("-{l}\n")));
        diff.extend(added.iter().map(|l| format!("+{l}\n")));
        diff
    }

    #[test]
    fn a_restored_return_value_no_title_speaks_of_is_unstated() {
        let reference = patch("ttlcache/snapshot.py", &[], &["    return len(triples)"]);
        let unfair = request(&["restore puts each entry with remaining ttl"]);
        assert_eq!(
            unstated_contract(&unfair, &reference),
            Some(Unstated::ReturnValue)
        );
        // A title that names the result states the contract.
        let fair = request(&["restore returns how many entries were put"]);
        assert_eq!(unstated_contract(&fair, &reference), None);
        // One title among several is enough.
        let mixed = request(&["restore puts each entry", "restore reports the count"]);
        assert_eq!(unstated_contract(&mixed, &reference), None);
    }

    #[test]
    fn a_changed_return_is_not_a_removed_one() {
        let reference = patch("a.py", &["    return a - b"], &["    return a + b"]);
        assert_eq!(unstated_contract(&request(&["adjusts"]), &reference), None);
        // A return of nothing is not a value.
        let bare = patch("a.py", &[], &["    return None"]);
        assert_eq!(unstated_contract(&request(&["adjusts"]), &bare), None);
    }

    #[test]
    fn a_restored_truth_value_is_not_judged() {
        let reference = patch("a.py", &[], &["    return bool(flag)", "    return a >= b"]);
        assert_eq!(
            unstated_contract(&request(&["a flag is ready"]), &reference),
            None
        );
    }

    #[test]
    fn a_restored_error_no_title_speaks_of_is_unstated() {
        let reference = patch(
            "store.py",
            &[],
            &["    if ttl < 0:", "        raise ValueError('ttl')"],
        );
        let unfair = request(&["default ttl applies", "delete removes the entry"]);
        assert_eq!(
            unstated_contract(&unfair, &reference),
            Some(Unstated::RaisedError("ValueError".into()))
        );
        for fair in [
            "negative ttl is rejected",
            "a bad ttl raises",
            "value error on negative ttl",
        ] {
            assert_eq!(unstated_contract(&request(&[fair]), &reference), None);
        }
    }

    #[test]
    fn what_is_not_python_or_not_a_request_is_not_judged() {
        let rust = patch("src/lib.rs", &[], &["    return 1"]);
        assert_eq!(unstated_contract(&request(&["adjusts"]), &rust), None);
        let python = patch("a.py", &[], &["    return 1"]);
        assert_eq!(unstated_contract("fix the bug", &python), None);
    }
}
