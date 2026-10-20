// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a solver is told of a task built from a fault: the symptom, in the
//! words of the tests that catch it, and nothing of the cause.
//!
//! The request is composed by code from a constant, never written by a
//! model that has seen the answer, because a helper-written request
//! restates the answer by paraphrase. What is taken from the repository is
//! the name of each failing test and, where a test has a short docstring,
//! that docstring. A name or docstring of fewer than [`MIN_SYMPTOM_WORDS`]
//! words (`databricks`, `time`, `cast`) says nothing a solver can act on, so
//! the symptom is then rendered from what the failed check observed.
//!
//! # What an assertion may show, and what it may not
//!
//! Shown: the exception's own message as the test runner printed it, when it
//! is an equality (`first != second`, the two values, each at most
//! [`MAX_VALUE_CHARS`] characters on one line) or an error the code under
//! test raised (`Type: message`). These are behaviour: what the code returned
//! and what it was required to return, as an issue report would say.
//!
//! Never shown: a traceback frame (file paths and the lines of the test that
//! ran, which are the hidden test's code), the failure's diff lines, the
//! test's file path, or the assertion's call. A failure whose message is
//! neither of the two shapes cannot be rendered, and a task whose symptom
//! needs it is refused ([`Unactionable`]).
//!
//! The values come from the hidden test's literals, so they can repeat a run
//! of its text. They are the one disclosure the task makes on purpose: the
//! request carries them in [`Request::shown`], the admission check masks them
//! before it looks for a quotation of the hidden tests, and the exposure
//! audit counts the request as text the solver may see.
//!
//! The test command the request names is the one the solver can run: a
//! module the tests of this task live in is not in the tree, so it is left
//! out of the command.

use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
use crate::tasks::dedup::Seen;
use splinter_core::selfcontained::MIN_QUOTED_WORDS;

/// The most symptoms an instruction lists; the rest are counted.
const MAX_SYMPTOMS: usize = 12;

/// The fewest words a symptom label needs to tell a solver anything.
pub const MIN_SYMPTOM_WORDS: usize = 3;

/// The most characters of one value an assertion's message may show.
pub const MAX_VALUE_CHARS: usize = 300;

/// A request, and the values of the hidden checks it shows on purpose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Request {
    pub(super) text: String,
    /// The assertion values the request quotes (see the module docs).
    pub(super) shown: Vec<String>,
}

impl Request {
    /// The request with each shown value replaced by a placeholder: what the
    /// quotation check reads, so that only text beyond the disclosed values
    /// can count as quoting a hidden test.
    pub(super) fn without_values(&self) -> String {
        let mut text = self.text.clone();
        for value in &self.shown {
            text = text.replace(value, "<value>");
        }
        text
    }
}

/// A failing test whose name says too little and whose assertion cannot be
/// rendered as behaviour.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Unactionable {
    pub(super) test: String,
}

/// The request for a fault whose symptoms are `failing`.
///
/// `test_command` is how the repository's tests are run, `hidden` the files
/// taken out of the solver's tree, `docs` the first docstring line of the
/// tests that have one and `messages` the exception message of each failed
/// test.
pub(super) fn compose(
    test_command: &[String],
    hidden: &BTreeSet<String>,
    failing: &BTreeSet<String>,
    docs: &BTreeMap<String, String>,
    messages: &BTreeMap<String, String>,
) -> Result<Request, Unactionable> {
    let mut shown = Vec::new();
    let mut symptoms = BTreeSet::new();
    for id in failing {
        let doc = docs.get(id).map(String::as_str);
        let mut label = symptom(id, doc);
        if label.split_whitespace().count() < MIN_SYMPTOM_WORDS {
            // A short docstring does not outrank a name that says more.
            let named = symptom(id, None);
            if named.split_whitespace().count() >= MIN_SYMPTOM_WORDS {
                label = named;
            }
        }
        if label.split_whitespace().count() >= MIN_SYMPTOM_WORDS {
            symptoms.insert(label);
            continue;
        }
        let rendered = messages
            .get(id)
            .and_then(|message| observed(message))
            .ok_or_else(|| Unactionable { test: id.clone() })?;
        shown.extend(rendered.values);
        symptoms.insert(format!("{label}: {}", rendered.text));
    }
    let mut listed: Vec<String> = symptoms
        .iter()
        .take(MAX_SYMPTOMS)
        .map(|s| format!("- {s}"))
        .collect();
    if symptoms.len() > MAX_SYMPTOMS {
        listed.push(format!(
            "- and {} more checks of the same behaviour",
            symptoms.len() - MAX_SYMPTOMS
        ));
    }
    let text = format!(
        "Some behaviour of this repository is broken. The tests that check it are not in the \
         tree you were given; they fail on it, and each names a symptom:\n\n{}\n\nFind the cause \
         in the repository's code and fix it so that this behaviour works again. The tests that \
         are in the tree must keep passing and are not to be edited; they are run with `{}`.",
        listed.join("\n"),
        visible_command(test_command, hidden).join(" ")
    );
    Ok(Request { text, shown })
}

/// `command` without the test modules the solver was not given: for a
/// `unittest` run, each dotted module argument whose file is in `hidden`.
/// Any other command is the project's own and is returned whole.
fn visible_command(command: &[String], hidden: &BTreeSet<String>) -> Vec<String> {
    let Some(at) = command.iter().position(|a| a == "unittest") else {
        return command.to_vec();
    };
    command
        .iter()
        .enumerate()
        .filter(|(i, arg)| {
            *i <= at
                || arg.starts_with('-')
                || !hidden.contains(&format!("{}.py", arg.replace('.', "/")))
        })
        .map(|(_, arg)| arg.clone())
        .collect()
}

/// What a failed check observed, as words, with the values it quotes.
struct Observed {
    text: String,
    values: Vec<String>,
}

/// The behaviour a failure's exception message states, or `None` when it is
/// neither an equality nor an error the code raised.
fn observed(message: &str) -> Option<Observed> {
    let first = message.lines().next()?.trim();
    if let Some(rest) = first.strip_prefix("AssertionError: ") {
        let (left, right) = split_inequality(rest)?;
        let (left, right) = (value(left)?, value(right)?);
        return Some(Observed {
            text: format!("two values that must be equal are `{left}` and `{right}`"),
            values: vec![left, right],
        });
    }
    let (kind, text) = first.split_once(": ")?;
    let is_error = kind.rsplit('.').next().is_some_and(|name| {
        name.ends_with("Error") || name.ends_with("Exception") || name.ends_with("Warning")
    });
    if !is_error || kind.starts_with("AssertionError") || text.trim().is_empty() {
        return None;
    }
    let shown = value(text)?;
    Some(Observed {
        text: format!("raises `{kind}: {shown}`"),
        values: vec![shown],
    })
}

/// One value, on one line and short enough to be a value and not a listing;
/// `None` for an empty one.
fn value(text: &str) -> Option<String> {
    let text = text.trim();
    if text.is_empty() || text.chars().count() > MAX_VALUE_CHARS || text.contains('`') {
        return None;
    }
    Some(text.to_string())
}

/// `first != second` split at the ` != ` that is outside every quotation and
/// bracket, so that a value which contains the operator is kept whole.
fn split_inequality(text: &str) -> Option<(&str, &str)> {
    let mut depth = 0_i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (at, c) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match (quote, c) {
            (Some(_), '\\') => escaped = true,
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '\'' | '"') => quote = Some(c),
            (None, '(' | '[' | '{') => depth += 1,
            (None, ')' | ']' | '}') => depth -= 1,
            _ => {}
        }
        if quote.is_none() && depth == 0 && text[at..].starts_with(" != ") {
            return Some((&text[..at], &text[at + 4..]));
        }
    }
    None
}

/// The part of a request that says what is broken: its list of symptoms.
/// The rest of every request is the same sentence and the project's test
/// command, which can be hundreds of words, so two faults of one project
/// would overlap almost completely when whole requests are compared. A text
/// that is not a request is returned whole.
pub(super) fn symptoms_of(request: &str) -> &str {
    const OPENS: &str = "each names a symptom:\n\n";
    const CLOSES: &str = "\n\nFind the cause";
    let Some((_, after)) = request.split_once(OPENS) else {
        return request;
    };
    after.split_once(CLOSES).map_or(after, |(list, _)| list)
}

/// What one failing test says, in words: its docstring when it has a short
/// one, else its name.
fn symptom(id: &str, doc: Option<&str>) -> String {
    if let Some(doc) = doc.map(|d| d.trim().trim_end_matches('.')) {
        let words = doc.split_whitespace().count();
        if words > 0 && words < MIN_QUOTED_WORDS {
            return doc.to_string();
        }
    }
    let name = id.rsplit("::").next().unwrap_or(id);
    let name = name.rsplit('.').next().unwrap_or(name);
    let stem = name
        .strip_prefix("test_")
        .or_else(|| name.strip_prefix("test"))
        .filter(|s| !s.is_empty())
        .unwrap_or(name);
    stem.replace('_', " ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> (
        BTreeSet<String>,
        BTreeMap<String, String>,
        BTreeMap<String, String>,
    ) {
        (BTreeSet::new(), BTreeMap::new(), BTreeMap::new())
    }

    fn command(modules: &[&str]) -> Vec<String> {
        ["python3", "-m", "unittest"]
            .into_iter()
            .map(String::from)
            .chain(modules.iter().map(|m| (*m).to_string()))
            .collect()
    }

    #[test]
    fn a_symptom_is_a_short_docstring_or_the_name_in_words() {
        assert_eq!(
            symptom("t.A.test_empty_list_sums_to_zero", None),
            "empty list sums to zero"
        );
        assert_eq!(
            symptom("tests/total.rs::prices_are_added_up", None),
            "prices are added up"
        );
        assert_eq!(symptom("t.A.test_x", Some("Adds one.")), "Adds one");
        // A docstring long enough to be a quotation of the test is not used.
        let long = "This docstring has more than enough words to be a quoted passage";
        assert_eq!(symptom("t.A.test_long_one", Some(long)), "long one");
    }

    #[test]
    fn the_command_never_names_a_module_the_solver_was_not_given() {
        let (hidden, docs, messages) = none();
        let hidden: BTreeSet<String> = ["tests/dialects/test_teradata.py".to_string()]
            .into_iter()
            .chain(hidden)
            .collect();
        let failing: BTreeSet<String> = ["t.A.test_empty_list_sums_to_zero".to_string()].into();
        let request = compose(
            &command(&[
                "tests.test_a",
                "tests.dialects.test_teradata",
                "tests.test_b",
            ]),
            &hidden,
            &failing,
            &docs,
            &messages,
        )
        .unwrap();
        assert!(
            request
                .text
                .contains("`python3 -m unittest tests.test_a tests.test_b`"),
            "{}",
            request.text
        );
        assert!(!request.text.contains("teradata"));
        // A command that is not a list of modules is the project's own.
        let own = vec!["cargo".to_string(), "test".to_string()];
        let text = compose(&own, &hidden, &failing, &docs, &messages)
            .unwrap()
            .text;
        assert!(text.contains("`cargo test`"));
    }

    #[test]
    fn a_label_too_short_to_act_on_is_rendered_from_what_the_assertion_observed() {
        let failing: BTreeSet<String> = ["t.A.test_databricks".to_string()].into();
        let messages: BTreeMap<String, String> = [(
            "t.A.test_databricks".to_string(),
            "AssertionError: 'SELECT A FROM t' != 'SELECT a FROM t'\n- SELECT A FROM t\n+ SELECT a FROM t"
                .to_string(),
        )]
        .into();
        let request = compose(
            &command(&[]),
            &BTreeSet::new(),
            &failing,
            &BTreeMap::new(),
            &messages,
        )
        .unwrap();
        assert!(
            request.text.contains(
                "- databricks: two values that must be equal are `'SELECT A FROM t'` and `'SELECT a FROM t'`"
            ),
            "{}",
            request.text
        );
        // The diff lines, the exception name and any frame are not shown.
        assert!(!request.text.contains("AssertionError"));
        assert_eq!(request.shown, ["'SELECT A FROM t'", "'SELECT a FROM t'"]);
        assert!(!request.without_values().contains("SELECT"));
    }

    #[test]
    fn an_error_the_code_raised_is_a_symptom_and_a_frame_never_is() {
        let failing: BTreeSet<String> = ["t.A.test_cast".to_string()].into();
        let raised = |m: &str| -> BTreeMap<String, String> {
            [("t.A.test_cast".to_string(), m.to_string())].into()
        };
        let ok = compose(
            &command(&[]),
            &BTreeSet::new(),
            &failing,
            &BTreeMap::new(),
            &raised("sqlglot.errors.ParseError: Expecting ) near 'x'"),
        )
        .unwrap();
        assert!(ok
            .text
            .contains("cast: raises `sqlglot.errors.ParseError: Expecting ) near 'x'`"));
        // Neither shape: a bare assertion, a truthiness check, an empty value.
        for message in ["AssertionError", "AssertionError: False is not true", ""] {
            assert!(
                compose(
                    &command(&[]),
                    &BTreeSet::new(),
                    &failing,
                    &BTreeMap::new(),
                    &raised(message)
                )
                .is_err(),
                "{message:?}"
            );
        }
        // A value that is a listing, not a value, is not shown.
        let long = format!("AssertionError: '{}' != 'b'", "a".repeat(MAX_VALUE_CHARS));
        assert!(compose(
            &command(&[]),
            &BTreeSet::new(),
            &failing,
            &BTreeMap::new(),
            &raised(&long)
        )
        .is_err());
    }

    #[test]
    fn a_failing_test_without_a_message_and_a_short_label_is_refused_and_a_clear_name_is_not() {
        let short: BTreeSet<String> = ["t.A.test_time".to_string()].into();
        let refused = compose(
            &command(&[]),
            &BTreeSet::new(),
            &short,
            &BTreeMap::new(),
            &BTreeMap::new(),
        );
        assert_eq!(
            refused,
            Err(Unactionable {
                test: "t.A.test_time".into()
            })
        );
        let clear: BTreeSet<String> = ["t.A.test_empty_list_sums_to_zero".to_string()].into();
        let request = compose(
            &command(&[]),
            &BTreeSet::new(),
            &clear,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(request.text.contains("- empty list sums to zero\n"));
        assert!(request.shown.is_empty());
    }

    #[test]
    fn an_equality_whose_value_contains_the_operator_is_split_at_the_right_place() {
        assert_eq!(
            split_inequality("'a != b' != 'c'"),
            Some(("'a != b'", "'c'"))
        );
        assert_eq!(
            split_inequality("[1, (2 != 3)] != []"),
            Some(("[1, (2 != 3)]", "[]"))
        );
        assert_eq!(split_inequality("no operator"), None);
    }

    #[test]
    fn two_different_faults_are_not_near_duplicates_because_they_share_a_long_command() {
        let modules: Vec<String> = (0..55).map(|n| format!("tests.test_module_{n}")).collect();
        let modules: Vec<&str> = modules.iter().map(String::as_str).collect();
        let command = command(&modules);
        let request = |test: &str| {
            let failing: BTreeSet<String> = [format!("t.A.{test}")].into();
            compose(
                &command,
                &BTreeSet::new(),
                &failing,
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .unwrap()
            .text
        };
        let (first, second) = (
            request("test_logarithm_of_one"),
            request("test_round_half_even"),
        );
        assert_eq!(symptoms_of(&first), "- logarithm of one");

        let mut seen = Seen::new(
            crate::tasks::DEFAULT_SHINGLE_WORDS,
            crate::tasks::DEFAULT_MAX_OVERLAP,
        );
        seen.admit(symptoms_of(&first));
        assert!(seen.repeats(symptoms_of(&second)).is_none());
        assert!(seen.repeats(symptoms_of(&first)).is_some());
        // Compared whole, the shared sentence and command make them repeats.
        let mut whole = Seen::new(
            crate::tasks::DEFAULT_SHINGLE_WORDS,
            crate::tasks::DEFAULT_MAX_OVERLAP,
        );
        whole.admit(&first);
        assert!(whole.repeats(&second).is_some());
        assert_eq!(symptoms_of("not a request"), "not a request");
    }

    #[test]
    fn the_request_lists_symptoms_and_never_the_failure_output() {
        let failing: BTreeSet<String> = ["t.A.test_b_second_thing", "t.A.test_a_first_thing"]
            .map(String::from)
            .into();
        let request = compose(
            &command(&[]),
            &BTreeSet::new(),
            &failing,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(request.text.contains("- a first thing\n- b second thing"));
        assert!(request.text.contains("`python3 -m unittest`"));
        assert!(!request.text.contains("Traceback") && !request.text.contains("AssertionError"));
    }
}
