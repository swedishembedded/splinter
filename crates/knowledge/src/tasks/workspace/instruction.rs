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
//! restates the answer by paraphrase. The only text taken from the
//! repository is the name of each failing test and, where a test has a
//! short docstring, that docstring; the failing output is not quoted, since
//! a traceback prints the source line that was changed.

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::selfcontained::MIN_QUOTED_WORDS;

/// The most symptoms an instruction lists; the rest are counted.
const MAX_SYMPTOMS: usize = 12;

/// The request for a fault whose symptoms are `failing`.
///
/// `test_command` is how the repository's tests are run, `docs` the first
/// docstring line of the tests that have one.
pub(super) fn compose(
    test_command: &[String],
    failing: &BTreeSet<String>,
    docs: &BTreeMap<String, String>,
) -> String {
    let symptoms: BTreeSet<String> = failing
        .iter()
        .map(|id| symptom(id, docs.get(id).map(String::as_str)))
        .collect();
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
    format!(
        "Some behaviour of this repository is broken. The tests that check it are not in the \
         tree you were given; they fail on it, and each names a symptom:\n\n{}\n\nFind the cause \
         in the repository's code and fix it so that this behaviour works again. The tests that \
         are in the tree must keep passing and are not to be edited; they are run with `{}`.",
        listed.join("\n"),
        test_command.join(" ")
    )
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
    fn the_request_lists_symptoms_and_never_the_failure_output() {
        let failing: BTreeSet<String> = ["t.A.test_b_second", "t.A.test_a_first"]
            .map(String::from)
            .into();
        let text = compose(
            &["python3".into(), "-m".into(), "unittest".into()],
            &failing,
            &BTreeMap::new(),
        );
        assert!(text.contains("- a first\n- b second"));
        assert!(text.contains("`python3 -m unittest`"));
        assert!(!text.contains("Traceback") && !text.contains("AssertionError"));
    }
}
