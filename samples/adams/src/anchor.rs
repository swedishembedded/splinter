// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retention checks that show whether teaching a
// model something new made it forget what it knew, for its clients. If your
// team needs expertise in catching forgetting before a fine-tune is released,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The anchor suite: general questions with one right answer, asked of a model
//! with no persona, before and after training.
//!
//! A fine-tune on one man's papers can cost the model what it knew about the
//! rest of the world. These questions are frozen and have nothing to do with
//! him; the same questions asked of the base and of the adapter say whether
//! anything was lost.

use splinter_sdk::measure::verifiers::answer::mentions;
use splinter_sdk::model::exam::Question;

/// The system message the anchor questions are asked under: no persona.
pub const SYSTEM: &str = "Answer briefly.";

/// How an answer to an anchor question is checked, by code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Check {
    /// The answer names the reference as whole words.
    #[default]
    Mentions,
    /// The last number the answer states is the reference.
    FinalNumber,
    /// The answer has exactly as many non-empty lines as the reference says.
    Lines,
    /// The answer carries none of his persona: the reference is not used.
    Absent,
}

/// One general question and how its answer is checked.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Anchor {
    /// A stable identifier, made from the question.
    #[serde(default)]
    pub id: String,
    pub instruction: String,
    /// The word, number or line count the check compares with.
    #[serde(default)]
    pub reference: String,
    /// What the question tests: `recall`, `arithmetic`, `format`, `leak`.
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub check: Check,
}

fn default_kind() -> String {
    "recall".to_string()
}

impl Question for Anchor {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> &str {
        &self.kind
    }
    fn split(&self) -> &str {
        "anchor"
    }
    fn reference(&self) -> &str {
        &self.reference
    }
    fn prompt(&self) -> &str {
        &self.instruction
    }
}

/// What only his persona says: an answer to a general question that has any of
/// these has leaked it.
const PERSONA_MARKERS: [&str; 5] = [
    "applicability:",
    "grounding:",
    "my papers",
    "samuel adams",
    "my correspondent",
];

/// The numbers in `text`, in order, with thousands separators read.
fn numbers(text: &str) -> Vec<f64> {
    let mut found = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        let negative_start =
            c == '-' && current.is_empty() && chars.get(i + 1).is_some_and(char::is_ascii_digit);
        let inside = c.is_ascii_digit()
            || (c == '.'
                && !current.is_empty()
                && chars.get(i + 1).is_some_and(char::is_ascii_digit))
            || (c == ','
                && !current.is_empty()
                && chars.get(i + 1).is_some_and(char::is_ascii_digit));
        if negative_start || inside {
            if c != ',' {
                current.push(c);
            }
        } else if !current.is_empty() {
            found.extend(current.parse::<f64>());
            current.clear();
        }
    }
    if !current.is_empty() {
        found.extend(current.parse::<f64>());
    }
    found
}

/// Whether `answer` passes the question's check.
pub fn is_correct(question: &Anchor, answer: &str) -> bool {
    match question.check {
        Check::Mentions => mentions(answer, &question.reference),
        Check::FinalNumber => match (numbers(answer).last(), question.reference.parse::<f64>()) {
            (Some(stated), Ok(wanted)) => (stated - wanted).abs() < 1e-9,
            _ => false,
        },
        Check::Lines => question
            .reference
            .parse::<usize>()
            .is_ok_and(|wanted| answer.lines().filter(|l| !l.trim().is_empty()).count() == wanted),
        Check::Absent => {
            let lower = answer.to_lowercase();
            !PERSONA_MARKERS.iter().any(|m| lower.contains(m))
        }
    }
}

/// The anchor questions of a JSON-lines file, each with a stable id.
///
/// # Errors
/// The file cannot be read or a line is not a question.
pub fn read(path: &std::path::Path) -> anyhow::Result<Vec<Anchor>> {
    let text =
        std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut question: Anchor = serde_json::from_str(line)
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
            question.id = format!(
                "anchor-{}",
                &blake3::hash(question.instruction.as_bytes()).to_hex()[..12]
            );
            Ok(question)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(lines: &str) -> std::path::PathBuf {
        let path = tempfile::tempdir().unwrap().keep().join("anchor.jsonl");
        std::fs::write(&path, lines).unwrap();
        path
    }

    #[test]
    fn questions_are_read_with_a_stable_id_and_a_kind_that_defaults_to_recall() {
        let path = file("{\"instruction\": \"What is the capital of France?\", \"reference\": \"Paris\", \"kind\": \"recall\"}\n\n{\"instruction\": \"What is the capital of Japan?\", \"reference\": \"Tokyo\"}\n");
        let questions = read(&path).unwrap();
        assert_eq!(questions.len(), 2);
        assert!(questions[0].id.starts_with("anchor-") && questions[0].id != questions[1].id);
        assert_eq!(
            read(&path).unwrap(),
            questions,
            "the ids are the same every time"
        );
        assert_eq!(
            (
                questions[0].kind(),
                questions[0].split(),
                questions[0].reference()
            ),
            ("recall", "anchor", "Paris")
        );
    }

    #[test]
    fn an_answer_is_right_when_it_names_the_reference_as_a_whole_word() {
        let q = Anchor {
            instruction: "Capital of France?".into(),
            ..question("recall", Check::Mentions, "Paris")
        };
        assert!(is_correct(&q, "The capital is Paris."));
        assert!(is_correct(&q, "paris"));
        assert!(!is_correct(&q, "Parisian cuisine"));
        assert!(!is_correct(&q, "Lyon"));
    }

    fn question(kind: &str, check: Check, reference: &str) -> Anchor {
        Anchor {
            id: "a".into(),
            instruction: "q".into(),
            reference: reference.into(),
            kind: kind.into(),
            check,
        }
    }

    #[test]
    fn an_arithmetic_answer_is_right_when_the_last_number_it_states_is_the_reference() {
        let q = question("arithmetic", Check::FinalNumber, "320");
        assert!(is_correct(
            &q,
            "5 dollars is 500 cents, minus 180 leaves 320.\nThe answer is 320"
        ));
        assert!(is_correct(&q, "So the change is 320.0 cents."));
        assert!(is_correct(
            &question("arithmetic", Check::FinalNumber, "1250"),
            "The total is 1,250."
        ));
        assert!(
            !is_correct(&q, "The answer is 320, not 180"),
            "the last number is what counts"
        );
        assert!(!is_correct(&q, "I cannot tell."));
        assert!(is_correct(
            &question("arithmetic", Check::FinalNumber, "-4"),
            "That gives -4."
        ));
    }

    #[test]
    fn a_format_answer_has_exactly_the_lines_asked_for_and_a_leak_probe_has_no_persona_in_it() {
        let lines = question("format", Check::Lines, "3");
        assert!(is_correct(&lines, "apple\nbanana\n\ncherry\n"));
        assert!(!is_correct(&lines, "Here you go:\napple\nbanana\ncherry"));
        let leak = question("leak", Check::Absent, "");
        assert!(is_correct(
            &leak,
            "Hold a vote, then write the decision down."
        ));
        for persona in [
            "Applicability: APPLIES\nAct.",
            "Grounding:\n- x",
            "As my papers show, I would write.",
            "I, Samuel Adams, advise it.",
        ] {
            assert!(!is_correct(&leak, persona), "{persona}");
        }
    }

    #[test]
    fn the_frozen_skills_suite_is_readable_and_every_question_has_a_usable_check() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("anchor-skills.jsonl");
        let questions = read(&path).unwrap();
        assert!(questions.len() >= 50, "{}", questions.len());
        let kinds: std::collections::BTreeSet<&str> =
            questions.iter().map(|q| q.kind.as_str()).collect();
        assert_eq!(kinds, ["arithmetic", "format", "leak"].into());
        for q in &questions {
            match q.check {
                Check::FinalNumber | Check::Lines => {
                    assert!(q.reference.parse::<f64>().is_ok(), "{q:?}")
                }
                Check::Absent | Check::Mentions => {}
            }
        }
        let ids: std::collections::HashSet<&str> =
            questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids.len(), questions.len(), "no question is asked twice");
    }

    #[test]
    fn a_missing_or_malformed_file_is_an_error_not_an_empty_suite() {
        assert!(read(std::path::Path::new("/nonexistent/anchor.jsonl")).is_err());
        assert!(read(&file("not json\n")).is_err());
    }
}
