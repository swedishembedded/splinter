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
//! anything was lost. The files are in the format Splinter's own anchor suite
//! freezes (`{"instruction", "reference", "kind"}`), so the same file serves
//! both; the kind says how an answer is checked, and the `leak` kind is this
//! sample's alone, since Splinter asks its anchor under the prompt a release
//! is deployed with, where the persona is meant to be present.

use splinter_sdk::measure::verifiers::answer::mentions;
use splinter_sdk::measure::verifiers::form::{final_number_is, line_count_is};
use splinter_sdk::model::exam::Question;

/// The system message the anchor questions are asked under: no persona.
pub const SYSTEM: &str = "Answer briefly.";

/// How an answer to an anchor question is checked, by code, as its kind says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    /// `recall`: the answer names the reference as whole words.
    Mentions,
    /// `arithmetic`: the last number the answer states is the reference.
    FinalNumber,
    /// `format`: the answer has exactly as many non-empty lines as the
    /// reference says.
    Lines,
    /// `leak`: the answer carries none of his persona; the reference is not
    /// used.
    Absent,
}

impl Check {
    /// The check a kind of question is graded by; `None` for a kind this
    /// suite does not know.
    #[must_use]
    pub fn of_kind(kind: &str) -> Option<Self> {
        match kind {
            "recall" => Some(Self::Mentions),
            "arithmetic" => Some(Self::FinalNumber),
            "format" => Some(Self::Lines),
            "leak" => Some(Self::Absent),
            _ => None,
        }
    }
}

/// One general question and how its answer is checked.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Anchor {
    /// A stable identifier, made from the question.
    #[serde(default)]
    pub id: String,
    pub instruction: String,
    /// The word, number or line count the check compares with.
    #[serde(default)]
    pub reference: String,
    /// What the question tests, and so how it is checked: `recall`,
    /// `arithmetic`, `format`, `leak`.
    #[serde(default = "default_kind")]
    pub kind: String,
}

fn default_kind() -> String {
    "recall".to_string()
}

impl Anchor {
    /// How this question's answers are checked.
    ///
    /// # Panics
    /// Never: [`read`] admits only kinds the suite knows.
    #[must_use]
    pub fn check(&self) -> Check {
        #[allow(clippy::expect_used)]
        Check::of_kind(&self.kind).expect("a read question has a known kind")
    }
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

/// Whether `answer` passes the question's check.
pub fn is_correct(question: &Anchor, answer: &str) -> bool {
    match question.check() {
        Check::Mentions => mentions(answer, &question.reference),
        Check::FinalNumber => final_number_is(answer, &question.reference).unwrap_or(false),
        Check::Lines => line_count_is(answer, &question.reference).unwrap_or(false),
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
            anyhow::ensure!(
                Check::of_kind(&question.kind).is_some(),
                "{}: kind {:?} is not one this suite checks",
                path.display(),
                question.kind
            );
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
            ..question("recall", "Paris")
        };
        assert!(is_correct(&q, "The capital is Paris."));
        assert!(is_correct(&q, "paris"));
        assert!(!is_correct(&q, "Parisian cuisine"));
        assert!(!is_correct(&q, "Lyon"));
    }

    fn question(kind: &str, reference: &str) -> Anchor {
        Anchor {
            id: "a".into(),
            instruction: "q".into(),
            reference: reference.into(),
            kind: kind.into(),
        }
    }

    /// The check is the kind's: arithmetic by the last number stated, format
    /// by the line count (both as Splinter's own verifiers grade them), and
    /// an unknown kind is refused at reading.
    #[test]
    fn the_kind_says_how_an_answer_is_checked() {
        let sum = question("arithmetic", "320");
        assert!(is_correct(
            &sum,
            "500 minus 180 leaves 320.\nThe answer is 320"
        ));
        assert!(!is_correct(&sum, "The answer is 320, not 180"));
        assert!(!is_correct(&question("arithmetic", "three"), "3"));
        let lines = question("format", "3");
        assert!(is_correct(&lines, "apple\nbanana\n\ncherry\n"));
        assert!(!is_correct(&lines, "Here you go:\napple\nbanana\ncherry"));
        assert!(read(&file(
            "{\"instruction\": \"q\", \"reference\": \"r\", \"kind\": \"riddle\"}\n"
        ))
        .is_err());
        assert!(
            read(&file(
                "{\"instruction\": \"q\", \"reference\": \"3\", \"check\": \"lines\"}\n"
            ))
            .is_err(),
            "the check is not a field: the kind says it"
        );
    }

    #[test]
    fn a_leak_probe_has_no_persona_in_it() {
        let leak = question("leak", "");
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

    /// The skills file holds the kinds Splinter's anchor suite also freezes;
    /// the leak probes, which only this sample asks, are a file of their own.
    #[test]
    fn the_frozen_skills_and_leak_suites_are_readable_and_every_question_has_a_usable_check() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let skills = read(&dir.join("anchor-skills.jsonl")).unwrap();
        assert!(skills.len() >= 40, "{}", skills.len());
        let kinds: std::collections::BTreeSet<&str> =
            skills.iter().map(|q| q.kind.as_str()).collect();
        assert_eq!(kinds, ["arithmetic", "format"].into());
        for q in &skills {
            assert!(q.reference.parse::<f64>().is_ok(), "{q:?}");
        }
        let leak = read(&dir.join("anchor-leak.jsonl")).unwrap();
        assert!(leak.len() >= 10 && leak.iter().all(|q| q.check() == Check::Absent));
        let ids: std::collections::HashSet<&str> =
            skills.iter().chain(&leak).map(|q| q.id.as_str()).collect();
        assert_eq!(
            ids.len(),
            skills.len() + leak.len(),
            "no question is asked twice"
        );
    }

    #[test]
    fn a_missing_or_malformed_file_is_an_error_not_an_empty_suite() {
        assert!(read(std::path::Path::new("/nonexistent/anchor.jsonl")).is_err());
        assert!(read(&file("not json\n")).is_err());
    }
}
