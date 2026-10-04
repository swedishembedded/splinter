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

/// One general question and the word or words its answer must contain.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Anchor {
    /// A stable identifier, made from the question.
    #[serde(default)]
    pub id: String,
    pub instruction: String,
    pub reference: String,
}

impl Question for Anchor {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> &str {
        "anchor"
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

/// Whether `answer` names the reference as whole words.
pub fn is_correct(question: &Anchor, answer: &str) -> bool {
    mentions(answer, &question.reference)
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
    fn questions_are_read_with_a_stable_id_and_the_extra_fields_of_the_suite_are_ignored() {
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
            ("anchor", "anchor", "Paris")
        );
    }

    #[test]
    fn an_answer_is_right_when_it_names_the_reference_as_a_whole_word() {
        let q = Anchor {
            id: "a".into(),
            instruction: "Capital of France?".into(),
            reference: "Paris".into(),
        };
        assert!(is_correct(&q, "The capital is Paris."));
        assert!(is_correct(&q, "paris"));
        assert!(!is_correct(&q, "Parisian cuisine"));
        assert!(!is_correct(&q, "Lyon"));
    }

    #[test]
    fn a_missing_or_malformed_file_is_an_error_not_an_empty_suite() {
        assert!(read(std::path::Path::new("/nonexistent/anchor.jsonl")).is_err());
        assert!(read(&file("not json\n")).is_err());
    }
}
