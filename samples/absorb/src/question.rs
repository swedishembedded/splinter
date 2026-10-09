// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements question writing for checkable facts, with
// every question admitted by code, for its clients. If your team needs
// expertise in generating test questions a model cannot answer by echoing
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The one thing a model is asked when a fact is selected: to word a natural
//! question whose answer is a sentence's specific.
//!
//! Code admits the question or sends it back: it asks for the specific and
//! does not contain any key of it, it copies no stretch of the sentence, it is
//! a question the answer can fill (not yes or no), and it names its subject
//! instead of pointing at the letter.

use std::time::Duration;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::schemars::{self, JsonSchema};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::typed::TypedCall;
use splinter_sdk::measure::verifiers::quotation::words;
use splinter_sdk::Context;

use crate::keys::present;
use crate::probes::shares_a_run;
use crate::sentences::Candidate;

/// The longest a question may be, in characters.
pub const MAX_QUESTION_CHARS: usize = 300;

/// How long writing one question may take, every correction included and the
/// first call of a phase loading the model.
const WRITE_DEADLINE: Duration = Duration::from_secs(3600);

/// Output tokens one call may run to.
const WRITE_MAX_OUTPUT_TOKENS: u64 = 256;

/// Corrections a rejected question is sent back for.
const REPAIRS: u32 = 1;

const ROLE: &str = "You write one clear question about a single sentence, for someone who is \
checking whether a person knows what it says.";

const TASK: &str = "Write one natural question whose answer is `answer` and that `sentence` \
answers. The sentence is from a letter written by Thomas Jefferson, and the question is put to \
him: say \"you\" for him (for example \"Which port did you ask Mr. Barclay to meet you at?\"), \
never \"the writer\", \"the author\" or \"he\". Make the question make sense to someone who has \
never seen the sentence: name the people, places and things it concerns, and never point at \
\"the letter\", \"the passage\", \"this\" or \"he\". `heading` says whom the letter is to and where \
and when it was written, and may help you name them. Never put any of `answer_parts` in the \
question, never copy a stretch of `sentence`, and start with a question word (who, what, which, \
where, when, how many, how much, why), not with a verb. End with one question mark.";

/// What the writer is shown.
#[derive(Serialize)]
struct Shown<'a> {
    sentence: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    heading: Option<&'a str>,
    answer: &'a str,
    answer_parts: Vec<&'a str>,
}

/// The question the writer returns.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "schemars")]
struct Written {
    /// The question.
    question: String,
}

/// Phrases that point at the letter instead of naming the subject.
const POINTING: &[&str] = &[
    "the letter",
    "this letter",
    "the passage",
    "the sentence",
    "the text",
    "the excerpt",
    "the document",
    "the author",
    "the writer",
];

/// Pronouns that stand for someone the question has not named.
const UNNAMED: &[&str] = &[
    "he", "she", "him", "they", "them", "we", "us", "this", "these", "those",
];

/// Words that point at a text instead of asking for the fact.
const TEXTUAL: &[&str] = &[
    "mentioned",
    "described",
    "stated",
    "referred",
    "discussed",
    "implied",
];

/// Words that start a question a bare yes or no would answer.
const YES_NO_OPENERS: &[&str] = &[
    "is", "was", "were", "are", "did", "do", "does", "had", "has", "have", "can", "could", "would",
    "will", "should", "shall", "may", "might",
];

/// Why `question` may not be asked for `candidate`; `Ok` when it may.
///
/// # Errors
/// The reason, worded as the correction sent back to the writer.
pub fn check(question: &str, candidate: &Candidate, persona: &str) -> Result<(), String> {
    let chars = question.chars().count();
    if chars == 0 || chars > MAX_QUESTION_CHARS {
        return Err(format!(
            "write between 1 and {MAX_QUESTION_CHARS} characters"
        ));
    }
    if question.matches('?').count() != 1 || !question.trim_end().ends_with('?') {
        return Err("write exactly one question, ending with a question mark".into());
    }
    if let Some(key) = candidate
        .specific
        .keys
        .iter()
        .find(|k| present(question, k))
    {
        return Err(format!(
            "the question must not contain {:?}: the answer is what holds it",
            key.text
        ));
    }
    if shares_a_run(question, &candidate.sentence) {
        return Err("do not copy a stretch of the sentence into the question".into());
    }
    let lowered = words(question);
    let padded = format!(" {} ", lowered.join(" "));
    if let Some(pointing) = POINTING.iter().find(|p| padded.contains(&format!(" {p} "))) {
        return Err(if matches!(*pointing, "the writer" | "the author") {
            format!(
                "do not say {pointing:?}: say \"you\" for Jefferson, to whom the question is put"
            )
        } else {
            format!("do not say {pointing:?}: name what the question is about")
        });
    }
    if let Some(word) = lowered.iter().find(|w| TEXTUAL.contains(&w.as_str())) {
        return Err(format!(
            "do not ask what was {word}: ask for the fact itself"
        ));
    }
    let own: Vec<String> = persona.split_whitespace().map(str::to_lowercase).collect();
    let anchored = question.split_whitespace().skip(1).any(|raw| {
        let word = raw.trim_matches(|c: char| !c.is_alphanumeric());
        word.chars().count() > 2
            && word.chars().next().is_some_and(char::is_uppercase)
            && !own.contains(&word.to_lowercase())
    });
    if !anchored {
        return Err("name the person, place or thing the question is about".into());
    }
    if let Some(pronoun) = lowered.iter().find(|w| UNNAMED.contains(&w.as_str())) {
        return Err(format!("do not say {pronoun:?}: name the person or thing"));
    }
    if lowered
        .first()
        .is_some_and(|w| YES_NO_OPENERS.contains(&w.as_str()))
    {
        return Err(
            "start with a question word, so that the answer is a fact, not yes or no".into(),
        );
    }
    Ok(())
}

/// The question the generator writes for `candidate`.
///
/// # Errors
/// The generator cannot run, or its question cannot be admitted within the
/// corrections allowed.
pub fn write(
    ctx: &Context,
    generator: &Model,
    candidate: &Candidate,
    persona: &str,
) -> anyhow::Result<String> {
    let (checked, persona) = (candidate.clone(), persona.to_string());
    let call = TypedCall::<Written>::new("write_question", TASK, ROLE, WRITE_DEADLINE)
        .max_output_tokens(WRITE_MAX_OUTPUT_TOKENS)
        .repairs(REPAIRS)
        .postcondition(move |w: &Written| check(w.question.trim(), &checked, &persona));
    let reply = ctx
        .block_on(
            call.run(
                generator,
                &Shown {
                    sentence: &candidate.sentence,
                    heading: candidate.heading.as_deref(),
                    answer: &candidate.specific.answer,
                    answer_parts: candidate
                        .specific
                        .keys
                        .iter()
                        .map(|k| k.text.as_str())
                        .collect(),
                },
            ),
        )
        .with_context(|| format!("writing a question for {:?}", candidate.specific.answer))?;
    Ok(reply.question.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{Key, KeyKind};
    use crate::sentences::{Specific, SpecificKind};

    const PERSONA: &str = "Thomas Jefferson";

    fn candidate() -> Candidate {
        Candidate {
            family: "f.txt".into(),
            sentence: "In 1785 Mr. Short sailed from Paris with the books that he had bought for \
                       the library of Mr. Madison."
                .into(),
            specific: Specific {
                kind: SpecificKind::Year,
                answer: "1785".into(),
                keys: vec![Key {
                    text: "1785".into(),
                    kind: KeyKind::Number,
                }],
            },
            heading: None,
        }
    }

    #[test]
    fn a_question_asks_for_the_specific_and_names_its_subject() {
        let c = candidate();
        assert_eq!(
            check(
                "In which year did Mr. Short leave Paris with Madison's books?",
                &c,
                PERSONA
            ),
            Ok(())
        );
        for (question, why) in [
            (
                "When did Short sail from Paris in 1785?",
                "must not contain",
            ),
            ("When did Short sail?\nWhy?", "exactly one question"),
            ("When did Short sail", "exactly one question"),
            ("Did Mr. Short sail from Paris?", "yes or no"),
            ("What did the letter say about Short?", "the letter"),
            ("When did he leave Paris with the books?", "name the person"),
            (
                "What was mentioned about Short in Paris?",
                "ask for the fact",
            ),
            (
                "Which year did the minister leave with the books?",
                "name the person, place",
            ),
            (
                "Who sailed from Paris with the books that he had bought for the library?",
                "copy",
            ),
        ] {
            let error = check(question, &c, PERSONA).expect_err(why);
            assert!(error.contains(why), "{question}: {error}");
        }
    }

    #[test]
    fn a_figure_written_as_a_word_is_given_away_too() {
        let mut c = candidate();
        c.specific.keys = vec![Key {
            text: "forty".into(),
            kind: KeyKind::Number,
        }];
        assert!(check("Did he pay 40 dollars?", &c, PERSONA).is_err());
        assert!(check(
            "How many dollars did Mr. Short's clerk cost per month?",
            &c,
            PERSONA
        )
        .is_ok());
    }
}
