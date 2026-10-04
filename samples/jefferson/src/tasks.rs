// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded task generation for
// language-model fine-tuning for its clients. If your team needs expertise in
// compiling verifiable training tasks from a text corpus, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The checkable tasks: who a letter was written to, in what year, and which
//! work a passage comes from. Each has an exact reference built from the
//! source by code, so no model writes a question or an answer and nothing a
//! model says is trusted.

use serde::{Deserialize, Serialize};

use crate::corpus::{is_exam_family, without_salutation, Letter};
use crate::works::{Passage, WORKS};

/// The system message every training record and every exam question carries:
/// the voice, and the rule that a claim about what he wrote has a source.
pub const PERSONA: &str =
    "You are Thomas Jefferson. Answer in the first person. Say plainly what you cannot source.";

/// What a task asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// Who a letter was written to.
    Recipient,
    /// In what year a letter was written.
    Year,
    /// Which of the listed works a passage comes from.
    Work,
}

impl Kind {
    /// The kind's name as an exam record carries it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Kind::Recipient => "recipient",
            Kind::Year => "year",
            Kind::Work => "work",
        }
    }
}

/// One task: the question, the reference answer and the answer in the
/// model's voice that training uses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// A stable identifier: the kind and a hash of the question.
    pub id: String,
    /// What it asks.
    pub kind: Kind,
    /// The family of the letter, or the work of the passage: the unit the
    /// exam split keeps apart.
    pub family: String,
    /// `train` or `exam`.
    pub split: String,
    /// The question.
    pub prompt: String,
    /// What the grader compares an answer with: a surname, a year or the
    /// index of the work in the closed set.
    pub reference: String,
    /// The training answer, in the first person.
    pub voice: String,
}

/// The first `count` words of what the letter says, cut at a word boundary.
#[must_use]
pub fn excerpt(letter: &Letter, count: usize) -> String {
    without_salutation(&letter.body)
        .split_whitespace()
        .take(count)
        .collect::<Vec<_>>()
        .join(" ")
}

fn short_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex()[..12].to_string()
}

/// The two tasks a letter supports, or none when its opening gives the answer
/// away, its recipient is not one person or its year is not of the era.
#[must_use]
pub fn letter_tasks(letter: &Letter, family_key: &str, seed: u64) -> Vec<Task> {
    let Some(surname) = splinter_sdk::measure::verifiers::names::surname_of(&letter.recipient)
    else {
        return Vec::new();
    };
    if !(1760..=1826).contains(&letter.year) {
        return Vec::new();
    }
    let opening = excerpt(letter, 50);
    if opening.split_whitespace().count() < 40
        || splinter_sdk::measure::verifiers::answer::mentions(&opening, &surname)
        || splinter_sdk::measure::verifiers::answer::mentions(&opening, &letter.year.to_string())
    {
        return Vec::new();
    }
    let split = if is_exam_family(family_key, seed, 20) {
        "exam"
    } else {
        "train"
    };
    let shown = format!("Here is the opening of one of your letters:\n\n\"{opening}\"\n");
    let mut tasks = Vec::new();
    let recipient_prompt = format!("{shown}\nWho did you write it to?");
    tasks.push(Task {
        id: format!("recipient-{}", short_hash(&recipient_prompt)),
        kind: Kind::Recipient,
        family: family_key.to_string(),
        split: split.to_string(),
        prompt: recipient_prompt,
        reference: surname,
        voice: format!("I wrote this letter to {}.", letter.recipient),
    });
    let year_prompt = format!("{shown}\nIn what year did you write it?");
    let place = if letter.place.chars().count() <= 25
        && letter.place.chars().all(|c| c.is_alphabetic() || c == ' ')
    {
        format!(", from {}", letter.place)
    } else {
        String::new()
    };
    tasks.push(Task {
        id: format!("year-{}", short_hash(&year_prompt)),
        kind: Kind::Year,
        family: family_key.to_string(),
        split: split.to_string(),
        prompt: year_prompt,
        reference: letter.year.to_string(),
        voice: format!("I wrote it in {}{place}.", letter.year),
    });
    tasks
}

/// The works every which-work question offers, in a fixed order, as one line
/// of the names an answer may use.
#[must_use]
pub fn work_list() -> String {
    WORKS
        .iter()
        .map(|w| {
            let name = w.names[0];
            let mut chars = name.chars();
            chars
                .next()
                .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The which-work task a passage supports; its split is a hash of the passage
/// text, so the same passage is on the same side whichever file order.
#[must_use]
pub fn passage_task(passage: &Passage, seed: u64) -> Task {
    let work = &WORKS[passage.work];
    let prompt = format!(
        "Here is a passage from one of these works: {}.\n\n\"{}\"\n\nWhich work is it from?",
        work_list(),
        passage.text
    );
    let key = format!("passage-{}", short_hash(&passage.text));
    let split = if is_exam_family(&key, seed, 20) {
        "exam"
    } else {
        "train"
    };
    Task {
        id: format!("work-{}", short_hash(&prompt)),
        kind: Kind::Work,
        family: key,
        split: split.to_string(),
        prompt,
        reference: passage.work.to_string(),
        voice: format!("That passage is from {}, by {}.", work.title, work.author),
    }
}

/// One training record in brain's chat-dataset format: the persona and the
/// question are not trained on, the answer is.
#[must_use]
pub fn sft_record(task: &Task) -> serde_json::Value {
    serde_json::json!({
        "messages": [
            {"role": "system", "content": PERSONA, "train": false},
            {"role": "user", "content": task.prompt, "train": false},
            {"role": "assistant", "content": task.voice, "train": true},
        ],
        "tools": [],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn letter(recipient: &str, year: u16, body: &str) -> Letter {
        Letter {
            id: "w-0".into(),
            edition: "w".into(),
            recipient: recipient.into(),
            place: "Paris".into(),
            year,
            body: body.into(),
        }
    }

    const BODY: &str = "Dear Sir,--The assembly of the notables has dissolved itself and the \
        nation looks to the states general for a constitution founded on the consent of the \
        governed and I find the temper of the people favourable to a change that will give \
        them security in their persons and their property against the arbitrary will of any one man.";

    #[test]
    fn a_letter_yields_a_recipient_task_and_a_year_task_with_exact_references() {
        let tasks = letter_tasks(&letter("James Madison", 1789, BODY), "w-0", 1);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].kind, Kind::Recipient);
        assert_eq!(tasks[0].reference, "Madison");
        assert_eq!(tasks[0].voice, "I wrote this letter to James Madison.");
        assert_eq!(tasks[1].kind, Kind::Year);
        assert_eq!(tasks[1].reference, "1789");
        assert_eq!(tasks[1].voice, "I wrote it in 1789, from Paris.");
        assert!(
            tasks[0].prompt.contains("The assembly of the notables")
                || tasks[0].prompt.contains("The assembly")
        );
        assert!(
            !tasks[0].prompt.contains("Dear Sir"),
            "the salutation is not shown"
        );
    }

    #[test]
    fn a_letter_that_gives_its_answer_away_yields_no_task() {
        let leaky = BODY.replace(
            "The assembly of the notables",
            "My dear Madison, the assembly of the notables",
        );
        assert!(letter_tasks(&letter("James Madison", 1789, &leaky), "w-0", 1).is_empty());
        let dated = BODY.replace("the nation", "the nation in 1789");
        assert!(letter_tasks(&letter("James Madison", 1789, &dated), "w-0", 1).is_empty());
        assert!(letter_tasks(&letter("the President", 1789, BODY), "w-0", 1).is_empty());
        assert!(letter_tasks(&letter("James Madison", 1888, BODY), "w-0", 1).is_empty());
    }

    #[test]
    fn a_task_lands_on_the_side_its_family_does_and_both_of_a_letter_agree() {
        let tasks = letter_tasks(&letter("James Madison", 1789, BODY), "family-3", 1);
        assert_eq!(tasks[0].split, tasks[1].split);
        let again = letter_tasks(&letter("James Madison", 1789, BODY), "family-3", 1);
        assert_eq!(tasks, again, "compiling twice gives the same tasks");
    }

    #[test]
    fn the_training_record_trains_only_on_the_answer() {
        let task = &letter_tasks(&letter("James Madison", 1789, BODY), "w-0", 1)[0];
        let record = sft_record(task);
        let trained: Vec<bool> = record["messages"]
            .as_array()
            .map(|m| m.iter().map(|x| x["train"] == true).collect())
            .unwrap_or_default();
        assert_eq!(trained, vec![false, false, true]);
        assert_eq!(record["messages"][2]["content"], task.voice.as_str());
    }

    #[test]
    fn a_which_work_question_lists_the_closed_set_and_names_the_answer() {
        let passage = Passage {
            work: 5,
            text: "the life of man solitary poor nasty brutish and short".into(),
        };
        let task = passage_task(&passage, 1);
        assert_eq!(task.reference, "5");
        assert!(task.prompt.contains("Leviathan; ") && task.prompt.contains("Common sense; "));
        assert!(task.voice.contains("Leviathan") && task.voice.contains("Thomas Hobbes"));
        assert_eq!(task, passage_task(&passage, 1));
    }
}
