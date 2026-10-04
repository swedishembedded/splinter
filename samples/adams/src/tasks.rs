// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded evaluation of language models
// against historical records for its clients. If your team needs expertise in
// building questions whose answers a model cannot get without the source, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Questions built by code from the curated documents, and the answers training
//! shows a model for them.
//!
//! No model writes a question or a reference: each is read off a document's
//! own fields. A question is asked only when its opening does not already
//! contain its answer, and an exam question comes only from a document the
//! split holds out.

use splinter_sdk::measure::verifiers::answer::{mentions, year_ok};
use splinter_sdk::measure::verifiers::names::surname_of;

use crate::curate::Document;
use crate::split::{Assignment, Split};

/// The system message every question is asked under, and the rule that a claim
/// about what he wrote has a source.
pub const PERSONA: &str = "You are Samuel Adams (1722-1803). Answer in the first person, only from what your own papers show. Say plainly what the record does not show.";

/// What a task asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// Who a letter was written to.
    Recipient,
    /// In what year a document was written.
    Year,
    /// What kind of document it is: his own, a committee's, a newspaper piece.
    Attribution,
    /// How a passage goes on.
    Continuation,
}

impl Kind {
    /// The kind's name as an exam record carries it.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Recipient => "recipient",
            Kind::Year => "year",
            Kind::Attribution => "attribution",
            Kind::Continuation => "continuation",
        }
    }
}

/// One task: the question, the reference and the answer training shows.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Task {
    /// The kind and a hash of the question.
    pub id: String,
    pub kind: Kind,
    /// The document's family: the unit the exam split keeps apart.
    pub family: String,
    pub doc_id: String,
    /// `train`, `exam` or `seen`.
    pub split: String,
    pub prompt: String,
    /// What the grader compares an answer with: a surname, a year, a letter, or
    /// the words a passage goes on with.
    pub reference: String,
    /// The training answer, in the first person.
    pub voice: String,
}

/// Words of the opening a question shows.
const OPENING_WORDS: usize = 50;
/// The fewest words an opening must have to be worth asking about.
const MIN_OPENING_WORDS: usize = 40;
/// A paragraph this short at the top is a salutation, not the text.
const MAX_SALUTATION_WORDS: usize = 12;
/// Words a continuation question shows, and asks for.
const CONTINUATION_WORDS: usize = 40;
/// Share of a reference's four-word runs an answer must carry.
const CONTINUATION_SHARE: f64 = 0.5;
const RUN: usize = 4;
/// Where in the text a minority class's training openings start, in words.
const MINORITY_OPENINGS: [usize; 3] = [0, 150, 300];

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

fn short_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex()[..12].to_string()
}

/// The text's words after a salutation, as one list.
fn words_after_salutation(body: &str) -> Vec<&str> {
    let text = match body.split_once("\n\n") {
        Some((first, rest)) if first.split_whitespace().count() <= MAX_SALUTATION_WORDS => rest,
        _ => body,
    };
    text.split_whitespace().collect()
}

/// The option an attribution question's answer names for a class of document.
fn attribution_option(doc: &Document) -> Option<(char, &'static str)> {
    use crate::document::Authorship::{
        CommitteeCoauthored, DirectAutograph, DraftInHand, PseudonymousAttributed, SignedScribal,
    };
    match doc.authorship {
        DraftInHand => Some(('A', "It is a letter of my own, a draft or manuscript kept in my papers.")),
        SignedScribal | DirectAutograph => Some(('A', "It is a letter of my own, as sent or in a copy of it.")),
        CommitteeCoauthored => Some(('B', "A committee or town I sat on adopted it, so it is not my private voice alone.")),
        PseudonymousAttributed => Some(('C', "It was printed in a newspaper under a pseudonym; that I wrote it rests on the editor's attribution.")),
        _ => None,
    }
}

/// The tasks one document supports, on `split` (`train` or `exam`).
pub fn document_tasks(doc: &Document, family: &str, split: &str) -> Vec<Task> {
    let words = words_after_salutation(&doc.body);
    if words.len() < MIN_OPENING_WORDS {
        return Vec::new();
    }
    let opening = words[..OPENING_WORDS.min(words.len())].join(" ");
    let make = |kind: Kind, prompt: String, reference: String, voice: String| Task {
        id: format!(
            "{}-{}",
            kind.name(),
            short_hash(&format!("{}\n{prompt}", doc.id))
        ),
        kind,
        family: family.to_string(),
        doc_id: doc.id.clone(),
        split: split.to_string(),
        prompt,
        reference,
        voice,
    };
    let shown = format!("Here is the opening of a document from your papers:\n\n\"{opening}\"\n");
    let mut tasks = Vec::new();

    if doc.authorship.is_voice() {
        if let Some(surname) = doc.recipient.as_deref().and_then(surname_of) {
            if !mentions(&opening, &surname) {
                let who = doc.recipient.clone().unwrap_or_default();
                tasks.push(make(
                    Kind::Recipient,
                    format!("{shown}\nWho did you write it to?"),
                    surname,
                    format!("I wrote this letter to {who}."),
                ));
            }
        }
    }
    if doc.authorship.has_a_settled_kind() && !mentions(&opening, &doc.date.year.to_string()) {
        let when = doc
            .date
            .month
            .and_then(|m| MONTHS.get(usize::from(m) - 1))
            .map_or(doc.date.year.to_string(), |month| {
                format!("{month} {}", doc.date.year)
            });
        tasks.push(make(
            Kind::Year,
            format!("{shown}\nIn what year did you write it?"),
            doc.date.year.to_string(),
            format!("I wrote it in {when}."),
        ));
    }
    if let Some((option, why)) = attribution_option(doc) {
        // A class that is rare in the corpus is shown through several openings in
        // training, so the model sees more of it without a record repeated.
        let starts: &[usize] = if split == "train" && option != 'A' {
            &MINORITY_OPENINGS
        } else {
            &[0]
        };
        for &start in starts
            .iter()
            .filter(|&&start| words.len() >= start + MIN_OPENING_WORDS)
        {
            let window = words[start..(start + OPENING_WORDS).min(words.len())].join(" ");
            let prompt = format!(
                "Here is the opening of a document from your papers:\n\n\"{window}\"\n\nWhich is it? (A) a letter or draft of your own, or a copy of it; (B) a text adopted by a committee or town you sat on; (C) a newspaper piece printed under a pseudonym. Answer with the letter and one sentence of why."
            );
            tasks.push(make(
                Kind::Attribution,
                prompt,
                option.to_string(),
                format!("{option}. {why}"),
            ));
        }
    }
    if split == "train" && doc.authorship.is_voice() && words.len() >= 2 * CONTINUATION_WORDS {
        let shown_words = words[..CONTINUATION_WORDS].join(" ");
        let reference = words[CONTINUATION_WORDS..2 * CONTINUATION_WORDS].join(" ");
        let prompt = format!("Here is the opening of a document from your papers:\n\n\"{shown_words}\"\n\nContinue it as it goes in your papers, for about forty words.");
        tasks.push(make(
            Kind::Continuation,
            prompt,
            reference.clone(),
            reference,
        ));
    }
    tasks
}

/// The option a model's answer picks in an attribution question: `A`, `B` or
/// `C` when the answer opens with exactly one of them.
pub fn choice_of(answer: &str) -> Option<char> {
    let mut chars = answer.trim_start().trim_start_matches('(').chars();
    let letter = chars.next().filter(|c| matches!(c, 'A' | 'B' | 'C'))?;
    match chars.next() {
        None | Some('.' | ')' | ':' | ',' | '-' | '\u{2013}' | '\u{2014}') => Some(letter),
        Some(_) => None,
    }
}

/// Whether `answer` goes on as the passage does: enough of the reference's
/// four-word runs appear in it, in any wording around them.
pub fn continuation_ok(answer: &str, reference: &str) -> bool {
    let reference = normalised_words(reference);
    if reference.len() < RUN {
        return false;
    }
    let answer = normalised_words(answer);
    let runs: Vec<&[String]> = reference.windows(RUN).collect();
    let found = runs
        .iter()
        .filter(|run| answer.windows(RUN).any(|w| w == **run))
        .count();
    found as f64 >= CONTINUATION_SHARE * runs.len() as f64
}

fn normalised_words(text: &str) -> Vec<String> {
    splinter_sdk::measure::verifiers::quotation::words(text)
}

/// Whether `answer` is a right answer to `task`.
pub fn is_correct(task: &Task, answer: &str) -> bool {
    match task.kind {
        Kind::Recipient => mentions(answer, &task.reference),
        Kind::Year => task
            .reference
            .parse()
            .is_ok_and(|year| year_ok(answer, year)),
        Kind::Attribution => choice_of(answer).is_some_and(|c| task.reference.starts_with(c)),
        Kind::Continuation => continuation_ok(answer, &task.reference),
    }
}

/// One training record in brain's chat-dataset format: the persona and the
/// question are not trained on, the answer is.
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

impl splinter_sdk::model::exam::Question for Task {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> &str {
        self.kind.name()
    }
    fn split(&self) -> &str {
        &self.split
    }
    fn reference(&self) -> &str {
        &self.reference
    }
    fn prompt(&self) -> &str {
        &self.prompt
    }
}

/// The chance level of a kind of question, where one exists: an attribution
/// offers three options; a name, a year and a passage have no fixed chance.
pub fn chance(kind: &str) -> Option<f64> {
    (kind == Kind::Attribution.name()).then_some(1.0 / 3.0)
}

/// The order the kinds of question are reported in.
pub const KIND_ORDER: [&str; 4] = ["recipient", "year", "attribution", "continuation"];

/// The training set, the frozen exam and a sample of what training contains.
#[derive(Debug, Default)]
pub struct Built {
    pub train: Vec<Task>,
    pub exam: Vec<Task>,
    pub seen: Vec<Task>,
}

/// Build every task from the assigned documents. `seen_per_kind` training
/// tasks of each kind are also kept as the `seen` sample, chosen by hash so
/// the sample does not depend on file order. Documents kept for the temporal
/// test yield nothing.
pub fn build(docs: &[Document], assignments: &[Assignment], seen_per_kind: usize) -> Built {
    let mut built = Built::default();
    let mut ids = std::collections::HashSet::new();
    for doc in docs {
        let Some(assignment) = assignments.iter().find(|a| a.doc_id == doc.id) else {
            continue;
        };
        let (split, into) = match assignment.split {
            Split::Train => ("train", &mut built.train),
            Split::Exam => ("exam", &mut built.exam),
            Split::Temporal => continue,
        };
        for task in document_tasks(doc, &assignment.family, split) {
            if ids.insert(task.id.clone()) {
                into.push(task);
            }
        }
    }
    for kind in [
        Kind::Recipient,
        Kind::Year,
        Kind::Attribution,
        Kind::Continuation,
    ] {
        let mut of_kind: Vec<&Task> = built.train.iter().filter(|t| t.kind == kind).collect();
        of_kind.sort_by_key(|t| short_hash(&t.id));
        built
            .seen
            .extend(of_kind.into_iter().take(seen_per_kind).map(|t| Task {
                split: "seen".into(),
                ..t.clone()
            }));
    }
    built
}

/// Write `sft.jsonl` (training records), `exam.jsonl` and `seen.jsonl` (tasks)
/// under `dir`, each atomically.
///
/// # Errors
/// A file cannot be written.
pub fn write_all(built: &Built, dir: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let lines = |rows: Vec<serde_json::Value>| -> String {
        rows.into_iter().map(|r| format!("{r}\n")).collect()
    };
    let tasks = |tasks: &[Task]| lines(tasks.iter().map(|t| serde_json::json!(t)).collect());
    let files = [
        (
            "sft.jsonl",
            lines(built.train.iter().map(sft_record).collect()),
        ),
        ("exam.jsonl", tasks(&built.exam)),
        ("seen.jsonl", tasks(&built.seen)),
    ];
    for (name, text) in files {
        let path = dir.join(name);
        if name == "exam.jsonl" {
            if let Ok(existing) = std::fs::read_to_string(&path) {
                anyhow::ensure!(existing == text, "{} is the frozen exam and these tasks would change it; the exam is never rewritten", path.display());
            }
        }
        let tmp = path.with_extension("part");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
    }
    Ok(())
}

/// The tasks of a JSON-lines file.
///
/// # Errors
/// The file cannot be read or a line is not a task.
pub fn read_tasks(path: &std::path::Path) -> anyhow::Result<Vec<Task>> {
    std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

#[cfg(test)]
mod tests;
