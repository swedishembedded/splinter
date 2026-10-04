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
mod tests {
    use super::*;
    use crate::document::{Authorship, Date, Period};

    /// `n` distinct words.
    fn words(prefix: &str, n: usize) -> String {
        (0..n)
            .map(|i| format!("{prefix}{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn doc(
        id: &str,
        authorship: Authorship,
        recipient: Option<&str>,
        year: u16,
        body: String,
    ) -> Document {
        Document {
            schema: crate::curate::SCHEMA,
            id: id.into(),
            source_id: "cushing-1".into(),
            heading: recipient.map_or("ARTICLE".into(), |r| format!("TO {}.", r.to_uppercase())),
            recipient: recipient.map(str::to_string),
            note: "[MS.]".into(),
            date: Date {
                year,
                month: Some(4),
                day: Some(2),
            },
            period: Period::of(year),
            authorship,
            authorship_confidence: 0.8,
            authorship_basis: "test".into(),
            temporal_holdout: crate::document::in_temporal_holdout(year),
            body,
        }
    }

    fn letter(id: &str) -> Document {
        doc(
            id,
            Authorship::DraftInHand,
            Some("James Otis"),
            1770,
            format!("My dear Sir,\n\n{}", words("alpha", 200)),
        )
    }

    fn by_kind(tasks: &[Task], kind: Kind) -> Vec<&Task> {
        tasks.iter().filter(|t| t.kind == kind).collect()
    }

    #[test]
    fn a_letter_supports_every_kind_with_references_read_off_its_fields() {
        let tasks = document_tasks(&letter("d1"), "fam-d1", "train");
        let recipient = by_kind(&tasks, Kind::Recipient);
        assert_eq!(recipient[0].reference, "Otis");
        assert_eq!(by_kind(&tasks, Kind::Year)[0].reference, "1770");
        assert_eq!(by_kind(&tasks, Kind::Attribution)[0].reference, "A");
        let cont = by_kind(&tasks, Kind::Continuation)[0];
        assert!(cont.reference.starts_with("alpha40 "), "{}", cont.reference);
        assert!(tasks
            .iter()
            .all(|t| t.split == "train" && t.family == "fam-d1" && t.doc_id == "d1"));
    }

    #[test]
    fn the_salutation_is_not_the_opening_a_question_shows() {
        let tasks = document_tasks(&letter("d1"), "f", "train");
        assert!(
            !tasks[0].prompt.contains("My dear Sir"),
            "{}",
            tasks[0].prompt
        );
        assert!(tasks.iter().any(|t| t.prompt.contains("alpha0 alpha1")));
    }

    #[test]
    fn the_opening_starts_at_the_first_word_after_the_salutation_however_many_paragraphs_follow() {
        let mut d = letter("d1");
        d.body = format!(
            "My dear Sir,\n\n{}\n\n{}\n\n{}",
            words("alpha", 30),
            words("beta", 30),
            words("gamma", 30)
        );
        let tasks = document_tasks(&d, "f", "train");
        assert!(
            tasks
                .iter()
                .all(|t| t.kind == Kind::Continuation || t.prompt.contains("\"alpha0 alpha1 ")),
            "{:?}",
            tasks.iter().map(|t| &t.prompt).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_question_whose_opening_names_its_answer_is_not_asked() {
        let mut d = letter("d1");
        d.body = format!(
            "My dear Sir,\n\nWhen I wrote to Otis in 1770 I said {}",
            words("beta", 120)
        );
        let kinds: Vec<Kind> = document_tasks(&d, "f", "train")
            .iter()
            .map(|t| t.kind)
            .collect();
        assert!(
            !kinds.contains(&Kind::Recipient) && !kinds.contains(&Kind::Year),
            "{kinds:?}"
        );
        assert!(kinds.contains(&Kind::Attribution));
    }

    #[test]
    fn a_recipient_who_is_not_one_person_gets_no_recipient_question() {
        let mut d = letter("d1");
        d.recipient = Some("the Committee of Correspondence of Lancaster".into());
        assert!(by_kind(&document_tasks(&d, "f", "train"), Kind::Recipient).is_empty());
    }

    #[test]
    fn the_attribution_answer_follows_the_authorship_class() {
        let class = |a| {
            let d = doc(
                "d",
                a,
                Some("James Otis"),
                1770,
                format!("Sir,\n\n{}", words("g", 100)),
            );
            by_kind(&document_tasks(&d, "f", "train"), Kind::Attribution)
                .first()
                .map(|t| t.reference.clone())
        };
        assert_eq!(class(Authorship::DraftInHand).as_deref(), Some("A"));
        assert_eq!(class(Authorship::SignedScribal).as_deref(), Some("A"));
        assert_eq!(class(Authorship::CommitteeCoauthored).as_deref(), Some("B"));
        assert_eq!(
            class(Authorship::PseudonymousAttributed).as_deref(),
            Some("C")
        );
        assert_eq!(class(Authorship::EditorAttributed), None);
    }

    #[test]
    fn a_newspaper_piece_is_not_taught_as_a_letter_or_given_a_recipient_or_voice_task() {
        let d = doc(
            "n",
            Authorship::PseudonymousAttributed,
            None,
            1770,
            format!("Messieurs,\n\n{}", words("p", 150)),
        );
        let tasks = document_tasks(&d, "f", "train");
        let kinds: Vec<Kind> = tasks.iter().map(|t| t.kind).collect();
        assert!(
            !kinds.is_empty()
                && kinds
                    .iter()
                    .all(|k| matches!(k, Kind::Attribution | Kind::Year)),
            "{kinds:?}"
        );
        assert!(by_kind(&tasks, Kind::Attribution)[0]
            .voice
            .contains("pseudonym"));
        assert!(
            by_kind(&tasks, Kind::Attribution)[0]
                .voice
                .contains("editor"),
            "a newspaper attribution is the editor's"
        );
        assert!(
            !kinds.contains(&Kind::Continuation),
            "a piece he may not have written is not taught as his words"
        );
    }

    #[test]
    fn a_short_document_supports_nothing() {
        let d = doc(
            "s",
            Authorship::DraftInHand,
            Some("James Otis"),
            1770,
            format!("Sir,\n\n{}", words("s", 30)),
        );
        assert!(document_tasks(&d, "f", "train").is_empty());
    }

    #[test]
    fn a_minority_class_is_shown_through_several_openings_in_training_and_the_exam_keeps_one() {
        let committee = doc(
            "c",
            Authorship::CommitteeCoauthored,
            Some("James Otis"),
            1770,
            format!("Sir,\n\n{}", words("c", 400)),
        );
        let attribution =
            |split: &str| by_kind(&document_tasks(&committee, "f", split), Kind::Attribution).len();
        assert_eq!(
            attribution("train"),
            3,
            "windows at the start, 150 and 300 words in"
        );
        assert_eq!(
            attribution("exam"),
            1,
            "the frozen exam asks one question per document"
        );
        let letter = doc(
            "l",
            Authorship::DraftInHand,
            Some("James Otis"),
            1770,
            format!("Sir,\n\n{}", words("l", 400)),
        );
        assert_eq!(
            by_kind(&document_tasks(&letter, "f", "train"), Kind::Attribution).len(),
            1,
            "the majority class is not multiplied"
        );
    }

    #[test]
    fn the_extra_openings_are_different_text_and_a_short_document_gets_only_those_that_fit() {
        let committee = doc(
            "c",
            Authorship::CommitteeCoauthored,
            Some("James Otis"),
            1770,
            format!("Sir,\n\n{}", words("c", 400)),
        );
        let prompts: Vec<String> =
            by_kind(&document_tasks(&committee, "f", "train"), Kind::Attribution)
                .iter()
                .map(|t| t.prompt.clone())
                .collect();
        assert!(
            prompts[0].contains("c0 c1 ")
                && prompts[1].contains("c150 c151 ")
                && prompts[2].contains("c300 c301 ")
        );
        let short = doc(
            "s",
            Authorship::CommitteeCoauthored,
            Some("James Otis"),
            1770,
            format!("Sir,\n\n{}", words("s", 200)),
        );
        assert_eq!(
            by_kind(&document_tasks(&short, "f", "train"), Kind::Attribution).len(),
            2,
            "windows at 0 and 150 fit in 200 words; 300 does not"
        );
    }

    #[test]
    fn writing_the_tasks_again_never_changes_a_frozen_exam() {
        let (docs, assignments) = corpus();
        let built = build(&docs, &assignments, 3);
        let dir = tempfile::tempdir().unwrap();
        write_all(&built, dir.path()).unwrap();
        write_all(&built, dir.path()).unwrap();
        let mut changed = build(&docs, &assignments, 3);
        changed.exam[0].prompt.push_str(" altered");
        let err = write_all(&changed, dir.path()).unwrap_err().to_string();
        assert!(err.contains("frozen exam"), "{err}");
        assert_eq!(
            read_tasks(&dir.path().join("exam.jsonl")).unwrap(),
            built.exam
        );
    }

    #[test]
    fn a_continuation_is_never_asked_on_the_exam_since_unseen_text_cannot_be_continued() {
        assert!(by_kind(
            &document_tasks(&letter("d"), "f", "exam"),
            Kind::Continuation
        )
        .is_empty());
    }

    #[test]
    fn the_year_answer_names_the_month_when_the_edition_does() {
        let t = document_tasks(&letter("d"), "f", "train");
        assert!(
            by_kind(&t, Kind::Year)[0].voice.contains("April 1770"),
            "{}",
            by_kind(&t, Kind::Year)[0].voice
        );
    }

    #[test]
    fn a_choice_is_read_from_the_opening_of_the_answer_only() {
        assert_eq!(choice_of("A. It is a letter of mine."), Some('A'));
        assert_eq!(choice_of("(B) a committee text"), Some('B'));
        assert_eq!(
            choice_of("  c: a newspaper piece"),
            None,
            "lowercase prose is not an option"
        );
        assert_eq!(choice_of("C"), Some('C'));
        assert_eq!(
            choice_of("A letter of mine, I think."),
            None,
            "the article is not an option"
        );
        assert_eq!(choice_of("I think it is B."), None);
        assert_eq!(choice_of(""), None);
    }

    #[test]
    fn a_continuation_is_right_when_it_carries_the_passages_own_runs_in_any_wording() {
        let reference =
            "we have been long and patiently waiting for redress of every grievance we complain of";
        assert!(continuation_ok(
            &format!("As I wrote: {reference} and so on."),
            reference
        ));
        assert!(!continuation_ok(
            "something about taxation and representation entirely unlike it",
            reference
        ));
        assert!(!continuation_ok("", reference));
    }

    #[test]
    fn grading_follows_each_kinds_reference() {
        let tasks = document_tasks(&letter("d"), "f", "train");
        let task = |k| by_kind(&tasks, k)[0].clone();
        assert!(is_correct(
            &task(Kind::Recipient),
            "I wrote it to James Otis."
        ));
        assert!(!is_correct(&task(Kind::Recipient), "To Hancock."));
        assert!(is_correct(&task(Kind::Year), "It was 1770."));
        assert!(!is_correct(&task(Kind::Year), "1770 or 1771"));
        assert!(is_correct(&task(Kind::Attribution), "A. Mine."));
        assert!(!is_correct(&task(Kind::Attribution), "B. A committee."));
    }

    #[test]
    fn a_training_record_trains_the_answer_and_nothing_else() {
        let t = document_tasks(&letter("d"), "f", "train").remove(0);
        let r = sft_record(&t);
        let m = r["messages"].as_array().unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(
            (m[0]["role"].as_str(), m[0]["train"].as_bool()),
            (Some("system"), Some(false))
        );
        assert_eq!(
            (m[1]["role"].as_str(), m[1]["train"].as_bool()),
            (Some("user"), Some(false))
        );
        assert_eq!(
            (m[2]["role"].as_str(), m[2]["train"].as_bool()),
            (Some("assistant"), Some(true))
        );
        assert_eq!(m[2]["content"].as_str(), Some(t.voice.as_str()));
        assert_eq!(m[0]["content"].as_str(), Some(PERSONA));
    }

    fn corpus() -> (Vec<Document>, Vec<Assignment>) {
        let docs: Vec<Document> = (0..30)
            .map(|i| {
                doc(
                    &format!("d{i}"),
                    Authorship::DraftInHand,
                    Some("James Otis"),
                    1770,
                    format!("Sir,\n\n{}", words(&format!("t{i}x"), 200)),
                )
            })
            .collect();
        let mut late = doc(
            "late",
            Authorship::DraftInHand,
            Some("James Otis"),
            1795,
            format!("Sir,\n\n{}", words("late", 200)),
        );
        late.temporal_holdout = true;
        let mut all = docs;
        all.push(late);
        let assignments = all
            .iter()
            .enumerate()
            .map(|(i, d)| Assignment {
                doc_id: d.id.clone(),
                family: format!("f{i}"),
                split: if d.id == "late" {
                    Split::Temporal
                } else if i % 5 == 0 {
                    Split::Exam
                } else {
                    Split::Train
                },
            })
            .collect();
        (all, assignments)
    }

    #[test]
    fn exam_questions_come_only_from_held_out_documents_and_training_only_from_the_rest() {
        let (docs, assignments) = corpus();
        let built = build(&docs, &assignments, 3);
        let split_of = |id: &str| assignments.iter().find(|a| a.doc_id == id).unwrap().split;
        assert!(!built.exam.is_empty() && !built.train.is_empty());
        assert!(built
            .exam
            .iter()
            .all(|t| split_of(&t.doc_id) == Split::Exam && t.split == "exam"));
        assert!(built
            .train
            .iter()
            .all(|t| split_of(&t.doc_id) == Split::Train && t.split == "train"));
    }

    #[test]
    fn the_late_documents_yield_no_task_at_all() {
        let (docs, assignments) = corpus();
        let built = build(&docs, &assignments, 3);
        assert!(built
            .train
            .iter()
            .chain(&built.exam)
            .chain(&built.seen)
            .all(|t| t.doc_id != "late"));
    }

    #[test]
    fn the_seen_sample_is_taken_from_what_training_contains_a_few_per_kind() {
        let (docs, assignments) = corpus();
        let built = build(&docs, &assignments, 3);
        for kind in [
            Kind::Recipient,
            Kind::Year,
            Kind::Attribution,
            Kind::Continuation,
        ] {
            let seen = by_kind(&built.seen, kind);
            assert_eq!(seen.len(), 3, "{kind:?}");
            assert!(seen
                .iter()
                .all(|t| t.split == "seen" && built.train.iter().any(|x| x.id == t.id)));
        }
    }

    #[test]
    fn no_task_id_is_used_twice_and_none_of_an_exam_prompt_appears_in_training() {
        let (docs, assignments) = corpus();
        let built = build(&docs, &assignments, 3);
        let mut ids: Vec<&str> = built
            .train
            .iter()
            .chain(&built.exam)
            .map(|t| t.id.as_str())
            .collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before);
        for e in &built.exam {
            assert!(built.train.iter().all(|t| t.prompt != e.prompt));
        }
    }

    #[test]
    fn the_three_files_round_trip_and_the_training_file_holds_only_chat_records() {
        let (docs, assignments) = corpus();
        let built = build(&docs, &assignments, 3);
        let dir = tempfile::tempdir().unwrap();
        write_all(&built, dir.path()).unwrap();
        assert_eq!(
            read_tasks(&dir.path().join("exam.jsonl")).unwrap(),
            built.exam
        );
        assert_eq!(
            read_tasks(&dir.path().join("seen.jsonl")).unwrap(),
            built.seen
        );
        let sft = std::fs::read_to_string(dir.path().join("sft.jsonl")).unwrap();
        assert_eq!(sft.lines().count(), built.train.len());
        assert!(sft
            .lines()
            .all(
                |l| serde_json::from_str::<serde_json::Value>(l).unwrap()["messages"][2]["train"]
                    == true
            ));
        assert!(!dir.path().join("sft.part").exists());
    }

    #[test]
    fn chance_is_named_only_for_the_three_way_choice() {
        assert_eq!(chance("attribution"), Some(1.0 / 3.0));
        assert_eq!(chance("year"), None);
    }
}
