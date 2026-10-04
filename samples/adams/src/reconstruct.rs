// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements historical-behavior reconstruction tests for
// language models for its clients. If your team needs expertise in measuring
// whether a model reproduces what a person actually did, from what they were
// shown and not from what they wrote, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Reconstructing what he actually did: a held-out letter of his is turned into
//! a briefing of the situation he answered, the model writes the reply, and the
//! reply is held against what the real letter does.
//!
//! The briefing must not contain his words: the situation is told in the
//! helper's own, and shares no passage with the letter. The rubric is a short
//! list of what the real letter does, the issues it recognises, the action it
//! recommends, what it asks for, and each point must carry a quotation from the
//! letter that code finds there, so the rubric is grounded in the letter and
//! not in a helper's reading of it.

use splinter_sdk::agent::schemars::JsonSchema;
use splinter_sdk::agent::typed::TypedCall;
use splinter_sdk::measure::verifiers::quotation::words;

use crate::curate::Document;
use crate::helper::Helper;
use crate::principles::occurs;

/// The system message the student writes the reply under.
pub const SYSTEM: &str = "You are Samuel Adams (1722-1803), writing in your own time. Write the reply you would send, in the first person, as a letter. Do not quote your own earlier writing. Say plainly what you do not know.";

/// Words a situation must run to be one, and a request to be one.
const MIN_SITUATION_WORDS: usize = 60;
const MIN_REQUEST_WORDS: usize = 5;
/// How many things the rubric names, and the fewest words of a quotation that
/// grounds one.
const KEY_POINTS: std::ops::RangeInclusive<usize> = 3..=6;
const MIN_QUOTE_WORDS: usize = 6;
/// A run this long shared with the letter is the letter's own wording.
const LEAK_RUN_WORDS: usize = 8;

/// One thing the real letter does, and the words of the letter that show it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
pub struct KeyPoint {
    /// What the letter does: the issue it recognises, the action it recommends,
    /// what it asks for, the value it invokes.
    pub point: String,
    /// Words of the letter, copied exactly, that show it.
    pub quote: String,
}

/// What the helper drafts from a letter.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
pub struct Drafted {
    /// The situation he was answering, told in your own words, without his.
    pub situation: String,
    /// What the correspondent asks of him, or what the moment calls for.
    pub request: String,
    /// What the real letter does, three to six things, each with its quotation.
    pub key_points: Vec<KeyPoint>,
}

/// A briefing that passed the gate.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Briefing {
    pub id: String,
    pub doc_id: String,
    pub situation: String,
    pub request: String,
    pub key_points: Vec<KeyPoint>,
}

/// Whether `text` shares a run of [`LEAK_RUN_WORDS`] words with `letter`.
fn repeats(text: &str, letter: &[String]) -> bool {
    let words = words(text);
    words.windows(LEAK_RUN_WORDS).any(|run| occurs(letter, run))
}

/// Refuse a draft that cannot be used, naming why.
pub fn gate(drafted: &Drafted, letter: &Document) -> Result<(), String> {
    let body = words(&letter.body);
    if words(&drafted.situation).len() < MIN_SITUATION_WORDS {
        return Err(format!("the situation is too short to answer: write at least {MIN_SITUATION_WORDS} words about what he was facing"));
    }
    if words(&drafted.request).len() < MIN_REQUEST_WORDS {
        return Err("the request is too short".to_string());
    }
    if !KEY_POINTS.contains(&drafted.key_points.len()) {
        return Err(format!(
            "give {} to {} key points",
            KEY_POINTS.start(),
            KEY_POINTS.end()
        ));
    }
    for point in &drafted.key_points {
        let quote = words(&point.quote);
        if quote.len() < MIN_QUOTE_WORDS {
            return Err(format!("the quotation for {:?} is too short: copy at least {MIN_QUOTE_WORDS} words of the letter", point.point));
        }
        if !occurs(&body, &quote) {
            return Err(format!(
                "the quotation for {:?} is not in the letter: copy it exactly",
                point.point
            ));
        }
    }
    if repeats(&drafted.situation, &body) || repeats(&drafted.request, &body) {
        return Err("the situation or the request repeats the letter's own words: tell it in your own words, so the answer is not given away".to_string());
    }
    Ok(())
}

/// How the student is asked: the situation and the request, nothing else.
pub fn prompt(briefing: &Briefing) -> String {
    format!(
        "Situation: {}\n\n{}\n",
        briefing.situation, briefing.request
    )
}

/// The role the helper is given.
const ROLE: &str = "a historian who reads a letter closely and says what situation it answered and what it does, without quoting it";

/// The task the helper is given.
const TASK: &str = "You are given a letter Samuel Adams wrote. Tell, in your own words and without repeating any passage of his, the situation he was answering, as a briefing a person could act on: who is asking, what has happened, what is at stake. Say what is asked of him. Then list three to six things the real letter does: an issue it recognises, an action it recommends, something it asks for, a value it invokes. For each, copy exactly the words of the letter that show it.";

/// What the helper reads of a letter, and who it went to.
#[derive(serde::Serialize)]
struct Reading<'a> {
    year: u16,
    recipient: Option<&'a str>,
    letter: &'a str,
}

const OUTPUT_TOKENS: u64 = 1500;
const REPAIRS: u32 = 2;

/// Draft a briefing of the situation `letter` answered, and gate it.
///
/// # Errors
/// The helper could not be reached, or no draft passed the gate after the
/// permitted corrections.
pub fn brief(helper: &Helper, letter: &Document) -> anyhow::Result<Briefing> {
    let checked = letter.clone();
    let call = TypedCall::<Drafted>::new("brief_letter", TASK, ROLE, crate::helper::CALL_DEADLINE)
        .max_output_tokens(OUTPUT_TOKENS)
        .repairs(REPAIRS)
        .postcondition(move |drafted| gate(drafted, &checked));
    let d = helper.call(
        call,
        &Reading {
            year: letter.date.year,
            recipient: letter.recipient.as_deref(),
            letter: &letter.body,
        },
    )?;
    let id = format!(
        "recon-{}",
        &blake3::hash(letter.id.as_bytes()).to_hex()[..12]
    );
    Ok(Briefing {
        id,
        doc_id: letter.id.clone(),
        situation: d.situation,
        request: d.request,
        key_points: d.key_points,
    })
}

/// What one letter came to when briefed, one line of the briefings file.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BriefingResult {
    pub doc_id: String,
    pub briefing: Option<Briefing>,
    /// Set when the helper failed; the letter is retried until it has failed
    /// [`MAX_BRIEF_ATTEMPTS`] times.
    pub error: Option<String>,
}

/// Attempts a letter gets before the run stops asking about it.
pub const MAX_BRIEF_ATTEMPTS: usize = 2;

/// The briefings already made in `path`, those that succeeded.
///
/// # Errors
/// The file exists and a line is not a result.
pub fn read_briefings(path: &std::path::Path) -> anyhow::Result<Vec<Briefing>> {
    Ok(read_lines::<BriefingResult>(path)?
        .into_iter()
        .filter_map(|r| r.briefing)
        .collect())
}

/// Brief every letter not yet briefed, at most `limit`, appending each result
/// as it finishes and recording a letter the helper fails on. Returns how many
/// were asked.
///
/// # Errors
/// `out` cannot be read or written.
pub fn brief_all(
    helper: &Helper,
    letters: &[&Document],
    out: &std::path::Path,
    limit: Option<usize>,
) -> anyhow::Result<usize> {
    use std::io::Write;
    let existing: Vec<BriefingResult> = read_lines(out)?;
    let todo: Vec<&&Document> = letters
        .iter()
        .filter(|l| {
            let mine = existing.iter().filter(|r| r.doc_id == l.id);
            !mine.clone().any(|r| r.error.is_none())
                && mine.filter(|r| r.error.is_some()).count() < MAX_BRIEF_ATTEMPTS
        })
        .collect();
    let todo = &todo[..limit.map_or(todo.len(), |n| n.min(todo.len()))];
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)?;
    for (n, letter) in todo.iter().enumerate() {
        let result = match brief(helper, letter) {
            Ok(b) => BriefingResult {
                doc_id: letter.id.clone(),
                briefing: Some(b),
                error: None,
            },
            Err(e) => BriefingResult {
                doc_id: letter.id.clone(),
                briefing: None,
                error: Some(format!("{e:#}")),
            },
        };
        writeln!(file, "{}", serde_json::to_string(&result)?)?;
        file.flush()?;
        eprintln!(
            "[{}/{}] {}{}",
            n + 1,
            todo.len(),
            letter.id,
            result
                .error
                .as_deref()
                .map_or(String::new(), |e| format!(" FAILED: {e}"))
        );
    }
    Ok(todo.len())
}

/// What one arm wrote in reply to one briefing, in full.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Reply {
    pub briefing_id: String,
    /// Which model wrote it: `base`, `tuned`.
    pub arm: String,
    /// The reply, or empty when the model gave none.
    pub text: String,
    pub tokens: u32,
    pub truncated: bool,
}

/// Lines of a JSON-lines file; none when there is no file.
fn read_lines<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> anyhow::Result<Vec<T>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

/// The replies already in `path`.
///
/// # Errors
/// The file exists and a line is not a reply.
pub fn read_replies(path: &std::path::Path) -> anyhow::Result<Vec<Reply>> {
    read_lines(path)
}

/// Ask `ask` for a reply to every briefing not yet answered by `arm` in `out`,
/// at most `limit`, appending each in full as it comes. Returns how many.
///
/// # Errors
/// `out` cannot be written or `ask` fails.
pub fn answer_all(
    briefings: &[Briefing],
    out: &std::path::Path,
    arm: &str,
    limit: Option<usize>,
    ask: &mut dyn FnMut(&Briefing) -> anyhow::Result<splinter_sdk::model::answer::Reply>,
) -> anyhow::Result<usize> {
    use std::io::Write;
    let done: std::collections::HashSet<String> = read_replies(out)?
        .into_iter()
        .filter(|r| r.arm == arm)
        .map(|r| r.briefing_id)
        .collect();
    let todo: Vec<&Briefing> = briefings.iter().filter(|b| !done.contains(&b.id)).collect();
    let todo = &todo[..limit.map_or(todo.len(), |n| n.min(todo.len()))];
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)?;
    for (n, briefing) in todo.iter().enumerate() {
        let reply = ask(briefing)?;
        let text = splinter_sdk::measure::verifiers::answer::final_answer(
            &reply.thinking,
            &reply.text,
            reply.truncated,
        )
        .unwrap_or_default();
        let record = Reply {
            briefing_id: briefing.id.clone(),
            arm: arm.to_string(),
            text,
            tokens: reply.tokens,
            truncated: reply.truncated,
        };
        writeln!(file, "{}", serde_json::to_string(&record)?)?;
        file.flush()?;
        eprintln!(
            "[{}/{}] {arm} {}: {} tokens",
            n + 1,
            todo.len(),
            briefing.id,
            reply.tokens
        );
    }
    Ok(todo.len())
}

/// A reply, scored.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Scored {
    pub briefing_id: String,
    pub arm: String,
    /// Which rubric items the judge says the reply does; empty on an error.
    pub covered: Vec<bool>,
    /// Quotations of eight words or more in the reply that are in no document,
    /// and how many quotations there are.
    pub fabricated: usize,
    pub quotations: usize,
    /// A word from after his death the reply uses, if any.
    pub anachronism: Option<String>,
    /// Set when the judge failed; the reply is retried until it has failed twice.
    pub error: Option<String>,
}

/// Attempts a reply gets from the judge before the run stops asking.
const MAX_JUDGE_ATTEMPTS: usize = 2;

/// Score every reply not yet scored, at most `limit`: the judge's coverage of
/// the rubric, and the two checks code makes. Returns how many were scored.
///
/// # Errors
/// `out` cannot be written.
pub fn score_all(
    helper: &Helper,
    briefings: &[Briefing],
    replies: &[Reply],
    corpus: &splinter_sdk::measure::verifiers::quotation::TextIndex,
    out: &std::path::Path,
    limit: Option<usize>,
) -> anyhow::Result<usize> {
    use std::io::Write;
    let existing: Vec<Scored> = read_lines(out)?;
    let todo: Vec<&Reply> = replies
        .iter()
        .filter(|r| {
            let mine = existing
                .iter()
                .filter(|s| s.briefing_id == r.briefing_id && s.arm == r.arm);
            !mine.clone().any(|s| s.error.is_none())
                && mine.filter(|s| s.error.is_some()).count() < MAX_JUDGE_ATTEMPTS
        })
        .collect();
    let todo = &todo[..limit.map_or(todo.len(), |n| n.min(todo.len()))];
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)?;
    for reply in todo {
        let Some(briefing) = briefings.iter().find(|b| b.id == reply.briefing_id) else {
            continue;
        };
        let (fabricated, quotations) =
            splinter_sdk::measure::verifiers::answer::fabricated(&reply.text, corpus);
        let judged = if reply.text.trim().is_empty() {
            Ok(vec![false; briefing.key_points.len()])
        } else {
            crate::judge::cover(helper, briefing, &reply.text)
        };
        let (covered, error) = match judged {
            Ok(covered) => (covered, None),
            Err(e) => (Vec::new(), Some(format!("{e:#}"))),
        };
        let scored = Scored {
            briefing_id: reply.briefing_id.clone(),
            arm: reply.arm.clone(),
            covered,
            fabricated,
            quotations,
            anachronism: crate::grounding::anachronism(&reply.text),
            error,
        };
        writeln!(file, "{}", serde_json::to_string(&scored)?)?;
        file.flush()?;
    }
    Ok(todo.len())
}

/// The scored replies in `path`.
///
/// # Errors
/// The file exists and a line is not a scored reply.
pub fn read_scores(path: &std::path::Path) -> anyhow::Result<Vec<Scored>> {
    read_lines(path)
}

/// Two arms compared on the briefings both were scored on.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub n: usize,
    /// Mean share of rubric items done, per arm.
    pub coverage_before: f64,
    pub coverage_after: f64,
    /// Briefings where only the second arm covered more, and only the first.
    pub gained: usize,
    pub lost: usize,
    pub p_value: f64,
    /// Share of replies with a fabricated quotation, and with an anachronism.
    pub fabricated_before: f64,
    pub fabricated_after: f64,
    pub anachronism_before: f64,
    pub anachronism_after: f64,
}

/// Compare two arms' scores on the briefings both were scored on.
pub fn summarize(before: &[Scored], after: &[Scored]) -> Summary {
    let ok = |s: &&Scored| s.error.is_none();
    let after_by: std::collections::HashMap<&str, &Scored> = after
        .iter()
        .filter(ok)
        .map(|s| (s.briefing_id.as_str(), s))
        .collect();
    let pairs: Vec<(&Scored, &Scored)> = before
        .iter()
        .filter(ok)
        .filter_map(|b| after_by.get(b.briefing_id.as_str()).map(|a| (b, *a)))
        .collect();
    let n = pairs.len().max(1) as f64;
    let mean = |pick: &dyn Fn(&(&Scored, &Scored)) -> f64| pairs.iter().map(pick).sum::<f64>() / n;
    let rate = |pick: &dyn Fn(&(&Scored, &Scored)) -> bool| {
        pairs.iter().filter(|p| pick(p)).count() as f64 / n
    };
    let outcomes: Vec<(bool, bool)> = pairs
        .iter()
        .map(|(b, a)| {
            (
                crate::judge::share(&a.covered) > crate::judge::share(&b.covered),
                crate::judge::share(&b.covered) > crate::judge::share(&a.covered),
            )
        })
        .collect();
    Summary {
        n: pairs.len(),
        coverage_before: mean(&|(b, _)| crate::judge::share(&b.covered)),
        coverage_after: mean(&|(_, a)| crate::judge::share(&a.covered)),
        gained: outcomes.iter().filter(|o| o.0).count(),
        lost: outcomes.iter().filter(|o| o.1).count(),
        p_value: splinter_sdk::model::stats::sign_test(&outcomes).p_value,
        fabricated_before: rate(&|(b, _)| b.fabricated > 0),
        fabricated_after: rate(&|(_, a)| a.fabricated > 0),
        anachronism_before: rate(&|(b, _)| b.anachronism.is_some()),
        anachronism_after: rate(&|(_, a)| a.anachronism.is_some()),
    }
}

/// The comparison as text. When the judge did not pass its controls no claim is
/// made about coverage, and only what code measured is reported.
pub fn render(
    summary: &Summary,
    calibration: &crate::judge::Calibration,
    before: &str,
    after: &str,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Replies to {} briefings of held-out letters, {before} against {after}.",
        summary.n
    );
    let _ = writeln!(out, "Judge controls on {} briefings: the real letter is credited with {:.0}% of the items, another letter with {:.0}%.", calibration.briefings, 100.0 * calibration.real_letter, 100.0 * calibration.other_letter);
    if calibration.passes() {
        let _ = writeln!(out, "Coverage of what the real letter does: {:.0}% against {:.0}% (gained {}, lost {}, p {:.4} that {after} covers more).", 100.0 * summary.coverage_before, 100.0 * summary.coverage_after, summary.gained, summary.lost, summary.p_value);
    } else {
        let _ = writeln!(out, "The judge did not tell the real letter from another well enough: no claim is made about coverage.");
    }
    let _ = writeln!(
        out,
        "Replies with a fabricated quotation: {:.0}% against {:.0}%.",
        100.0 * summary.fabricated_before,
        100.0 * summary.fabricated_after
    );
    let _ = writeln!(
        out,
        "Replies with a word from after his death: {:.0}% against {:.0}%.",
        100.0 * summary.anachronism_before,
        100.0 * summary.anachronism_after
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{doc, helper, words as distinct};

    const LETTER: &str = "My dear Sir, I have received your Favor of the tenth, and I am of Opinion that the Towns ought not to wait upon the Ministry. Let each Town choose a Committee to write to the rest, that the Sense of the People may be known before the Assembly meets. I beg you will send me what Number of Towns have already acted.";

    fn letter() -> Document {
        doc("l1", 1773, "James Warren", LETTER)
    }

    fn good() -> Drafted {
        Drafted {
            situation: format!("In the autumn of 1773 the Province is uneasy over the Governor's salary and the shipments of tea expected at the port. A correspondent in Plymouth reports that his neighbours are divided on whether to petition or to act, and asks how the towns should proceed. {}", distinct("ctx", 20)),
            request: "Advise your correspondent how the towns should proceed.".into(),
            key_points: vec![
                KeyPoint { point: "he advises not waiting on the ministry".into(), quote: "the Towns ought not to wait upon the Ministry".into() },
                KeyPoint { point: "he proposes committees in each town to write to the rest".into(), quote: "Let each Town choose a Committee to write to the rest".into() },
                KeyPoint { point: "he asks how many towns have already acted".into(), quote: "send me what Number of Towns have already acted".into() },
            ],
        }
    }

    #[test]
    fn a_situation_told_in_the_helpers_words_with_a_grounded_rubric_passes() {
        assert_eq!(gate(&good(), &letter()), Ok(()));
    }

    #[test]
    fn a_rubric_point_whose_quotation_is_not_in_the_letter_is_refused() {
        let mut d = good();
        d.key_points[1].quote = "the committees shall assemble at Boston in December".into();
        let why = gate(&d, &letter()).unwrap_err();
        assert!(why.contains("not in the letter"), "{why}");
    }

    #[test]
    fn a_quotation_too_short_to_ground_a_point_is_refused() {
        let mut d = good();
        d.key_points[0].quote = "not wait".into();
        assert!(gate(&d, &letter()).unwrap_err().contains("short"));
    }

    #[test]
    fn the_rubric_names_three_to_six_things() {
        let mut few = good();
        few.key_points.truncate(2);
        assert!(gate(&few, &letter()).unwrap_err().contains("key points"));
        let mut many = good();
        let extra = many.key_points[0].clone();
        many.key_points.extend(std::iter::repeat_n(extra, 5));
        assert!(gate(&many, &letter()).unwrap_err().contains("key points"));
    }

    #[test]
    fn a_situation_that_repeats_the_letters_words_gives_the_answer_away_and_is_refused() {
        let mut d = good();
        d.situation.push_str(" Let each Town choose a Committee to write to the rest, that the Sense of the People may be known.");
        assert!(gate(&d, &letter())
            .unwrap_err()
            .contains("letter's own words"));
        let mut request = good();
        request.request = "Let each Town choose a Committee to write to the rest of them".into();
        assert!(gate(&request, &letter())
            .unwrap_err()
            .contains("letter's own words"));
    }

    #[test]
    fn a_situation_too_thin_to_answer_is_refused() {
        let mut d = good();
        d.situation = "A man writes to him.".into();
        assert!(gate(&d, &letter()).unwrap_err().contains("short"));
    }

    #[test]
    fn the_student_is_shown_the_situation_and_the_request_and_never_the_rubric() {
        let b = Briefing {
            id: "recon-1".into(),
            doc_id: "l1".into(),
            situation: good().situation,
            request: good().request,
            key_points: good().key_points,
        };
        let p = prompt(&b);
        assert!(p.contains("autumn of 1773") && p.contains("Advise your correspondent"));
        assert!(
            !p.contains("Committee to write to the rest") && !p.contains("ought not to wait"),
            "{p}"
        );
    }

    #[test]
    fn a_briefing_is_drafted_from_the_letter_and_the_answer_is_not_in_what_the_student_sees() {
        let json = serde_json::to_string(&good()).unwrap();
        let (h, provider) = helper(&[&json]);
        let b = brief(&h, &letter()).unwrap();
        assert_eq!((b.doc_id.as_str(), b.key_points.len()), ("l1", 3));
        assert!(b.id.starts_with("recon-"));
        assert!(
            provider.sent().contains("Towns ought not to wait"),
            "the helper reads the letter"
        );
        assert!(!prompt(&b).contains("Towns ought not to wait"));
    }

    #[test]
    fn a_draft_that_fails_the_gate_is_corrected_and_one_that_never_passes_is_an_error() {
        let mut leaky = good();
        leaky.situation.push_str(" Let each Town choose a Committee to write to the rest, that the Sense of the People may be known.");
        let (h, provider) = helper(&[
            &serde_json::to_string(&leaky).unwrap(),
            &serde_json::to_string(&good()).unwrap(),
        ]);
        assert_eq!(brief(&h, &letter()).unwrap().situation, good().situation);
        assert!(provider.requests() >= 2);
        let (stuck, _) = helper(&[&serde_json::to_string(&leaky).unwrap()]);
        assert!(brief(&stuck, &letter()).is_err());
    }

    // ---- replies, scores and the comparison ----

    use splinter_sdk::measure::verifiers::quotation::TextIndex;

    fn briefing(id: &str) -> Briefing {
        Briefing {
            id: id.into(),
            doc_id: "l1".into(),
            situation: good().situation,
            request: good().request,
            key_points: good().key_points,
        }
    }

    fn reply_of(text: &str) -> splinter_sdk::model::answer::Reply {
        splinter_sdk::model::answer::Reply {
            text: text.into(),
            tokens: 7,
            ..Default::default()
        }
    }

    fn out() -> std::path::PathBuf {
        tempfile::tempdir().unwrap().keep().join("out.jsonl")
    }

    #[test]
    fn a_reply_is_kept_in_full_per_arm_and_a_run_resumes_where_it_stopped() {
        let briefings = [briefing("b1"), briefing("b2"), briefing("b3")];
        let path = out();
        let long = "x ".repeat(900);
        assert_eq!(
            answer_all(&briefings, &path, "base", Some(1), &mut |_| Ok(reply_of(
                &long
            )))
            .unwrap(),
            1
        );
        let first = read_replies(&path).unwrap();
        assert!(
            first[0].text.len() > 1000,
            "the reply is not cut for the record"
        );
        let mut asked = Vec::new();
        answer_all(&briefings, &path, "base", None, &mut |b| {
            asked.push(b.id.clone());
            Ok(reply_of("ok"))
        })
        .unwrap();
        assert_eq!(asked, ["b2", "b3"]);
        assert_eq!(
            answer_all(&briefings, &path, "tuned", None, &mut |_| Ok(reply_of(
                "ok"
            )))
            .unwrap(),
            3,
            "another arm answers the same briefings"
        );
    }

    #[test]
    fn a_reply_the_model_gave_inside_its_reasoning_block_counts_and_one_that_ran_out_does_not() {
        let path = out();
        answer_all(&[briefing("b1")], &path, "base", None, &mut |_| {
            Ok(splinter_sdk::model::answer::Reply {
                thinking: "Dear Sir, I answer.".into(),
                ..Default::default()
            })
        })
        .unwrap();
        answer_all(&[briefing("b2")], &path, "base", None, &mut |_| {
            Ok(splinter_sdk::model::answer::Reply {
                thinking: "still weighing".into(),
                truncated: true,
                ..Default::default()
            })
        })
        .unwrap();
        let replies = read_replies(&path).unwrap();
        assert_eq!(replies[0].text, "Dear Sir, I answer.");
        assert_eq!(replies[1].text, "");
    }

    fn corpus() -> TextIndex {
        TextIndex::new([LETTER])
    }

    fn reply(id: &str, arm: &str, text: &str) -> Reply {
        Reply {
            briefing_id: id.into(),
            arm: arm.into(),
            text: text.into(),
            tokens: 5,
            truncated: false,
        }
    }

    fn judging(covered: &[bool]) -> Helper {
        let v = serde_json::json!({"covered": covered}).to_string();
        crate::testkit::responding(move |_| v.clone()).0
    }

    #[test]
    fn a_reply_is_scored_by_the_judge_and_by_code() {
        let b = [briefing("b1")];
        let quote = "As I wrote, \"Let each Town choose a Committee to write to the rest\" always.";
        let invented = "As I wrote, \"liberty is the first gift of nature to every citizen of the commonwealth\" always, and I would use the internet.";
        let path = out();
        score_all(
            &judging(&[true, false, true]),
            &b,
            &[reply("b1", "base", invented)],
            &corpus(),
            &path,
            None,
        )
        .unwrap();
        let s = &read_scores(&path).unwrap()[0];
        assert_eq!(s.covered, [true, false, true]);
        assert_eq!((s.fabricated, s.quotations), (1, 1));
        assert_eq!(s.anachronism.as_deref(), Some("internet"));
        let real = out();
        score_all(
            &judging(&[true, true, true]),
            &b,
            &[reply("b1", "base", quote)],
            &corpus(),
            &real,
            None,
        )
        .unwrap();
        assert_eq!(
            read_scores(&real).unwrap()[0].fabricated,
            0,
            "a quotation that is in the letter is not fabricated"
        );
    }

    #[test]
    fn an_empty_reply_scores_nothing_without_asking_the_judge() {
        let (h, provider) = helper(&["not used"]);
        let path = out();
        score_all(
            &h,
            &[briefing("b1")],
            &[reply("b1", "base", "  ")],
            &corpus(),
            &path,
            None,
        )
        .unwrap();
        assert_eq!(
            read_scores(&path).unwrap()[0].covered,
            [false, false, false]
        );
        assert_eq!(provider.requests(), 0);
    }

    #[test]
    fn a_judge_that_fails_on_a_reply_is_recorded_retried_twice_and_then_given_up_on() {
        let (h, _) = helper(&["not json"]);
        let path = out();
        let replies = [reply("b1", "base", "Dear Sir, a reply.")];
        for _ in 0..3 {
            score_all(&h, &[briefing("b1")], &replies, &corpus(), &path, None).unwrap();
        }
        let scores = read_scores(&path).unwrap();
        assert_eq!(scores.len(), 2, "two attempts and no more");
        assert!(scores.iter().all(|s| s.error.is_some()));
    }

    #[test]
    fn a_scored_reply_is_not_scored_again() {
        let path = out();
        let replies = [reply("b1", "base", "Dear Sir, a reply.")];
        let h = judging(&[true, true, false]);
        assert_eq!(
            score_all(&h, &[briefing("b1")], &replies, &corpus(), &path, None).unwrap(),
            1
        );
        assert_eq!(
            score_all(&h, &[briefing("b1")], &replies, &corpus(), &path, None).unwrap(),
            0
        );
    }

    fn scored(id: &str, covered: &[bool], fabricated: usize, anachronism: bool) -> Scored {
        Scored {
            briefing_id: id.into(),
            arm: "x".into(),
            covered: covered.to_vec(),
            fabricated,
            quotations: fabricated,
            anachronism: anachronism.then(|| "internet".to_string()),
            error: None,
        }
    }

    #[test]
    fn two_arms_are_compared_on_the_briefings_both_were_scored_on() {
        let before = [
            scored("b1", &[true, false, false], 1, true),
            scored("b2", &[false, false, false], 0, false),
            scored("only-before", &[true, true, true], 0, false),
        ];
        let after = [
            scored("b1", &[true, true, true], 0, false),
            scored("b2", &[false, false, false], 0, false),
            Scored {
                error: Some("x".into()),
                covered: vec![],
                ..scored("b3", &[], 0, false)
            },
        ];
        let s = summarize(&before, &after);
        assert_eq!(s.n, 2);
        assert!(
            (s.coverage_before - 1.0 / 6.0).abs() < 1e-9 && (s.coverage_after - 0.5).abs() < 1e-9
        );
        assert_eq!((s.gained, s.lost), (1, 0));
        assert!((s.fabricated_before - 0.5).abs() < 1e-9 && s.fabricated_after == 0.0);
        assert!((s.anachronism_before - 0.5).abs() < 1e-9 && s.anachronism_after == 0.0);
    }

    #[test]
    fn a_judge_that_failed_its_controls_leaves_no_claim_about_coverage_but_the_code_measures_stand()
    {
        let s = summarize(
            &[scored("b1", &[true, false, false], 1, true)],
            &[scored("b1", &[true, true, true], 0, false)],
        );
        let failed = crate::judge::Calibration {
            real_letter: 0.4,
            other_letter: 0.6,
            briefings: 10,
        };
        let text = render(&s, &failed, "base", "tuned");
        assert!(
            text.contains("no claim is made about coverage")
                && !text.contains("Coverage of what the real letter does"),
            "{text}"
        );
        assert!(text.contains("fabricated quotation") && text.contains("after his death"));
        let passed = crate::judge::Calibration {
            real_letter: 0.9,
            other_letter: 0.1,
            briefings: 10,
        };
        let text = render(&s, &passed, "base", "tuned");
        assert!(
            text.contains("Coverage of what the real letter does") && !text.contains("no claim"),
            "{text}"
        );
    }

    #[test]
    fn letters_are_briefed_once_a_failure_is_recorded_and_retried_a_bounded_number_of_times() {
        let letters = [letter()];
        let refs: Vec<&Document> = letters.iter().collect();
        let path = out();
        let (bad, _) = helper(&["not json"]);
        for _ in 0..3 {
            brief_all(&bad, &refs, &path, None).unwrap();
        }
        let lines = std::fs::read_to_string(&path).unwrap().lines().count();
        assert_eq!(lines, 2, "two attempts and no more");
        assert!(read_briefings(&path).unwrap().is_empty());
        let good_json = serde_json::to_string(&good()).unwrap();
        let (good_helper, _) = helper(&[&good_json]);
        let other = out();
        assert_eq!(brief_all(&good_helper, &refs, &other, None).unwrap(), 1);
        assert_eq!(
            brief_all(&good_helper, &refs, &other, None).unwrap(),
            0,
            "a briefed letter is not briefed again"
        );
        assert_eq!(read_briefings(&other).unwrap().len(), 1);
    }
}
