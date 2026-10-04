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

/// Words a situation must run to be one.
const MIN_SITUATION_WORDS: usize = 60;
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
    /// What the real letter does, three to six things, each with its quotation.
    pub key_points: Vec<KeyPoint>,
}

/// A briefing that passed the gate.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Briefing {
    pub id: String,
    pub doc_id: String,
    pub situation: String,
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
    if repeats(&drafted.situation, &body) {
        return Err("the situation repeats the letter's own words: tell it in your own words, so the answer is not given away".to_string());
    }
    match gives_away(&drafted.situation, &drafted.key_points) {
        Some(why) => Err(why),
        None => Ok(()),
    }
}

/// A key point is given away when this share of its content words is already
/// in the situation.
const GIVEN_AWAY_SHARE: f64 = 0.5;
/// Words shorter than this carry no content.
const CONTENT_WORD_LEN: usize = 4;

/// Verbs that say what Adams does, which is the letter's content.
const ADAMS_ACTS: &[&str] = &[
    "asks",
    "requests",
    "urges",
    "writes",
    "advises",
    "recommends",
    "proposes",
    "argues",
    "states",
    "answers",
    "replies",
    "insists",
    "suggests",
    "warns",
    "calls",
    "encourages",
    "declares",
];

/// Whether `text` says what Adams does: "Adams" followed by one of [`ADAMS_ACTS`].
fn says_what_adams_does(text: &str) -> bool {
    let words = words(text);
    words
        .windows(2)
        .any(|pair| pair[0] == "adams" && ADAMS_ACTS.contains(&pair[1].as_str()))
}

/// The share of `of`'s content words found among `given`.
fn share_present(of: &str, given: &std::collections::HashSet<String>) -> f64 {
    let content: Vec<String> = words(of)
        .into_iter()
        .filter(|w| w.chars().count() >= CONTENT_WORD_LEN)
        .collect();
    if content.is_empty() {
        return 0.0;
    }
    content.iter().filter(|w| given.contains(*w)).count() as f64 / content.len() as f64
}

/// Why the situation hands over the answer, if it does: it says what Adams
/// does, or already states one of the key points.
fn gives_away(situation: &str, points: &[KeyPoint]) -> Option<String> {
    if says_what_adams_does(situation) {
        return Some("the situation says what Adams does or asks: put to him only what he was facing and what the correspondent wants to know, so the answer is not given away".to_string());
    }
    let given: std::collections::HashSet<String> = words(situation).into_iter().collect();
    points.iter().find_map(|p| {
        let share = share_present(&p.quote, &given).max(share_present(&p.point, &given));
        (share >= GIVEN_AWAY_SHARE).then(|| {
            format!(
                "the situation already says {:?}: it gives away what the letter does, so leave it for the reply",
                p.point
            )
        })
    })
}

/// Why a finished briefing hands over its answer, if it does.
#[must_use]
pub fn leaks(briefing: &Briefing) -> Option<String> {
    gives_away(&briefing.situation, &briefing.key_points)
}

/// What every briefing asks, said in code so that no helper can put the
/// answer into it: the situation carries who is asking and what they want to
/// know, and this only says to answer.
pub const REQUEST: &str = "Write the reply you would send.";

/// How the student is asked: the situation and the request, nothing else.
pub fn prompt(briefing: &Briefing) -> String {
    format!("Situation: {}\n\n{REQUEST}\n", briefing.situation)
}

/// The role the helper is given.
const ROLE: &str = "a historian who reads a letter closely and says what situation it answered and what it does, without quoting it";

/// The task the helper is given.
const TASK: &str = "You are given a letter Samuel Adams wrote. Tell, in your own words and without repeating any passage of his, the situation he was answering, as a briefing a person could act on: who is asking, what has happened, what is at stake, and what the correspondent wants to know. Do not say what he answered, advised or asked for: that is what his reply must supply. Then list three to six things the real letter does: an issue it recognises, an action it recommends, something it asks for, a value it invokes. For each, copy exactly the words of the letter that show it.";

/// What the helper reads of a letter, and who it went to.
#[derive(serde::Serialize)]
struct Reading<'a> {
    year: u16,
    recipient: Option<&'a str>,
    letter: &'a str,
}

/// What the helper may write across every attempt, repairs included.
const OUTPUT_TOKENS: u64 = 3500;
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
        .filter(|b| leaks(b).is_none())
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
            !mine.clone().any(|r| {
                r.error.is_none() && r.briefing.as_ref().is_some_and(|b| leaks(b).is_none())
            }) && mine.filter(|r| r.error.is_some()).count() < MAX_BRIEF_ATTEMPTS
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
mod tests;
