// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reading what an author decided out of what the author wrote.
//!
//! A model reads an article and, for each case it tells, writes a neutral
//! briefing of the situation, names the way the narrator acted and quotes the
//! sentences in which he says why. Code decides what is kept: a quotation must
//! be in the article word for word, a briefing must not give the decision away,
//! and a case the reader cannot read is turned away.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::typed::{Measured, TypedCall};
use splinter_sdk::agent::CallError;
use splinter_sdk::measure::verifiers::quotation::contains_words as contains_run;

use crate::cache::StoredArticle;
pub use crate::gates::MinedFacts;
use crate::gates::{told_paragraphs, Article, Chose, ReadCase, Reading, QUOTE_RUN_WORDS};
use crate::scenario::Archetype;

/// The fewest words a briefing may have.
pub const MIN_BRIEFING_WORDS: usize = 20;

const ROLE: &str = "You read a steward's essay and set out what it says, exactly.";

const TASK: &str = "Below is a steward's essay and the list of the cases it tells. For each case in `cases` give: `briefing`, 40 to 90 words, plain and neutral, of the facts the essay states about the situation BEFORE he acted - the stakes, the pressure of time, what the charter allowed, how long the matter would bind, whether it could be undone, whether families were hurt, whether an outside house offered help, and which ways were open to him - and NOTHING about what he did or why, in your own words and not the essay's; `offer_made`: whether an outside house offered help; `facts`: whether the essay states that the charter gave no power to do what was wanted (`exceeds_authority`), that delay would cost dearly (`urgent`), that what was settled would bind the community for years (`binds_future`), that what was done could not be undone (`irreversible`) and that families were already suffering (`members_harmed`) - true, false, or null if the essay does not say; `open`: every way the essay says was open to the narrator in that case, the one he took included and no way the essay does not mention, each as `act_alone`, `consult_then_act`, `defer_to_outsider` or `hold_and_wait`; `chose`: `act_alone`, `consult_then_act`, `defer_to_outsider`, `hold_and_wait` or `unclear`; `reasons`: up to three sentences copied word for word from the essay in which the narrator says why he did what he did. Reply as JSON with `cases`.";

const REQUEST_DEADLINE: Duration = Duration::from_secs(900);
const MAX_TOKENS: u64 = 6000;

/// What the reader wrote for one case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MinedCase {
    /// The case, named as given.
    pub case: String,
    /// The situation before the decision.
    pub briefing: String,
    /// Whether an outside house offered help.
    pub offer_made: bool,
    /// The other facts of the case, as the essay states them. Asked for, never
    /// defaulted: a field the schema lists as optional came back empty, and the
    /// essay then failed the gate for facts it states.
    pub facts: MinedFacts,
    /// What the narrator did.
    pub chose: Chose,
    /// Sentences from the essay that give his reasons.
    pub reasons: Vec<String>,
    /// The ways to act the essay says were open to him, the one he took among
    /// them. Asked for, never defaulted, like `facts`.
    pub open: Vec<Archetype>,
}

/// What the reader wrote for an article.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Mined {
    /// One entry per case.
    pub cases: Vec<MinedCase>,
}

/// A decision read out of an article and checked.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verified {
    /// The case.
    pub case: String,
    /// The situation before the decision.
    pub briefing: String,
    /// Whether an outside house offered help.
    pub offer_made: bool,
    /// The other facts of the case, as the essay states them.
    #[serde(default)]
    pub facts: MinedFacts,
    /// What the narrator did.
    pub chose: Archetype,
    /// The ways to act that were open to him, the one he took among them.
    pub open: Vec<Archetype>,
    /// The reasons found word for word in the article.
    pub reasons: Vec<String>,
    /// How many quoted reasons were not found and were dropped.
    pub dropped: usize,
}

fn normal(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// What of `mined` stands, for each case of `article` in order: the case as
/// verified, or why it is turned away.
#[must_use]
pub fn verify(article: &StoredArticle, mined: &Mined) -> Vec<Result<Verified, String>> {
    let text = Article {
        title: article.title.clone(),
        body: article.body.clone(),
    };
    article
        .cases
        .iter()
        .map(|name| {
            let found = mined
                .cases
                .iter()
                .find(|c| c.case.eq_ignore_ascii_case(name))
                .ok_or_else(|| format!("{name}: the reader did not write it up"))?;
            let chose = found
                .chose
                .archetype()
                .ok_or_else(|| format!("{name}: the reader found the decision unclear"))?;
            if found.briefing.split_whitespace().count() < MIN_BRIEFING_WORDS {
                return Err(format!("{name}: the briefing is too short"));
            }
            if !found.open.contains(&chose)
                || found.open.contains(&Archetype::DeferToOutsider) != found.offer_made
            {
                return Err(format!(
                    "{name}: the ways open do not match what he did or the offer"
                ));
            }
            let told = normal(&told_paragraphs(&text, name));
            let (kept, lost): (Vec<_>, Vec<_>) = found
                .reasons
                .iter()
                .map(|r| (r.trim().to_string(), normal(r)))
                .filter(|(_, words)| !words.is_empty())
                .partition(|(_, words)| contains_run(&told, words));
            let briefing = normal(&found.briefing);
            for (_, words) in &kept {
                if words
                    .windows(QUOTE_RUN_WORDS)
                    .any(|w| contains_run(&briefing, w))
                {
                    return Err(format!("{name}: the briefing gives away the decision"));
                }
            }
            Ok(Verified {
                case: name.clone(),
                briefing: found.briefing.trim().to_string(),
                offer_made: found.offer_made,
                facts: found.facts,
                chose,
                open: found.open.clone(),
                reasons: kept.into_iter().map(|(r, _)| r).collect(),
                dropped: lost.len(),
            })
        })
        .collect()
}

/// The call that reads `body`, and what the model is shown with it.
fn reading_call(body: &str, cases: &[String]) -> (TypedCall<Mined>, serde_json::Value) {
    let shown = serde_json::json!({"essay": body, "cases": cases});
    let call = TypedCall::<Mined>::new("mine_decisions", TASK, ROLE, REQUEST_DEADLINE)
        .max_output_tokens(MAX_TOKENS)
        .repairs(1);
    (call, shown)
}

/// Asks `model` to read the essay `body`, which tells `cases`, and returns what
/// it wrote, unchecked, with the tokens the call used. The result is an error
/// when the model could not be reached (`CallError::Infrastructure`) or did
/// not answer in the shape asked for.
pub fn read_raw(model: &BlockingModel, body: &str, cases: &[String]) -> Measured<Mined> {
    let (call, shown) = reading_call(body, cases);
    model.call_measured(call, &shown)
}

/// [`read_raw`] for the reader that makes the training records: a briefing
/// under [`MIN_BRIEFING_WORDS`] words is sent back once to be written again,
/// since a case whose briefing stays short is turned away by [`verify`] and
/// leaves no record. An answer that is still short after the correction is
/// returned as it is, for `verify` to turn away.
pub fn read_decisions(model: &BlockingModel, body: &str, cases: &[String]) -> Measured<Mined> {
    let (call, shown) = reading_call(body, cases);
    let last: Arc<Mutex<Option<Mined>>> = Arc::default();
    let seen = Arc::clone(&last);
    let call = call.postcondition(move |mined: &Mined| {
        *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(mined.clone());
        let short: Vec<String> = mined
            .cases
            .iter()
            .filter(|c| c.briefing.split_whitespace().count() < MIN_BRIEFING_WORDS)
            .map(|c| {
                format!(
                    "the briefing of {} has {} words; write each briefing in 40 to 90 words",
                    c.case,
                    c.briefing.split_whitespace().count()
                )
            })
            .collect();
        if short.is_empty() {
            Ok(())
        } else {
            Err(short.join("; "))
        }
    });
    let mut measured = model.call_measured(call, &shown);
    if matches!(measured.result, Err(CallError::Postcondition { .. })) {
        if let Some(mined) = last.lock().unwrap_or_else(PoisonError::into_inner).take() {
            measured.result = Ok(mined);
        }
    }
    measured
}

impl Mined {
    /// What was read, as a reading of the decisions and facts alone.
    #[must_use]
    pub fn as_reading(&self) -> Reading {
        Reading {
            cases: self
                .cases
                .iter()
                .map(|c| ReadCase {
                    case: c.case.clone(),
                    chose: c.chose,
                    offer_made: Some(c.offer_made),
                    facts: c.facts,
                    open: c.open.clone(),
                })
                .collect(),
        }
    }
}

/// The most tokens a reading of an essay may run to.
pub const MAX_READING_TOKENS: u64 = MAX_TOKENS;
