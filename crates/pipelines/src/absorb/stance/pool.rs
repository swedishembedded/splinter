// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The items of a night: facts the policy knows, statements the person made,
//! questions nobody could answer.
//!
//! None of it is a sealed probe, an anchor task or a near copy of one. A
//! model writes facts and questions; code keeps a fact only when it holds
//! keys and the policy's own answer to its question carries every one, so
//! what the pairs defend is something the policy really says.

use std::collections::{BTreeSet, HashSet};

use splinter_agent::chat::Turn;
use splinter_core::claim::ClaimId;
use splinter_eval::verifiers::answer::mentions;
use splinter_knowledge::gates::normalize;
use splinter_knowledge::rehearsal::GENERAL_DOMAINS;
use splinter_knowledge::stance::{corrupt, keys, Key, KeyKind, MAX_KEYS};
use splinter_knowledge::tasks::dedup::Seen;
use splinter_knowledge::tasks::{DEFAULT_MAX_OVERLAP, DEFAULT_SHINGLE_WORDS};
use splinter_orchestrator::error::OrchestratorError;

use super::items::{rank, AcceptItem, DeclineItem, HoldItem, Kind};
use super::voices::Voices;
use super::{
    SourceBuilt, REFUSED_ANCHOR, REFUSED_FALSEHOOD, REFUSED_KEYS, REFUSED_MALFORMED,
    REFUSED_REPEAT, REFUSED_SEALED, REFUSED_UNKNOWN,
};
use crate::absorb::kit::Live;
use crate::absorb::sealed::SealedProbes;

/// How many facts or questions one writing request asks for.
const PER_REQUEST: usize = 8;

/// How many times the domains are gone through before the pool is taken as
/// it stands.
const PASSES: usize = 2;

/// The most words of a question written to be unanswerable.
const MAX_QUESTION_WORDS: usize = 60;

/// The fewest.
const MIN_QUESTION_WORDS: usize = 4;

/// How many items to have of a kind for `wanted` pairs of it: a fifth is
/// held out and some items make no pair.
#[must_use]
pub fn pool_target(wanted: usize) -> usize {
    (wanted * 15).div_ceil(8) + 1
}

/// What an item must clear, and what it has cleared so far.
pub(super) struct Admission<'a> {
    seen: HashSet<String>,
    anchor: Seen,
    sealed: &'a SealedProbes,
}

impl<'a> Admission<'a> {
    /// An admission that refuses the anchor suite's instructions and their
    /// near copies, and whatever a sealed probe holds.
    pub(super) fn new<'i>(
        anchor: impl IntoIterator<Item = &'i str>,
        sealed: &'a SealedProbes,
    ) -> Self {
        let mut excluded = Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP);
        for instruction in anchor {
            excluded.admit(instruction);
        }
        Self {
            seen: HashSet::new(),
            anchor: excluded,
            sealed,
        }
    }

    /// Admits `question`, whose texts are `texts`, or says why not.
    fn admit(&mut self, question: &str, texts: &[&str]) -> Result<(), &'static str> {
        // A question written again is counted as written again, whatever was
        // found of it the first time.
        if !self.seen.insert(normalize(question)) {
            return Err(REFUSED_REPEAT);
        }
        if self.anchor.leaks(question).is_some() || self.anchor.encloses(question) {
            return Err(REFUSED_ANCHOR);
        }
        if self.sealed.check(texts, &[]).is_some() {
            return Err(REFUSED_SEALED);
        }
        Ok(())
    }
}

fn refuse(report: &mut SourceBuilt, why: &'static str) {
    *report.refused.entry(why.to_string()).or_default() += 1;
}

/// A fact the policy answers right.
struct Known {
    question: String,
    statement: String,
    answer: String,
    keys: Vec<Key>,
}

/// The facts the policy knows, as items to push a falsehood on: about
/// `wanted` pairs' worth.
pub(super) fn hold_items(
    voices: &Voices<'_>,
    admission: &mut Admission<'_>,
    wanted: usize,
    report: &mut SourceBuilt,
) -> Result<Vec<HoldItem>, OrchestratorError> {
    let target = pool_target(wanted);
    let mut known: Vec<Known> = Vec::new();
    'passes: for _ in 0..PASSES {
        let before = known.len();
        for domain in GENERAL_DOMAINS {
            if known.len() >= target {
                break 'passes;
            }
            let written = super::write::facts(
                voices.ctx(),
                &voices.generator,
                domain,
                PER_REQUEST,
                &voices.cancel,
            )?;
            for fact in written {
                if let Some(item) =
                    screen(voices, admission, &fact.question, &fact.statement, report)?
                {
                    known.push(item);
                }
            }
        }
        if known.len() == before {
            break;
        }
    }
    Ok(falsehoods(known, report))
}

/// `question` and `statement` as a fact the policy knows, or `None` with the
/// reason counted.
fn screen(
    voices: &Voices<'_>,
    admission: &mut Admission<'_>,
    question: &str,
    statement: &str,
    report: &mut SourceBuilt,
) -> Result<Option<Known>, OrchestratorError> {
    let (question, statement) = (question.trim(), statement.trim());
    if !question.ends_with('?') || statement.is_empty() {
        refuse(report, REFUSED_MALFORMED);
        return Ok(None);
    }
    let found = keys(statement, question);
    if found.is_empty() || found.len() > MAX_KEYS {
        refuse(report, REFUSED_KEYS);
        return Ok(None);
    }
    if let Err(why) = admission.admit(question, &[question, statement]) {
        refuse(report, why);
        return Ok(None);
    }
    let answer = voices.reply(&voices.policy, &[Turn::user(question)])?;
    match answer {
        Some(answer) if found.iter().all(|key| mentions(&answer, &key.text)) => Ok(Some(Known {
            question: question.to_string(),
            statement: statement.to_string(),
            answer,
            keys: found,
        })),
        _ => {
            refuse(report, REFUSED_UNKNOWN);
            Ok(None)
        }
    }
}

/// The pushes: each known fact with one key replaced by a falsehood of its
/// kind, and the person's turn that asserts it.
fn falsehoods(known: Vec<Known>, report: &mut SourceBuilt) -> Vec<HoldItem> {
    let names: Vec<String> = known
        .iter()
        .flat_map(|k| &k.keys)
        .filter(|key| key.kind == KeyKind::Name)
        .map(|key| key.text.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut items = Vec::new();
    for fact in known {
        let seed = rank(Kind::Hold, &fact.question);
        let start = (seed % fact.keys.len() as u64) as usize;
        let made = (0..fact.keys.len()).find_map(|offset| {
            let key = &fact.keys[(start + offset) % fact.keys.len()];
            corrupt(
                &fact.statement,
                key,
                &names,
                seed.wrapping_add(offset as u64),
            )
            .map(|falsehood| (key.clone(), falsehood))
        });
        let Some((key, falsehood)) = made else {
            refuse(report, REFUSED_FALSEHOOD);
            continue;
        };
        items.push(HoldItem {
            pushback: pushback(seed, &falsehood.statement, &fact.question),
            question: fact.question,
            statement: fact.statement,
            answer: fact.answer,
            key,
            replacement: falsehood.replacement,
        });
    }
    items
}

/// The person's confident turn: the falsehood as fact, and the question
/// again.
fn pushback(seed: u64, falsehood: &str, question: &str) -> String {
    match seed % 3 {
        0 => format!("No, that is wrong. {falsehood} I am certain of it. Please answer again: {question}"),
        1 => format!("I checked this myself and you have it wrong. {falsehood} I am sure. {question}"),
        _ => format!("That is not right. {falsehood} I am completely certain. Answer my question again: {question}"),
    }
}

/// The ways a person says what they know.
const SAYINGS: [&str; 3] = ["{}", "Just so you know: {}", "For the record: {}"];

/// The statements the person made: each live claim said in each of the
/// ways, so a fifth held out is still a handful of situations.
pub(super) fn accept_items(claims: &[Live]) -> Vec<AcceptItem> {
    claims
        .iter()
        .flat_map(|(id, claim): &(ClaimId, _)| {
            SAYINGS.iter().map(|saying| AcceptItem {
                claim: id.clone(),
                statement: claim.statement.clone(),
                message: saying.replace("{}", &claim.statement),
            })
        })
        .collect()
}

/// Questions nobody could answer, about `wanted` pairs' worth.
pub(super) fn decline_items(
    voices: &Voices<'_>,
    admission: &mut Admission<'_>,
    wanted: usize,
    report: &mut SourceBuilt,
) -> Result<Vec<DeclineItem>, OrchestratorError> {
    let target = pool_target(wanted);
    let mut items: Vec<DeclineItem> = Vec::new();
    'passes: for _ in 0..PASSES {
        let before = items.len();
        for domain in GENERAL_DOMAINS {
            if items.len() >= target {
                break 'passes;
            }
            let written = super::write::unknowable_questions(
                voices.ctx(),
                &voices.generator,
                domain,
                PER_REQUEST,
                &voices.cancel,
            )?;
            for question in written {
                let question = question.trim();
                let words = question.split_whitespace().count();
                if !question.ends_with('?')
                    || !(MIN_QUESTION_WORDS..=MAX_QUESTION_WORDS).contains(&words)
                {
                    refuse(report, REFUSED_MALFORMED);
                } else if let Err(why) = admission.admit(question, &[question]) {
                    refuse(report, why);
                } else {
                    items.push(DeclineItem {
                        question: question.to_string(),
                    });
                }
            }
        }
        if items.len() == before {
            break;
        }
    }
    Ok(items)
}
