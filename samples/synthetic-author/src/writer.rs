// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Writes the corpus of an author: one long article per set of episodes, asked
//! of a model, kept only if code can see it enacts them.
//!
//! The writer is told the author's convictions privately and the decision each
//! episode ends in. It is told never to state a conviction as a rule, so the
//! convictions can only be learned from what is decided. A run has a budget in
//! requests and in tokens and stops before it is exceeded.

use std::time::Duration;

use serde_json::json;
use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::typed::TypedCall;

use crate::cache::{Cache, StoredArticle, Usage};
use crate::episode::{episodes_for, Episode};
use crate::gates::{check, Article};
use crate::persona::Persona;
use crate::scenario::{Archetype, TOPICS};

/// Topics an author writes about. The rest are kept back to test what was
/// learned on topics it never wrote about.
pub const TRAIN_TOPICS: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Topics no article is written about.
pub const HELD_OUT_TOPICS: [usize; 2] = [10, 11];

/// The most tokens one article may run to.
const MAX_ARTICLE_TOKENS: u64 = 3500;
/// How long one request may take, corrections included.
const REQUEST_DEADLINE: Duration = Duration::from_secs(900);
/// Consecutive failed requests after which the run stops.
const MAX_CONSECUTIVE_ERRORS: usize = 3;

const ROLE: &str =
    "You are a retired steward of a small community, writing in plain, concrete prose.";

const TASK: &str = "Write one long first-person essay - a steward's reflections for a younger colleague - of at least `min_words` words on the topic given. Tell EVERY episode in `episodes`: give each its own paragraph or paragraphs that name the case exactly as given in `case`, say what you did (`you_did` says what it was; put it in your own words so that it is unmistakable), and say why, from experience, as a story and as observation. Your private `convictions` shape what you notice and how you judge. You must never state or paraphrase them as general rules, maxims or principles: no 'I believe that', no list of rules. Let them show only in what you do and how you account for it. Separate paragraphs with a blank line. Reply as JSON with `title` and `body`.";

/// What a run may spend and how it writes.
#[derive(Clone, Debug)]
pub struct Options {
    /// How many articles the corpus has.
    pub articles: usize,
    /// How many episodes each article tells.
    pub episodes_per_article: usize,
    /// The fewest words an article may have.
    pub min_words: usize,
    /// How many times an article is asked for before it is given up on.
    pub attempts_per_article: usize,
    /// The most requests this run may make.
    pub max_calls: usize,
    /// The most output tokens this run may write, estimated.
    pub max_output_tokens: u64,
}

/// What a run did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Articles written by this run.
    pub written: usize,
    /// Articles already kept.
    pub skipped: usize,
    /// Articles given up on after their attempts.
    pub abandoned: usize,
    /// Attempts turned away by the gates.
    pub rejected: usize,
    /// Requests made by this run.
    pub calls: usize,
    /// Estimated tokens shown to the writer by this run.
    pub input_tokens: u64,
    /// Estimated tokens the writer wrote for this run.
    pub output_tokens: u64,
    /// Why the run stopped early, if it did.
    pub stopped: Option<String>,
}

fn tokens(text: &str) -> u64 {
    text.chars().count().div_ceil(4) as u64
}

fn you_did(option: Archetype, episode: &Episode) -> String {
    match option {
        Archetype::ActAlone => "settled it yourself, at once, without waiting".into(),
        Archetype::ConsultThenAct => "put it before the assembly and acted on its word".into(),
        Archetype::DeferToOutsider => format!("accepted the offer of {}", episode.case.house),
        Archetype::HoldAndWait => "waited, and did nothing yet".into(),
    }
}

/// What the writer is shown for one article.
fn input(
    persona: &Persona,
    topic: usize,
    episodes: &[Episode],
    min_words: usize,
) -> serde_json::Value {
    json!({
        "author": persona.name,
        "topic": TOPICS[topic],
        "min_words": min_words,
        "convictions": persona.beliefs.iter().map(|b| b.statement()).collect::<Vec<_>>(),
        "episodes": episodes.iter().map(|e| json!({
            "case": e.case.name,
            "situation": e.situation,
            "you_did": you_did(persona.decide(&e.scenario).choice, e),
        })).collect::<Vec<_>>(),
    })
}

/// Writes the corpus of `persona` into `cache`, within `opts`.
///
/// # Errors
/// The cache cannot be written, or the writer fails on every request.
pub fn write_corpus(
    persona: &Persona,
    seed: u64,
    writer: &BlockingModel,
    cache: &Cache,
    opts: &Options,
) -> anyhow::Result<Summary> {
    cache.write_persona(persona)?;
    let mut summary = Summary::default();
    let mut total = cache.usage();
    let mut used = Vec::new();
    let mut errors = 0;
    'articles: for index in 0..opts.articles {
        let topic = TRAIN_TOPICS[index % TRAIN_TOPICS.len()];
        // Drawn for every index, kept or not, so that what article `index` tells
        // does not depend on what was written before it.
        let episodes = episodes_for(persona, seed, topic, opts.episodes_per_article, &mut used);
        if cache.has_article(index) {
            summary.skipped += 1;
            continue;
        }
        let shown = input(persona, topic, &episodes, opts.min_words);
        let shown_tokens = tokens(TASK) + tokens(&shown.to_string());
        for attempt in 1..=opts.attempts_per_article {
            if summary.calls >= opts.max_calls {
                summary.stopped = Some(format!("the budget of {} calls is spent", opts.max_calls));
                break 'articles;
            }
            if summary.output_tokens + MAX_ARTICLE_TOKENS > opts.max_output_tokens {
                summary.stopped = Some(format!(
                    "another article could pass the budget of {} output tokens",
                    opts.max_output_tokens
                ));
                break 'articles;
            }
            let call = TypedCall::<Article>::new("write_article", TASK, ROLE, REQUEST_DEADLINE)
                .max_output_tokens(MAX_ARTICLE_TOKENS)
                .repairs(1);
            summary.calls += 1;
            summary.input_tokens += shown_tokens;
            let article = match writer.call(call, &shown) {
                Ok(article) => {
                    errors = 0;
                    article
                }
                Err(e) => {
                    errors += 1;
                    cache.note_rejected(index, attempt, &format!("the writer failed: {e}"))?;
                    summary.rejected += 1;
                    if errors >= MAX_CONSECUTIVE_ERRORS {
                        summary.stopped = Some(format!("{errors} requests in a row failed: {e}"));
                        break 'articles;
                    }
                    continue;
                }
            };
            summary.output_tokens += tokens(&article.title) + tokens(&article.body);
            match check(&article, persona, &episodes, opts.min_words) {
                Ok(()) => {
                    cache.write_article(&StoredArticle {
                        index,
                        topic,
                        title: article.title,
                        body: article.body,
                        cases: episodes.iter().map(|e| e.case.name.clone()).collect(),
                        attempts: attempt,
                    })?;
                    summary.written += 1;
                    eprintln!(
                        "article {}/{}: kept on attempt {attempt}",
                        index + 1,
                        opts.articles
                    );
                    continue 'articles;
                }
                Err(reason) => {
                    cache.note_rejected(index, attempt, &reason.to_string())?;
                    summary.rejected += 1;
                    eprintln!(
                        "article {}/{}: attempt {attempt} turned away: {reason}",
                        index + 1,
                        opts.articles
                    );
                }
            }
        }
        summary.abandoned += 1;
    }
    total.calls += summary.calls;
    total.input_tokens += summary.input_tokens;
    total.output_tokens += summary.output_tokens;
    cache.write_usage(&total)?;
    Ok(summary)
}

/// What a cache has cost so far.
#[must_use]
pub fn spent(cache: &Cache) -> Usage {
    cache.usage()
}
