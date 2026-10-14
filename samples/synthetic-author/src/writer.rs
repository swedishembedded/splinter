// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Writes the corpus of an author: one long article per set of episodes, asked
//! of a model, kept only if code can see it enacts them.
//!
//! The writer is told the author's convictions privately and the decision each
//! episode ends in. It is told never to state a conviction as a rule, so the
//! convictions can only be learned from what is decided. A run has a budget in
//! requests and in tokens and stops before it is exceeded.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use serde_json::json;
use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::typed::TypedCall;

use crate::cache::{Cache, StoredArticle, Usage};
use crate::episode::{did_phrase, episodes_for, Episode};
use crate::gates::{
    check_form, check_markers, check_reading, told_paragraphs, Article, Reading, Rejection,
};
use crate::persona::Persona;
use crate::scenario::TOPICS;

/// Topics an author writes about. The rest are kept back to test what was
/// learned on topics it never wrote about.
pub const TRAIN_TOPICS: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Topics no article is written about.
pub const HELD_OUT_TOPICS: [usize; 2] = [10, 11];

/// The most tokens one article may run to.
const MAX_ARTICLE_TOKENS: u64 = 6000;
/// How long one request may take, corrections included.
const REQUEST_DEADLINE: Duration = Duration::from_secs(900);
/// Consecutive failed requests after which the run stops.
const MAX_CONSECUTIVE_ERRORS_PER_WORKER: usize = 4;

const ROLE: &str =
    "You are a retired steward of a small community, writing in plain, concrete prose.";

const TASK: &str = "Write one long first-person essay - a steward's reflections for a younger colleague - of at least `min_words` words on the topic given. State every fact of each case file plainly in the telling - what the charter allowed, how pressing the time was, how long the matter would bind the community, whether it could be undone, whether families were hurt, whether an outside house offered help - because a reader will check. Tell EVERY episode in `episodes`: give each its own paragraph or paragraphs that name the case exactly as given in `case_file.case` and tell its facts (the `case_file` is a list of facts, not wording: put every fact in your own words), say what you did (`you_did` says what it was; put it in your own words so that it is unmistakable), and say why, from experience, as a story and as observation. Your private `convictions` shape what you notice and how you judge. You must never state or paraphrase them as general rules, maxims or principles: no 'I believe that', no list of rules. Let them show only in what you do and how you account for it. Do not reuse the wording of `case_file` or `you_did`: retell each episode in your own words, as someone remembering it, and vary how you put what you did from one episode to the next. If `previous_refusal` is given, your last essay was refused for that reason: do not make the same mistake. Separate paragraphs with a blank line. Reply as JSON with `title` and `body`.";

const READER_ROLE: &str = "You read a steward's recollections and say plainly what he did.";

const READER_TASK: &str = "Each entry of `cases` is the part of a steward's essay that tells one case. For each, say which way the narrator himself acted in that case: `act_alone` (he decided and acted on his own), `consult_then_act` (he took it to the assembly or the community and acted on its word), `defer_to_outsider` (he accepted the offer of the outside house), `hold_and_wait` (he waited and did nothing yet), or `unclear` if the text does not show it. Also say, for each case, whether an outside house offered help (`offer_made`) and whether the essay states that the charter gave no power to do what was wanted (`exceeds_authority`), that delay would cost dearly (`urgent`), that what was settled would bind the community for years (`binds_future`), that what was done could not be undone (`irreversible`) and that families were already suffering (`members_harmed`), as `facts`: true if it says so, false if it says the opposite, null if it does not say. Ignore the ways he says he did not act. Reply as JSON with `cases`, each with the `case` as given, `chose`, `offer_made` and `facts`.";

/// The most tokens a reading may run to.
const MAX_READING_TOKENS: u64 = 1500;

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
    /// How many articles are written at a time.
    pub workers: usize,
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

/// What the writer is shown for one article.
fn input(
    persona: &Persona,
    topic: usize,
    episodes: &[Episode],
    min_words: usize,
    refusal: Option<&str>,
) -> serde_json::Value {
    json!({
        "author": persona.name,
        "topic": TOPICS[topic],
        "min_words": min_words,
        "previous_refusal": refusal,
        "convictions": persona.beliefs.iter().map(|b| b.note()).collect::<Vec<_>>(),
        "episodes": episodes.iter().map(|e| json!({
            "case_file": e.case_file(),
            "you_did": did_phrase(persona.decide(&e.scenario).choice, &e.case),
        })).collect::<Vec<_>>(),
    })
}

/// The topic and episodes of each article of a corpus: a function of the
/// author and the seed alone, so that article `i` tells the same episodes
/// whether or not the ones before it were written in this run.
#[must_use]
pub fn plan(
    persona: &Persona,
    seed: u64,
    articles: usize,
    per_article: usize,
) -> Vec<(usize, Vec<Episode>)> {
    let mut used = Vec::new();
    (0..articles)
        .map(|index| {
            let topic = TRAIN_TOPICS[index % TRAIN_TOPICS.len()];
            (
                topic,
                episodes_for(persona, seed, topic, per_article, &mut used),
            )
        })
        .collect()
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
    write_corpus_read(persona, seed, writer, None, cache, opts)
}

/// Writes the corpus of `persona` into `cache`, within `opts`, with the
/// decisions of each article judged by `reader` when there is one and by their
/// marker words when there is not. `opts.workers` articles are written at a
/// time; the budget is reserved before a request is made, so it is never
/// passed however many are in flight.
///
/// # Errors
/// The cache cannot be written, or the writer fails on every request.
pub fn write_corpus_read(
    persona: &Persona,
    seed: u64,
    writer: &BlockingModel,
    reader: Option<&BlockingModel>,
    cache: &Cache,
    opts: &Options,
) -> anyhow::Result<Summary> {
    cache.write_persona(persona)?;
    let run = Run {
        persona,
        writer,
        reader,
        cache,
        opts,
        planned: plan(persona, seed, opts.articles, opts.episodes_per_article),
        next: AtomicUsize::new(0),
        state: Mutex::new(State::default()),
    };
    let failed: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..opts.workers.max(1) {
            scope.spawn(|| {
                while let Some(index) = run.take() {
                    if let Err(e) = run.write_article(index) {
                        run.stop(format!("an article could not be kept: {e}"));
                        failed
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .get_or_insert(e);
                    }
                }
            });
        }
    });
    if let Some(e) = failed.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(e);
    }
    let summary = run
        .state
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .summary;
    let mut total = cache.usage();
    total.calls += summary.calls;
    total.input_tokens += summary.input_tokens;
    total.output_tokens += summary.output_tokens;
    cache.write_usage(&total)?;
    Ok(summary)
}

#[derive(Default)]
struct State {
    summary: Summary,
    errors: usize,
    stopped: bool,
}

struct Run<'a> {
    persona: &'a Persona,
    writer: &'a BlockingModel,
    reader: Option<&'a BlockingModel>,
    cache: &'a Cache,
    opts: &'a Options,
    planned: Vec<(usize, Vec<Episode>)>,
    next: AtomicUsize,
    state: Mutex<State>,
}

impl Run<'_> {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The next article nobody has taken, unless the run has stopped.
    fn take(&self) -> Option<usize> {
        if self.state().stopped {
            return None;
        }
        let index = self.next.fetch_add(1, Ordering::SeqCst);
        (index < self.planned.len()).then_some(index)
    }

    fn stop(&self, why: String) {
        let mut state = self.state();
        state.stopped = true;
        state.summary.stopped.get_or_insert(why);
    }

    /// Reserves one request that may write `max_output` tokens, or says the
    /// run has to stop. Reserved tokens are replaced by what was written.
    fn reserve(&self, max_output: u64, shown_tokens: u64) -> bool {
        let mut state = self.state();
        if state.stopped {
            return false;
        }
        let why = if state.summary.calls >= self.opts.max_calls {
            Some(format!(
                "the budget of {} calls is spent",
                self.opts.max_calls
            ))
        } else if state.summary.output_tokens + max_output > self.opts.max_output_tokens {
            Some(format!(
                "another request could pass the budget of {} output tokens",
                self.opts.max_output_tokens
            ))
        } else {
            None
        };
        if let Some(why) = why {
            state.stopped = true;
            state.summary.stopped.get_or_insert(why);
            return false;
        }
        state.summary.calls += 1;
        state.summary.input_tokens += shown_tokens;
        state.summary.output_tokens += max_output;
        true
    }

    /// Replaces a reservation of `reserved` tokens with the `written` ones.
    fn settle(&self, reserved: u64, written: u64) {
        let mut state = self.state();
        state.summary.output_tokens = state.summary.output_tokens - reserved + written;
    }

    fn write_article(&self, index: usize) -> anyhow::Result<()> {
        let (topic, episodes) = (self.planned[index].0, &self.planned[index].1);
        if self.cache.has_article(index) {
            self.state().summary.skipped += 1;
            return Ok(());
        }
        let opts = self.opts;
        let mut refusal: Option<String> = None;
        for attempt in 1..=opts.attempts_per_article {
            let shown = input(
                self.persona,
                topic,
                episodes,
                opts.min_words,
                refusal.as_deref(),
            );
            if !self.reserve(
                MAX_ARTICLE_TOKENS,
                tokens(TASK) + tokens(&shown.to_string()),
            ) {
                return Ok(());
            }
            let call = TypedCall::<Article>::new("write_article", TASK, ROLE, REQUEST_DEADLINE)
                .max_output_tokens(MAX_ARTICLE_TOKENS)
                .repairs(1);
            let article = match self.writer.call(call, &shown) {
                Ok(article) => {
                    self.state().errors = 0;
                    article
                }
                Err(e) => {
                    self.settle(MAX_ARTICLE_TOKENS, 0);
                    self.cache.note_rejected(
                        index,
                        attempt,
                        &format!("the writer failed: {e}"),
                        None,
                    )?;
                    let mut state = self.state();
                    state.summary.rejected += 1;
                    state.errors += 1;
                    if state.errors >= MAX_CONSECUTIVE_ERRORS_PER_WORKER * self.opts.workers.max(1)
                    {
                        state.stopped = true;
                        let errors = state.errors;
                        state
                            .summary
                            .stopped
                            .get_or_insert(format!("{errors} requests in a row failed: {e}"));
                    }
                    continue;
                }
            };
            self.settle(
                MAX_ARTICLE_TOKENS,
                tokens(&article.title) + tokens(&article.body),
            );
            match self.judge(&article, episodes) {
                Ok(()) => {
                    self.cache.write_article(&StoredArticle {
                        index,
                        topic,
                        title: article.title,
                        body: article.body,
                        cases: episodes.iter().map(|e| e.case.name.clone()).collect(),
                        attempts: attempt,
                    })?;
                    self.state().summary.written += 1;
                    eprintln!(
                        "article {}/{}: kept on attempt {attempt}",
                        index + 1,
                        opts.articles
                    );
                    return Ok(());
                }
                Err(reason) => {
                    self.cache.note_rejected(
                        index,
                        attempt,
                        &reason.to_string(),
                        Some(&article),
                    )?;
                    refusal = Some(reason.to_string());
                    self.state().summary.rejected += 1;
                    eprintln!(
                        "article {}/{}: attempt {attempt} turned away: {reason}",
                        index + 1,
                        opts.articles
                    );
                }
            }
        }
        self.state().summary.abandoned += 1;
        Ok(())
    }

    /// Whether the article may join the corpus: its form, then its decisions,
    /// by a reader when there is one and by marker words when there is not.
    fn judge(&self, article: &Article, episodes: &[Episode]) -> Result<(), Rejection> {
        check_form(article, self.persona, episodes, self.opts.min_words)?;
        let Some(reader) = self.reader else {
            return check_markers(article, self.persona, episodes);
        };
        let cases: Vec<_> = episodes
            .iter()
            .map(|e| json!({"case": e.case.name, "text": told_paragraphs(article, &e.case.name)}))
            .collect();
        let shown = json!({ "cases": cases });
        if !self.reserve(
            MAX_READING_TOKENS,
            tokens(READER_TASK) + tokens(&shown.to_string()),
        ) {
            return Err(Rejection::WrongDecision {
                case: "(the budget ended before it was read)".into(),
                expected: self.persona.decide(&episodes[0].scenario).choice,
            });
        }
        let call =
            TypedCall::<Reading>::new("read_decisions", READER_TASK, READER_ROLE, REQUEST_DEADLINE)
                .max_output_tokens(MAX_READING_TOKENS)
                .repairs(1);
        match reader.call(call, &shown) {
            Ok(reading) => {
                self.settle(
                    MAX_READING_TOKENS,
                    tokens(&serde_json::to_string(&reading).unwrap_or_default()),
                );
                check_reading(&reading, self.persona, episodes)
            }
            Err(e) => {
                self.settle(MAX_READING_TOKENS, 0);
                eprintln!("reading failed: {e}");
                Err(Rejection::WrongDecision {
                    case: "(the reader could not read it)".into(),
                    expected: self.persona.decide(&episodes[0].scenario).choice,
                })
            }
        }
    }
}

/// What a cache has cost so far.
#[must_use]
pub fn spent(cache: &Cache) -> Usage {
    cache.usage()
}
