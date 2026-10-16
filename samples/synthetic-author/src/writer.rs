// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Writes the corpus of an author: one long article per set of episodes, asked
//! of a model, kept only if code can see it enacts them.
//!
//! The writer is told the author's convictions privately and the decision each
//! episode ends in. It is told never to state a conviction as a rule, so the
//! convictions can only be learned from what is decided. A run has a budget in
//! requests and in tokens and stops before it is exceeded.

use std::cell::Cell;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use serde_json::json;
use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::agent::typed::{Measured, Spent, TypedCall};
use splinter_sdk::agent::CallError;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::cache::{Cache, Kind, StoredArticle, Usage};
use crate::episode::{did_phrase, episodes_for, Episode};
use crate::gates::{check_form, check_markers, check_reading, Article, Rejection};
use crate::mine::{read_raw, MAX_READING_TOKENS};
use crate::persona::Persona;
use crate::scenario::TOPICS;
use crate::transit::{self, Retried, Transit};

/// Topics an author writes about. The rest are kept back to test what was
/// learned on topics it never wrote about.
pub const TRAIN_TOPICS: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Topics no article is written about.
pub const HELD_OUT_TOPICS: [usize; 2] = [10, 11];

/// The most tokens one article may run to, the reasoning a model does before
/// it writes included: a reasoning model spends several times the essay's own
/// length thinking, and an essay cut off by the cap is lost whole.
const MAX_ARTICLE_TOKENS: u64 = 16_000;
/// How long one request may take, corrections included.
const REQUEST_DEADLINE: Duration = Duration::from_secs(900);
/// Consecutive unusable answers from the writer after which the run stops.
const MAX_CONSECUTIVE_ERRORS_PER_WORKER: usize = 4;

const ROLE: &str =
    "You are a retired steward of a small community, writing in plain, concrete prose.";

/// The instruction the writer is given with every article.
pub const TASK: &str = "Write one long first-person essay - a steward's reflections for a younger colleague - of at least `min_words` words on the topic given. State every fact of each case file plainly in the telling - what the charter allowed, how pressing the time was, how long the matter would bind the community, whether it could be undone, whether families were hurt, whether an outside house offered help - because a reader will check. Say too which ways were open to you, as `case_file.open` lists them: tell each of them in your own words, the one you took and the ones you left, and name no way that is not listed. Tell EVERY episode in `episodes`: give each its own paragraph or paragraphs that name the case exactly as given in `case_file.case` and tell its facts (the `case_file` is a list of facts, not wording: put every fact in your own words), say what you did (`you_did` says what it was; put it in your own words so that it is unmistakable), and say why, from experience, as a story and as observation. Your private `convictions` shape what you notice and how you judge. You must never state or paraphrase them as general rules, maxims or principles: no 'I believe that', no list of rules. Let them show only in what you do and how you account for it. Do not reuse the wording of `case_file` or `you_did`: retell each episode in your own words, as someone remembering it, and vary how you put what you did from one episode to the next. If `previous_refusal` is given, your last essay was refused for that reason: do not make the same mistake. Separate paragraphs with a blank line. Reply as JSON with `title` and `body`.";

/// Keeps the writer's seeds apart from the reader's.
const WRITER_SALT: u64 = 0x5752_4954_4552_0001;
/// Keeps the reader's seeds apart from the writer's.
const READER_SALT: u64 = 0x5245_4144_4552_0002;
/// Keeps the miner's seeds apart from the writer's and the reader's.
const MINER_SALT: u64 = 0x4d49_4e45_5200_0003;

/// Which model of a run is being asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The model that writes the articles.
    Writer,
    /// The model that reads each article back.
    Reader,
    /// The model that reads the decisions out of the kept articles.
    Miner,
}

/// One request of a run: the article, the attempt at it (from one) and who is
/// asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestKey {
    /// The article.
    pub article: usize,
    /// The attempt at the article, from one.
    pub attempt: usize,
    /// Who is asked.
    pub role: Role,
}

impl RequestKey {
    /// The seed of this request's decoding, derived from the run's seed: a
    /// pure function of the run seed and the request, so a request can be
    /// repeated from the manifest, and two attempts at one article draw
    /// differently.
    #[must_use]
    pub fn seed(self, run_seed: u64) -> u64 {
        let salt = match self.role {
            Role::Writer => WRITER_SALT,
            Role::Reader => READER_SALT,
            Role::Miner => MINER_SALT,
        };
        let place = ((self.article as u64) << 24) ^ self.attempt as u64;
        SplitMix64::new(run_seed ^ salt ^ place).next_u64()
    }
}

/// A model lent for one request: borrowed when the run has one model for all
/// requests, owned when it is made for the request.
pub enum Lent<'a> {
    /// The run's own model.
    Borrowed(&'a BlockingModel),
    /// A model made for this request.
    Owned(Box<BlockingModel>),
}

impl Deref for Lent<'_> {
    type Target = BlockingModel;

    fn deref(&self) -> &BlockingModel {
        match self {
            Self::Borrowed(model) => model,
            Self::Owned(model) => model,
        }
    }
}

/// Where a run gets the model that answers a request, so that a model that
/// samples can be given a seed of its own for every request.
pub trait Models: Sync {
    /// The model that answers `request`.
    ///
    /// # Errors
    /// The model cannot be made.
    fn model(&self, request: RequestKey) -> anyhow::Result<Lent<'_>>;
}

impl Models for BlockingModel {
    fn model(&self, _: RequestKey) -> anyhow::Result<Lent<'_>> {
        Ok(Lent::Borrowed(self))
    }
}

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
    /// How many times a request that fails in transit is repeated before the
    /// model counts as unreachable. A repeat uses no attempt at the article.
    pub transport_retries: usize,
    /// The pause before the first repeat; it doubles with each repeat up to a
    /// minute.
    pub transport_backoff: Duration,
}

impl Options {
    /// How requests lost in transit are repeated.
    #[must_use]
    pub fn transit(&self) -> Transit {
        Transit::new(self.transport_retries, self.transport_backoff)
    }
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
    /// Articles left unfinished because the writer or the reader could not be
    /// reached, which says nothing about the articles.
    pub unavailable: usize,
    /// Attempts turned away by the gates.
    pub rejected: usize,
    /// Requests made by this run.
    pub calls: usize,
    /// Tokens shown to the models by this run, as their providers counted
    /// them, else estimated from the text.
    pub input_tokens: u64,
    /// Tokens the models wrote for this run, reasoning included, as their
    /// providers counted them, else estimated from the text.
    pub output_tokens: u64,
    /// Why the run stopped early, if it did.
    pub stopped: Option<String>,
    /// The reason of the last attempt turned away, or of the last answer that
    /// could not be used: what a run that kept nothing has to say for itself.
    pub last_turned_away: Option<String>,
}

fn tokens(text: &str) -> u64 {
    text.chars().count().div_ceil(4) as u64
}

/// What the writer is shown for one article.
#[must_use]
pub fn writer_input(
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
/// The cache cannot be read or written.
pub fn write_corpus(
    persona: &Persona,
    seed: u64,
    writer: &dyn Models,
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
/// The cache cannot be read or written, or its record of what was spent is
/// unreadable.
pub fn write_corpus_read(
    persona: &Persona,
    seed: u64,
    writer: &dyn Models,
    reader: Option<&dyn Models>,
    cache: &Cache,
    opts: &Options,
) -> anyhow::Result<Summary> {
    let spent_before = cache.usage()?;
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
    let mut total = spent_before;
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
    writer: &'a dyn Models,
    reader: Option<&'a dyn Models>,
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

    /// Replaces the reservation `request` made with what the call used: the
    /// tokens its provider counted, else, for an answer, `answer_estimate`
    /// output tokens and the request's estimated input; nothing known of a
    /// call that was not answered counts as nothing spent.
    fn settle(&self, request: &Request<'_>, spent: &Spent, answer_estimate: Option<u64>) {
        let input = spent
            .input_tokens
            .or_else(|| answer_estimate.map(|_| request.shown_tokens))
            .unwrap_or(0);
        let output = spent.output_tokens.or(answer_estimate).unwrap_or(0);
        let mut state = self.state();
        state.summary.input_tokens = state.summary.input_tokens - request.shown_tokens + input;
        state.summary.output_tokens = state.summary.output_tokens - request.max_output + output;
    }

    /// Puts one request to `models`, repeating it, without using an attempt,
    /// while it is lost in transit and the bound allows (see [`transit`]).
    /// Every request made, repeats included, is taken from the budget, and
    /// every loss is noted under the kind of its role with `draft`: in one
    /// line, and with the provider's whole answer once, when the bound is
    /// spent. A reservation is settled here unless the model answered; the
    /// caller settles an answer.
    fn ask<T>(
        &self,
        models: &dyn Models,
        request: &Request<'_>,
        call: impl Fn(&BlockingModel) -> Measured<T>,
    ) -> anyhow::Result<Asked<T>> {
        let (who, kind) = match request.key.role {
            Role::Writer => ("writer", Kind::WriterUnavailable),
            Role::Reader | Role::Miner => ("reader", Kind::ReaderUnavailable),
        };
        let spent = Cell::new(Spent::default());
        let outcome = transit::retry(
            &self.opts.transit(),
            || self.reserve(request.max_output, request.shown_tokens),
            || {
                spent.set(Spent::default());
                let model = models
                    .model(request.key)
                    .map_err(|e| CallError::Precondition(format!("{e:#}")))?;
                let measured = call(&model);
                spent.set(measured.spent);
                measured.result
            },
            |loss| {
                self.settle(request, &spent.get(), None);
                let shown = if loss.last {
                    &loss.detail
                } else {
                    &loss.summary
                };
                self.cache.note_rejected(
                    request.key.article,
                    request.key.attempt,
                    kind,
                    &format!("the {who} could not be reached: {shown}"),
                    request.draft,
                )?;
                eprintln!(
                    "article {}: attempt {}: the {who} could not be reached: {}",
                    request.key.article + 1,
                    request.key.attempt,
                    loss.summary
                );
                Ok(())
            },
        )?;
        Ok(match outcome {
            Retried::Answered(value) => Asked::Answered(value, spent.get()),
            Retried::Refused(e) => {
                self.settle(request, &spent.get(), None);
                Asked::Unreadable(e)
            }
            Retried::Exhausted { last, requests } => Asked::Unreachable(format!(
                "the {who} could not be reached: {last}, in {requests} requests"
            )),
            Retried::Halted => Asked::OutOfBudget,
        })
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
            let shown = writer_input(
                self.persona,
                topic,
                episodes,
                opts.min_words,
                refusal.as_deref(),
            );
            let request = Request {
                key: RequestKey {
                    article: index,
                    attempt,
                    role: Role::Writer,
                },
                max_output: MAX_ARTICLE_TOKENS,
                shown_tokens: tokens(TASK) + tokens(&shown.to_string()),
                draft: None,
            };
            let asked = self.ask(self.writer, &request, |writer| {
                let call = TypedCall::<Article>::new("write_article", TASK, ROLE, REQUEST_DEADLINE)
                    .max_output_tokens(MAX_ARTICLE_TOKENS)
                    .repairs(1);
                writer.call_measured(call, &shown)
            })?;
            let article = match asked {
                Asked::Answered(article, spent) => {
                    self.state().errors = 0;
                    self.settle(
                        &request,
                        &spent,
                        Some(tokens(&article.title) + tokens(&article.body)),
                    );
                    article
                }
                Asked::OutOfBudget => return Ok(()),
                Asked::Unreachable(why) => {
                    self.give_up_unreachable(index, &why);
                    return Ok(());
                }
                Asked::Unreadable(e) => {
                    let why = format!("the writer failed: {e}");
                    self.cache
                        .note_rejected(index, attempt, Kind::Content, &why, None)?;
                    let mut state = self.state();
                    state.summary.rejected += 1;
                    state.summary.last_turned_away = Some(why);
                    state.errors += 1;
                    if state.errors >= MAX_CONSECUTIVE_ERRORS_PER_WORKER * self.opts.workers.max(1)
                    {
                        state.stopped = true;
                        let errors = state.errors;
                        state.summary.stopped.get_or_insert(format!(
                            "{errors} answers in a row could not be used: {e}"
                        ));
                    }
                    continue;
                }
            };
            match self.judge(&article, episodes, index, attempt)? {
                Verdict::Accepted => {
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
                Verdict::Rejected(reason) => {
                    self.cache.note_rejected(
                        index,
                        attempt,
                        Kind::Content,
                        &reason.to_string(),
                        Some(&article),
                    )?;
                    // A reader's unreadable answer is no fault of the essay,
                    // so the writer is not told of it.
                    refusal = (!matches!(reason, Rejection::Unreadable { .. }))
                        .then(|| reason.to_string());
                    {
                        let mut state = self.state();
                        state.summary.rejected += 1;
                        state.summary.last_turned_away = Some(reason.to_string());
                    }
                    eprintln!(
                        "article {}/{}: attempt {attempt} turned away: {reason}",
                        index + 1,
                        opts.articles
                    );
                }
                Verdict::Unreachable(why) => {
                    self.give_up_unreachable(index, &why);
                    return Ok(());
                }
                Verdict::OutOfBudget => {
                    self.cache.note_rejected(
                        index,
                        attempt,
                        Kind::Budget,
                        "the budget ended before the draft was read",
                        Some(&article),
                    )?;
                    return Ok(());
                }
            }
        }
        self.state().summary.abandoned += 1;
        Ok(())
    }

    /// Ends the run because a model stayed unreachable: the article is left
    /// for a later run and reported apart from those given up on.
    fn give_up_unreachable(&self, index: usize, why: &str) {
        self.state().summary.unavailable += 1;
        self.stop(format!("article {}: {why}", index + 1));
    }

    /// Whether the article may join the corpus: its form, then its decisions,
    /// by a reader when there is one and by marker words when there is not.
    fn judge(
        &self,
        article: &Article,
        episodes: &[Episode],
        index: usize,
        attempt: usize,
    ) -> anyhow::Result<Verdict> {
        let turned_away = |result: Result<(), Rejection>| match result {
            Ok(()) => Verdict::Accepted,
            Err(reason) => Verdict::Rejected(reason),
        };
        if let Err(reason) = check_form(article, self.persona, episodes, self.opts.min_words) {
            return Ok(Verdict::Rejected(reason));
        }
        let Some(reader) = self.reader else {
            return Ok(turned_away(check_markers(article, self.persona, episodes)));
        };
        let cases: Vec<String> = episodes.iter().map(|e| e.case.name.clone()).collect();
        let request = Request {
            key: RequestKey {
                article: index,
                attempt,
                role: Role::Reader,
            },
            max_output: MAX_READING_TOKENS,
            shown_tokens: tokens(&article.body) + tokens(&cases.join(" ")),
            draft: Some(article),
        };
        Ok(
            match self.ask(reader, &request, |r| read_raw(r, &article.body, &cases))? {
                Asked::Answered(mined, spent) => {
                    self.settle(
                        &request,
                        &spent,
                        Some(tokens(&serde_json::to_string(&mined).unwrap_or_default())),
                    );
                    let reading = mined.as_reading();
                    let result = check_reading(&reading, self.persona, episodes);
                    self.cache
                        .note_reading(index, attempt, result.is_ok(), &reading)?;
                    turned_away(result)
                }
                Asked::Unreadable(e) => Verdict::Rejected(Rejection::Unreadable {
                    detail: e.to_string(),
                }),
                Asked::Unreachable(why) => Verdict::Unreachable(why),
                Asked::OutOfBudget => Verdict::OutOfBudget,
            },
        )
    }
}

/// One request of an article: who is asked, what it may cost and the draft it
/// concerns, if there is one.
struct Request<'a> {
    key: RequestKey,
    max_output: u64,
    shown_tokens: u64,
    draft: Option<&'a Article>,
}

/// How a request, repeated as far as the bound allows, ended.
enum Asked<T> {
    /// The model answered, using this many tokens.
    Answered(T, Spent),
    /// The model answered in no shape that can be read.
    Unreadable(CallError),
    /// Every request was lost in transit; the text says how.
    Unreachable(String),
    /// The budget ended before a request could be made.
    OutOfBudget,
}

/// What came of judging a draft.
enum Verdict {
    Accepted,
    Rejected(Rejection),
    Unreachable(String),
    OutOfBudget,
}

/// What a cache has cost so far.
///
/// # Errors
/// The record of what was spent cannot be read.
pub fn spent(cache: &Cache) -> anyhow::Result<Usage> {
    cache.usage()
}
