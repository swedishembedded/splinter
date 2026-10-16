// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in running studies against hosted
// language models without losing a run to a rate limit, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Requests lost in transit: the ones a hosted model never answered because
//! of its provider or the network, as opposed to the ones it answered badly
//! or refused as malformed.
//!
//! A lost request says nothing about the article it concerned, so it is
//! waited out and repeated without using an attempt, within a bound. A rate
//! limit is waited out at the pace the provider asks for when it says, else
//! at a pace that doubles from a base to a cap.

use std::time::Duration;

use splinter_sdk::agent::{CallError, RunConclusion};

/// How many times a request lost in transit is repeated, unless the caller
/// asks for another bound. With the default pause this waits about five
/// minutes in all, which is what a provider's shared-pool rate limit takes to
/// clear.
pub const DEFAULT_RETRIES: usize = 8;
/// Seconds paused before the first repeat of a lost request, unless the
/// caller asks for another pause. Each further repeat waits twice as long as
/// the last, up to [`MAX_PAUSE`].
pub const DEFAULT_BACKOFF_SECS: u64 = 5;
/// The longest pause before a repeat, however many have been made and
/// whatever the provider asks for.
pub const MAX_PAUSE: Duration = Duration::from_secs(60);
/// The longest a one-line account of a provider's reason runs to.
const REASON_CHARS: usize = 140;

/// How a lost request is repeated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transit {
    /// How many times a request is repeated before it counts as unreachable.
    pub retries: usize,
    /// The pause before the first repeat; it doubles with each repeat up to
    /// [`MAX_PAUSE`].
    pub backoff: Duration,
}

impl Default for Transit {
    fn default() -> Self {
        Self::new(DEFAULT_RETRIES, Duration::from_secs(DEFAULT_BACKOFF_SECS))
    }
}

impl Transit {
    /// Repeats a request up to `retries` times, first after `backoff`.
    #[must_use]
    pub const fn new(retries: usize, backoff: Duration) -> Self {
        Self { retries, backoff }
    }

    /// The pause before repeat number `repeat` (from one): the provider's
    /// `hint` when it gave one, else the backoff doubled once for each repeat
    /// before this one; never more than [`MAX_PAUSE`].
    #[must_use]
    pub fn pause_before(&self, repeat: usize, hint: Option<Duration>) -> Duration {
        let doublings = u32::try_from(repeat.saturating_sub(1))
            .unwrap_or(u32::MAX)
            .min(16);
        hint.unwrap_or_else(|| self.backoff.saturating_mul(1 << doublings))
            .min(MAX_PAUSE)
    }

    /// How long all the repeats wait when the provider gives no hint.
    #[must_use]
    pub fn total_wait(&self) -> Duration {
        (1..=self.retries)
            .map(|repeat| self.pause_before(repeat, None))
            .sum()
    }
}

/// What a failed call says about the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The provider or the network failed: the request may be repeated, after
    /// the pause the provider asked for if it asked.
    Lost {
        /// How long the provider asked to be left alone.
        retry_after: Option<Duration>,
    },
    /// The model answered badly or refused the request: repeating it as it is
    /// would not help.
    Fault,
}

/// Decides whether `error` is a request lost in transit: a call that never
/// reached an answer because of a rate limit (429), a server error (5xx), a
/// timeout (408, 425 or the call's own deadline) or a connection that failed.
/// Any other 4xx status is the request's fault, as is every error that is
/// about the answer rather than the way to it.
#[must_use]
pub fn classify(error: &CallError) -> Failure {
    match error {
        CallError::Infrastructure(e) => {
            let text = format!("{e:#}");
            match status_of(&text) {
                Some(status) if is_fault(status) && !rerouted_after_loss(&text) => Failure::Fault,
                _ => Failure::Lost {
                    retry_after: hint_of(&text),
                },
            }
        }
        CallError::Stopped {
            conclusion: RunConclusion::Timeout,
        } => Failure::Lost { retry_after: None },
        _ => Failure::Fault,
    }
}

/// Whether the provider that refused the request was only the fallback of a
/// router whose first choice had failed in transit (`previous_errors` names a
/// rate limit, a timeout or a server error). The fallback may not support what
/// the request asks for, which says nothing about the request: the first
/// choice will serve it once it recovers.
fn rerouted_after_loss(text: &str) -> bool {
    let Some(at) = text.find('{') else {
        return false;
    };
    let body = serde_json::Deserializer::from_str(&text[at..])
        .into_iter::<serde_json::Value>()
        .next()
        .and_then(Result::ok);
    body.is_some_and(|body| {
        body["error"]["metadata"]["previous_errors"]
            .as_array()
            .is_some_and(|errors| {
                errors
                    .iter()
                    .filter_map(|e| e["code"].as_u64())
                    .any(|code| matches!(code, 408 | 425 | 429 | 500..=599))
            })
    })
}

/// Whether the HTTP status is the request's fault: a 4xx that is not a rate
/// limit or a timeout.
fn is_fault(status: u16) -> bool {
    (400..500).contains(&status) && !matches!(status, 408 | 425 | 429)
}

/// The HTTP status a driver's error text names, in the form `error 429 ...`
/// every hosted driver reports it in.
fn status_of(text: &str) -> Option<u16> {
    text.match_indices("error ").find_map(|(at, key)| {
        let digits = text.get(at + key.len()..)?.get(..3)?;
        digits.parse().ok()
    })
}

/// How long the error text asks to wait, if it says: a number of seconds after
/// `retry-after`, `retry_after` or `retry after`.
fn hint_of(text: &str) -> Option<Duration> {
    let lower = text.to_lowercase();
    ["retry-after", "retry_after", "retry after"]
        .iter()
        .find_map(|key| {
            let rest = lower.get(lower.find(key)? + key.len()..)?;
            let rest = rest.trim_start_matches([':', '=', '"', ' ']);
            let end = rest
                .find(|c: char| !c.is_ascii_digit() && c != '.')
                .unwrap_or(rest.len());
            let seconds: f64 = rest[..end].parse().ok()?;
            Duration::try_from_secs_f64(seconds).ok()
        })
}

/// What a failed request is called in a log or a report, in one line: the
/// status, the provider and a short reason, instead of the provider's whole
/// JSON body. Text that is not a provider's error is cut to its first line.
#[must_use]
pub fn one_line(error: &anyhow::Error) -> String {
    let text = format!("{error:#}");
    let Some(body_at) = text.find('{') else {
        return clip(first_line(&text), REASON_CHARS + 60);
    };
    let head = text[..body_at].trim_end().trim_end_matches(':').trim_end();
    let body = serde_json::Deserializer::from_str(&text[body_at..])
        .into_iter::<serde_json::Value>()
        .next()
        .and_then(Result::ok);
    let Some(body) = body else {
        return clip(first_line(&text), REASON_CHARS + 60);
    };
    let metadata = &body["error"]["metadata"];
    // A router passes its upstream's answer on as a string of JSON.
    let raw = metadata["raw"].as_str().map(|raw| {
        serde_json::from_str::<serde_json::Value>(raw)
            .ok()
            .and_then(|inner| inner["message"].as_str().map(str::to_string))
            .unwrap_or_else(|| raw.to_string())
    });
    let reason = raw
        .as_deref()
        .or_else(|| body["error"]["message"].as_str())
        .unwrap_or_default();
    let reason = clip(reason.split(". ").next().unwrap_or(reason), REASON_CHARS);
    let provider = metadata["provider_name"]
        .as_str()
        .map_or_else(String::new, |p| format!(" ({p})"));
    format!("{head}{provider}: {reason}")
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default().trim()
}

fn clip(text: &str, chars: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(chars).collect();
    format!("{}...", kept.trim_end())
}

/// One request that was lost, as the caller is told of it.
#[derive(Debug)]
pub struct Loss {
    /// The loss in one line.
    pub summary: String,
    /// The loss with the provider's whole answer.
    pub detail: String,
    /// Whether no more repeats follow: the bound is spent.
    pub last: bool,
}

/// How a request, repeated as far as the bound allows, ended.
#[derive(Debug)]
pub enum Retried<T> {
    /// The model answered.
    Answered(T),
    /// The call failed in a way repeating would not mend.
    Refused(CallError),
    /// Every request was lost in transit.
    Exhausted {
        /// The last loss in one line.
        last: String,
        /// How many requests were made.
        requests: usize,
    },
    /// The caller declined to make another request.
    Halted,
}

/// Puts a request, repeating it while it is lost in transit and `transit`
/// allows, pausing before each repeat. `admit` is asked before every request,
/// the first included, and may decline it; `lost` is told of every loss.
///
/// # Errors
/// `lost` failed.
pub fn retry<T>(
    transit: &Transit,
    mut admit: impl FnMut() -> bool,
    mut request: impl FnMut() -> Result<T, CallError>,
    mut lost: impl FnMut(&Loss) -> anyhow::Result<()>,
) -> anyhow::Result<Retried<T>> {
    let mut last = String::new();
    let mut hint = None;
    for repeat in 0..=transit.retries {
        if repeat > 0 {
            std::thread::sleep(transit.pause_before(repeat, hint));
        }
        if !admit() {
            return Ok(Retried::Halted);
        }
        let error = match request() {
            Ok(value) => return Ok(Retried::Answered(value)),
            Err(error) => error,
        };
        let Failure::Lost { retry_after } = classify(&error) else {
            return Ok(Retried::Refused(error));
        };
        hint = retry_after;
        let (summary, detail) = match &error {
            CallError::Infrastructure(e) => (one_line(e), format!("{e:#}")),
            other => (other.to_string(), other.to_string()),
        };
        lost(&Loss {
            summary: summary.clone(),
            detail,
            last: repeat == transit.retries,
        })?;
        last = summary;
    }
    Ok(Retried::Exhausted {
        last,
        requests: transit.retries + 1,
    })
}
