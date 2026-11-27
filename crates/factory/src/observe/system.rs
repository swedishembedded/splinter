// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every generation can be
// explained from its own records, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! How fast a run's attempt stages went: attempts per hour over the span
//! from the first attempt's start to the last one's end, the attempts the
//! infrastructure decided (uncounted) by cause, and the tokens per second
//! the agents' models produced. A run records no card utilisation; a log of
//! `nvidia-smi dmon -o DT` is joined by time when one is given.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::clock::{days_from_civil, parse_utc};
use splinter_core::factory::product::TrialResult;

/// One attempt stage's (or evaluation arm's) pace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pace {
    /// The stage, or `<evaluation>/<arm>`.
    pub stage: String,
    /// Its attempts.
    pub trials: usize,
    /// Those the grader counted.
    pub counted: usize,
    /// The uncounted ones by `status:cause`.
    pub uncounted: BTreeMap<String, usize>,
    /// From the first attempt's start to the last one's end.
    pub span_seconds: Option<f64>,
    /// Attempts per hour over the span.
    pub attempts_per_hour: Option<f64>,
    /// Completion tokens the agents' models produced per second of the span.
    pub completion_tokens_per_second: Option<f64>,
    /// Prompt tokens the agents' models read per second of the span.
    pub prompt_tokens_per_second: Option<f64>,
    /// Per card, its mean use over the span, when a utilisation log is given.
    pub cards: Vec<CardUse>,
}

/// One card's mean use over a span.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CardUse {
    /// The card's index.
    pub gpu: String,
    /// Samples within the span.
    pub samples: usize,
    /// Mean streaming-multiprocessor use, percent.
    pub sm: Option<f64>,
    /// Mean memory-controller use, percent.
    pub mem: Option<f64>,
}

/// The span of `trials`: the first start and the last end, in Unix seconds.
#[must_use]
pub fn span(trials: &[TrialResult]) -> Option<(f64, f64)> {
    let start = trials
        .iter()
        .filter_map(|t| t.started_at.as_deref().and_then(parse_utc))
        .min_by(f64::total_cmp)?;
    let end = trials
        .iter()
        .filter_map(|t| t.finished_at.as_deref().and_then(parse_utc))
        .max_by(f64::total_cmp)?;
    (end > start).then_some((start, end))
}

/// The pace of `trials`, the attempts of `stage`.
#[must_use]
pub fn pace(stage: &str, trials: &[TrialResult], utilisation: Option<&Utilisation>) -> Pace {
    let mut uncounted = BTreeMap::new();
    for t in trials.iter().filter(|t| !t.grade.counted) {
        let status = serde_json::to_value(t.grade.status)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        let cause = t.grade.cause.as_ref().map_or("", |c| c.code.as_str());
        *uncounted.entry(format!("{status}:{cause}")).or_default() += 1;
    }
    let window = span(trials);
    let seconds = window.map(|(s, e)| e - s);
    let per_second = |total: u64| seconds.map(|s| total as f64 / s);
    let completion: u64 = trials
        .iter()
        .filter_map(|t| t.usage.completion_tokens)
        .sum();
    let prompt: u64 = trials.iter().filter_map(|t| t.usage.prompt_tokens).sum();
    Pace {
        stage: stage.to_owned(),
        trials: trials.len(),
        counted: trials.iter().filter(|t| t.grade.counted).count(),
        uncounted,
        span_seconds: seconds,
        attempts_per_hour: seconds.map(|s| trials.len() as f64 * 3600.0 / s),
        completion_tokens_per_second: per_second(completion),
        prompt_tokens_per_second: per_second(prompt),
        cards: match (utilisation, window) {
            (Some(u), Some((s, e))) => u.over(s, e),
            _ => Vec::new(),
        },
    }
}

/// A card utilisation log, `nvidia-smi dmon -o DT` (a `#Date Time gpu sm mem
/// ...` header, then one line per card and sample), its local times read at
/// `offset_minutes` east of UTC.
#[derive(Clone, Debug, Default)]
pub struct Utilisation {
    /// (Unix seconds, card, sm, mem) per sample.
    samples: Vec<(f64, String, Option<f64>, Option<f64>)>,
}

impl Utilisation {
    /// The samples of the log `text`.
    ///
    /// # Errors
    /// The log has no header naming `Date`, `Time` and `gpu`.
    pub fn parse(text: &str, offset_minutes: i64) -> Result<Self, String> {
        let mut columns: Option<Vec<String>> = None;
        let mut samples = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if let Some(header) = line.strip_prefix('#') {
                let names: Vec<String> = header.split_whitespace().map(str::to_lowercase).collect();
                if names.iter().any(|n| n == "gpu") && names.iter().any(|n| n == "date") {
                    columns = Some(names);
                }
                continue;
            }
            let Some(names) = &columns else { continue };
            let fields: Vec<&str> = line.split_whitespace().collect();
            let at = |name: &str| {
                names
                    .iter()
                    .position(|n| n == name)
                    .and_then(|i| fields.get(i).copied())
            };
            let (Some(date), Some(time), Some(gpu)) = (at("date"), at("time"), at("gpu")) else {
                continue;
            };
            let Some(local) = local_seconds(date, time) else {
                continue;
            };
            let number = |name: &str| at(name).and_then(|v| v.parse::<f64>().ok());
            samples.push((
                local - (offset_minutes * 60) as f64,
                gpu.to_owned(),
                number("sm"),
                number("mem"),
            ));
        }
        if columns.is_none() {
            return Err(
                "no `#Date Time gpu ...` header: not a `nvidia-smi dmon -o DT` log".to_owned(),
            );
        }
        Ok(Self { samples })
    }

    /// Per card, its mean use over `[start, end]`.
    #[must_use]
    pub fn over(&self, start: f64, end: f64) -> Vec<CardUse> {
        let mut by: BTreeMap<&str, (usize, f64, usize, f64, usize)> = BTreeMap::new();
        for (at, gpu, sm, mem) in &self.samples {
            if *at < start || *at > end {
                continue;
            }
            let e = by.entry(gpu.as_str()).or_default();
            e.0 += 1;
            if let Some(sm) = sm {
                e.1 += sm;
                e.2 += 1;
            }
            if let Some(mem) = mem {
                e.3 += mem;
                e.4 += 1;
            }
        }
        by.into_iter()
            .map(|(gpu, (n, sm, nsm, mem, nmem))| CardUse {
                gpu: gpu.to_owned(),
                samples: n,
                sm: (nsm > 0).then(|| sm / nsm as f64),
                mem: (nmem > 0).then(|| mem / nmem as f64),
            })
            .collect()
    }
}

/// `YYYYMMDD` and `HH:MM:SS` as seconds since the epoch, read as UTC.
fn local_seconds(date: &str, time: &str) -> Option<f64> {
    if date.len() != 8 {
        return None;
    }
    let (y, m, d) = (
        date[..4].parse::<i64>().ok()?,
        date[4..6].parse::<i64>().ok()?,
        date[6..].parse::<i64>().ok()?,
    );
    let mut t = time.split(':').map(str::parse::<i64>);
    let (h, mi, s) = (t.next()?.ok()?, t.next()?.ok()?, t.next()?.ok()?);
    Some((days_from_civil(y, m, d) * 86_400 + h * 3_600 + mi * 60 + s) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "#Date       Time        gpu    pwr  gtemp  mtemp     sm    mem\n\
                       #YYYYMMDD   HH:MM:SS    Idx      W      C      C      %      %\n\
                       20261127   06:34:50      0    300     50     50     10     20\n\
                       20261127   06:34:55      0    300     50     50     90     40\n\
                       20261127   06:34:55      1    100     50     50      -      -\n\
                       20261127   06:40:00      0    300     50     50     50     50\n";

    #[test]
    fn samples_within_the_span_are_averaged_per_card_in_utc() {
        // The log is at UTC+2 (120 minutes east); the span is 04:34:52 to 04:35:00 UTC.
        let log = Utilisation::parse(LOG, 120).unwrap();
        let start = parse_utc("2026-11-27T04:34:52Z").unwrap();
        let end = parse_utc("2026-11-27T04:35:00Z").unwrap();
        let cards = log.over(start, end);
        assert_eq!(cards.len(), 2);
        assert_eq!(
            (
                cards[0].gpu.as_str(),
                cards[0].samples,
                cards[0].sm,
                cards[0].mem
            ),
            ("0", 1, Some(90.0), Some(40.0))
        );
        assert_eq!(
            (cards[1].samples, cards[1].sm),
            (1, None),
            "an unmeasured column is absent, never 0"
        );
        assert!(Utilisation::parse("no header\n", 0).is_err());
    }
}
