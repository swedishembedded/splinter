// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that finish within the time
// they are given, for its clients. If your team needs expertise in turning an
// open-ended learning job into a bounded one, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A wall-clock budget as a command line names it.

use std::time::Duration;

use splinter_orchestrator::error::OrchestratorError;

/// A duration as a `--budget` flag takes it: whole units of `s`, `m` and
/// `h`, alone or combined largest first (`30m`, `1h30m`), or bare
/// seconds.
pub fn parse_budget(text: &str) -> Result<Duration, OrchestratorError> {
    let refuse = |why: &str| {
        OrchestratorError::Refused(format!(
            "{text:?} is not a duration ({why}): e.g. 30m, 2h, 1h30m; units h, m and s"
        ))
    };
    if text.is_empty() {
        return Err(refuse("it is empty"));
    }
    if let Ok(seconds) = text.parse::<u64>() {
        return positive(Duration::from_secs(seconds)).ok_or_else(|| refuse("it is zero"));
    }
    let mut total: u64 = 0;
    let mut number = String::new();
    let mut last_unit = u64::MAX;
    for c in text.chars() {
        if c.is_ascii_digit() {
            number.push(c);
            continue;
        }
        let unit = match c {
            'h' => 3600,
            'm' => 60,
            's' => 1,
            _ => return Err(refuse("units are h, m and s")),
        };
        if number.is_empty() {
            return Err(refuse("a unit needs a number before it"));
        }
        if unit >= last_unit {
            return Err(refuse("units go largest first, each once"));
        }
        let value: u64 = number
            .parse()
            .map_err(|_| refuse("the number is too large"))?;
        total = value
            .checked_mul(unit)
            .and_then(|v| total.checked_add(v))
            .ok_or_else(|| refuse("it is too long"))?;
        number.clear();
        last_unit = unit;
    }
    if !number.is_empty() {
        return Err(refuse("a number needs a unit after it"));
    }
    positive(Duration::from_secs(total)).ok_or_else(|| refuse("it is zero"))
}

fn positive(duration: Duration) -> Option<Duration> {
    (!duration.is_zero()).then_some(duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_budget_is_whole_units_largest_first() {
        let secs = |text: &str| parse_budget(text).ok().map(|d| d.as_secs());
        assert_eq!(secs("90"), Some(90));
        assert_eq!(secs("90s"), Some(90));
        assert_eq!(secs("30m"), Some(1800));
        assert_eq!(secs("1h30m"), Some(5400));
        assert_eq!(secs("2h5s"), Some(7205));
        for refused in ["", "0", "0s", "m", "1.5h", "30m1h", "5x", "10mm", "3m3m"] {
            assert!(parse_budget(refused).is_err(), "{refused}");
        }
    }
}
