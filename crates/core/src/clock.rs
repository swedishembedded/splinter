// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents whose every
// record carries a trustworthy UTC timestamp. If your team needs expertise
// in agent infrastructure or durable operating state, you can procure our
// services by sending an email to info@swedishembedded.com.

//! UTC wall-clock time, rendered without a date dependency.
//!
//! The trace's event timestamps and the store's run ids are the same instant
//! seen through two formats, so the arithmetic lives here once: [`now`]
//! gives the components, [`utc_now`] the RFC-3339-shaped string the trace
//! writes, and a run id is the same components in their compact order.

/// One UTC instant, broken into the parts both consumers need.
#[derive(Clone, Copy, Debug)]
pub struct Stamp {
    /// Proleptic Gregorian year, e.g. `2026`.
    pub year: i64,
    /// 1-12.
    pub month: i64,
    /// 1-31.
    pub day: i64,
    /// Hour of the UTC day, 0-23.
    pub hour: i64,
    /// Minute of the hour, 0-59.
    pub min: i64,
    /// Second of the minute, 0-59 (leap seconds are not represented).
    pub sec: i64,
    /// Milliseconds into the second, 0-999.
    pub millis: u32,
}

/// The current UTC instant.
#[must_use]
pub fn now() -> Stamp {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let (y, m, d) = civil_from_days((secs / 86400) as i64);
    let rem = secs % 86400;
    Stamp {
        year: y,
        month: m,
        day: d,
        hour: (rem / 3600) as i64,
        min: ((rem % 3600) / 60) as i64,
        sec: (rem % 60) as i64,
        millis,
    }
}

/// A UTC timestamp with millisecond precision, `YYYY-MM-DDTHH:MM:SS.mmmZ` -
/// the form every trace event carries.
#[must_use]
pub fn utc_now() -> String {
    let t = now();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        t.year, t.month, t.day, t.hour, t.min, t.sec, t.millis
    )
}

/// The current UTC instant in whole seconds since the Unix epoch, which is
/// what a lease or a heartbeat compares instead of parsing a timestamp.
#[must_use]
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// A source of [`utc_now`]-shaped timestamps, passed to whatever stamps a
/// durable record, so a test or a replay can pin time instead of reading the
/// wall clock.
pub trait Clock {
    /// The current instant as `YYYY-MM-DDTHH:MM:SS.mmmZ`.
    fn utc_now(&self) -> String;
}

/// The wall clock: [`utc_now`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn utc_now(&self) -> String {
        utc_now()
    }
}

/// A clock that always reads the same instant.
#[derive(Clone, Debug)]
pub struct FixedClock(String);

impl FixedClock {
    /// A clock stopped at `stamp`, which is returned verbatim.
    #[must_use]
    pub fn new(stamp: impl Into<String>) -> Self {
        Self(stamp.into())
    }
}

impl Clock for FixedClock {
    fn utc_now(&self) -> String {
        self.0.clone()
    }
}

/// (year, month, day) to days since the epoch: the inverse of
/// [`civil_from_days`], by the same algorithm.
#[must_use]
pub fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Seconds since the Unix epoch of a UTC timestamp `YYYY-MM-DDTHH:MM:SS`,
/// with any fraction of a second and a trailing `Z`; `None` for any other
/// shape.
#[must_use]
pub fn parse_utc(stamp: &str) -> Option<f64> {
    let stamp = stamp.strip_suffix('Z').unwrap_or(stamp);
    let (date, time) = stamp.split_once('T')?;
    let mut d = date.split('-').map(str::parse::<i64>);
    let (year, month, day) = (d.next()?.ok()?, d.next()?.ok()?, d.next()?.ok()?);
    let mut t = time.split(':');
    let (hour, minute) = (
        t.next()?.parse::<i64>().ok()?,
        t.next()?.parse::<i64>().ok()?,
    );
    let second = t.next()?.parse::<f64>().ok()?;
    if d.next().is_some()
        || t.next().is_some()
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
    {
        return None;
    }
    Some((days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60) as f64 + second)
}

/// Days-since-epoch to (year, month, day) - Howard Hinnant's algorithm, the
/// same arithmetic a date library would apply, without the dependency.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timestamp_parses_to_the_seconds_it_names() {
        for days in [0, 20_723, 20_784, -1, 11_016] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(parse_utc("1970-01-01T00:00:00Z"), Some(0.0));
        // 2026-11-27 is day 20_784.
        assert_eq!(
            parse_utc("2026-11-27T04:34:53.5Z"),
            Some(20_784.0 * 86_400.0 + 16_493.5)
        );
        assert_eq!(parse_utc("2026-11-27 04:34:53"), None);
        assert_eq!(parse_utc("2026-13-27T04:34:53Z"), None);
    }

    #[test]
    fn the_epoch_and_a_pinned_date_render_exactly() {
        // 1970-01-01, via the same arithmetic now() uses.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2026-09-27: pins the leap-year correction (the `yoe / 100` term -
        // miswritten once as `yoe / 146_100`, which shifted dates up to
        // three days inside part of every 400-year era).
        assert_eq!(civil_from_days(20_723), (2026, 9, 27));
        let stamp = utc_now();
        assert_eq!(stamp.len(), 24, "{stamp}");
        assert!(stamp.ends_with('Z'), "{stamp}");
        assert_eq!(&stamp[4..5], "-");
        assert_eq!(&stamp[10..11], "T");
    }
}
