// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for turning sensitive longitudinal
// records into training data without leaking identities or the future, for
// its clients. If your team needs expertise in privacy-preserving data
// pipelines or leakage-free evaluation, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Longitudinal records: one participant's irregular history, as a source
//! file states it and as Splinter keeps it.
//!
//! A [`Record`] is one line of an input file: brain's `timeline-v1` fields
//! (measurements, events, observation windows) plus the interventions a
//! participant was given, each stating whether it was randomised. It names
//! the participant, which is the one thing Splinter never keeps. A
//! [`History`] is the participant-safe form: the raw identifiers are replaced
//! by opaque keys derived with a secret ([`ParticipantKeying`]), and every
//! item is tied to the file line it came from ([`Provenance`]) and to the
//! terms its source came under. A history's [`History::address`] is the
//! content address of one stored episode.
//!
//! Times share one unit for a whole dataset and sit on the participant's own
//! clock; `calendar_at_entry` places that clock on the calendar. A measurement
//! that was not made is absent: nothing here supplies a default value.

use serde::{Deserialize, Serialize};

use crate::digest::{canonical_json, Digest};
use crate::terms::Terms;

/// Nanosecond ticks the experience database counts per unit of dataset time.
/// A dataset's unit (years, for a health cohort) is the caller's; this only
/// fixes how a real-valued time becomes an integer tick for ordering.
pub const TICKS_PER_UNIT: f64 = 1e9;

/// The largest time magnitude, in dataset units, whose tick fits an `i64`.
const MAX_TIME: f64 = 9.0e9;

/// The tick of a time, when the time is finite and inside the range ticks
/// cover.
#[must_use]
pub fn ticks(t: f64) -> Option<i64> {
    (t.is_finite() && t.abs() <= MAX_TIME).then(|| (t * TICKS_PER_UNIT).round() as i64)
}

/// One measured value: a number, a category, or a number known only to lie
/// beyond a detection limit - which is information about the value, not a
/// missing one. The serialized shape is `timeline-v1`'s.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// An exact numeric value.
    Number(f64),
    /// The true value is at or below this limit.
    Below {
        /// The detection limit.
        below: f64,
    },
    /// The true value is at or above this limit.
    Above {
        /// The detection limit.
        above: f64,
    },
    /// A categorical level.
    Category(String),
}

/// A measurement of one variable at one time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    /// When it was measured, on the participant's clock.
    pub t: f64,
    /// The variable's name in the dataset's vocabulary.
    pub var: String,
    /// The value.
    pub value: Value,
    /// The unit a numeric `value` is stated in (`"mmol/L"`), when the source
    /// states one. It is carried into the projected record unchanged, so a
    /// model trained on the data records it and refuses another at
    /// prediction; nothing here converts between units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// The longest unit a record may state, as brain's timeline format limits it.
pub const MAX_UNIT_LEN: usize = 32;

/// Why `unit` cannot be recorded for a measurement: empty, longer than
/// [`MAX_UNIT_LEN`], padded with whitespace or holding a control character.
fn unit_problem(unit: &str) -> Option<&'static str> {
    if unit.is_empty() {
        Some("is empty")
    } else if unit.chars().count() > MAX_UNIT_LEN {
        Some("is longer than 32 characters")
    } else if unit.trim() != unit {
        Some("has surrounding whitespace")
    } else if unit.chars().any(char::is_control) {
        Some("holds a control character")
    } else {
        None
    }
}

/// An event of one code at one time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    /// When it happened, on the participant's clock.
    pub t: f64,
    /// The event code.
    pub code: String,
}

/// The window in which an event of `code` would have been observed had it
/// happened: entry into observation (left truncation) to exit (censoring, or
/// the event itself). `*` covers every code without a window of its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtRisk {
    /// The event code, or `*`.
    pub code: String,
    /// Start of observation.
    pub from: f64,
    /// End of observation.
    pub to: f64,
}

/// An intervention as a source file states it. Whether it was randomised
/// has no default: a file that does not say is refused, because assuming
/// either answer would mix observational exposures with assigned treatments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionRecord {
    /// When it was given, on the participant's clock.
    pub t: f64,
    /// What it was.
    pub code: String,
    /// Whether it was assigned by randomisation (`true`) or is an exposure
    /// that was observed (`false`).
    pub randomised: bool,
}

/// How an intervention came to be given. The two never share a flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Assignment {
    /// Assigned by randomisation.
    Randomised,
    /// An exposure that was observed, not assigned.
    Observational,
}

impl Assignment {
    /// The label an intervention is given as a variable level.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Assignment::Randomised => "randomised",
            Assignment::Observational => "observational",
        }
    }
}

/// An intervention in a [`History`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Intervention {
    /// When it was given, on the participant's clock.
    pub t: f64,
    /// What it was.
    pub code: String,
    /// How it came to be given.
    pub assignment: Assignment,
}

/// One line of a longitudinal input file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    /// The participant, unique within the file. Never stored.
    pub subject_id: String,
    /// Records sharing a group are never split between training and test (a
    /// household, a site). Never stored.
    #[serde(default)]
    pub group_id: Option<String>,
    /// Sampling weight; 1 when absent.
    #[serde(default = "one")]
    pub weight: f64,
    /// Where the record came from within the dataset (a survey cycle, a site).
    pub source: String,
    /// The prediction time on the participant's clock.
    pub entry: f64,
    /// The calendar time at `entry`.
    pub calendar_at_entry: f64,
    /// Measurements.
    #[serde(default)]
    pub observations: Vec<Observation>,
    /// Events.
    #[serde(default)]
    pub events: Vec<Event>,
    /// Observation windows for outcome codes.
    #[serde(default)]
    pub at_risk: Vec<AtRisk>,
    /// Interventions given.
    #[serde(default)]
    pub interventions: Vec<InterventionRecord>,
}

fn one() -> f64 {
    1.0
}

fn finite(who: &str, what: &str, v: f64) -> Result<(), String> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(format!("{who}: {what} is not finite ({v})"))
    }
}

fn timed(who: &str, what: &str, t: f64) -> Result<(), String> {
    finite(who, what, t)?;
    ticks(t)
        .map(|_| ())
        .ok_or_else(|| format!("{who}: {what} ({t}) is outside the range of representable times"))
}

impl Record {
    /// Parses and validates one input line. The error names the field that
    /// is wrong, never the participant.
    pub fn from_json_line(line: &str) -> Result<Record, String> {
        let record: Record = serde_json::from_str(line).map_err(|e| e.to_string())?;
        record.validate()?;
        Ok(record)
    }

    /// The checks `serde` cannot make: finite, representable times, a
    /// positive weight, non-empty names, and observation windows that are
    /// intervals opening no earlier than entry.
    pub fn validate(&self) -> Result<(), String> {
        let who = "record";
        if self.subject_id.is_empty() {
            return Err("record: subject_id is empty".into());
        }
        if self.source.is_empty() {
            return Err("record: source is empty".into());
        }
        if !(self.weight.is_finite() && self.weight > 0.0) {
            return Err(format!(
                "record: weight must be positive and finite, got {}",
                self.weight
            ));
        }
        timed(who, "entry", self.entry)?;
        finite(who, "calendar_at_entry", self.calendar_at_entry)?;
        for o in &self.observations {
            if o.var.is_empty() {
                return Err("record: an observation has an empty var".into());
            }
            timed(who, &format!("observation {} time", o.var), o.t)?;
            if let Value::Number(v) | Value::Below { below: v } | Value::Above { above: v } =
                &o.value
            {
                finite(who, &format!("observation {} value", o.var), *v)?;
            }
            if let Some(unit) = &o.unit {
                if matches!(o.value, Value::Category(_)) {
                    return Err(format!(
                        "record: observation {} is a category and cannot have a unit",
                        o.var
                    ));
                }
                if let Some(why) = unit_problem(unit) {
                    return Err(format!("record: the unit of observation {} {why}", o.var));
                }
            }
        }
        for e in &self.events {
            if e.code.is_empty() {
                return Err("record: an event has an empty code".into());
            }
            timed(who, &format!("event {} time", e.code), e.t)?;
        }
        for i in &self.interventions {
            if i.code.is_empty() {
                return Err("record: an intervention has an empty code".into());
            }
            timed(who, &format!("intervention {} time", i.code), i.t)?;
        }
        for w in &self.at_risk {
            timed(who, &format!("at_risk {} from", w.code), w.from)?;
            timed(who, &format!("at_risk {} to", w.code), w.to)?;
            if w.to < w.from {
                return Err(format!(
                    "record: at_risk {} ends ({}) before it starts ({})",
                    w.code, w.to, w.from
                ));
            }
            if w.from < self.entry {
                return Err(format!(
                    "record: at_risk {} opens at {} before entry {}",
                    w.code, w.from, self.entry
                ));
            }
        }
        Ok(())
    }
}

/// The secret that makes opaque participant keys. A key derived from the
/// same secret, dataset and identifier is always the same; without the
/// secret it cannot be recomputed from an identifier or turned back into one.
#[derive(Clone)]
pub struct ParticipantKeying([u8; 32]);

impl std::fmt::Debug for ParticipantKeying {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ParticipantKeying(..)")
    }
}

/// The opaque, stable key of one participant of one dataset: only for
/// grouping, splitting and lineage, never a model input.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ParticipantKey(String);

/// The opaque key of a group of participants (a household, a site, or the
/// participant alone when the record names no group).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GroupKey(String);

macro_rules! key_text {
    ($name:ident) => {
        impl $name {
            /// The key as 64 lowercase hex digits.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}
key_text!(ParticipantKey);
key_text!(GroupKey);

impl ParticipantKeying {
    /// A keying derived from `secret`, which the campaign keeps; losing it
    /// means later imports cannot reproduce earlier keys.
    #[must_use]
    pub fn from_secret(secret: &[u8]) -> Self {
        Self(blake3::derive_key(
            "splinter longitudinal participant keying v1",
            secret,
        ))
    }

    fn derive(&self, domain: &str, dataset: &str, id: &str) -> String {
        let mut hasher = blake3::Hasher::new_keyed(&self.0);
        for part in [domain, dataset, id] {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    /// The key of participant `subject_id` of dataset `dataset`.
    #[must_use]
    pub fn participant(&self, dataset: &str, subject_id: &str) -> ParticipantKey {
        ParticipantKey(self.derive("participant", dataset, subject_id))
    }

    /// The key of the group `group_id` of dataset `dataset`, or, when the
    /// record names none, of the participant alone.
    #[must_use]
    pub fn group(&self, dataset: &str, subject_id: &str, group_id: Option<&str>) -> GroupKey {
        GroupKey(match group_id {
            Some(group) => self.derive("group", dataset, group),
            None => self.derive("solo", dataset, subject_id),
        })
    }
}

/// Where an item came from and under which terms.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The source dataset's id.
    pub dataset: String,
    /// The digest of the source file's bytes.
    pub file: Digest,
    /// The line of the file the item came from, counting from 1.
    pub line: u64,
    /// The address of the usage terms the dataset came under ([`terms_address`]).
    pub terms: Digest,
}

/// The content address of `terms`.
pub fn terms_address(terms: &Terms) -> Result<Digest, serde_json::Error> {
    Ok(Digest::of(&canonical_json(terms)?))
}

/// One participant's history with every identifier replaced by an opaque
/// key. Items are kept in a canonical order, so equal content has equal
/// [`History::address`] however the source file ordered it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct History {
    /// The participant's opaque key.
    pub participant: ParticipantKey,
    /// The group's opaque key.
    pub group: GroupKey,
    /// Where the record came from within the dataset.
    pub source: String,
    /// Sampling weight.
    pub weight: f64,
    /// The prediction time the source file gave, on the participant's clock.
    pub entry: f64,
    /// The calendar time at `entry`.
    pub calendar_at_entry: f64,
    /// Measurements, by time then variable.
    pub observations: Vec<Observation>,
    /// Events, by time then code.
    pub events: Vec<Event>,
    /// Observation windows, by code then start.
    pub at_risk: Vec<AtRisk>,
    /// Interventions, by time then code.
    pub interventions: Vec<Intervention>,
    /// Where every item of this history came from.
    pub provenance: Provenance,
    /// The terms the dataset came under.
    pub terms: Terms,
}

impl History {
    /// The participant-safe form of `record`, read from line `line` of the
    /// file `file` of dataset `dataset`, which came under `terms`.
    pub fn from_record(
        record: Record,
        keying: &ParticipantKeying,
        dataset: &str,
        file: &Digest,
        line: u64,
        terms: &Terms,
    ) -> Result<History, serde_json::Error> {
        let interventions: Vec<Intervention> = record
            .interventions
            .into_iter()
            .map(|i| Intervention {
                t: i.t,
                code: i.code,
                assignment: if i.randomised {
                    Assignment::Randomised
                } else {
                    Assignment::Observational
                },
            })
            .collect();
        let mut history = History {
            participant: keying.participant(dataset, &record.subject_id),
            group: keying.group(dataset, &record.subject_id, record.group_id.as_deref()),
            source: record.source,
            weight: record.weight,
            entry: record.entry,
            calendar_at_entry: record.calendar_at_entry,
            observations: record.observations,
            events: record.events,
            at_risk: record.at_risk,
            interventions,
            provenance: Provenance {
                dataset: dataset.to_owned(),
                file: file.clone(),
                line,
                terms: terms_address(terms)?,
            },
            terms: terms.clone(),
        };
        history.sort_canonically();
        Ok(history)
    }

    /// Puts every list in its canonical order (see the field docs), the one
    /// order [`History::address`] is computed over.
    pub fn sort_canonically(&mut self) {
        self.observations
            .sort_by(|a, b| a.t.total_cmp(&b.t).then_with(|| a.var.cmp(&b.var)));
        self.events
            .sort_by(|a, b| a.t.total_cmp(&b.t).then_with(|| a.code.cmp(&b.code)));
        self.at_risk
            .sort_by(|a, b| a.code.cmp(&b.code).then(a.from.total_cmp(&b.from)));
        self.interventions.sort_by(|a, b| {
            a.t.total_cmp(&b.t)
                .then_with(|| a.code.cmp(&b.code))
                .then(a.assignment.cmp(&b.assignment))
        });
    }

    /// The content address of this history: the identity of its episode.
    pub fn address(&self) -> Result<Digest, serde_json::Error> {
        Ok(Digest::of(&canonical_json(self)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unit_is_optional_numeric_only_and_held_to_brains_limits() {
        let line = |obs: &str| {
            format!(
                r#"{{"subject_id":"a","source":"s","entry":1.0,"calendar_at_entry":2000.0,"observations":[{obs}]}}"#
            )
        };
        let ok = Record::from_json_line(&line(
            r#"{"t":1.0,"var":"ldl","value":3.1,"unit":"mmol/L"},{"t":1.0,"var":"crp","value":{"below":0.2},"unit":"mg/L"}"#,
        ))
        .unwrap();
        assert_eq!(ok.observations[0].unit.as_deref(), Some("mmol/L"));
        for (obs, why) in [
            (
                r#"{"t":1.0,"var":"s","value":"never","unit":"x"}"#,
                "category",
            ),
            (r#"{"t":1.0,"var":"a","value":1,"unit":""}"#, "empty"),
            (
                r#"{"t":1.0,"var":"a","value":1,"unit":" kg"}"#,
                "whitespace",
            ),
            (r#"{"t":1.0,"var":"a","value":1,"unit":"k\tg"}"#, "control"),
            (
                r#"{"t":1.0,"var":"a","value":1,"unit":"123456789012345678901234567890123"}"#,
                "longer",
            ),
        ] {
            let err = Record::from_json_line(&line(obs)).unwrap_err();
            assert!(err.contains(why), "{err}");
        }
        // No unit, no change: the address of a record without one is what it was.
        let plain = Record::from_json_line(&line(r#"{"t":1.0,"var":"a","value":1}"#)).unwrap();
        assert!(!serde_json::to_string(&plain).unwrap().contains("unit"));
    }

    const LINE: &str = r#"{"subject_id":"alice-17","group_id":"home-4","weight":2.5,"source":"cycle-a","entry":50.0,"calendar_at_entry":2003.5,
        "observations":[{"t":50.0,"var":"sbp","value":131},{"t":50.0,"var":"crp","value":{"below":0.2}},{"t":50.0,"var":"smoking","value":"never"}],
        "events":[{"t":62.5,"code":"death:heart"},{"t":44.0,"code":"dx:hypertension"}],
        "at_risk":[{"code":"*","from":50.0,"to":62.5}],
        "interventions":[{"t":50.0,"code":"statin","randomised":true}]}"#;

    fn history(line: &str, secret: &[u8], n: u64) -> History {
        History::from_record(
            Record::from_json_line(line).unwrap(),
            &ParticipantKeying::from_secret(secret),
            "cohort",
            &Digest::of(b"file"),
            n,
            &Terms::public_domain("cohort"),
        )
        .unwrap()
    }

    #[test]
    fn a_history_keeps_no_identifier_and_its_keys_depend_on_the_secret() {
        let h = history(LINE, b"one", 1);
        let text = serde_json::to_string(&h).unwrap();
        assert!(
            !text.contains("alice-17") && !text.contains("home-4"),
            "{text}"
        );
        assert_eq!(h.participant, history(LINE, b"one", 9).participant);
        assert_ne!(h.participant, history(LINE, b"two", 1).participant);
        assert_ne!(h.participant.as_str(), h.group.as_str());
    }

    #[test]
    fn the_address_ignores_source_ordering_and_follows_content() {
        let reordered = LINE.replace(
            r#""events":[{"t":62.5,"code":"death:heart"},{"t":44.0,"code":"dx:hypertension"}]"#,
            r#""events":[{"t":44.0,"code":"dx:hypertension"},{"t":62.5,"code":"death:heart"}]"#,
        );
        assert_eq!(
            history(LINE, b"k", 1).address().unwrap(),
            history(&reordered, b"k", 1).address().unwrap()
        );
        assert_ne!(
            history(LINE, b"k", 1).address().unwrap(),
            history(LINE, b"k", 2).address().unwrap(),
            "another line is another item"
        );
    }

    #[test]
    fn an_intervention_must_say_whether_it_was_randomised() {
        let undecided = LINE.replace(r#","randomised":true"#, "");
        assert!(Record::from_json_line(&undecided)
            .unwrap_err()
            .contains("randomised"));
    }

    #[test]
    fn contradictory_times_and_unknown_fields_are_refused() {
        let early = LINE.replace(r#""from":50.0"#, r#""from":40.0"#);
        assert!(Record::from_json_line(&early)
            .unwrap_err()
            .contains("before entry"));
        let extra = LINE.replace(r#""weight":2.5,"#, r#""weight":2.5,"name":"x","#);
        assert!(Record::from_json_line(&extra).is_err());
        let zero = LINE.replace(r#""weight":2.5"#, r#""weight":0"#);
        assert!(Record::from_json_line(&zero)
            .unwrap_err()
            .contains("weight"));
    }
}
