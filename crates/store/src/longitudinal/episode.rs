// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for importing sensitive
// longitudinal datasets as immutable, traceable episodes, for its clients. If
// your team needs expertise in privacy-preserving data pipelines you can
// procure our services by sending an email to info@swedishembedded.com.

//! How one participant's [`History`] is laid out as an episode of the
//! experience database, and read back.
//!
//! - The episode sits on the participant's own clock domain
//!   (`subject/<opaque key>`); a calendar clock domain is shared, and a clock
//!   mapping at entry relates the two.
//! - A `record` event carries what belongs to the participant as a whole: the
//!   keys, the source, the weight, the entry time and the terms.
//! - Each measured variable is a stream (`obs:<variable>`) of instant samples;
//!   its one chunk holds that variable's samples from the record, so a
//!   variable that was not measured has no stream.
//! - Each event is an `event:<code>` event; each observation window an
//!   interval event `at_risk:<code>`; each intervention an action whose
//!   payload says whether it was randomised.
//!
//! Every item's payload repeats the provenance of the line it came from;
//! reading back refuses an item whose provenance disagrees with the record's.
//! Exact times are kept in the payloads: the ticks the database orders by
//! are only for ordering.

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::longitudinal::{
    ticks, Assignment, AtRisk, Event as HistoryEvent, GroupKey, History, Intervention, Observation,
    ParticipantKey, Provenance, Value,
};
use splinter_core::terms::Terms;
use splinter_expdb::ingest::Collector;
use splinter_expdb::manifest::Snapshot;
use splinter_expdb::model::{
    ActionKind, ActionSegment, ClockDomain, ClockMapping, Content, EpisodeKind, Event,
    ModalitySchema, StreamSpec, TimeRange, TimeSemantics,
};
use splinter_expdb::{Error, RecordId, Result};

const RECORD_EVENT: &str = "record";
const EVENT_PREFIX: &str = "event:";
const WINDOW_PREFIX: &str = "at_risk:";
const STREAM_PREFIX: &str = "obs:";

/// The calendar clock domain every episode's clock is mapped to.
#[must_use]
pub fn calendar_clock() -> ClockDomain {
    ClockDomain {
        name: "calendar".into(),
        description: "calendar time, in the dataset's time unit".into(),
    }
}

/// The clock domain of one participant: attained time in the dataset's time
/// unit, on a clock of its own.
#[must_use]
pub fn subject_clock(key: &ParticipantKey) -> ClockDomain {
    ClockDomain {
        name: format!("subject/{key}"),
        description: "the participant's own clock, in the dataset's time unit".into(),
    }
}

fn observation_schema() -> ModalitySchema {
    ModalitySchema {
        time: TimeSemantics::Instant,
        ..ModalitySchema::custom("timeline/observation", "u8", &[], "json")
    }
}

fn tick(what: &str, t: f64) -> Result<i64> {
    ticks(t).ok_or_else(|| Error::corrupt(what.to_owned(), format!("time {t} has no tick")))
}

fn encode<T: Serialize>(what: &'static str, value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|source| Error::Encode { what, source })
}

fn decode<T: for<'de> Deserialize<'de>>(what: &str, bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|source| Error::Decode {
        what: what.to_owned(),
        source,
    })
}

#[derive(Serialize, Deserialize)]
struct Header {
    participant: ParticipantKey,
    group: GroupKey,
    source: String,
    weight: f64,
    entry: f64,
    calendar_at_entry: f64,
    provenance: Provenance,
    terms: Terms,
}

#[derive(Serialize, Deserialize)]
struct Sample {
    t: f64,
    value: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unit: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Samples {
    provenance: Provenance,
    samples: Vec<Sample>,
}

#[derive(Serialize, Deserialize)]
struct EventPayload {
    t: f64,
    provenance: Provenance,
}

#[derive(Serialize, Deserialize)]
struct WindowPayload {
    from: f64,
    to: f64,
    provenance: Provenance,
}

#[derive(Serialize, Deserialize)]
struct InterventionPayload {
    t: f64,
    assignment: Assignment,
    provenance: Provenance,
}

/// Writes `history` as an episode labelled with its content `address`.
pub(super) fn write_history(
    c: &mut Collector,
    history: &History,
    address: &Digest,
) -> Result<RecordId> {
    let subject = c.register_clock(&subject_clock(&history.participant))?;
    let calendar = c.register_clock(&calendar_clock())?;
    c.map_clock(ClockMapping {
        source: subject,
        destination: calendar,
        src_ref_ns: tick("entry", history.entry)?,
        dst_ref_ns: tick("calendar_at_entry", history.calendar_at_entry)?,
        slope: 1.0,
        uncertainty_ns: 0,
    })?;
    let kind = if history
        .interventions
        .iter()
        .any(|i| i.assignment == Assignment::Randomised)
    {
        EpisodeKind::Experimental
    } else {
        EpisodeKind::Observational
    };
    let episode = c.start_episode(kind, address.as_str(), subject)?;
    let provenance = &history.provenance;

    let header = Header {
        participant: history.participant.clone(),
        group: history.group.clone(),
        source: history.source.clone(),
        weight: history.weight,
        entry: history.entry,
        calendar_at_entry: history.calendar_at_entry,
        provenance: provenance.clone(),
        terms: history.terms.clone(),
    };
    c.add_event(
        episode,
        Event::at(RECORD_EVENT, tick("entry", history.entry)?, subject).payload(to_value(&header)?),
    )?;

    let mut variables: Vec<&str> = history
        .observations
        .iter()
        .map(|o| o.var.as_str())
        .collect();
    variables.sort_unstable();
    variables.dedup();
    let schema = observation_schema();
    for var in variables {
        let samples: Vec<&Observation> = history
            .observations
            .iter()
            .filter(|o| o.var == var)
            .collect();
        let first = tick(var, samples[0].t)?;
        let last = tick(var, samples[samples.len() - 1].t)?;
        let body = Samples {
            provenance: provenance.clone(),
            samples: samples
                .iter()
                .map(|o| Sample {
                    t: o.t,
                    value: o.value.clone(),
                    unit: o.unit.clone(),
                })
                .collect(),
        };
        let stream = c.add_stream(
            episode,
            &StreamSpec::new(&format!("{STREAM_PREFIX}{var}"), &schema, subject),
        )?;
        c.add_chunk(
            episode,
            stream,
            TimeRange::new(first, last)?,
            samples.len() as u64,
            &encode("observation samples", &body)?,
        )?;
    }
    for event in &history.events {
        let payload = EventPayload {
            t: event.t,
            provenance: provenance.clone(),
        };
        c.add_event(
            episode,
            Event::at(
                &format!("{EVENT_PREFIX}{}", event.code),
                tick(&event.code, event.t)?,
                subject,
            )
            .payload(to_value(&payload)?),
        )?;
    }
    for window in &history.at_risk {
        let mut event = Event::at(
            &format!("{WINDOW_PREFIX}{}", window.code),
            tick(&window.code, window.from)?,
            subject,
        );
        event.at = TimeRange::new(
            tick(&window.code, window.from)?,
            tick(&window.code, window.to)?,
        )?;
        let payload = WindowPayload {
            from: window.from,
            to: window.to,
            provenance: provenance.clone(),
        };
        c.add_event(episode, event.payload(to_value(&payload)?))?;
    }
    for intervention in &history.interventions {
        let payload = InterventionPayload {
            t: intervention.t,
            assignment: intervention.assignment,
            provenance: provenance.clone(),
        };
        let at = TimeRange::instant(tick(&intervention.code, intervention.t)?);
        c.add_action(
            episode,
            ActionSegment::new(ActionKind::Discrete, &intervention.code, at, subject).payload(
                Content::text(
                    String::from_utf8_lossy(&encode("intervention", &payload)?).into_owned(),
                ),
            ),
        )?;
    }
    Ok(episode)
}

fn to_value<T: Serialize>(value: &T) -> Result<serde_json::Value> {
    serde_json::to_value(value).map_err(|source| Error::Encode {
        what: "episode payload",
        source,
    })
}

fn same_provenance(what: &str, found: &Provenance, expected: &Provenance) -> Result<()> {
    if found == expected {
        Ok(())
    } else {
        Err(Error::corrupt(
            what.to_owned(),
            "its provenance differs from the record's own",
        ))
    }
}

/// Reads the episode `id` back as the history it was written from.
pub(super) fn read_history(snapshot: &Snapshot, id: RecordId) -> Result<History> {
    let what = format!("episode {id}");
    let view = snapshot
        .episode(id)?
        .ok_or_else(|| Error::corrupt(what.clone(), "it is not in the snapshot"))?;
    let header_event = view
        .events
        .iter()
        .find(|e| e.event.name == RECORD_EVENT)
        .ok_or_else(|| Error::corrupt(what.clone(), "it has no record event"))?;
    let header: Header =
        serde_json::from_value(header_event.event.payload.clone()).map_err(|source| {
            Error::Decode {
                what: format!("the record event of {what}"),
                source,
            }
        })?;
    let expected = &header.provenance;

    let mut history = History {
        participant: header.participant,
        group: header.group,
        source: header.source,
        weight: header.weight,
        entry: header.entry,
        calendar_at_entry: header.calendar_at_entry,
        observations: Vec::new(),
        events: Vec::new(),
        at_risk: Vec::new(),
        interventions: Vec::new(),
        provenance: header.provenance.clone(),
        terms: header.terms,
    };
    for stream in &view.streams {
        let Some(var) = stream.stream.name.strip_prefix(STREAM_PREFIX) else {
            continue;
        };
        for chunk in &stream.chunks {
            let bytes = snapshot.read_chunk(chunk)?;
            let body: Samples = decode(&format!("a chunk of {what}"), &bytes)?;
            same_provenance(&what, &body.provenance, expected)?;
            history
                .observations
                .extend(body.samples.into_iter().map(|s| Observation {
                    t: s.t,
                    var: var.to_owned(),
                    value: s.value,
                    unit: s.unit,
                }));
        }
    }
    for item in &view.events {
        let name = &item.event.name;
        let payload = item.event.payload.clone();
        let wrong = |source| Error::Decode {
            what: format!("event {name} of {what}"),
            source,
        };
        if let Some(code) = name.strip_prefix(EVENT_PREFIX) {
            let p: EventPayload = serde_json::from_value(payload).map_err(wrong)?;
            same_provenance(&what, &p.provenance, expected)?;
            history.events.push(HistoryEvent {
                t: p.t,
                code: code.to_owned(),
            });
        } else if let Some(code) = name.strip_prefix(WINDOW_PREFIX) {
            let p: WindowPayload = serde_json::from_value(payload).map_err(wrong)?;
            same_provenance(&what, &p.provenance, expected)?;
            history.at_risk.push(AtRisk {
                code: code.to_owned(),
                from: p.from,
                to: p.to,
            });
        }
    }
    for item in &view.actions {
        let Content::Text { text } = &item.action.payload else {
            return Err(Error::corrupt(
                what.clone(),
                "an intervention's payload is not inline text",
            ));
        };
        let p: InterventionPayload =
            decode(&format!("an intervention of {what}"), text.as_bytes())?;
        same_provenance(&what, &p.provenance, expected)?;
        history.interventions.push(Intervention {
            t: p.t,
            code: item.action.name.clone(),
            assignment: p.assignment,
        });
    }
    history.sort_canonically();
    Ok(history)
}
