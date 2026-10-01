// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: experience is time-varying streams of any modality plus actions,
//! events and outcomes, kept as one episode. Modalities are registered
//! schemas, raw streams stay canonical, and everything a model makes of them
//! is a derivation that names its source.
#![allow(clippy::unwrap_used)]

mod common;

use common::{noise, robot_episode, Scratch, SEC};
use splinter_expdb::model::{
    ActionKind, ClockDomain, Correspondence, CorrespondenceRelation, Derivation, EpisodeKind,
    Epistemic, Event, ModalitySchema, SpanRef, StreamOrigin, StreamSpec, TimeRange,
};
use splinter_expdb::WriterIdentity;

fn collector(db: &splinter_expdb::Database) -> splinter_expdb::ingest::Collector {
    db.collector(&WriterIdentity::new("exp", "job", "robot-1", 0))
        .unwrap()
}

#[test]
fn a_robot_episode_is_one_experience_with_every_stream_action_and_event_in_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let robot = robot_episode(&mut c, "make breakfast", 1);

    let view = db
        .snapshot()
        .unwrap()
        .episode(robot.episode)
        .unwrap()
        .unwrap();
    assert_eq!(view.episode.kind, EpisodeKind::Interactive);
    let mut names: Vec<_> = view
        .streams
        .iter()
        .map(|s| s.stream.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["camera", "imu", "joints", "mic"]);
    let chunks = |name: &str| {
        view.streams
            .iter()
            .find(|s| s.stream.name == name)
            .unwrap()
            .chunks
            .len()
    };
    assert_eq!(
        (
            chunks("camera"),
            chunks("mic"),
            chunks("joints"),
            chunks("imu")
        ),
        (10, 10, 10, 12)
    );
    assert_eq!(
        view.events
            .iter()
            .filter(|e| e.event.name == "contact")
            .count(),
        2
    );
    assert_eq!(view.actions.len(), 2);
    assert!(view
        .actions
        .iter()
        .any(|a| a.action.kind == ActionKind::Continuous));
    // The synchronisation is the information: all of it shares one episode.
    let index = db.snapshot().unwrap().index().unwrap();
    assert!(index.by_attempt(robot.episode).len() > 40);
}

#[test]
fn what_was_recorded_is_what_is_read_back() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let robot = robot_episode(&mut c, "make breakfast", 2);
    let snapshot = db.snapshot().unwrap();
    let view = snapshot.episode(robot.episode).unwrap().unwrap();
    let mic = view
        .streams
        .iter()
        .find(|s| s.stream.name == "mic")
        .unwrap();
    for (second, expected) in robot.audio_chunks.iter().enumerate() {
        assert_eq!(&snapshot.read_chunk(&mic.chunks[second]).unwrap(), expected);
    }
}

#[test]
fn modalities_are_registered_schemas_so_a_new_sensor_needs_no_change_to_the_store() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    for required in [
        "text/utf8",
        "audio/pcm",
        "video/rgb",
        "image/rgb",
        "sensor/imu",
        "sensor/depth",
        "robot/joint_state",
        "robot/action",
        "geometry/pointcloud",
    ] {
        assert!(
            ModalitySchema::builtin(required).is_some(),
            "{required} is built in"
        );
    }
    assert!(ModalitySchema::builtin("radar/fmcw").is_none());

    let radar = ModalitySchema::custom("radar/fmcw", "f32", &[64, 4], "raw");
    let clock = c
        .register_clock(&ClockDomain {
            name: "lab".into(),
            description: String::new(),
        })
        .unwrap();
    let episode = c
        .start_episode(EpisodeKind::Experimental, "range test", clock)
        .unwrap();
    let stream = c
        .add_stream(episode, &StreamSpec::new("radar", &radar, clock).rate(20.0))
        .unwrap();
    c.add_chunk(
        episode,
        stream,
        TimeRange::new(0, SEC).unwrap(),
        20,
        &noise(20 * 64 * 4 * 4, 1),
    )
    .unwrap();
    c.flush().unwrap();

    let view = db.snapshot().unwrap().episode(episode).unwrap().unwrap();
    assert_eq!(view.streams[0].schema.name, "radar/fmcw");
    assert_eq!(view.streams[0].schema.sample_bytes(), Some(64 * 4 * 4));
}

#[test]
fn identical_chunks_are_stored_once() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let clock = c
        .register_clock(&ClockDomain {
            name: "lab".into(),
            description: String::new(),
        })
        .unwrap();
    let episode = c
        .start_episode(EpisodeKind::Observational, "a quiet room", clock)
        .unwrap();
    let mic = c
        .add_stream(
            episode,
            &StreamSpec::new("mic", &ModalitySchema::builtin("audio/pcm").unwrap(), clock)
                .rate(8000.0),
        )
        .unwrap();
    let silence = vec![0u8; 8000 * 4];
    c.add_chunk(
        episode,
        mic,
        TimeRange::new(0, SEC).unwrap(),
        8000,
        &silence,
    )
    .unwrap();
    let after_one = c.blob_stats().chunks_written;
    for second in 1..10 {
        c.add_chunk(
            episode,
            mic,
            TimeRange::new(second * SEC, (second + 1) * SEC).unwrap(),
            8000,
            &silence,
        )
        .unwrap();
    }
    assert_eq!(
        c.blob_stats().chunks_written,
        after_one,
        "nine more seconds of silence cost nothing"
    );
}

#[test]
fn passive_observation_needs_no_actions_rewards_or_goals() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let clock = c
        .register_clock(&ClockDomain {
            name: "cam".into(),
            description: String::new(),
        })
        .unwrap();
    let episode = c
        .start_episode(EpisodeKind::Observational, "street corner", clock)
        .unwrap();
    let video = c
        .add_stream(
            episode,
            &StreamSpec::new(
                "camera",
                &ModalitySchema::builtin("video/rgb").unwrap(),
                clock,
            )
            .rate(30.0),
        )
        .unwrap();
    c.add_chunk(
        episode,
        video,
        TimeRange::new(0, SEC).unwrap(),
        30,
        &noise(500, 1),
    )
    .unwrap();
    c.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    let view = snapshot.episode(episode).unwrap().unwrap();
    assert!(view.actions.is_empty() && view.events.is_empty());
    assert_eq!(
        snapshot
            .episodes_of_kind(EpisodeKind::Observational)
            .unwrap(),
        [episode]
    );
    assert!(snapshot
        .episodes_of_kind(EpisodeKind::Interactive)
        .unwrap()
        .is_empty());
}

#[test]
fn the_raw_waveform_stays_canonical_and_a_transcript_is_a_derivation_that_names_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let robot = robot_episode(&mut c, "make breakfast", 3);

    let transcript = c
        .derived_stream(
            robot.episode,
            Derivation {
                algorithm: "asr".into(),
                version: "2".into(),
                inputs: vec![robot.audio],
                code_ref: None,
                model: Some("whisper-like".into()),
                params: serde_json::Value::Null,
                seed: None,
            },
            &StreamSpec::new(
                "transcript",
                &ModalitySchema::builtin("text/utf8").unwrap(),
                robot.experiment_clock,
            ),
            &[(
                TimeRange::new(3 * SEC, 4 * SEC).unwrap(),
                1,
                b"I've got it".as_slice(),
            )],
        )
        .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let view = snapshot.episode(robot.episode).unwrap().unwrap();
    let raw = view.streams.iter().find(|s| s.id == robot.audio).unwrap();
    let derived = view.streams.iter().find(|s| s.id == transcript).unwrap();
    assert_eq!(raw.stream.origin, StreamOrigin::Raw);
    assert_eq!(derived.stream.origin, StreamOrigin::Derived);
    assert_eq!(
        snapshot.get(robot.audio).unwrap().unwrap().body.epistemic(),
        Epistemic::Fact
    );
    assert_eq!(
        snapshot.get(transcript).unwrap().unwrap().body.epistemic(),
        Epistemic::Derived
    );
    assert!(snapshot
        .lineage_back(transcript)
        .unwrap()
        .contains(&robot.audio));
    assert_eq!(
        snapshot
            .derivation_of(transcript)
            .unwrap()
            .unwrap()
            .1
            .algorithm,
        "asr"
    );
    // The waveform is untouched and still all there.
    assert_eq!(raw.chunks.len(), 10);
    assert_eq!(
        snapshot.read_chunk(&derived.chunks[0]).unwrap(),
        b"I've got it"
    );
}

#[test]
fn text_is_an_annotation_of_a_span_with_a_confidence_and_a_better_alignment_adds_rather_than_replaces(
) {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let robot = robot_episode(&mut c, "make breakfast", 4);
    let caption = c
        .add_stream(
            robot.episode,
            &StreamSpec::new(
                "caption",
                &ModalitySchema::builtin("text/utf8").unwrap(),
                robot.experiment_clock,
            ),
        )
        .unwrap();
    c.add_chunk(
        robot.episode,
        caption,
        TimeRange::new(3 * SEC, 4 * SEC).unwrap(),
        1,
        b"grasp failed",
    )
    .unwrap();

    let sentence = SpanRef {
        stream: caption,
        interval: TimeRange::new(3 * SEC, 4 * SEC).unwrap(),
    };
    let coarse = SpanRef {
        stream: robot.video,
        interval: TimeRange::new(3 * SEC, 5 * SEC).unwrap(),
    };
    let fine = SpanRef {
        stream: robot.video,
        interval: TimeRange::new(3_300_000_000, 3_900_000_000).unwrap(),
    };
    let first = c
        .correspond(
            robot.episode,
            Correspondence {
                a: sentence,
                b: coarse,
                relation: CorrespondenceRelation::Describes,
                confidence: 0.78,
                derivation: None,
            },
        )
        .unwrap();
    let better = c
        .correspond(
            robot.episode,
            Correspondence {
                a: sentence,
                b: fine,
                relation: CorrespondenceRelation::Describes,
                confidence: 0.93,
                derivation: None,
            },
        )
        .unwrap();
    c.flush().unwrap();

    let found = db
        .snapshot()
        .unwrap()
        .correspondences(robot.video, TimeRange::new(0, 10 * SEC).unwrap())
        .unwrap();
    let ids: Vec<_> = found.iter().map(|f| f.id).collect();
    assert_eq!(ids, [better, first], "both stand, the more confident first");
    assert_eq!(found[0].correspondence.confidence, 0.93);
}

#[test]
fn an_instant_and_an_interval_are_the_same_kind_of_thing() {
    let still = TimeRange::instant(5 * SEC);
    assert_eq!(still.len_ns(), 0);
    assert!(still.overlaps(&TimeRange::new(4 * SEC, 6 * SEC).unwrap()));
    assert!(!still.overlaps(&TimeRange::new(6 * SEC, 7 * SEC).unwrap()));
    assert!(TimeRange::new(5, 3).is_err());

    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let clock = c
        .register_clock(&ClockDomain {
            name: "cam".into(),
            description: String::new(),
        })
        .unwrap();
    let episode = c
        .start_episode(EpisodeKind::Observational, "photo", clock)
        .unwrap();
    let photo = c
        .add_stream(
            episode,
            &StreamSpec::new(
                "photo",
                &ModalitySchema::builtin("image/rgb").unwrap(),
                clock,
            ),
        )
        .unwrap();
    c.add_chunk(episode, photo, still, 1, &noise(300, 1))
        .unwrap();
    c.add_event(episode, Event::at("shutter", 5 * SEC, clock))
        .unwrap();
    c.flush().unwrap();
    assert_eq!(
        db.snapshot()
            .unwrap()
            .episode(episode)
            .unwrap()
            .unwrap()
            .streams[0]
            .chunks
            .len(),
        1
    );
}
