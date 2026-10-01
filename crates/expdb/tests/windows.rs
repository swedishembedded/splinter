// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: experience is queried by time and by event: every stream around a
//! successful contact, the exact samples of an interval across chunks, and
//! the actions that were followed by something.
#![allow(clippy::unwrap_used)]

mod common;

use common::{robot_episode, Scratch, AUDIO_HZ, SEC};
use splinter_expdb::model::TimeRange;
use splinter_expdb::timeline::{EventQuery, StreamFilter};
use splinter_expdb::WriterIdentity;

fn fixture(scratch: &Scratch) -> (splinter_expdb::Database, common::RobotEpisode) {
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "robot-1", 0))
        .unwrap();
    let robot = robot_episode(&mut c, "make breakfast", 1);
    (db, robot)
}

#[test]
fn a_window_holds_the_chunks_of_every_stream_that_overlap_it() {
    let scratch = Scratch::new();
    let (db, robot) = fixture(&scratch);
    let snapshot = db.snapshot().unwrap();
    let window = snapshot
        .window(
            robot.episode,
            TimeRange::new(2_500_000_000, 4_500_000_000).unwrap(),
            &StreamFilter::all(),
        )
        .unwrap();

    let chunks = |name: &str| window.iter().find(|w| w.name == name).unwrap().chunks.len();
    assert_eq!(chunks("mic"), 3, "seconds 2, 3 and 4 overlap [2.5s, 4.5s)");
    assert_eq!(chunks("camera"), 3);
    assert_eq!(
        chunks("imu"),
        2,
        "device time runs half a second ahead, so seconds 3 and 4 of it overlap"
    );
    // Streams can be chosen by name or by modality.
    let only_audio = snapshot
        .window(
            robot.episode,
            TimeRange::new(0, 10 * SEC).unwrap(),
            &StreamFilter::all().modalities(&["audio/pcm"]),
        )
        .unwrap();
    assert_eq!(only_audio.len(), 1);
    assert_eq!(only_audio[0].name, "mic");
}

#[test]
fn the_samples_of_an_interval_are_cut_exactly_even_across_chunk_boundaries() {
    let scratch = Scratch::new();
    let (db, robot) = fixture(&scratch);
    let snapshot = db.snapshot().unwrap();

    // 0.75 s to 1.25 s of 8 kHz mono float audio: 4000 samples across two chunks.
    let slice = snapshot
        .read_samples(
            robot.audio,
            TimeRange::new(750_000_000, 1_250_000_000).unwrap(),
        )
        .unwrap();
    assert_eq!(slice.count, 4000);
    assert_eq!(
        slice.interval,
        TimeRange::new(750_000_000, 1_250_000_000).unwrap()
    );
    let per_second = AUDIO_HZ as usize * 4;
    let mut expected = robot.audio_chunks[0][per_second * 3 / 4..].to_vec();
    expected.extend_from_slice(&robot.audio_chunks[1][..per_second / 4]);
    assert_eq!(slice.bytes, expected);
}

#[test]
fn samples_are_cut_on_the_experiment_clock_whichever_clock_the_device_used() {
    let scratch = Scratch::new();
    let (db, robot) = fixture(&scratch);
    let snapshot = db.snapshot().unwrap();
    // 1.0s..2.0s experiment time is 1.5s..2.5s on the IMU: 200 six-axis samples.
    let slice = snapshot
        .read_samples(robot.imu, TimeRange::new(SEC, 2 * SEC).unwrap())
        .unwrap();
    assert_eq!(slice.count, 200);
    assert_eq!(slice.bytes.len(), 200 * 24);
}

#[test]
fn compressed_streams_cannot_be_cut_to_a_sample_and_say_so() {
    let scratch = Scratch::new();
    let (db, robot) = fixture(&scratch);
    let snapshot = db.snapshot().unwrap();
    assert!(snapshot
        .read_samples(robot.video, TimeRange::new(0, SEC / 2).unwrap())
        .is_err());
    // Chunk-aligned access is still available.
    let window = snapshot
        .window(
            robot.episode,
            TimeRange::new(0, SEC / 2).unwrap(),
            &StreamFilter::all().names(&["camera"]),
        )
        .unwrap();
    assert_eq!(window[0].chunks.len(), 1);
}

#[test]
fn every_stream_around_a_successful_contact_is_one_query() {
    let scratch = Scratch::new();
    let (db, robot) = fixture(&scratch);
    let snapshot = db.snapshot().unwrap();

    let found = snapshot
        .windows_around(
            &EventQuery::new("contact")
                .before_ns(2 * SEC)
                .after_ns(2 * SEC)
                .min_value(1.0),
        )
        .unwrap();
    assert_eq!(found.len(), 1, "the contact that failed is not wanted");
    let window = &found[0];
    assert_eq!(window.episode, robot.episode);
    assert_eq!(
        window.interval,
        TimeRange::new(3_693_000_000 - 2 * SEC, 3_693_000_000 + 2 * SEC).unwrap()
    );
    let names: Vec<_> = window.streams.iter().map(|s| s.name.as_str()).collect();
    for stream in ["camera", "mic", "imu", "joints"] {
        assert!(names.contains(&stream), "{stream} is in the window");
    }
    let all = snapshot
        .windows_around(&EventQuery::new("contact").before_ns(SEC).after_ns(SEC))
        .unwrap();
    assert_eq!(all.len(), 2);
    assert!(snapshot
        .windows_around(&EventQuery::new("nothing happened"))
        .unwrap()
        .is_empty());
}

#[test]
fn the_actions_that_were_followed_by_an_event_within_a_time_can_be_found() {
    let scratch = Scratch::new();
    let (db, _) = fixture(&scratch);
    let snapshot = db.snapshot().unwrap();

    // The gripper closed at 3.4-3.7 s and the cup was lifted at 4.2 s.
    let within_a_second = snapshot.actions_followed_by("lifted", SEC).unwrap();
    let names: Vec<_> = within_a_second
        .iter()
        .map(|a| a.action.name.as_str())
        .collect();
    assert_eq!(names, ["close_gripper"]);
    let within_three = snapshot.actions_followed_by("lifted", 3 * SEC).unwrap();
    assert_eq!(within_three.len(), 2, "the arm move began 2.2 s before");
    assert!(snapshot
        .actions_followed_by("never", SEC)
        .unwrap()
        .is_empty());
}
