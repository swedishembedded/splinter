// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: devices disagree about time, so time is addressed through explicit
//! clock domains and mappings that carry their own uncertainty, and a query
//! in one clock finds data recorded in another.
#![allow(clippy::unwrap_used)]

mod common;

use common::{robot_episode, Scratch, SEC};
use splinter_expdb::model::{ClockDomain, ClockMapping, TimeRange};
use splinter_expdb::WriterIdentity;

fn collector(db: &splinter_expdb::Database) -> splinter_expdb::ingest::Collector {
    db.collector(&WriterIdentity::new("exp", "job", "robot-1", 0))
        .unwrap()
}

#[test]
fn a_mapping_converts_a_device_time_to_the_experiment_clock_and_back() {
    let (a, b) = (
        splinter_expdb::ContentId::of(b"imu"),
        splinter_expdb::ContentId::of(b"experiment"),
    );
    let mapping = ClockMapping {
        source: a,
        destination: b,
        src_ref_ns: 1_000 * SEC,
        dst_ref_ns: 1_000 * SEC - 12_300_000,
        slope: 1.000_003_7,
        uncertainty_ns: 85_000,
    };
    let t = 1_000 * SEC + 3_600 * SEC; // an hour after the reference
    let global = mapping.to_destination(t);
    // An hour of drift at 3.7 ppm is 13.32 ms.
    assert_eq!(global - t, -12_300_000 + 13_320_000);
    assert!(
        (mapping.to_source(global) - t).abs() <= 1,
        "the round trip is exact to a nanosecond"
    );
}

#[test]
fn clocks_resolve_through_a_chain_of_mappings_and_uncertainty_adds_up() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let domain = |c: &mut splinter_expdb::ingest::Collector, name: &str| {
        c.register_clock(&ClockDomain {
            name: name.into(),
            description: String::new(),
        })
        .unwrap()
    };
    let (camera, host, lab) = (
        domain(&mut c, "camera"),
        domain(&mut c, "host"),
        domain(&mut c, "lab"),
    );
    c.map_clock(ClockMapping {
        source: camera,
        destination: host,
        src_ref_ns: 0,
        dst_ref_ns: 2 * SEC,
        slope: 1.0,
        uncertainty_ns: 50_000,
    })
    .unwrap();
    c.map_clock(ClockMapping {
        source: host,
        destination: lab,
        src_ref_ns: 0,
        dst_ref_ns: -SEC,
        slope: 1.0,
        uncertainty_ns: 30_000,
    })
    .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let transform = snapshot.resolve_clock(camera, lab).unwrap().unwrap();
    assert_eq!(transform.apply(10 * SEC), 11 * SEC);
    assert_eq!(transform.uncertainty_ns, 80_000);
    // Mappings run both ways.
    assert_eq!(
        snapshot
            .resolve_clock(lab, camera)
            .unwrap()
            .unwrap()
            .apply(11 * SEC),
        10 * SEC
    );
    assert_eq!(
        snapshot
            .resolve_clock(camera, camera)
            .unwrap()
            .unwrap()
            .apply(7),
        7
    );
    let stranger = splinter_expdb::ContentId::of(b"no mapping to this one");
    assert!(snapshot.resolve_clock(camera, stranger).unwrap().is_none());
}

#[test]
fn the_best_known_mapping_between_two_clocks_wins() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let a = c
        .register_clock(&ClockDomain {
            name: "a".into(),
            description: String::new(),
        })
        .unwrap();
    let b = c
        .register_clock(&ClockDomain {
            name: "b".into(),
            description: String::new(),
        })
        .unwrap();
    let rough = ClockMapping {
        source: a,
        destination: b,
        src_ref_ns: 0,
        dst_ref_ns: 5_000,
        slope: 1.0,
        uncertainty_ns: 2_000_000,
    };
    let fine = ClockMapping {
        dst_ref_ns: 7_000,
        uncertainty_ns: 1_000,
        ..rough
    };
    c.map_clock(fine).unwrap();
    c.map_clock(rough).unwrap();
    c.flush().unwrap();
    let transform = db.snapshot().unwrap().resolve_clock(a, b).unwrap().unwrap();
    assert_eq!(
        (transform.apply(0), transform.uncertainty_ns),
        (7_000, 1_000)
    );
}

#[test]
fn a_stream_on_another_clock_is_found_at_the_experiment_time_it_really_happened() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let robot = robot_episode(&mut c, "make breakfast", 1);
    let snapshot = db.snapshot().unwrap();

    // IMU chunk 4 covers device time [4s, 5s) which is experiment time [3.5s, 4.5s).
    let window = snapshot
        .window(
            robot.episode,
            TimeRange::new(4_100_000_000, 4_200_000_000).unwrap(),
            &splinter_expdb::timeline::StreamFilter::all(),
        )
        .unwrap();
    let imu = window.iter().find(|w| w.name == "imu").unwrap();
    assert_eq!(imu.chunks.len(), 1);
    assert_eq!(
        imu.chunks[0].chunk.interval,
        TimeRange::new(4 * SEC, 5 * SEC).unwrap(),
        "stored in device time"
    );
    // The same instant on the experiment clock finds the camera's fifth second.
    let camera = window.iter().find(|w| w.name == "camera").unwrap();
    assert_eq!(
        camera.chunks[0].chunk.interval,
        TimeRange::new(4 * SEC, 5 * SEC).unwrap()
    );
    // Experiment time 4.1s is device time 4.6s, inside chunk 4 and not chunk 3 or 5.
    assert_eq!(imu.chunks[0].chunk.samples, 200);
}
