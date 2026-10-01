// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the same raw multimodal experience compiles into whatever a learner
//! needs - predicting the future from the past and an action, aligning two
//! modalities, filling in a masked stream, producing an action chunk from
//! observations and an instruction - as plans of references to windows of the
//! raw streams. Nothing is copied or pre-noised.
#![allow(clippy::unwrap_used)]

mod common;

use common::{noise, robot_episode, Scratch, AUDIO_HZ, SEC};
use splinter_expdb::model::{ClockDomain, EpisodeKind, ModalitySchema, StreamSpec, TimeRange};
use splinter_expdb::train::{flow_matching, DataRef, Materialized, Recipe, SampleBody};
use splinter_expdb::{Database, WriterIdentity};

fn collector(db: &Database) -> splinter_expdb::ingest::Collector {
    db.collector(&WriterIdentity::new("exp", "job", "robot-1", 0))
        .unwrap()
}

fn window(sample: &DataRef) -> (splinter_expdb::RecordId, TimeRange) {
    match sample {
        DataRef::Window { stream, interval } => (*stream, *interval),
        other => panic!("not a window: {other:?}"),
    }
}

#[test]
fn a_world_model_sample_is_the_past_and_an_action_and_the_future_it_led_to() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let robot = robot_episode(&mut c, "make breakfast", 1);
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot
        .compile(
            &Recipe::world_model()
                .streams(&["camera", "mic"])
                .past_ns(2 * SEC)
                .future_ns(2 * SEC),
        )
        .unwrap();
    assert_eq!(
        plan.samples.len(),
        2,
        "one per action that has two seconds of history and of future"
    );
    let SampleBody::WorldModel {
        context,
        actions,
        target,
    } = &plan.samples[0].body
    else {
        panic!("not a world-model sample")
    };
    assert_eq!(context.len(), 2);
    assert_eq!(window(&context[0]).1, TimeRange::new(0, 2 * SEC).unwrap());
    assert_eq!(
        window(&target[0]).1,
        TimeRange::new(2 * SEC, 4 * SEC).unwrap()
    );
    assert_eq!(
        actions.len(),
        2,
        "the arm move and the grasp both begin in the two seconds ahead"
    );
    let SampleBody::WorldModel {
        actions: later,
        target: later_target,
        ..
    } = &plan.samples[1].body
    else {
        panic!()
    };
    assert_eq!(later.len(), 1);
    assert_eq!(
        window(&later_target[0]).1,
        TimeRange::new(3_400_000_000, 5_400_000_000).unwrap()
    );

    // Materialised, the windows are the raw bytes of the stream.
    let Materialized::WorldModel {
        context, actions, ..
    } = snapshot.materialize(&plan.samples[0]).unwrap()
    else {
        panic!()
    };
    let mic = context.iter().find(|s| s.label == "mic").unwrap();
    let expected: Vec<u8> = robot.audio_chunks[..2].concat();
    assert_eq!(mic.bytes, expected);
    assert_eq!(mic.count, 2 * AUDIO_HZ as u64);
    assert!(mic.exact);
    let camera = context.iter().find(|s| s.label == "camera").unwrap();
    assert!(
        !camera.exact,
        "compressed video is available a chunk at a time"
    );
    assert_eq!(
        actions.iter().map(|a| a.label.as_str()).collect::<Vec<_>>(),
        ["move_arm", "close_gripper"]
    );
}

#[test]
fn passive_observation_makes_world_model_data_with_no_action_at_all() {
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
    for s in 0..10 {
        c.add_chunk(
            episode,
            video,
            TimeRange::new(s * SEC, (s + 1) * SEC).unwrap(),
            30,
            &noise(400, s as u64),
        )
        .unwrap();
    }
    c.flush().unwrap();

    let plan = db
        .snapshot()
        .unwrap()
        .compile(
            &Recipe::world_model()
                .past_ns(SEC)
                .future_ns(SEC)
                .stride_ns(2 * SEC),
        )
        .unwrap();
    assert_eq!(plan.samples.len(), 5, "anchors at 1, 3, 5, 7 and 9 seconds");
    for sample in &plan.samples {
        let SampleBody::WorldModel { actions, .. } = &sample.body else {
            panic!()
        };
        assert!(actions.is_empty());
    }
}

#[test]
fn contrastive_pairs_align_two_modalities_and_take_hard_negatives_from_nearby_in_time() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let first = robot_episode(&mut c, "first", 1);
    let second = robot_episode(&mut c, "second", 2);
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot
        .compile(
            &Recipe::contrastive()
                .pair("camera", "mic")
                .window_ns(SEC)
                .stride_ns(3 * SEC)
                .hard_shift_ns(5 * SEC)
                .negatives(2),
        )
        .unwrap();
    assert_eq!(
        plan.samples.len(),
        8,
        "windows at 0, 3, 6 and 9 seconds in each of two episodes"
    );

    let mut hard_shifts = Vec::new();
    for sample in &plan.samples {
        let SampleBody::Contrastive {
            anchor,
            positive,
            negatives,
        } = &sample.body
        else {
            panic!("not a contrastive sample")
        };
        let ((a_stream, a_at), (p_stream, p_at)) = (window(anchor), window(positive));
        assert_eq!(
            a_at, p_at,
            "the positive is the other modality at the same time"
        );
        assert_ne!(a_stream, p_stream);
        assert_eq!(negatives.len(), 2);
        let (hard, easy) = (&negatives[0], &negatives[1]);
        assert!(hard.hard && !easy.hard);
        let (h_stream, h_at) = window(&hard.data);
        assert_eq!(
            h_stream, p_stream,
            "same stream, same episode, a different moment"
        );
        hard_shifts.push((h_at.start_ns - p_at.start_ns).abs());
        let (e_stream, _) = window(&easy.data);
        assert_ne!(
            e_stream, p_stream,
            "the easy negative comes from another episode"
        );
        assert!([first.audio, second.audio].contains(&e_stream));
    }
    assert!(hard_shifts.iter().all(|s| *s == 5 * SEC));
    // The same recipe and snapshot give the same pairs.
    assert_eq!(
        plan.id().unwrap(),
        snapshot
            .compile(
                &Recipe::contrastive()
                    .pair("camera", "mic")
                    .window_ns(SEC)
                    .stride_ns(3 * SEC)
                    .hard_shift_ns(5 * SEC)
                    .negatives(2)
            )
            .unwrap()
            .id()
            .unwrap()
    );
}

#[test]
fn a_single_episode_has_no_other_episode_to_borrow_an_easy_negative_from() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    robot_episode(&mut c, "only", 1);
    let plan = db
        .snapshot()
        .unwrap()
        .compile(
            &Recipe::contrastive()
                .pair("camera", "mic")
                .window_ns(SEC)
                .stride_ns(3 * SEC)
                .hard_shift_ns(5 * SEC)
                .negatives(3),
        )
        .unwrap();
    let SampleBody::Contrastive { negatives, .. } = &plan.samples[0].body else {
        panic!()
    };
    assert_eq!(
        negatives.len(),
        1,
        "only the hard negative exists, and no negative is invented"
    );
}

#[test]
fn masked_prediction_hides_one_stream_and_leaves_everything_around_it_visible() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let robot = robot_episode(&mut c, "make breakfast", 1);
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot
        .compile(
            &Recipe::masked()
                .target_stream("mic")
                .streams(&["camera"])
                .window_ns(SEC)
                .context_ns(2 * SEC)
                .stride_ns(4 * SEC),
        )
        .unwrap();
    assert_eq!(plan.samples.len(), 2, "windows at 2 and 6 seconds");
    for sample in &plan.samples {
        let SampleBody::Masked { visible, masked } = &sample.body else {
            panic!("not a masked sample")
        };
        let (m_stream, m_at) = window(masked);
        assert_eq!(m_stream, robot.audio);
        for shown in visible {
            let (stream, at) = window(shown);
            assert!(
                stream != robot.audio || !at.overlaps(&m_at),
                "the hidden audio is not visible"
            );
        }
        assert!(
            visible.iter().any(|v| window(v).0 == robot.audio),
            "the audio on either side is"
        );
        assert!(
            visible.iter().any(|v| window(v).0 == robot.video),
            "and so is the picture"
        );
    }
    let Materialized::Masked { masked, .. } = snapshot.materialize(&plan.samples[0]).unwrap()
    else {
        panic!()
    };
    assert_eq!(
        masked.bytes, robot.audio_chunks[2],
        "the target is exactly the audio of second two"
    );
}

#[test]
fn an_action_chunk_is_what_the_robot_should_do_next_given_what_it_saw_and_was_told() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    robot_episode(&mut c, "make breakfast", 1);
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot
        .compile(
            &Recipe::action_chunk()
                .streams(&["camera", "joints"])
                .past_ns(SEC)
                .horizon_ns(2 * SEC)
                .stride_ns(SEC),
        )
        .unwrap();
    assert_eq!(
        plan.samples.len(),
        3,
        "anchors at 1, 2 and 3 seconds have an action within two seconds"
    );
    let SampleBody::ActionChunk {
        observations,
        actions,
        ..
    } = &plan.samples[1].body
    else {
        panic!("not an action chunk")
    };
    assert_eq!(observations.len(), 2);
    assert_eq!(
        window(&observations[0]).1,
        TimeRange::new(SEC, 2 * SEC).unwrap()
    );
    assert_eq!(actions.len(), 2);

    let Materialized::ActionChunk {
        instruction,
        actions,
        observations,
    } = snapshot.materialize(&plan.samples[1]).unwrap()
    else {
        panic!()
    };
    assert_eq!(instruction.as_deref(), Some("pick up the cup"));
    assert_eq!(
        actions.iter().map(|a| a.label.as_str()).collect::<Vec<_>>(),
        ["move_arm", "close_gripper"]
    );
    assert_eq!(
        observations
            .iter()
            .find(|o| o.label == "joints")
            .unwrap()
            .count,
        100
    );
}

#[test]
fn flow_matching_targets_are_derived_from_clean_actions_and_a_seed_not_stored() {
    let clean: Vec<f32> = (0..7).map(|i| i as f32 * 0.1).collect();
    let a = flow_matching(&clean, 42, 0);
    assert_eq!(
        a,
        flow_matching(&clean, 42, 0),
        "the same seed and step give the same noise"
    );
    assert_ne!(a.noise, flow_matching(&clean, 42, 1).noise);
    assert!(a.t > 0.0 && a.t < 1.0);
    for (i, action) in clean.iter().enumerate() {
        let expected = (1.0 - a.t) * a.noise[i] + a.t * action;
        assert!(
            (a.noisy[i] - expected).abs() < 1e-6,
            "x_t = (1 - t) noise + t action"
        );
        assert!(
            (a.target[i] - (action - a.noise[i])).abs() < 1e-6,
            "the field to learn points from noise to action"
        );
    }
    // The noise is standard normal.
    let big = flow_matching(&vec![0.0; 20_000], 7, 3);
    let mean = big.noise.iter().sum::<f32>() / big.noise.len() as f32;
    let var = big.noise.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / big.noise.len() as f32;
    assert!(
        mean.abs() < 0.05 && (var - 1.0).abs() < 0.1,
        "mean {mean} variance {var}"
    );
}

#[test]
fn a_multimodal_plan_is_exported_with_its_bytes_as_hex_and_is_still_only_a_projection() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    robot_episode(&mut c, "make breakfast", 1);
    let snapshot = db.snapshot().unwrap();
    let plan = snapshot
        .compile(
            &Recipe::world_model()
                .streams(&["mic"])
                .past_ns(SEC)
                .future_ns(SEC),
        )
        .unwrap();

    let mut out = Vec::new();
    plan.export_jsonl(&snapshot, &mut out).unwrap();
    let line: serde_json::Value =
        serde_json::from_str(String::from_utf8(out).unwrap().lines().next().unwrap()).unwrap();
    assert_eq!(line["type"], "world_model");
    let hex = line["context"][0]["bytes"].as_str().unwrap();
    assert_eq!(
        hex.len(),
        2 * AUDIO_HZ as usize * 4,
        "one second of float audio, in hex"
    );
}
