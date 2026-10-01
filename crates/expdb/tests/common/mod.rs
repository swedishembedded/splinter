// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Shared helpers for the specs: a scratch database root per test.
#![allow(dead_code, clippy::unwrap_used)]

use std::collections::BTreeMap;

use splinter_expdb::model::{
    family_key, Action, Body, Decision, Edge, Observation, Outcome, PolicyRef, Record, Rel,
    ReproLevel, State, TaskDefinition,
};
use splinter_expdb::{Config, ContentId, Database, RecordId, WriterId};
use tempfile::TempDir;

/// A record id of writer `writer`.
pub fn rid(writer: u64, seq: u64) -> RecordId {
    RecordId::new(WriterId::from_raw(writer), seq)
}

/// A state with one named part.
pub fn state(part: &str) -> State {
    State::new(
        BTreeMap::from([("fs".to_owned(), ContentId::of(part.as_bytes()))]),
        ReproLevel::Exact,
    )
}

/// A decision record with a distinctive action.
pub fn decision(writer: u64, seq: u64, family: ContentId, action: &str) -> Record {
    let st = state("s0").id().unwrap();
    Record::new(
        rid(writer, seq),
        seq,
        Body::Decision(Decision {
            state: st,
            observation: None,
            policy: PolicyRef::new("policy", "1"),
            context: None,
            action: Action::new(action, serde_json::json!({ "n": seq })),
            old_logprob: Some(-0.5),
            value_estimate: None,
        }),
    )
    .in_attempt(rid(writer, 0), family, ContentId::of(b"instance"))
}

/// A family key for test `n`.
pub fn family(n: u64) -> ContentId {
    family_key(&ContentId::of(&n.to_le_bytes()), &state("s0").id().unwrap())
}

/// A mixed set of records and edges covering several kinds.
pub fn mixed(writer: u64, count: u64) -> (Vec<Record>, Vec<Edge>) {
    let fam = family(1);
    let mut records = Vec::new();
    let mut edges = Vec::new();
    for seq in 0..count {
        let record = match seq % 4 {
            0 => Record::new(
                rid(writer, seq),
                seq,
                Body::TaskDefinition(TaskDefinition {
                    name: format!("task {seq}"),
                    description: "fix the bug".into(),
                    domain: "coding".into(),
                }),
            ),
            1 => Record::new(
                rid(writer, seq),
                seq,
                Body::Observation(Observation {
                    state: state("s0").id().unwrap(),
                    content: splinter_expdb::model::Content::text(format!("output {seq}")),
                }),
            ),
            2 => decision(writer, seq, fam, "grep"),
            _ => Record::new(
                rid(writer, seq),
                seq,
                Body::AttemptEnd {
                    attempt: rid(writer, 0),
                    outcome: Outcome::Pass,
                },
            )
            .with_parent(rid(writer, seq - 1)),
        };
        if seq > 0 {
            edges.push(Edge::new(
                rid(writer, seq),
                Rel::DerivedFrom,
                rid(writer, seq - 1),
            ));
        }
        records.push(record);
    }
    (records, edges)
}

/// A database in its own temporary directory, removed on drop.
pub struct Scratch {
    pub dir: TempDir,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            dir: TempDir::new().unwrap(),
        }
    }

    pub fn open(&self) -> Database {
        Database::open(self.dir.path(), Config::default()).unwrap()
    }

    pub fn open_with(&self, config: Config) -> Database {
        Database::open(self.dir.path(), config).unwrap()
    }
}

/// A task definition, instance and initial state for a coding task.
pub fn coding_task(n: u64) -> (TaskDefinition, splinter_expdb::model::TaskInstance, State) {
    let definition = TaskDefinition {
        name: "fix the bug".into(),
        description: "make the failing test pass".into(),
        domain: "coding".into(),
    };
    let instance = splinter_expdb::model::TaskInstance {
        definition: definition.id().unwrap(),
        params: serde_json::json!({ "issue": n }),
        environment: None,
    };
    (definition, instance, state("repo@abc123"))
}

/// Records one attempt of `chain` decisions, each followed by a transition,
/// ending as `outcome`. Returns the decisions made and the attempt id.
pub fn attempt(
    collector: &mut splinter_expdb::ingest::Collector,
    task: u64,
    policy: &str,
    chain: usize,
    outcome: Outcome,
) -> (Vec<splinter_expdb::ingest::DecisionRef>, RecordId) {
    let (definition, instance, initial) = coding_task(task);
    let mut run = collector
        .start_attempt(
            &definition,
            &instance,
            &initial,
            &PolicyRef::new(policy, "1"),
            None,
        )
        .unwrap();
    let attempt = run.attempt();
    let mut decisions = Vec::new();
    for step in 0..chain {
        let d = run
            .decision()
            .commit(Action::new(
                "act",
                serde_json::json!({ "step": step, "policy": policy }),
            ))
            .unwrap();
        run.transition(&d, &state(&format!("{policy}-{task}-{step}")), None, None)
            .unwrap();
        decisions.push(d);
    }
    run.finish(outcome).unwrap();
    (decisions, attempt)
}

/// A skill body with a distinctive name.
pub fn skill(name: &str) -> splinter_expdb::model::Skill {
    splinter_expdb::model::Skill {
        name: name.into(),
        description: format!("{name}: do it before anything else"),
        trigger: "an asynchronous operation fails".into(),
        action_pattern: "inspect the event source".into(),
        expected_effect: "the failing layer is found first".into(),
        parents: vec![],
        prerequisites: vec![],
    }
}

pub const SEC: i64 = 1_000_000_000;

/// Deterministic bytes that do not compress away.
pub fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

/// A robot making breakfast: synchronised video, audio, IMU and joint streams
/// on two clocks, an instruction, actions, a contact that worked and one that
/// did not, and a reward.
pub struct RobotEpisode {
    pub episode: RecordId,
    pub experiment_clock: ContentId,
    pub imu_clock: ContentId,
    pub video: RecordId,
    pub audio: RecordId,
    pub imu: RecordId,
    pub joints: RecordId,
    pub audio_chunks: Vec<Vec<u8>>,
}

pub const AUDIO_HZ: f64 = 8000.0;
pub const IMU_HZ: f64 = 200.0;
pub const JOINT_HZ: f64 = 100.0;

pub fn robot_episode(
    c: &mut splinter_expdb::ingest::Collector,
    label: &str,
    seed: u64,
) -> RobotEpisode {
    use splinter_expdb::model::*;
    let experiment_clock = c
        .register_clock(&ClockDomain {
            name: "experiment".into(),
            description: String::new(),
        })
        .unwrap();
    let imu_clock = c
        .register_clock(&ClockDomain {
            name: "imu-mcu".into(),
            description: "robot MCU".into(),
        })
        .unwrap();
    c.map_clock(ClockMapping {
        source: imu_clock,
        destination: experiment_clock,
        src_ref_ns: 0,
        dst_ref_ns: -SEC / 2,
        slope: 1.0,
        uncertainty_ns: 85_000,
    })
    .unwrap();
    let episode = c
        .start_episode(EpisodeKind::Interactive, label, experiment_clock)
        .unwrap();

    let video = c
        .add_stream(
            episode,
            &StreamSpec::new(
                "camera",
                &ModalitySchema::builtin("video/rgb").unwrap(),
                experiment_clock,
            )
            .rate(30.0),
        )
        .unwrap();
    let audio = c
        .add_stream(
            episode,
            &StreamSpec::new(
                "mic",
                &ModalitySchema::builtin("audio/pcm").unwrap(),
                experiment_clock,
            )
            .rate(AUDIO_HZ),
        )
        .unwrap();
    let imu = c
        .add_stream(
            episode,
            &StreamSpec::new(
                "imu",
                &ModalitySchema::builtin("sensor/imu").unwrap(),
                imu_clock,
            )
            .rate(IMU_HZ),
        )
        .unwrap();
    let joints = c
        .add_stream(
            episode,
            &StreamSpec::new(
                "joints",
                &ModalitySchema::builtin("robot/joint_state").unwrap(),
                experiment_clock,
            )
            .rate(JOINT_HZ),
        )
        .unwrap();

    let mut audio_chunks = Vec::new();
    for s in 0..10i64 {
        let whole = TimeRange::new(s * SEC, (s + 1) * SEC).unwrap();
        c.add_chunk(episode, video, whole, 30, &noise(2_000, seed + s as u64))
            .unwrap();
        let pcm = noise(AUDIO_HZ as usize * 4, seed * 100 + s as u64);
        c.add_chunk(episode, audio, whole, AUDIO_HZ as u64, &pcm)
            .unwrap();
        audio_chunks.push(pcm);
        c.add_chunk(
            episode,
            joints,
            whole,
            JOINT_HZ as u64,
            &noise(JOINT_HZ as usize * 28, seed + 7 * s as u64),
        )
        .unwrap();
    }
    // The IMU runs on its own clock, started half a second before the experiment.
    for s in 0..12i64 {
        let device = TimeRange::new(s * SEC, (s + 1) * SEC).unwrap();
        c.add_chunk(
            episode,
            imu,
            device,
            IMU_HZ as u64,
            &noise(IMU_HZ as usize * 24, seed + 13 * s as u64),
        )
        .unwrap();
    }

    let at = |ms: i64| ms * 1_000_000;
    c.add_event(
        episode,
        Event::at("instruction", at(1_000), experiment_clock)
            .payload(serde_json::json!({ "text": "pick up the cup" })),
    )
    .unwrap();
    c.add_action(
        episode,
        ActionSegment::new(
            ActionKind::Continuous,
            "move_arm",
            TimeRange::new(at(2_000), at(4_000)).unwrap(),
            experiment_clock,
        ),
    )
    .unwrap();
    c.add_action(
        episode,
        ActionSegment::new(
            ActionKind::Discrete,
            "close_gripper",
            TimeRange::new(at(3_400), at(3_700)).unwrap(),
            experiment_clock,
        ),
    )
    .unwrap();
    c.add_event(
        episode,
        Event::at("contact", at(3_693), experiment_clock).value(1.0),
    )
    .unwrap();
    c.add_event(episode, Event::at("lifted", at(4_200), experiment_clock))
        .unwrap();
    c.add_event(
        episode,
        Event::at("contact", at(7_200), experiment_clock).value(0.0),
    )
    .unwrap();
    c.add_event(
        episode,
        Event::at("reward", at(9_900), experiment_clock).value(1.0),
    )
    .unwrap();
    c.flush().unwrap();
    RobotEpisode {
        episode,
        experiment_clock,
        imu_clock,
        video,
        audio,
        imu,
        joints,
        audio_chunks,
    }
}
