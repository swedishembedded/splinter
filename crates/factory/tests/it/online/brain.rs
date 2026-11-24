// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning against a
// training service that serves each policy version it trains, for its
// clients. If your team needs expertise in streaming rollouts into a running
// policy-gradient job, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the learner on brain's streaming job. A group is sent as brain's
//! rollout records (each call with its member's reward, the group's key and
//! the member's rollout key) and answered as the seam says: accepted,
//! already held, or a full queue; the job's events come back as applied,
//! published, committed and step tokens. A job that restarted is an error
//! marked as a restart; the job started again under the same key answers
//! what it had committed as duplicates, which the learner reports as
//! committed.

use std::time::{Duration, Instant};

use serde_json::{json, Map};
use splinter_core::chat::WireMessage;
use splinter_core::factory::group::GroupKey;
use splinter_data::calls::{CallRecord, Sampler};
use splinter_data::{SampledTurn, SampledTurns};
use splinter_factory::online::brain::records_of;
use splinter_factory::online::seams::{
    LearnerEvent, Member, OutgoingGroup, SendOutcome, StreamLearner,
};
use splinter_factory::recipe::ModelRef;
use splinter_factory::training::{BrainTrainer, OnlineTrain, Trainer};
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::scripted_stream::StreamScript;
use splinter_model::served_train::ArtifactStore;

fn message(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        role: role.into(),
        content: content.into(),
        images: Vec::new(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        reasoning_content: None,
        train,
        loss_weight: None,
    }
}

/// One call of 5 sampled tokens at version 0, as the executor writes it.
fn call() -> String {
    let record = CallRecord {
        messages: vec![message("user", "q", false), message("assistant", "a", true)],
        tools: Vec::new(),
        sampled: SampledTurns {
            tokenizer_digest: format!("sha256:{}", "ab".repeat(32)),
            turns: vec![Some(SampledTurn {
                prompt_token_ids: vec![1, 2],
                token_ids: vec![3; 5],
                behavior_logprobs: vec![-0.5; 5],
                behavior_top_ids: None,
                behavior_top_logprobs: None,
            })],
        },
        policy_version: 0,
        sampler: Sampler::untruncated(1.0).unwrap(),
        truncated: false,
    };
    format!("{}\n", serde_json::to_string(&record).unwrap())
}

fn group(seq: u64) -> OutgoingGroup {
    OutgoingGroup {
        key: GroupKey::numbered("online", seq).unwrap(),
        task: "t".into(),
        members: (0..2)
            .map(|index| Member {
                index,
                reward: f64::from(index),
                records: call(),
            })
            .collect(),
    }
}

fn polled_until(
    learner: &mut dyn StreamLearner,
    enough: impl Fn(&[LearnerEvent]) -> bool,
) -> Vec<LearnerEvent> {
    let started = Instant::now();
    let mut events = Vec::new();
    while !enough(&events) {
        assert!(started.elapsed() < Duration::from_secs(30), "{events:?}");
        events.extend(learner.poll().unwrap());
        std::thread::sleep(Duration::from_millis(20));
    }
    events
}

fn committed(events: &[LearnerEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            LearnerEvent::Committed { groups } => Some(groups.clone()),
            _ => None,
        })
        .flatten()
        .map(|k| k.as_str().to_string())
        .collect()
}

#[test]
fn groups_reach_the_streaming_job_and_a_restart_is_resumed_from_its_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("artifacts");
    std::fs::create_dir_all(&store).unwrap();
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript {
        stream: StreamScript {
            restart_at_step: Some(3),
            ..StreamScript::default()
        },
        ..JobScript::over(store.clone())
    });
    let endpoints = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&store).unwrap());
    let trainer = BrainTrainer::new(endpoints, dir.path().join("models"));
    let base: ModelRef = "brain:Qwen/Toy".parse().unwrap();
    let params = Map::from_iter([("estimator".to_string(), json!("clipped"))]);
    let request = OnlineTrain {
        base: &base,
        params: &params,
        resume_key: "online-run",
        checkpoint_every: Some(2),
        publish_adapter: "splinter/online:s:abc:latest",
        publish_every: 1,
        step_tokens: 10,
        queue_tokens: 100,
        max_staleness: 4,
        max_steps: None,
        top_k: None,
        start: None,
    };

    // A group's records are brain's v2 lines of each member's calls.
    let lines: Vec<serde_json::Value> = records_of(&group(0))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        (
            lines[1]["group"].clone(),
            lines[1]["rollout"].clone(),
            lines[1]["reward"].clone()
        ),
        (json!("online:0"), json!("online:0/1"), json!(1.0))
    );

    let mut learner = trainer.stream(&request).unwrap();
    assert_eq!(learner.served_as(), "splinter/online:s:abc:latest");
    assert_eq!(learner.state().unwrap().policy_version, 0);
    assert_eq!(learner.send(&group(0)).unwrap(), SendOutcome::Accepted);
    assert_eq!(learner.send(&group(0)).unwrap(), SendOutcome::Duplicate);
    assert_eq!(learner.send(&group(1)).unwrap(), SendOutcome::Accepted);
    let events = polled_until(learner.as_mut(), |e| committed(e).len() == 2);
    assert_eq!(committed(&events), ["online:0", "online:1"]);
    assert!(events
        .iter()
        .any(|e| matches!(e, LearnerEvent::Step { tokens: 10 })));
    assert!(events
        .iter()
        .any(|e| matches!(e, LearnerEvent::Published { version: 2 })));
    let telemetry = learner.telemetry();
    let (version, digest) = telemetry.committed_version().unwrap();
    assert_eq!(version, 2);
    let adapter = dir.path().join("adapter.safetensors");
    learner.adapter(digest, &adapter).unwrap();
    assert!(adapter.is_file());

    // The third step takes the service down before its checkpoint.
    learner.send(&group(2)).unwrap();
    let restarted = (0..200).find_map(|_| {
        std::thread::sleep(Duration::from_millis(20));
        learner.poll().err()
    });
    assert!(
        restarted.as_ref().is_some_and(|e| e.restarted),
        "{restarted:?}"
    );
    drop(learner);

    let mut resumed = trainer.stream(&request).unwrap();
    assert_eq!(resumed.send(&group(0)).unwrap(), SendOutcome::Duplicate);
    assert_eq!(
        resumed.send(&group(2)).unwrap(),
        SendOutcome::Accepted,
        "lost, taken again"
    );
    let events = resumed.poll().unwrap();
    assert_eq!(
        committed(&events),
        ["online:0"],
        "committed before the restart"
    );
}
