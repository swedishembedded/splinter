// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning against a
// training service that serves each policy version it trains, for its
// clients. If your team needs expertise in streaming rollouts into a running
// policy-gradient job, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a streaming policy-gradient run takes complete groups while it
//! runs. Offered groups are queued and answered with the keys accepted and
//! those already held; a step trains once its tokens are queued, its groups
//! reported applied, every `publish_every` steps a version is published and
//! every checkpoint commits what was applied. A full queue is refused with
//! its allowance, a group of one rollout is refused as invalid, and a run
//! the server lost is a restart: the same run started again continues in the
//! next epoch and answers its committed keys as duplicates. Finishing trains
//! what is queued, commits it and ends the run.

#![allow(clippy::unwrap_used)]

use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use splinter_model::scripted_endpoint::ScriptedEndpoint;
use splinter_model::scripted_jobs::JobScript;
use splinter_model::scripted_stream::StreamScript;
use splinter_model::served_train::{
    Offered, StreamDoc, StreamError, StreamEvent, StreamJob, StreamSpec,
};

#[allow(dead_code)]
#[path = "support/served.rs"]
mod served;
use served::{endpoints, rig};

/// One record of rollout `rollout` of `group`, `tokens` policy tokens long.
fn record(group: &str, rollout: u32, reward: f64, tokens: usize) -> Value {
    json!({
        "reward": reward, "group": group, "rollout": format!("{group}/{rollout}"), "policy_version": 0,
        "messages": [
            {"role": "user", "content": "q", "train": false},
            {"role": "assistant", "content": "a", "train": true, "token_ids": vec![1; tokens],
             "behavior_logprobs": vec![-0.5; tokens]},
        ],
    })
}

/// A complete group of two rollouts of 5 tokens each.
fn group(key: &str) -> String {
    format!("{}\n{}\n", record(key, 0, 1.0, 5), record(key, 1, 0.0, 5))
}

fn spec(step_tokens: u64, queue_tokens: u64) -> StreamSpec {
    StreamSpec {
        base: "Qwen/Toy".into(),
        params: Map::from_iter([("estimator".to_string(), json!("clipped"))]),
        resume_key: Some("online-run".into()),
        checkpoint_every: Some(2),
        stream: StreamDoc {
            publish_adapter: "local/toy:splinter:online:v".into(),
            publish_every: 1,
            step_tokens,
            queue_tokens,
            max_staleness: Some(4),
            max_steps: None,
            top_k: None,
            start_adapter: None,
        },
    }
}

/// Every event until `done` says it has seen enough.
fn events_until(job: &mut StreamJob, done: impl Fn(&[StreamEvent]) -> bool) -> Vec<StreamEvent> {
    let started = Instant::now();
    let mut seen = Vec::new();
    while !done(&seen) {
        assert!(started.elapsed() < Duration::from_secs(30), "{seen:?}");
        seen.extend(job.events().unwrap());
        std::thread::sleep(Duration::from_millis(20));
    }
    seen
}

fn committed(events: &[StreamEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::Committed { groups, .. } => Some(groups.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

#[test]
fn groups_are_queued_trained_published_and_committed_and_a_full_queue_says_so() {
    let rig = rig();
    let script = JobScript::over(rig.store.clone());
    let paused = script.stream.paused.clone();
    let service = ScriptedEndpoint::start(vec![]).with_training(script);
    let brain = endpoints(&service, &rig);
    // A step trains 10 tokens (one group); the queue holds 20.
    let mut job = StreamJob::start(&brain, &spec(10, 20)).unwrap();
    assert_eq!((job.epoch(), job.version()), (1, 0));
    assert_eq!(job.publish_adapter(), "local/toy:splinter:online:v");

    let taken = job.offer(&group("g0")).unwrap();
    assert!(
        matches!(&taken, Offered::Taken { accepted, .. } if accepted == &["g0"]),
        "{taken:?}"
    );
    let again = job.offer(&group("g0")).unwrap();
    assert!(
        matches!(&again, Offered::Taken { duplicates, .. } if duplicates == &["g0"]),
        "{again:?}"
    );
    let seen = events_until(&mut job, |e| {
        e.iter().any(|e| matches!(e, StreamEvent::Published { .. }))
    });
    assert!(seen
        .iter()
        .any(|e| matches!(e, StreamEvent::Applied { step: 1, groups, skipped: None } if groups == &["g0"])));
    assert!(seen
        .iter()
        .any(|e| matches!(e, StreamEvent::Step { pg } if pg["policy_tokens"] == 10.0)));
    assert!(seen
        .iter()
        .any(|e| matches!(e, StreamEvent::Published { version: 1, .. })));

    let refused = job
        .offer(&format!("{}\n", record("solo", 0, 1.0, 5)))
        .unwrap_err();
    assert!(
        matches!(refused, StreamError::Refused(ref why) if why.contains("group_of_one")),
        "{refused}"
    );

    // The queue holds 20 tokens: a 30-token offer over a held step is refused.
    // The run holds its steps while the queue is filled.
    paused.store(true, std::sync::atomic::Ordering::SeqCst);
    let big = |key: &str| format!("{}\n{}\n", record(key, 0, 1.0, 15), record(key, 1, 0.0, 15));
    let _ = job.offer(&big("g1")).unwrap();
    let full = job.offer(&big("g2")).unwrap();
    assert!(
        matches!(full, Offered::Full { allowance: 0, .. }),
        "{full:?}"
    );
    paused.store(false, std::sync::atomic::Ordering::SeqCst);

    job.finish().unwrap();
    let seen = events_until(&mut job, |e| {
        e.iter().any(|e| matches!(e, StreamEvent::Done { .. }))
    });
    let mut all = committed(&seen);
    all.sort();
    assert_eq!(all, ["g0", "g1"], "{seen:?}");
}

#[test]
fn a_run_the_server_lost_is_a_restart_and_continues_in_the_next_epoch() {
    let rig = rig();
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript {
        stream: StreamScript {
            restart_at_step: Some(3),
            ..StreamScript::default()
        },
        ..JobScript::over(rig.store.clone())
    });
    let brain = endpoints(&service, &rig);
    let mut job = StreamJob::start(&brain, &spec(10, 100)).unwrap();
    for key in ["a", "b"] {
        job.offer(&group(key)).unwrap();
    }
    // Two steps, a checkpoint at step 2: a and b committed.
    let seen = events_until(&mut job, |e| committed(e).len() == 2);
    assert_eq!(committed(&seen), ["a", "b"]);
    // The third step brings the server down before its checkpoint.
    job.offer(&group("c")).unwrap();
    let restarted = (0..200).find_map(|_| {
        std::thread::sleep(Duration::from_millis(20));
        job.events().err().or_else(|| job.offer(&group("d")).err())
    });
    assert!(
        matches!(restarted, Some(StreamError::Restarted(_))),
        "{restarted:?}"
    );

    let mut resumed = StreamJob::start(&brain, &spec(10, 100)).unwrap();
    assert_eq!(resumed.epoch(), 2);
    let answer = resumed
        .offer(&format!("{}{}{}", group("a"), group("b"), group("c")))
        .unwrap();
    let Offered::Taken {
        accepted,
        duplicates,
        ..
    } = answer
    else {
        panic!("{answer:?}")
    };
    assert_eq!(duplicates, ["a", "b"], "committed before the restart");
    assert_eq!(accepted, ["c"], "lost with the restart, taken again");
}

#[test]
fn a_run_is_ready_once_its_server_serves_its_policy_and_one_that_never_does_is_refused_by_name() {
    // A real server lists the run's policy only once its base has loaded and its first version is published:
    // rollouts drawn before then are answered "model not found".
    let rig = rig();
    let withheld = |after: Duration| {
        ScriptedEndpoint::start(vec![]).with_training(JobScript {
            stream: StreamScript {
                withhold_policy: Some(after),
                ..StreamScript::default()
            },
            ..JobScript::over(rig.store.clone())
        })
    };
    let service = withheld(Duration::from_millis(600));
    let brain = endpoints(&service, &rig);
    let mut job = StreamJob::start(&brain, &spec(10, 20)).unwrap();
    let started = Instant::now();
    job.await_served(Duration::from_secs(30)).unwrap();
    assert!(
        started.elapsed() >= Duration::from_millis(300),
        "ready after {:?}, before the policy was served",
        started.elapsed()
    );
    // What the run reported while it was awaited is still reported.
    job.offer(&group("g0")).unwrap();
    let seen = events_until(&mut job, |e| {
        e.iter().any(|e| matches!(e, StreamEvent::Applied { .. }))
    });
    assert!(!seen.is_empty());

    let never = withheld(Duration::MAX);
    let brain = endpoints(&never, &rig);
    let mut job = StreamJob::start(&brain, &spec(10, 20)).unwrap();
    let refused = job.await_served(Duration::from_millis(500)).unwrap_err();
    assert!(
        refused.to_string().contains("local/toy:splinter:online:v"),
        "{refused}"
    );
}

#[test]
fn a_finished_runs_result_is_read_once_its_job_has_ended() {
    // brain answers 409 ("the job has not finished") while a finished run still writes its outputs: the receipts
    // of its last step can arrive before its result exists.
    let rig = rig();
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript {
        stream: StreamScript {
            result_pending: 3,
            ..StreamScript::default()
        },
        ..JobScript::over(rig.store.clone())
    });
    let brain = endpoints(&service, &rig);
    let mut job = StreamJob::start(&brain, &spec(10, 20)).unwrap();
    job.offer(&group("g0")).unwrap();
    job.finish().unwrap();
    let outputs = job.result().unwrap();
    assert!(outputs["adapter_digest"].is_string(), "{outputs}");
}

#[test]
fn a_finish_the_server_does_not_answer_is_made_again() {
    // A connection the server closes as the finish is sent (an idle keep-alive it dropped) is no answer: the run
    // still holds its groups and ends only when it is told again.
    let rig = rig();
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript {
        stream: StreamScript {
            unanswered_finishes: 2,
            ..StreamScript::default()
        },
        ..JobScript::over(rig.store.clone())
    });
    let brain = endpoints(&service, &rig);
    let mut job = StreamJob::start(&brain, &spec(10, 20)).unwrap();
    job.offer(&group("g0")).unwrap();
    job.finish().unwrap();
    let seen = events_until(&mut job, |e| {
        e.iter().any(|e| matches!(e, StreamEvent::Done { .. }))
    });
    assert_eq!(committed(&seen), ["g0"], "{seen:?}");
    assert_eq!(
        service.jobs().stream_runs.finishes(),
        1,
        "the run was finished once"
    );
}
