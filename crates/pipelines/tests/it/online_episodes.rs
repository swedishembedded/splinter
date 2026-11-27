// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning from the exact
// tokens an agent's policy sampled, for its clients. If your team needs
// expertise in training agents on their own rollouts while they run, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: practice episodes are an online run's rollouts. Each rollout runs
//! as its own sven session in a fresh copy of its task's tree, every request
//! asking for the policy trace and the sampler's top-K and sampling the whole
//! distribution at the run's temperature; the task's verifier rewards it
//! afterwards. Its records are one per model call, built from that rollout's
//! own requests as sent, each with the version that decoded it, so a
//! publication between two calls gives the span of both versions. A rollout
//! whose policy did not report a call's trace is not exact and carries no
//! records; a verifier that fails is an error of the rollout, never a reward
//! of zero; a sampler that cuts the distribution is refused before anything
//! runs; a cancelled run stops its rollouts.

#![allow(clippy::unwrap_used)]

use crate::common;
use crate::practice_episodes::{greet, MARKER};

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::repo::{call, request_of, tool_calls, write};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::factory::group::GroupKey;
use splinter_core::model_ref::ModelRef;
use splinter_data::calls::CallRecord;
use splinter_factory::online::seams::{
    RolloutEnd, RolloutExecutor, RolloutRequest, RunClock, SystemRunClock,
};
use splinter_orchestrator::Context;
use splinter_pipelines::episodes::online::{
    run_online, run_online_resolving, OnlineEpisodes, OnlinePolicy, OnlineSampling, OnlineSettings,
};
use splinter_pipelines::episodes::{load_collection, EpisodeError, EpisodeSettings, PracticeTask};
use sven_sdk::model::{
    CompletionRequest, LogprobKind, ModelProvider, ResponseEvent, ResponseStream, ServedIdentity,
};

const DIGEST: &str = "sha256:c7ae3118aa233013ee286be9c64eed92aeb1d113ebd470a1a3e646b3d78192a3";

/// A policy served by a learner that publishes: it writes the greeting, then
/// says it is done, reporting with every reply the version that decoded it,
/// its policy trace and, when asked, its top-K.
struct Served {
    version: AtomicU64,
    /// Publish the next version after each first request of a rollout.
    publish_mid_rollout: bool,
    /// Leave the identity off every reply that ends a rollout.
    drop_last_identity: bool,
    /// Never answer: the rollout runs until it is cancelled.
    hang: bool,
    /// Refuse every request after a rollout's first as longer than the
    /// served context, as a server does a conversation that outgrew it.
    refuse_context: bool,
    /// Render each request as the session so far: a later call's prompt ids
    /// extend the earlier call's prompt and answer (an append-only session).
    append_only: bool,
    asked: AtomicU64,
    saw_top: AtomicBool,
    temperatures: std::sync::Mutex<Vec<Option<f64>>>,
}

impl Served {
    fn new(version: u64) -> Self {
        Self {
            version: AtomicU64::new(version),
            publish_mid_rollout: false,
            drop_last_identity: false,
            hang: false,
            refuse_context: false,
            append_only: false,
            asked: AtomicU64::new(0),
            saw_top: AtomicBool::new(false),
            temperatures: std::sync::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for Served {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "policy-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        self.temperatures
            .lock()
            .unwrap()
            .push(req.sampling.temperature);
        if self.hang {
            tokio::time::sleep(Duration::from_secs(600)).await;
        }
        let first = tool_calls(&req).is_empty() && request_of(&req).contains("greeting.txt");
        if self.refuse_context && !first {
            anyhow::bail!(
                "server error (context_length_exceeded): the request needs 9000 tokens of a 8192 context"
            );
        }
        let version = self.version.load(Ordering::SeqCst);
        let answer = if first {
            if self.publish_mid_rollout {
                self.version.fetch_add(1, Ordering::SeqCst);
            }
            call(
                "c-greet",
                "shell",
                serde_json::json!({"description": "greet", "shell_command": "echo hello world > greeting.txt"}),
            )
        } else {
            vec![
                ResponseEvent::TextDelta("Done.".into()),
                ResponseEvent::Done,
            ]
        };
        let prompt: Vec<i64> = if self.append_only {
            // The first call reads [1, 2, 3]; a later one that and the first answer, then the tool's turn.
            if first {
                vec![1, 2, 3]
            } else {
                vec![1, 2, 3, 21, 22, 40, 41]
            }
        } else {
            (0..i64::try_from(req.messages.len()).unwrap() + 1).collect()
        };
        let mut events = vec![
            ResponseEvent::PromptTokens { ids: prompt },
            ResponseEvent::Tokens {
                ids: vec![21, 22],
                logprobs: vec![-0.5, -0.25],
            },
        ];
        if let Some(k) = req.top_logprobs {
            self.saw_top.store(true, Ordering::SeqCst);
            let n = usize::try_from(k).unwrap();
            let k = i64::from(k);
            events.push(ResponseEvent::TopTokens {
                ids: vec![(21..21 + k).collect(), (22..22 + k).collect()],
                logprobs: vec![vec![-0.5; n], vec![-0.25; n]],
            });
        }
        if !(self.drop_last_identity && !first) {
            events.push(ResponseEvent::Identity(ServedIdentity {
                policy_version: Some(version.to_string()),
                tokenizer_digest: Some(DIGEST.into()),
                logprob_kind: Some(LogprobKind::Policy),
                logprob_temperature: req.sampling.temperature,
                ..ServedIdentity::default()
            }));
        }
        events.extend(answer);
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

struct Setup {
    _tasks: Scratch,
    _scratch: Scratch,
    ctx: Context,
    tasks: Vec<PracticeTask>,
    policy: Arc<Served>,
    work: std::path::PathBuf,
}

fn setup(test: &str, policy: Served, verifier_crashes: bool) -> Setup {
    let tasks_dir = Scratch::new(&format!("{test}-tasks"));
    let root = tasks_dir.0.join("practice");
    let dir = greet(&root);
    if verifier_crashes {
        // It judges the starting files (reward 0) and crashes on a greeting.
        write(
            &dir.join("tests/test.sh"),
            &format!(
                "#!/bin/bash\n# {MARKER}\ncd \"$WORKSPACE_DIR\"\ngrep -q 'hello world' greeting.txt && exit 7\necho 0 > \"$LOGS_DIR/reward.txt\"\n"
            ),
        );
    }
    let policy = Arc::new(policy);
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-11-24T09:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(policy.clone(), "scripted/policy-1"),
        );
    let tasks = load_collection(&ctx, &root).unwrap();
    let work = scratch.0.join("online");
    Setup {
        _tasks: tasks_dir,
        _scratch: scratch,
        ctx,
        tasks,
        policy,
        work,
    }
}

fn settings(work: &Path, sampling: OnlineSampling) -> OnlineSettings {
    OnlineSettings {
        // seed: the rollouts' seeds derive from it; any value serves.
        episode: EpisodeSettings::new(1, 0x5eed),
        sampling,
        top_logprobs: Some(2),
        scratch: work.to_path_buf(),
        sessions: splinter_factory::online::rollout::Sessions::PerCall,
    }
}

/// Starts `count` rollouts of the run's one task in group `online:0` and
/// waits for every end.
fn rollouts(
    executor: &mut OnlineEpisodes<'_, '_>,
    task: &PracticeTask,
    count: u32,
) -> Vec<RolloutEnd> {
    for index in 0..count {
        executor
            .start(RolloutRequest {
                group: GroupKey::numbered("online", 0).unwrap(),
                task: task.task.to_string(),
                index,
                policy: None,
            })
            .unwrap();
    }
    let mut ended = Vec::new();
    while ended.len() < count as usize {
        let until = SystemRunClock.now_ms() + 60_000;
        let got = executor.wait(until).unwrap();
        assert!(!got.is_empty(), "a rollout ends within a minute");
        ended.extend(got);
    }
    ended.sort_by_key(|e| e.index);
    ended
}

fn records(end: &RolloutEnd) -> Vec<CallRecord> {
    let done = end.result.as_ref().unwrap();
    done.records
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn concurrent_rollouts_are_rewarded_by_the_verifier_and_carry_one_exact_record_per_call() {
    let s = setup("online-exact", Served::new(3), false);
    let ends = run_online(
        &s.ctx,
        s.tasks.clone(),
        &OnlinePolicy::Reference(ModelRef::policy_default()),
        settings(&s.work, OnlineSampling::untruncated(0.8)),
        &SystemRunClock,
        |executor| rollouts(executor, &s.tasks[0], 3),
    )
    .unwrap();
    assert_eq!(ends.len(), 3);
    for end in &ends {
        let done = end.result.as_ref().unwrap();
        assert_eq!(done.outcome.reward, Some(1.0), "{done:?}");
        assert!(done.outcome.exact, "{done:?}");
        assert_eq!(
            (done.outcome.span.oldest(), done.outcome.span.newest()),
            (3, 3)
        );
        assert_eq!(done.outcome.tokens, 4, "two ids in each of two calls");
        let calls = records(end);
        assert_eq!(calls.len(), 2, "one record per model call");
        for record in &calls {
            assert_eq!(record.policy_version, 3);
            assert!((record.sampler.temperature - 0.8).abs() < 1e-12);
            let answer = record.messages.last().unwrap();
            assert!(answer.train && record.messages.iter().filter(|m| m.train).count() == 1);
            let turn = record.sampled.turns[0].as_ref().unwrap();
            assert_eq!(turn.token_ids, [21, 22]);
            assert_eq!(turn.behavior_top_ids.as_ref().unwrap()[0], [21, 22]);
            assert_eq!(
                turn.prompt_token_ids.len(),
                record.messages.len(),
                "the ids the policy was given for this request"
            );
            assert_eq!(record.messages[0].role, "system", "the request as sent");
        }
        assert_eq!(calls[1].messages.last().unwrap().content, "Done.");
    }
    assert!(
        s.policy.saw_top.load(Ordering::SeqCst),
        "the top-K was asked of every request"
    );
    let temperatures = s.policy.temperatures.lock().unwrap().clone();
    assert!(
        temperatures
            .iter()
            .all(|t| t.is_some_and(|t| (t - 0.8).abs() < 1e-6)),
        "{temperatures:?}"
    );
    assert!(
        std::fs::read_dir(&s.work).unwrap().next().is_none(),
        "the request logs are filed on the trajectories, none left behind"
    );
}

#[test]
fn a_publication_between_two_calls_gives_the_span_of_both_versions() {
    let s = setup(
        "online-refit",
        Served {
            publish_mid_rollout: true,
            ..Served::new(3)
        },
        false,
    );
    let ends = run_online(
        &s.ctx,
        s.tasks.clone(),
        &OnlinePolicy::Reference(ModelRef::policy_default()),
        settings(&s.work, OnlineSampling::untruncated(0.8)),
        &SystemRunClock,
        |executor| rollouts(executor, &s.tasks[0], 1),
    )
    .unwrap();
    let done = ends[0].result.as_ref().unwrap();
    assert!(done.outcome.exact, "{done:?}");
    assert_eq!(
        (done.outcome.span.oldest(), done.outcome.span.newest()),
        (3, 4)
    );
    let versions: Vec<u64> = records(&ends[0]).iter().map(|r| r.policy_version).collect();
    assert_eq!(versions, [3, 4]);
}

#[test]
fn a_call_without_its_trace_makes_an_inexact_rollout_and_a_failed_verifier_is_an_error() {
    let s = setup(
        "online-inexact",
        Served {
            drop_last_identity: true,
            ..Served::new(3)
        },
        false,
    );
    let ends = run_online(
        &s.ctx,
        s.tasks.clone(),
        &OnlinePolicy::Reference(ModelRef::policy_default()),
        settings(&s.work, OnlineSampling::untruncated(0.8)),
        &SystemRunClock,
        |executor| rollouts(executor, &s.tasks[0], 1),
    )
    .unwrap();
    let done = ends[0].result.as_ref().unwrap();
    assert_eq!(
        done.outcome.reward,
        Some(1.0),
        "still rewarded, for the statistics"
    );
    assert!(!done.outcome.exact);
    assert!(
        done.records.is_empty(),
        "an inexact rollout is never training material"
    );

    let s = setup("online-outage", Served::new(3), true);
    let ends = run_online(
        &s.ctx,
        s.tasks.clone(),
        &OnlinePolicy::Reference(ModelRef::policy_default()),
        settings(&s.work, OnlineSampling::untruncated(0.8)),
        &SystemRunClock,
        |executor| rollouts(executor, &s.tasks[0], 1),
    )
    .unwrap();
    let why = ends[0].result.as_ref().unwrap_err();
    assert!(why.contains("verifier"), "{why}");
}

#[test]
fn a_cutting_sampler_is_refused_and_a_cancelled_run_stops_its_rollouts() {
    let s = setup("online-refused", Served::new(3), false);
    for cut in [
        OnlineSampling {
            top_k: 20,
            ..OnlineSampling::untruncated(0.8)
        },
        OnlineSampling {
            top_p: 0.95,
            ..OnlineSampling::untruncated(0.8)
        },
        OnlineSampling {
            min_p: 0.05,
            ..OnlineSampling::untruncated(0.8)
        },
    ] {
        let refused = run_online(
            &s.ctx,
            s.tasks.clone(),
            &OnlinePolicy::Reference(ModelRef::policy_default()),
            settings(&s.work, cut),
            &SystemRunClock,
            |_| (),
        )
        .unwrap_err();
        assert!(
            matches!(refused, EpisodeError::Settings(ref why) if why.contains("whole distribution")),
            "{refused}"
        );
    }
    assert_eq!(s.policy.asked.load(Ordering::SeqCst), 0, "nothing ran");

    let s = setup(
        "online-cancel",
        Served {
            hang: true,
            ..Served::new(3)
        },
        false,
    );
    let started = std::time::Instant::now();
    let ended = run_online(
        &s.ctx,
        s.tasks.clone(),
        &OnlinePolicy::Reference(ModelRef::policy_default()),
        settings(&s.work, OnlineSampling::untruncated(0.8)),
        &SystemRunClock,
        |executor| {
            executor
                .start(RolloutRequest {
                    group: GroupKey::numbered("online", 0).unwrap(),
                    task: s.tasks[0].task.to_string(),
                    index: 0,
                    policy: None,
                })
                .unwrap();
            let early = executor.wait(SystemRunClock.now_ms() + 300).unwrap();
            executor.cancel();
            let after = executor.wait(SystemRunClock.now_ms() + 2_000).unwrap();
            (early.len(), after.len())
        },
    )
    .unwrap();
    assert_eq!(ended, (0, 0), "a cancelled rollout reports no end");
    assert!(
        started.elapsed() < Duration::from_secs(60),
        "the run returned once cancelled"
    );
}

#[test]
fn a_conversation_that_outgrew_the_context_is_a_scored_failure_never_retried() {
    let s = setup(
        "online-context",
        Served {
            refuse_context: true,
            ..Served::new(2)
        },
        false,
    );
    let ends = run_online(
        &s.ctx,
        s.tasks.clone(),
        &OnlinePolicy::Reference(ModelRef::policy_default()),
        settings(&s.work, OnlineSampling::untruncated(1.0)),
        &SystemRunClock,
        |executor| rollouts(executor, &s.tasks[0], 1),
    )
    .unwrap();
    let done = ends[0]
        .result
        .as_ref()
        .expect("a limit is an outcome, not an infrastructure failure the controller retries");
    assert_eq!(done.outcome.reward, Some(0.0), "{done:?}");
    assert_eq!(
        done.outcome.truncation,
        Some(splinter_core::factory::group::Truncation::Context)
    );
    assert!(
        !done.outcome.exact && done.records.is_empty(),
        "the refused run kept no trajectory to pair its calls with: {done:?}"
    );
}

/// Spec: a rollout pinned to a version (its request names the version's own id) asks that version for every call of
/// its episode, though the run's policy publishes a newer version between two calls: its span is one version.
#[test]
fn a_rollout_pinned_to_a_version_asks_it_for_every_call_of_its_episode() {
    let s = setup(
        "online-pinned",
        Served {
            publish_mid_rollout: true,
            ..Served::new(3)
        },
        false,
    );
    // The pinned version: its own server, serving version 3 alone.
    let pinned = Arc::new(Served::new(3));
    let asked = std::sync::Mutex::new(Vec::new());
    let resolve = |id: &str| {
        asked.lock().unwrap().push(id.to_string());
        Ok(OnlinePolicy::Model(Model::new(
            pinned.clone(),
            "scripted/policy-1-v3",
        )))
    };
    let ends = run_online_resolving(
        &s.ctx,
        s.tasks.clone(),
        &OnlinePolicy::Reference(ModelRef::policy_default()),
        Some(&resolve),
        settings(&s.work, OnlineSampling::untruncated(0.8)),
        &SystemRunClock,
        |executor| {
            executor
                .start(RolloutRequest {
                    group: GroupKey::numbered("online", 0).unwrap(),
                    task: s.tasks[0].task.to_string(),
                    index: 0,
                    policy: Some("policy-1-v3".into()),
                })
                .unwrap();
            let mut ended = Vec::new();
            while ended.is_empty() {
                ended = executor.wait(SystemRunClock.now_ms() + 60_000).unwrap();
            }
            ended
        },
    )
    .unwrap();
    let done = ends[0].result.as_ref().unwrap();
    assert!(done.outcome.exact, "{done:?}");
    assert_eq!(
        (done.outcome.span.oldest(), done.outcome.span.newest()),
        (3, 3),
        "every call of the episode asked the version it started on"
    );
    let versions: Vec<u64> = records(&ends[0]).iter().map(|r| r.policy_version).collect();
    assert_eq!(versions, [3, 3], "two calls, one version");
    assert_eq!(*asked.lock().unwrap(), ["policy-1-v3"]);
    assert_eq!(
        s.policy.asked.load(Ordering::SeqCst),
        0,
        "the run's moving policy was never asked"
    );
}

/// Spec: a run that merges sessions writes an exact, append-only session as one record, its every answer
/// supervised where it stands, and says so; a session that is not append-only stays one record per call, and the
/// outcome names why.
#[test]
fn a_merged_run_writes_an_append_only_session_as_one_record_and_names_why_another_is_not() {
    for append_only in [true, false] {
        let s = setup(
            &format!("online-sessions-{append_only}"),
            Served {
                append_only,
                ..Served::new(3)
            },
            false,
        );
        let mut settings = settings(&s.work, OnlineSampling::untruncated(0.8));
        settings.sessions =
            splinter_factory::online::rollout::Sessions::Merged { block: Some(4096) };
        let ends = run_online(
            &s.ctx,
            s.tasks.clone(),
            &OnlinePolicy::Reference(ModelRef::policy_default()),
            settings,
            &SystemRunClock,
            |executor| rollouts(executor, &s.tasks[0], 1),
        )
        .unwrap();
        let done = ends[0].result.as_ref().unwrap();
        let lines: Vec<&str> = done.records.lines().collect();
        if append_only {
            assert_eq!(done.outcome.session.as_deref(), Some("merged"), "{done:?}");
            assert_eq!(lines.len(), 1, "the session is one record");
            let session: splinter_data::session::SessionRecord =
                serde_json::from_str(lines[0]).unwrap();
            assert_eq!(session.policy_version, 3);
        } else {
            assert_eq!(
                done.outcome.session.as_deref(),
                Some("not_append_only"),
                "{done:?}"
            );
            assert_eq!(lines.len(), 2, "one record per call");
        }
    }
}
