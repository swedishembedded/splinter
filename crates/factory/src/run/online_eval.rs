// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training loops that roll a model back only on
// evidence, for its clients. If your team needs expertise in evaluating
// continuously trained models while they train, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The watchdog's evaluations of an online stage: each version asked under
//! the id the learner serves it alone under while it retains it (so the
//! training that publishes on does not move it), its rollouts of the
//! held-out tasks run by the stage's own rollouts, beside the training's and
//! on their own groups, never seen by the task selector. A rollout whose infrastructure failed, or that was
//! not graded, is not evidence: its pair is left out.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use splinter_core::factory::group::GroupKey;

use crate::online::rollout::Sessions;
use crate::online::seams::{
    OnlineRollouts, RolloutRequest, RolloutsRequest, RunClock, SystemRunClock,
};
use crate::online::watchdog::{Evaluator, Pair, Version};

/// How long one wait for a rollout of an evaluation lasts before it is
/// asked again.
const WAIT_MS: u64 = 1_000;

/// What every evaluation of a stage shares.
pub(crate) struct Served<'a> {
    /// Runs the rollouts.
    pub rollouts: &'a dyn OnlineRollouts,
    /// The held-out task collections (the monitor pool, the confirm set),
    /// each a folder and the ids of its tasks.
    pub task_sets: Vec<(PathBuf, Vec<String>)>,
    /// Where the frozen copies, the rollouts' scratch files are kept.
    pub work: PathBuf,
    /// The groups' namespace prefix (the stage's).
    pub namespace: String,
    /// The sampling temperature.
    pub temperature: f32,
    /// The longest reply.
    pub max_output_tokens: Option<u64>,
    /// The seed the rollouts' draws derive from.
    pub seed: u64,
    /// Rollouts of one version an evaluation runs at once at most.
    pub slots: usize,
}

impl Served<'_> {
    /// The pass or fail of each task `version` was graded on: one rollout
    /// each, under the group namespace `namespace`.
    fn grade(
        &self,
        version: &Version,
        tasks: &[String],
        namespace: &str,
    ) -> Result<BTreeMap<String, bool>, String> {
        // The version itself, as the learner serves it while it retains it:
        // never a copy offered under the id the training publishes on.
        let id = version.id.clone().ok_or_else(|| {
            format!(
                "version {} is served under no id of its own: the learner retains no versions",
                version.version
            )
        })?;
        let dir = self
            .task_sets
            .iter()
            .find(|(_, ids)| tasks.iter().all(|t| ids.contains(t)))
            .map(|(dir, _)| dir.clone())
            .ok_or("no held-out task collection holds the tasks asked for")?;
        let mut graded = BTreeMap::new();
        let mut failure = None;
        self.rollouts.run(
            &RolloutsRequest {
                tasks: &dir,
                policy: &id,
                temperature: self.temperature,
                top_logprobs: None,
                max_output_tokens: self.max_output_tokens,
                seed: self.seed,
                scratch: &self.work,
                sessions: Sessions::PerCall,
            },
            &mut |executor, _| {
                // At most `slots` rollouts at once: an evaluation shares the
                // learner's server with the training's own rollouts.
                let mut queued = (0u64..).zip(tasks);
                let mut left = 0;
                let mut start = |executor: &mut dyn crate::online::seams::RolloutExecutor,
                                 left: &mut usize,
                                 failure: &mut Option<String>| {
                    while *left < self.slots {
                        let Some((seq, task)) = queued.next() else {
                            return;
                        };
                        let Ok(group) = GroupKey::numbered(namespace, seq) else {
                            continue;
                        };
                        match executor.start(RolloutRequest {
                            group,
                            task: task.clone(),
                            index: 0,
                            policy: None,
                        }) {
                            Ok(()) => *left += 1,
                            Err(e) => *failure = Some(e.to_string()),
                        }
                    }
                };
                start(executor, &mut left, &mut failure);
                while left > 0 {
                    let ends = match executor.wait(SystemRunClock.now_ms() + WAIT_MS) {
                        Ok(ends) => ends,
                        Err(e) => {
                            failure = Some(e.to_string());
                            return;
                        }
                    };
                    for end in ends {
                        left -= 1;
                        if let Err(why) = &end.result {
                            failure.get_or_insert_with(|| why.clone());
                        }
                        let seq: Option<usize> = end
                            .group
                            .as_str()
                            .rsplit(':')
                            .next()
                            .and_then(|s| s.parse().ok());
                        let reward = end.result.ok().and_then(|done| done.outcome.reward);
                        if let (Some(task), Some(reward)) = (seq.and_then(|s| tasks.get(s)), reward)
                        {
                            graded.insert(task.clone(), reward >= 1.0);
                        }
                    }
                    start(executor, &mut left, &mut failure);
                }
            },
        )?;
        match failure {
            Some(why) if graded.is_empty() => Err(why),
            _ => Ok(graded),
        }
    }
}

/// The evaluator of an online stage on scoped threads of `scope`.
pub(crate) struct ServedEvaluator<'s, 'e> {
    scope: &'s std::thread::Scope<'s, 'e>,
    served: &'e Served<'e>,
    pairs: mpsc::Receiver<Vec<Pair>>,
    sender: mpsc::Sender<Vec<Pair>>,
    running: usize,
    epoch: u64,
    started: u64,
}

impl<'s, 'e> ServedEvaluator<'s, 'e> {
    /// Evaluations of `served` in the run's `epoch`, on threads of `scope`:
    /// their groups' namespaces name the epoch and the evaluation.
    pub(crate) fn new(
        scope: &'s std::thread::Scope<'s, 'e>,
        served: &'e Served<'e>,
        epoch: u64,
    ) -> Self {
        let (sender, pairs) = mpsc::channel();
        Self {
            scope,
            served,
            pairs,
            sender,
            running: 0,
            epoch,
            started: 0,
        }
    }
}

impl Evaluator for ServedEvaluator<'_, '_> {
    fn start(&mut self, candidate: &Version, anchor: &Version, tasks: Vec<String>) {
        let served = self.served;
        let sender = self.sender.clone();
        let (candidate, anchor) = (candidate.clone(), anchor.clone());
        let (epoch, n) = (self.epoch, self.started);
        self.started += 1;
        self.running += 1;
        self.scope.spawn(move || {
            let namespace = |who: &str| format!("{}-e{epoch}-watch{n}-{who}", served.namespace);
            let graded = std::thread::scope(|inner| {
                let c = inner.spawn(|| served.grade(&candidate, &tasks, &namespace("c")));
                let a = inner.spawn(|| served.grade(&anchor, &tasks, &namespace("a")));
                (c.join(), a.join())
            });
            let pairs = match graded {
                (Ok(Ok(c)), Ok(Ok(a))) => tasks
                    .iter()
                    .filter_map(|t| {
                        Some(Pair {
                            task: t.clone(),
                            candidate: *c.get(t)?,
                            anchor: *a.get(t)?,
                        })
                    })
                    .collect(),
                (c, a) => {
                    eprintln!(
                        "watchdog: an evaluation could not be run (candidate {:?}, anchor {:?}); it decides nothing",
                        c.map(|r| r.err()),
                        a.map(|r| r.err())
                    );
                    Vec::new()
                }
            };
            // The receiver outlives every evaluation of the scope.
            let _ = sender.send(pairs);
        });
    }

    fn poll(&mut self) -> Vec<Pair> {
        let mut pairs = Vec::new();
        while let Ok(batch) = self.pairs.recv_timeout(Duration::ZERO) {
            self.running -= 1;
            pairs.extend(batch);
        }
        pairs
    }

    fn idle(&self) -> bool {
        self.running == 0
    }
}
