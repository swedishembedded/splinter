// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that train agents on their own
// sandboxed episodes with the exact tokens and log-probabilities the policy
// sampled, for its clients. If your team needs expertise in on-policy data
// capture for agent training you can procure our services by sending an
// email to info@swedishembedded.com.

//! The endpoint path: rollouts from an agent executor that runs tasks in
//! sandboxes against a served policy.
//!
//! This and the in-process path are the only ways a campaign collects training
//! episodes: it never drives a benchmark's runner on the benchmark's own
//! collections, because the benchmark is the sealed measurement authority and
//! is asked only for verdicts on finished candidates.
//!
//! The executor is a trait so that a campaign does not depend on how
//! episodes are run: an executor loads Harbor-format task folders
//! (instruction, environment, tests that write a reward), runs the policy
//! through an agent harness against a brain endpoint, and returns for every episode
//! the reward the task's verifier wrote and, for every model call, the prompt
//! token ids, the generated token ids and their log-probabilities as the
//! endpoint returned them. [`EpisodeCollector`] turns those episodes into
//! rollout groups; an episode whose chain of calls the executor cannot prove is
//! returned marked unprovable and is dropped from training, never repaired by
//! re-tokenizing. [`crate::episodes::SvenEpisodes`] implements it over practice
//! task folders run through sven: an attempt the task's verifier graded is an
//! [`Episode`] with its reward, one whose verifier or infrastructure failed
//! has none, and one whose calls do not all carry their token ids and
//! log-probabilities is `provable: false`.

use serde_json::{json, Value};
use splinter_core::chat::WireMessage;

use super::types::{
    CollectRequest, PolicyRef, Rollout, RolloutCollector, RolloutGroup, RolloutTrace, ServedError,
};

/// One episode the executor ran.
#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    /// The task it ran.
    pub task: String,
    /// The reward the task's verifier wrote, in `[0, 1]`; `None` when the
    /// episode could not be graded (an infrastructure failure, which is not a
    /// wrong answer and enters no group).
    pub reward: Option<f64>,
    /// Whether the verifier called it a pass.
    pub passed: bool,
    /// The conversation before the policy's final action.
    pub prompt: Vec<WireMessage>,
    /// The policy's final action.
    pub completion: String,
    /// What the verifier showed after a failure.
    pub feedback: Option<String>,
    /// What only the verifier knows.
    pub privileged: Option<String>,
    /// The sampled calls, with the policy version each was drawn from.
    pub trace: Option<RolloutTrace>,
}

/// An agent executor that runs tasks against a policy endpoint.
pub trait EpisodeExecutor {
    /// The executor's settings, for the configuration digest.
    fn describe(&self) -> Value;
    /// The tasks it can run.
    fn tasks(&self) -> Vec<String>;
    /// Makes the endpoint serve `policy`: loads the adapter without a restart
    /// and tags later samples with its version.
    ///
    /// # Errors
    /// The endpoint cannot serve the policy.
    fn serve(&mut self, policy: &PolicyRef) -> Result<(), ServedError>;
    /// Runs `attempts` episodes of each task, drawing under `seed`.
    ///
    /// # Errors
    /// The executor could not run the episodes at all; an episode that ran and
    /// could not be graded is an [`Episode`] with no reward.
    fn run(
        &mut self,
        tasks: &[String],
        attempts: usize,
        seed: u64,
    ) -> Result<Vec<Episode>, ServedError>;
}

/// Rollout groups from an [`EpisodeExecutor`].
pub struct EpisodeCollector<E> {
    executor: E,
}

impl<E: EpisodeExecutor> EpisodeCollector<E> {
    /// A collector over `executor`.
    #[must_use]
    pub fn new(executor: E) -> Self {
        Self { executor }
    }
}

impl<E: EpisodeExecutor> RolloutCollector for EpisodeCollector<E> {
    fn describe(&self) -> Value {
        json!({"path": "episode executor", "executor": self.executor.describe()})
    }

    fn tasks(&self) -> Vec<String> {
        self.executor.tasks()
    }

    fn set_policy(&mut self, policy: &PolicyRef) -> Result<(), ServedError> {
        self.executor.serve(policy)
    }

    fn collect(&mut self, request: &CollectRequest) -> Result<Vec<RolloutGroup>, ServedError> {
        let episodes = self
            .executor
            .run(&request.tasks, request.attempts, request.seed)?;
        let mut groups: Vec<RolloutGroup> = request
            .tasks
            .iter()
            .map(|task| RolloutGroup {
                task: task.clone(),
                rollouts: Vec::new(),
            })
            .collect();
        for episode in episodes {
            let Some(reward) = episode.reward else {
                continue;
            };
            let Some(group) = groups.iter_mut().find(|g| g.task == episode.task) else {
                return Err(ServedError::Collector(format!(
                    "the executor returned an episode of {:?}, which was not asked for",
                    episode.task
                )));
            };
            group.rollouts.push(Rollout {
                reward,
                passed: episode.passed,
                prompt: episode.prompt,
                completion: episode.completion,
                feedback: episode.feedback,
                privileged: episode.privileged,
                policy_version: request.policy.version,
                trace: episode.trace,
            });
        }
        Ok(groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::served::types::TraceSequence;

    struct Scripted {
        episodes: Vec<Episode>,
        served: Vec<u64>,
    }

    impl EpisodeExecutor for Scripted {
        fn describe(&self) -> Value {
            json!({"scripted": true})
        }
        fn tasks(&self) -> Vec<String> {
            vec!["a".into(), "b".into()]
        }
        fn serve(&mut self, policy: &PolicyRef) -> Result<(), ServedError> {
            self.served.push(policy.version);
            Ok(())
        }
        fn run(
            &mut self,
            _tasks: &[String],
            _attempts: usize,
            _seed: u64,
        ) -> Result<Vec<Episode>, ServedError> {
            Ok(self.episodes.clone())
        }
    }

    fn episode(task: &str, reward: Option<f64>, provable: bool) -> Episode {
        Episode {
            task: task.into(),
            reward,
            passed: reward == Some(1.0),
            prompt: Vec::new(),
            completion: "x".into(),
            feedback: None,
            privileged: None,
            trace: Some(RolloutTrace {
                sequences: vec![TraceSequence {
                    prompt_token_ids: vec![1, 2],
                    generation_token_ids: vec![3],
                    generation_log_probs: vec![-0.7],
                    truncated: false,
                }],
                provable,
            }),
        }
    }

    fn request() -> CollectRequest {
        CollectRequest {
            tasks: vec!["a".into(), "b".into()],
            attempts: 2,
            policy: PolicyRef {
                version: 3,
                ..PolicyRef::default()
            },
            seed: 0, // seed: the scripted executor ignores it
        }
    }

    #[test]
    fn episodes_become_groups_per_task_with_their_traces_and_the_policy_version() {
        let executor = Scripted {
            episodes: vec![
                episode("a", Some(1.0), true),
                episode("a", Some(0.0), true),
                episode("b", Some(0.0), true),
            ],
            served: Vec::new(),
        };
        let mut collector = EpisodeCollector::new(executor);
        collector.set_policy(&request().policy).unwrap();
        let groups = collector.collect(&request()).unwrap();
        assert_eq!(groups[0].rollouts.len(), 2);
        assert_eq!(groups[1].rollouts.len(), 1);
        assert!(groups[0].rollouts.iter().all(|r| r.policy_version == 3));
        assert_eq!(
            groups[0].rollouts[0].trace.as_ref().unwrap().sequences[0].generation_log_probs,
            [-0.7]
        );
    }

    #[test]
    fn an_episode_nothing_could_grade_enters_no_group_and_an_unasked_task_is_an_error() {
        let ungraded = Scripted {
            episodes: vec![episode("a", None, true), episode("a", Some(1.0), false)],
            served: Vec::new(),
        };
        let mut collector = EpisodeCollector::new(ungraded);
        let groups = collector.collect(&request()).unwrap();
        assert_eq!(
            groups[0].rollouts.len(),
            1,
            "an infrastructure failure is not a wrong answer"
        );
        assert!(!groups[0].rollouts[0].trace.as_ref().unwrap().provable);

        let stray = Scripted {
            episodes: vec![episode("z", Some(1.0), true)],
            served: Vec::new(),
        };
        let refused = EpisodeCollector::new(stray)
            .collect(&request())
            .unwrap_err();
        assert!(refused.to_string().contains("not asked for"));
    }
}
