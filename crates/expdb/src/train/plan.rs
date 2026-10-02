// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The training plan: samples that point at experience.

use serde::{Deserialize, Serialize};

use super::recipe::{Objective, Recipe};
use super::rng::Rng;
use crate::blob::BlobRef;
use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{DatasetNode, TimeRange};

/// Data a sample refers to, resolved to text only when it is needed.
/// Siblings refer to the same context, so it is stored once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "ref", rename_all = "snake_case")]
pub enum DataRef {
    /// A task instance's description.
    Task {
        /// The task instance.
        instance: ContentId,
    },
    /// Everything before a decision: the task, then each observation and
    /// action on the path to it.
    Context {
        /// The decision.
        decision: RecordId,
    },
    /// What a decision did.
    Action {
        /// The decision.
        decision: RecordId,
    },
    /// The whole path of an attempt: every observation and action.
    Episode {
        /// The attempt.
        attempt: RecordId,
    },
    /// What an observation showed.
    Observation {
        /// The observation record.
        record: RecordId,
    },
    /// Stored bytes.
    Blob {
        /// Where they are.
        blob: BlobRef,
    },
    /// What a stream held over an interval of its episode's clock.
    Window {
        /// The stream.
        stream: RecordId,
        /// The span, on the episode's clock.
        interval: TimeRange,
    },
    /// An action over an interval, with its payload.
    ActionSegment {
        /// The action record.
        record: RecordId,
    },
    /// The text or details an event carries.
    EventPayload {
        /// The event record.
        record: RecordId,
    },
}

/// A negative example and whether it was chosen to be hard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Negative {
    /// What is not a match.
    pub data: DataRef,
    /// Whether it was taken from nearby in time, where telling it from the
    /// positive takes more than noticing the moment is different.
    pub hard: bool,
}

/// One attempt of a group and how it compares with the rest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rollout {
    /// The attempt's whole path.
    pub response: DataRef,
    /// The reward it stood at.
    pub reward: f64,
    /// Its reward relative to the group: (reward - mean) / deviation.
    pub advantage: f64,
}

/// What a sample teaches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SampleBody {
    /// Imitate `target` given `context`.
    Sft {
        /// What came before.
        context: DataRef,
        /// What to produce.
        target: DataRef,
        /// How much the sample counts.
        weight: f64,
    },
    /// Prefer `chosen` over `rejected` given `context`.
    Preference {
        /// The shared context.
        context: DataRef,
        /// The better decision.
        chosen: DataRef,
        /// The worse decision.
        rejected: DataRef,
        /// How much better, in reward.
        reward_gap: f64,
    },
    /// A group of rollouts of one task.
    Group {
        /// The task.
        context: DataRef,
        /// The rollouts and their relative advantage.
        rollouts: Vec<Rollout>,
    },
    /// One step of an environment.
    Transition {
        /// What came before.
        context: DataRef,
        /// What was done.
        action: DataRef,
        /// The world before.
        state: ContentId,
        /// The world after.
        next_state: ContentId,
        /// The policy's log-probability of the action, if recorded.
        old_logprob: Option<f64>,
        /// The policy's value estimate, if recorded.
        value_estimate: Option<f64>,
        /// The reward, if the environment or the outcome gave one.
        reward: Option<f64>,
        /// Whether the episode ended here.
        done: bool,
    },
    /// A score for one step.
    StepLabel {
        /// What came before.
        context: DataRef,
        /// The step scored.
        step: DataRef,
        /// The score.
        score: f64,
        /// How sure the evaluator is.
        confidence: f64,
    },
    /// Predict `target` from `context` and `actions`.
    WorldModel {
        /// The past, a window of each stream.
        context: Vec<DataRef>,
        /// The actions that began in the span predicted.
        actions: Vec<DataRef>,
        /// The future, a window of each stream.
        target: Vec<DataRef>,
    },
    /// `positive` goes with `anchor`; `negatives` do not.
    Contrastive {
        /// A window of one modality.
        anchor: DataRef,
        /// The same moment of the other.
        positive: DataRef,
        /// Other moments, near and far.
        negatives: Vec<Negative>,
    },
    /// Fill in `masked` from `visible`.
    Masked {
        /// What is shown.
        visible: Vec<DataRef>,
        /// What is hidden.
        masked: DataRef,
    },
    /// Produce `actions` from what was seen and said.
    ActionChunk {
        /// The recent past of each observation stream.
        observations: Vec<DataRef>,
        /// What the robot was told, if it was told.
        instruction: Option<DataRef>,
        /// The actions that followed.
        actions: Vec<DataRef>,
    },
}

/// One sample and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// The record the sample is about, which says where to read it.
    pub provenance: RecordId,
    /// The family it belongs to.
    pub family: Option<ContentId>,
    /// What it teaches.
    pub body: SampleBody,
}

/// The compiled recipe: an immutable list of samples over one snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingPlan {
    /// The snapshot the recipe ran against.
    pub snapshot: ContentId,
    /// The recipe.
    pub recipe: Recipe,
    /// The samples, in canonical order.
    pub samples: Vec<Sample>,
}

impl TrainingPlan {
    /// The plan's identity: the recipe, the snapshot and every sample. Two
    /// compilations of one recipe against one snapshot have the same id.
    pub fn id(&self) -> Result<ContentId> {
        let bytes = serde_json::to_vec(self).map_err(|source| Error::Encode {
            what: "training plan",
            source,
        })?;
        Ok(ContentId::of(&bytes))
    }

    /// The records the samples are about, once each, in order.
    pub fn sources(&self) -> Vec<RecordId> {
        let mut sources: Vec<RecordId> = self.samples.iter().map(|s| s.provenance).collect();
        sources.sort_unstable();
        sources.dedup();
        sources
    }

    /// The dataset record that registers this plan in the graph.
    pub fn dataset_node(&self) -> Result<DatasetNode> {
        let recipe = serde_json::to_value(&self.recipe).map_err(|source| Error::Encode {
            what: "recipe",
            source,
        })?;
        Ok(DatasetNode {
            recipe,
            snapshot: self.snapshot,
            samples: self.samples.len() as u64,
        })
    }

    fn apply_limits(mut samples: Vec<Sample>, recipe: &Recipe) -> Vec<Sample> {
        samples.sort_by_key(|s| s.provenance);
        if let Some(cap) = recipe.max_per_family {
            let mut counts = std::collections::HashMap::new();
            samples.retain(|s| {
                let n = counts.entry(s.family).or_insert(0usize);
                *n += 1;
                *n <= cap
            });
        }
        if let Some(max) = recipe.max_samples.filter(|m| samples.len() > *m) {
            Rng::new(recipe.seed).shuffle(&mut samples);
            samples.truncate(max);
            samples.sort_by_key(|s| s.provenance);
        }
        samples
    }
}

impl Snapshot {
    /// Compiles a recipe against this snapshot.
    pub fn compile(&self, recipe: &Recipe) -> Result<TrainingPlan> {
        // The plan names this snapshot, so the snapshot must be reopenable.
        self.persist()?;
        let samples = match recipe.objective {
            Objective::Sft => self.compile_sft(recipe)?,
            Objective::Dpo => self.compile_dpo(recipe)?,
            Objective::Grpo => self.compile_grpo(recipe)?,
            Objective::Ppo => self.compile_ppo(recipe)?,
            Objective::Prm => self.compile_prm(recipe)?,
            Objective::WorldModel => self.compile_world_model(recipe)?,
            Objective::Contrastive => self.compile_contrastive(recipe)?,
            Objective::Masked => self.compile_masked(recipe)?,
            Objective::ActionChunk => self.compile_action_chunk(recipe)?,
        };
        Ok(TrainingPlan {
            snapshot: self.id(),
            recipe: recipe.clone(),
            samples: TrainingPlan::apply_limits(samples, recipe),
        })
    }
}
