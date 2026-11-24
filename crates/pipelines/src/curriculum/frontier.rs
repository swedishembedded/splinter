// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The parameters of a pass@k measurement ([`PassAtK`]: k attempts per
//! task and how the policy samples them) and the tally of a set's graded
//! attempts per task (`tally`), which the teach stage reads to find the
//! tasks a student never solved ([`super::teacher`]).

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_core::experience::ExperienceId;
use splinter_eval::frontier::PassCount;
pub use splinter_model::Sampling;
use splinter_model::AGENT_SAMPLING;
use splinter_store::experiences::SetId;

use crate::solving::SamplingChoice;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// Attempts per task when a command names none: enough for a task solved
/// about half the time to show both a pass and a fail most of the time,
/// few enough that measuring costs a handful of solves per task.
pub const DEFAULT_K: usize = 4;

/// How the policy samples its pass@k attempts when a command names
/// nothing: the agent sampling with a higher temperature, so the k attempts
/// can differ where the policy is unsure (at the agent's own low
/// temperature they are nearly one attempt repeated).
pub const DEFAULT_SAMPLING: Sampling = Sampling {
    temperature: 0.8,
    ..AGENT_SAMPLING
};

/// The parameters of a pass@k measurement.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PassAtK {
    /// Attempts per task; at least two, or no task could be on the
    /// frontier.
    pub k: usize,
    /// How the policy samples. `None`: [`DEFAULT_SAMPLING`] where its
    /// sampling can be set (a local model loaded here), as its provider
    /// does elsewhere. `Some`: exactly this, refused where it cannot be
    /// set.
    pub sampling: Option<Sampling>,
}

impl Default for PassAtK {
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            sampling: None,
        }
    }
}

impl PassAtK {
    /// k attempts per task. With a `temperature` or a `top_k` given, the
    /// policy samples exactly so (the other from [`DEFAULT_SAMPLING`]),
    /// refused where its sampling cannot be set; with neither, as
    /// [`Self::sampling`] `None` says.
    #[must_use]
    pub fn new(k: usize, temperature: Option<f32>, top_k: Option<u32>) -> Self {
        let sampling = (temperature.is_some() || top_k.is_some()).then(|| Sampling {
            temperature: temperature.unwrap_or(DEFAULT_SAMPLING.temperature),
            top_k: top_k.unwrap_or(DEFAULT_SAMPLING.top_k),
            ..DEFAULT_SAMPLING
        });
        Self { k, sampling }
    }

    /// Refuses a k below two, and a negative or non-finite temperature.
    pub fn validate(&self) -> Result<(), OrchestratorError> {
        if self.k < 2 {
            return Err(OrchestratorError::Refused(format!(
                "pass@k needs k of at least 2 to find a task sometimes solved; {} given",
                self.k
            )));
        }
        if let Some(sampling) = &self.sampling {
            if !(sampling.temperature.is_finite() && sampling.temperature >= 0.0) {
                return Err(OrchestratorError::Refused(format!(
                    "a temperature is finite and at least 0; {} given",
                    sampling.temperature
                )));
            }
        }
        Ok(())
    }

    /// How the solve stage samples.
    #[must_use]
    pub fn sampling_choice(&self) -> SamplingChoice {
        match self.sampling {
            Some(sampling) => SamplingChoice::Require(sampling),
            None => SamplingChoice::Prefer(DEFAULT_SAMPLING),
        }
    }
}

/// Each task's graded solves in `set`: how many, how many graded, how many
/// passed, and the experiences.
pub(crate) fn tally(
    ctx: &Context,
    set: &SetId,
) -> Result<BTreeMap<Digest, (PassCount, Vec<ExperienceId>)>, OrchestratorError> {
    let store = ctx.experiences();
    let mut tallied: BTreeMap<Digest, (PassCount, Vec<ExperienceId>)> = BTreeMap::new();
    let members = store.get_set(set)?.members;
    let decisions = store.decisions(&members)?;
    for id in members {
        let experience = store.get(&id)?;
        let outcome = decisions.get(&id).map(|d| d.passed);
        let (count, ids) = tallied.entry(experience.task.id).or_default();
        count.record(outcome);
        ids.push(id);
    }
    Ok(tallied)
}
