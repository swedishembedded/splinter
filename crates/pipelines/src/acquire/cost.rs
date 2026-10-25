// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! What a round spent: the compute side of a round's exploration yield.
//!
//! A round reports, beside what it found (tasks proposed, admitted,
//! attempted, passed, worth training on, trained on), what its two expensive
//! stages cost, so that the useful verified experiences per unit of
//! exploration compute can be computed from the run record alone.

use std::time::Instant;

use serde::Serialize;
use splinter_store::experiences::SetId;

use crate::curriculum::schedule::yield_of;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The time one stage of a round took.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StageCost {
    /// Wall-clock seconds.
    pub wall_seconds: f64,
    /// Device-busy seconds the stage's attempts recorded; absent when none
    /// recorded any (a stage that reports no device time is not a free
    /// one). A lower bound while `unmeasured_attempts` is not zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_seconds: Option<f64>,
    /// Attempts that recorded no device time.
    pub unmeasured_attempts: usize,
}

impl StageCost {
    /// A stage that began at `started` and reports no device time of its
    /// own: training holds the device for its whole duration but brain
    /// reports no busy figure for it.
    #[must_use]
    pub fn wall_only(started: Instant) -> Self {
        Self {
            wall_seconds: started.elapsed().as_secs_f64(),
            gpu_seconds: None,
            unmeasured_attempts: 0,
        }
    }

    /// A stage that began at `started` and made the attempts of `attempts`,
    /// each of which may have recorded its device time.
    pub fn of_attempts(
        ctx: &Context,
        started: Instant,
        attempts: &SetId,
    ) -> Result<Self, OrchestratorError> {
        let wall_seconds = started.elapsed().as_secs_f64();
        let made = ctx.experiences().get_set(attempts)?.members.len();
        let spent = yield_of(ctx, &[attempts], 0)?;
        Ok(Self {
            wall_seconds,
            gpu_seconds: (spent.unmeasured < made).then_some(spent.gpu_seconds),
            unmeasured_attempts: spent.unmeasured,
        })
    }
}
