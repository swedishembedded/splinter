// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for spending a reinforcement-learning
// budget only on prompts that teach, for its clients. If your team needs
// expertise in dynamic sampling, curricula for policy-gradient training or
// the statistics of repeated model gates, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The pure decisions of a continuous learning campaign: which prompts are
//! worth a rollout group, which groups carry a learning signal, whether the
//! training numbers of a step are healthy, whether an arm may be run at all,
//! and when discovery has stalled. No model, no files, no clock: the
//! controller in `splinter-pipelines` feeds these functions measurements and
//! acts on what they say.
//!
//! * [`pass_rate`] - each task's pass rate under the current policy, from an
//!   offline calibration at k = 8..16 and every group since, and the tasks the
//!   campaign drops for good (rate at or above 0.9).
//! * [`sampling`] - dynamic sampling: only groups whose attempts earned
//!   different rewards enter a batch, over-provisioned and refilled; groups
//!   that all failed are routed to a strategy.
//! * [`gate`] - the dev-validation gate: advance on a net dev gain, roll
//!   back on none or on a retention loss, with the level each gate may
//!   spend.
//! * [`signals`] - the per-step numbers a gate reads: informative-group
//!   fraction, KL, clip fraction and entropy.
//! * [`plan`] - power-first planning: an arm whose exam cannot detect the
//!   effect it targets is refused before it spends a GPU-hour, and repeated
//!   gates spend alpha by default.
//! * [`plateau`] - discovery per GPU-hour, and the switch when it falls.

pub mod gate;
pub mod pass_rate;
pub mod plan;
pub mod plateau;
pub mod sampling;
pub mod signals;
