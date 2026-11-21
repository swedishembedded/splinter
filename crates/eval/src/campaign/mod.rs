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
//! * [`learning_signal`] - how much a task's attempts can teach per token
//!   they cost (`Var(r) / mean cost`, the variance shrunk by a Beta
//!   posterior), per task and per family, and the variance-per-cost draw.
//! * [`pass_rate`] - each task's pass rate under the current policy, from an
//!   offline calibration at k = 8..16 and every group since, and the tasks the
//!   campaign drops for good (rate at or above 0.9).
//! * [`sampling`] - dynamic sampling: only groups whose attempts earned
//!   different rewards enter a batch, over-provisioned and refilled; groups
//!   that all failed are routed to a strategy.
//! * [`gate`] - the dev-validation gate: advance on a net dev gain, roll
//!   back on none or on a retention loss, with the level each gate may
//!   spend; or, opted into, advance only on an e-process's evidence.
//! * [`evidence`] - the evidence gate: a noninferiority e-process per round
//!   and a false evolution budget across rounds.
//! * [`dev_pool`] - the dev items a round of the evidence gate draws, and
//!   how exposed each is by earlier rounds.
//! * [`signals`] - the per-step numbers a gate reads: informative-group
//!   fraction, KL, clip fraction and entropy.
//! * [`plan`] - power-first planning: an arm whose exam cannot detect the
//!   effect it targets is refused before it spends a GPU-hour, and repeated
//!   gates spend alpha by default.
//! * [`plateau`] - discovery per GPU-hour, and the switch when it falls.
//! * [`problems`] - the problem state: diagnosed problems with evidence,
//!   candidate interventions with beliefs, best-first choice of the next
//!   single-variable hypothesis and the belief update of each verdict.

pub mod dev_pool;
pub mod evidence;
pub mod gate;
pub mod learning_signal;
pub mod pass_rate;
pub mod plan;
pub mod plateau;
pub mod problems;
pub mod sampling;
pub mod signals;
