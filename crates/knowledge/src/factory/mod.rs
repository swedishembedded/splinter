// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that compile a learning objective
// into validated, versioned training environments, for its clients. If your
// team needs expertise in environment factories for agent reinforcement
// learning, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The environment factory: a learning objective compiled into validated,
//! versioned training environments, without a hand-written pipeline.
//!
//! * [`objective`] - the objective (text, sources, constraints, budget) and
//!   the verifiability classification of each skill in it: objective check,
//!   proxy, calibrated judge or human evidence, recorded with its reasons.
//! * [`artifact`] - the environment artifact: a Harbor task folder with the
//!   schema version, verifier class and provenance, difficulty, split (train
//!   or dev, never a sealed exam) and seed written into `task.toml`.
//! * [`plugin`] and [`plugins`] - the generators: a generator, a reset, a
//!   verifier and a difficulty knob per kind of skill.
//! * [`host`] - runs a task folder on the host as a container would.
//! * [`admission`] - the gates an environment passes before a campaign may
//!   train on it, ending in a measured base pass-rate band.
//! * [`attack`] - the adversarial attempts on a verifier; an environment
//!   whose verifier is beaten is quarantined.

//!
//! Nothing here produces a sealed exam: those belong to the measurement
//! authority, and [`Split`](splinter_core::verifiability::Split) has no
//! variant for one.

pub mod admission;
pub mod artifact;
pub mod attack;
pub mod band;
pub mod folder;
pub mod host;
pub mod objective;
pub mod plugin;
pub mod plugins;

#[cfg(test)]
mod testing;
