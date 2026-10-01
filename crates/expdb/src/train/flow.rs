// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Flow-matching targets derived at the moment of training.
//!
//! The database keeps the clean action trajectory only. The noise, the time
//! `t`, the noisy action and the velocity to learn are computed from the clean
//! action and a seed when a sample is materialised, so nothing pre-noised is
//! ever stored and a different training objective needs no new dataset.

use super::rng::Rng;

/// One flow-matching training pair.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowSample {
    /// The interpolation time, strictly between 0 and 1.
    pub t: f32,
    /// The standard-normal noise the path starts from.
    pub noise: Vec<f32>,
    /// `(1 - t) * noise + t * action`.
    pub noisy: Vec<f32>,
    /// The velocity field to learn: `action - noise`.
    pub target: Vec<f32>,
}

/// Derives a flow-matching pair from a clean action, a sample seed and a step
/// number. The same inputs always give the same pair.
pub fn flow_matching(clean: &[f32], seed: u64, step: u64) -> FlowSample {
    let mut rng = Rng::new(seed ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let t = rng.unit() as f32;
    let mut noise = Vec::with_capacity(clean.len());
    while noise.len() < clean.len() {
        // Box-Muller: two uniform numbers make two independent normals.
        let (u1, u2) = (rng.unit(), rng.unit());
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        noise.push((radius * angle.cos()) as f32);
        if noise.len() < clean.len() {
            noise.push((radius * angle.sin()) as f32);
        }
    }
    let noisy = clean
        .iter()
        .zip(&noise)
        .map(|(a, n)| (1.0 - t) * n + t * a)
        .collect();
    let target = clean.iter().zip(&noise).map(|(a, n)| a - n).collect();
    FlowSample {
        t,
        noise,
        noisy,
        target,
    }
}
