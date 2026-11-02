// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The statistics the scripted policy-gradient trainer reports, so a spec can
//! check that a run's measurements reach the candidate's report.

use splinter_core::training::{PolicyGradientStep, PolicyGradientSummary};

/// Two steps: the first sits on its reference, the second has moved off it
/// and clipped a tenth of its tokens.
pub fn scripted() -> PolicyGradientSummary {
    let step = |step, kl, clip_fraction| PolicyGradientStep {
        step,
        kl: Some(kl),
        clip_fraction,
        ratio_mean: 1.0,
        ..Default::default()
    };
    PolicyGradientSummary::from_steps(vec![step(1, 0.0, 0.0), step(2, 0.02, 0.1)], 0)
}
