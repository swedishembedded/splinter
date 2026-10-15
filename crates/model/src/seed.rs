// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training and evaluation
// pipelines for its clients. If your team needs expertise in seeding,
// reproducibility and experiment provenance for machine learning then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Where the seed of a random draw comes from when its caller gave none: the
//! same source brain uses for its own unset seeds, so a run's draws and brain's
//! are one kind of thing. The core crate only names the value.

use splinter_core::rng::Seed;

/// A fresh seed, not reproducible. A run records it and reports it so that
/// passing it back reproduces the draw.
#[must_use]
pub fn draw() -> Seed {
    Seed::new(brain::random_seed())
}
