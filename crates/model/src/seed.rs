// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training and evaluation
// pipelines for its clients. If your team needs expertise in seeding,
// reproducibility and experiment provenance for machine learning then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Where the seed of a random draw comes from when its caller gave none: the
//! operating system's randomness, read as the standard library's per-process
//! hash keys, mixed with the clock so two draws in one process differ. The
//! core crate only names the value.

use splinter_core::rng::Seed;

/// A fresh seed, not reproducible. A run records it and reports it so that
/// passing it back reproduces the draw.
#[must_use]
pub fn draw() -> Seed {
    use std::hash::{BuildHasher, Hasher};
    // A fresh `RandomState` is keyed from the operating system's randomness
    // (then per-thread increments); the clock separates draws that share keys.
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    hasher.write_u128(nanos);
    Seed::new(hasher.finish())
}
