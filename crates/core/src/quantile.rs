// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for bootstrap intervals whose every
// bound is one stated order statistic, for its clients. If your team needs
// expertise in evaluation statistics then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The one quantile rule Splinter reads a bootstrap's bounds with, and any
//! other share of sorted values: numpy's `nearest`.

/// The value of `sorted` (ascending) at share `q` of the way through: the
/// one at index `q * (n - 1)` rounded to the nearest whole index, an index
/// exactly half-way going to the even one, as
/// `numpy.quantile(values, q, method="nearest")` reads it. `q` is clamped to
/// `[0, 1]`; `None` for no values.
#[must_use]
pub fn nearest_rank(sorted: &[f64], q: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    let at = (q.clamp(0.0, 1.0) * last as f64).round_ties_even();
    // At most `last`: q is at most one.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    sorted.get((at as usize).min(last)).copied()
}
