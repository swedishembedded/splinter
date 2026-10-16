// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in telling whether the hidden tests of a benchmark task would
// notice a wrong patch, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How strong a workspace task's hidden tests are: of the other faults
//! seeded near the one the task is about, how many do the hidden tests
//! catch?
//!
//! A hidden suite that fails on the seeded fault and on nothing else lets a
//! plausible wrong patch through as a pass. The measurement is recorded on
//! the task's check, so that admission can refuse a weak suite and a release
//! gate can weigh a pass by the strength of the tests behind it.

use serde::{Deserialize, Serialize};

/// What the hidden tests did on the variants of a fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenStrength {
    /// Variants of the fault that some test of the whole suite catches: the
    /// ones that are demonstrably wrong. A variant no test catches may be
    /// equivalent to the original and is not counted.
    pub variants: usize,
    /// Of those, the ones the hidden tests alone catch.
    pub caught: usize,
}

impl HiddenStrength {
    /// The share of variants the hidden tests catch; `None` when no variant
    /// was demonstrably wrong, which is no measurement and never `0`.
    #[must_use]
    pub fn kill_rate(&self) -> Option<f64> {
        (self.variants > 0).then(|| self.caught as f64 / self.variants as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_demonstrably_wrong_variant_is_unmeasured_not_zero() {
        assert_eq!(
            HiddenStrength {
                variants: 0,
                caught: 0
            }
            .kill_rate(),
            None
        );
        assert_eq!(
            HiddenStrength {
                variants: 4,
                caught: 3
            }
            .kill_rate(),
            Some(0.75)
        );
        assert_eq!(
            HiddenStrength {
                variants: 4,
                caught: 0
            }
            .kill_rate(),
            Some(0.0)
        );
    }
}
