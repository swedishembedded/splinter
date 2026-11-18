// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// signed, independent evidence, for its clients. If your team needs
// expertise in deciding when a retrained model may replace the one in
// service you can procure our services by sending an email to
// info@swedishembedded.com.

//! The promotion gate: whether a candidate replaces its predecessor as the
//! champion, decided on the examiner's signed aggregates and nothing else.
//!
//! The gate reads three verdicts of the examiner (`bench.examination/1`),
//! each comparing the candidate with its predecessor, each verified in this
//! process ([`verdict::verify`]): the signature under the examiner's key, the
//! schema, the examiner's own validity and integrity flags, and that the
//! verdict names this candidate and this predecessor. It never reads an item:
//! a verdict carries aggregates only. A candidate is promoted only when all
//! three rules hold:
//!
//! 1. **Dev gain.** On the development split, the two-sided 95%
//!    family-clustered interval of the paired gain lies above zero.
//! 2. **Retention.** On the retention suite, the lower bound of the one-sided
//!    95% interval of the paired change is above `-RETENTION_MARGIN`
//!    (-5 points): non-inferiority.
//! 3. **Sealed slice.** The verdict over a sealed exam slice drawn for this
//!    decision verifies, and its burn record names this candidate and this
//!    predecessor: the slice was fresh, spent by this decision alone. Its
//!    gain is recorded, never decided on: the sealed exam measures the
//!    decision, it does not make it.
//!
//! Every rule that fails adds a typed [`Reason`]; a candidate with any reason
//! is rejected and the champion stays. The examiner's intervals are at its
//! one-sided level `alpha`; the gate requires the verdicts it reads to be at
//! [`ALPHA`] (two-sided 95% intervals, one-sided 95% bounds) and refuses any
//! other level rather than reinterpret it.

pub mod canonical;
mod mac;
pub mod verdict;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use verdict::{verify, Aggregate, Pair, Rejection, VerdictKey, Verified};

/// The examiner's one-sided level the gate decides at: its `interval` is
/// then the two-sided 95% interval and its `one_sided_lower` the one-sided
/// 95% bound.
pub const ALPHA: f64 = 0.05;

/// The loss on the retention suite, as a share of items, a candidate may be
/// shown not to exceed: the non-inferiority margin.
pub const RETENTION_MARGIN: f64 = 0.05;

/// Which of the three verdicts a reason is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The development split.
    Dev,
    /// The retention suite.
    Retention,
    /// The sealed slice.
    Sealed,
}

/// Why a candidate was not promoted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Reason {
    /// A verdict was not accepted as evidence.
    #[error("{role:?} verdict: {rejection}")]
    Verdict {
        /// Which verdict.
        role: Role,
        /// Why.
        rejection: Rejection,
    },
    /// The dev split shows no gain: its interval reaches zero or below.
    #[error("no dev gain shown: gain {gain:?}, 95% interval [{low:.4}, {high:.4}] reaches 0")]
    NoDevGain {
        /// The paired gain.
        gain: Option<f64>,
        /// The interval's lower end.
        low: f64,
        /// Its upper end.
        high: f64,
    },
    /// The dev split's gain has no interval (fewer than two families).
    #[error("the dev gain has no interval: fewer than two families were compared")]
    DevUnmeasured,
    /// Retention is not shown non-inferior.
    #[error("retention not shown: the one-sided 95% bound {lower:.4} is not above -{margin}")]
    Inferior {
        /// The one-sided bound.
        lower: f64,
        /// The margin.
        margin: f64,
    },
    /// Retention has no bound (fewer than two families).
    #[error("retention has no bound: fewer than two families were compared")]
    RetentionUnmeasured,
}

/// The examiner's verdicts on one candidate, as received: unverified.
#[derive(Clone, Copy, Debug)]
pub struct Submitted<'a> {
    /// The verdict over the development split.
    pub dev: &'a Value,
    /// The verdict over the retention suite.
    pub retention: &'a Value,
    /// The verdict over the sealed slice drawn for this decision.
    pub sealed: &'a Value,
}

/// What the gate decided, with the aggregates it decided on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Promotion {
    /// Whether the candidate becomes the champion.
    pub promoted: bool,
    /// Why not; empty exactly when promoted.
    pub reasons: Vec<Reason>,
    /// The candidate and predecessor decided on.
    pub pair: Pair,
    /// The dev verdict's aggregates, when it verified.
    pub dev: Option<Aggregate>,
    /// The retention verdict's aggregates, when it verified.
    pub retention: Option<Aggregate>,
    /// The sealed slice verdict's aggregates, when it verified.
    pub sealed: Option<Aggregate>,
}

/// Decides whether `pair.candidate` replaces `pair.predecessor`, on the
/// examiner's `verdicts` verified under `key`.
#[must_use]
pub fn decide(key: &VerdictKey, pair: &Pair, verdicts: Submitted<'_>) -> Promotion {
    let mut reasons = Vec::new();
    let mut checked = |role: Role, document: &Value| match verify(document, key, pair, ALPHA) {
        Ok(verified) => Some(verified),
        Err(rejection) => {
            reasons.push(Reason::Verdict { role, rejection });
            None
        }
    };
    let dev = checked(Role::Dev, verdicts.dev);
    let retention = checked(Role::Retention, verdicts.retention);
    let sealed = checked(Role::Sealed, verdicts.sealed);
    if let Some(dev) = &dev {
        let a = dev.aggregate();
        match a.interval {
            Some((low, _)) if low > 0.0 => {}
            Some((low, high)) => reasons.push(Reason::NoDevGain {
                gain: a.gain,
                low,
                high,
            }),
            None => reasons.push(Reason::DevUnmeasured),
        }
    }
    if let Some(retention) = &retention {
        match retention.aggregate().one_sided_lower {
            Some(lower) if lower > -RETENTION_MARGIN => {}
            Some(lower) => reasons.push(Reason::Inferior {
                lower,
                margin: RETENTION_MARGIN,
            }),
            None => reasons.push(Reason::RetentionUnmeasured),
        }
    }
    if let Some(sealed) = &sealed {
        if let Err(rejection) = sealed.fresh_slice() {
            reasons.push(Reason::Verdict {
                role: Role::Sealed,
                rejection,
            });
        }
    }
    Promotion {
        promoted: reasons.is_empty(),
        reasons,
        pair: pair.clone(),
        dev: dev.map(|v| v.aggregate().clone()),
        retention: retention.map(|v| v.aggregate().clone()),
        sealed: sealed.map(|v| v.aggregate().clone()),
    }
}
