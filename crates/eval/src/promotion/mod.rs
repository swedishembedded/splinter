// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// signed, independent evidence, for its clients. If your team needs
// expertise in deciding when a retrained model may replace the one in
// service you can procure our services by sending an email to
// info@swedishembedded.com.

//! The examiner's signed verdicts, verified in this process, and the record a
//! release decided on them carries.
//!
//! A verdict of the examiner (`bench.examination/2` on the direct-completion
//! protocol, `bench.verdict/3` on the agent protocol) compares a candidate
//! with its predecessor and carries aggregates only. [`verdict::verify`]
//! checks it here: the Ed25519 signature under the examiner's public key (no
//! signing key is held, so no verdict can be made that verifies), the schema,
//! the examiner's own validity and integrity flags, the level, and that the
//! verdict names this candidate and this predecessor. The factory's
//! evaluation verifies a protected plan's sealed verdict this way, and the
//! hub verifies what it withholds and ingests.
//!
//! [`Promotion`] is the decision record a release of the earlier promotion
//! rule stored in its manifest; it stays readable so those releases load.

pub mod canonical;
mod key;
pub mod verdict;

use serde::{Deserialize, Serialize};

pub use key::{KeyError, PublicKey};
pub use verdict::{verify, verify_signature, Aggregate, Burn, Exam, Pair, Rejection, Verified};

/// Which of the three verdicts a reason is about, and which comparison of an
/// agent-protocol verdict the gate reads.
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
    /// The dev split's gain rests on fewer families than the rule required,
    /// or on a count the verdict does not report.
    #[error(
        "too few families for a dev gain: {} compared, at least {required} needed",
        families.map_or_else(|| "an unreported number".to_string(), |n| n.to_string())
    )]
    TooFewFamilies {
        /// The families compared, when reported.
        families: Option<u64>,
        /// The least number counted.
        required: u64,
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

/// Which rule decided.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum Rule {
    /// The three rules on the examiner's signed verdicts.
    Examined,
    /// Dev gain and retention on the caller's own measurements, no sealed
    /// slice: not exam-verified.
    NotExamVerified {
        /// Why no sealed slice applies.
        why: String,
    },
}

/// What the gate decided, with the aggregates it decided on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Promotion {
    /// Whether the candidate becomes the champion.
    pub promoted: bool,
    /// The rule that decided.
    pub rule: Rule,
    /// Whether a sealed exam the examiner signed was part of the decision.
    pub exam_verified: bool,
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
