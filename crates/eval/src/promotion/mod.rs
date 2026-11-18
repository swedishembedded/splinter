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
//! The gate reads three verdicts of the examiner (`bench.examination/2` on
//! the direct-completion protocol, `bench.verdict/3` on the agent protocol),
//! each comparing the candidate with its predecessor, each verified in this
//! process ([`verdict::verify`]): the Ed25519 signature under the examiner's
//! public key (the gate holds no signing key, so it cannot make a verdict
//! it accepts), the schema, the examiner's own validity and integrity
//! flags, and that the verdict names this candidate and this predecessor. It
//! never reads an item: a verdict carries aggregates only. A candidate is
//! promoted only when all three rules hold:
//!
//! 1. **Dev gain.** On the development split, the two-sided 95%
//!    family-clustered interval of the paired gain lies above zero, over at
//!    least [`MIN_GAIN_FAMILIES`] families (see below).
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
//!
//! **Why a least number of families.** A family-clustered interval says
//! nothing about how many families there are when they all agree: every
//! resample of unanimous families has the same mean (and a cluster
//! standard error of unanimous families is zero), so two families, both
//! won, give the interval [1, 1], above zero. The gain rule therefore also
//! asks for [`MIN_GAIN_FAMILIES`] families: the fewest whose unanimous
//! agreement the exact sign test over families finds significant at the
//! interval's tail `ALPHA / 2` (`2^-6 = 1/64 <= 0.025 < 1/32`). Below it no
//! outcome of the exam could be a shown gain at the stated level, so none
//! counts ([`Reason::TooFewFamilies`]). The minimum, not a sign test beside
//! the interval, is the rule because it is the one check that reads the
//! same on both paths: the examiner's verdicts carry aggregates and a family
//! count, never family signs, and the gate does not recompute or override
//! the examiner's interval. Above the minimum the interval decides.
//!
//! **Where no sealed slice applies.** A decision whose candidate the
//! examiner has no exam for (a release from `learn`, whose exam the learner
//! generated itself) is made by one reviewed, explicit rule,
//! [`decide_unexamined`]: rules 1 and 2 on the caller's own measurements,
//! no sealed slice, and the decision record says so ([`Rule::NotExamVerified`],
//! `exam_verified: false`, with why). It is never a silent bypass: a caller
//! that has examiner verdicts uses [`decide`].

pub mod canonical;
mod key;
pub mod verdict;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use key::{KeyError, PublicKey};
pub use verdict::{verify, verify_signature, Aggregate, Burn, Exam, Pair, Rejection, Verified};

/// The examiner's one-sided level the gate decides at: its `interval` is
/// then the two-sided 95% interval and its `one_sided_lower` the one-sided
/// 95% bound.
pub const ALPHA: f64 = 0.05;

/// The loss on the retention suite, as a share of items, a candidate may be
/// shown not to exceed: the non-inferiority margin.
pub const RETENTION_MARGIN: f64 = 0.05;

/// The fewest families a dev gain is counted over: the least `n` at which
/// `n` unanimous families are significant under the exact one-sided sign
/// test at the dev interval's tail `ALPHA / 2`, that is `2^-n <= ALPHA / 2`.
pub const MIN_GAIN_FAMILIES: u64 = sign_test_minimum(ALPHA / 2.0);

/// The least `n` with `2^-n <= level`.
const fn sign_test_minimum(level: f64) -> u64 {
    let (mut n, mut p) = (1, 0.5);
    while p > level {
        n += 1;
        p /= 2.0;
    }
    n
}

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
    /// The dev split's gain rests on fewer families than
    /// [`MIN_GAIN_FAMILIES`], or on a count the verdict does not report: no
    /// outcome over so few could be a gain shown at the gate's level.
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

/// Which rule decided.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum Rule {
    /// The three rules on the examiner's signed verdicts ([`decide`]).
    Examined,
    /// Dev gain and retention on the caller's own measurements, no sealed
    /// slice ([`decide_unexamined`]): not exam-verified.
    NotExamVerified {
        /// Why no sealed slice applies.
        why: String,
    },
}

/// One paired comparison the caller measured itself, in the examiner's
/// terms at the gate's level [`ALPHA`]: the change in the share of items
/// right, its two-sided family-clustered interval and the lower bound of its
/// one-sided interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    /// Items compared.
    pub items: u64,
    /// The families the items fall in, the units the interval resamples.
    pub families: u64,
    /// The paired change, candidate less predecessor.
    pub gain: Option<f64>,
    /// Its two-sided interval at the gate's level; `None` when it has none.
    pub interval: Option<(f64, f64)>,
    /// The lower bound of its one-sided interval at the gate's level.
    pub one_sided_lower: Option<f64>,
}

/// What the unexamined rule decides on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Measured {
    /// The development comparison.
    pub dev: Measurement,
    /// The retention comparison.
    pub retention: Measurement,
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

/// Rule 1 on a comparison over `families`: the reason it fails, if it does.
/// No interval (fewer than two families) and an interval at or below zero
/// are reported as such; an interval above zero over too few families is
/// refused for its count.
fn dev_reason(
    gain: Option<f64>,
    interval: Option<(f64, f64)>,
    families: Option<u64>,
) -> Option<Reason> {
    match interval {
        Some((low, high)) if low <= 0.0 => Some(Reason::NoDevGain { gain, low, high }),
        Some(_) if families.is_some_and(|n| n >= MIN_GAIN_FAMILIES) => None,
        Some(_) => Some(Reason::TooFewFamilies {
            families,
            required: MIN_GAIN_FAMILIES,
        }),
        None => Some(Reason::DevUnmeasured),
    }
}

/// Rule 2 on a comparison's one-sided bound: the reason it fails, if it does.
fn retention_reason(lower: Option<f64>) -> Option<Reason> {
    match lower {
        Some(lower) if lower > -RETENTION_MARGIN => None,
        Some(lower) => Some(Reason::Inferior {
            lower,
            margin: RETENTION_MARGIN,
        }),
        None => Some(Reason::RetentionUnmeasured),
    }
}

/// Decides, where no sealed slice applies, whether `pair.candidate` replaces
/// `pair.predecessor` on the caller's own `measured` dev gain and retention:
/// rules 1 and 2 as [`decide`] applies them, no sealed slice. The record is
/// labelled not exam-verified, with `why`.
#[must_use]
pub fn decide_unexamined(pair: &Pair, measured: &Measured, why: &str) -> Promotion {
    let reasons: Vec<Reason> = [
        dev_reason(
            measured.dev.gain,
            measured.dev.interval,
            Some(measured.dev.families),
        ),
        retention_reason(measured.retention.one_sided_lower),
    ]
    .into_iter()
    .flatten()
    .collect();
    let aggregate = |m: &Measurement| Aggregate {
        schema: "splinter.measurement".into(),
        collection: String::new(),
        exams: Vec::new(),
        sealed: false,
        items: m.items,
        alpha: ALPHA,
        gain: m.gain,
        interval: m.interval,
        one_sided_lower: m.one_sided_lower,
        families: Some(m.families),
        burns: Vec::new(),
    };
    Promotion {
        promoted: reasons.is_empty(),
        rule: Rule::NotExamVerified { why: why.into() },
        exam_verified: false,
        reasons,
        pair: pair.clone(),
        dev: Some(aggregate(&measured.dev)),
        retention: Some(aggregate(&measured.retention)),
        sealed: None,
    }
}

/// Decides whether `pair.candidate` replaces `pair.predecessor`, on the
/// examiner's `verdicts` verified under its public `key`.
#[must_use]
pub fn decide(key: &PublicKey, pair: &Pair, verdicts: Submitted<'_>) -> Promotion {
    let mut reasons = Vec::new();
    let mut checked = |role: Role, document: &Value| match verify(document, key, pair, ALPHA, role)
    {
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
        let aggregate = dev.aggregate();
        reasons.extend(dev_reason(
            aggregate.gain,
            aggregate.interval,
            aggregate.families,
        ));
    }
    if let Some(retention) = &retention {
        reasons.extend(retention_reason(retention.aggregate().one_sided_lower));
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
        rule: Rule::Examined,
        exam_verified: true,
        reasons,
        pair: pair.clone(),
        dev: dev.map(|v| v.aggregate().clone()),
        retention: retention.map(|v| v.aggregate().clone()),
        sealed: sealed.map(|v| v.aggregate().clone()),
    }
}
