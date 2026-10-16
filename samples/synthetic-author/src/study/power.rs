// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! How many items the examination needs to settle a comparison between two
//! arms, by an exact computation.
//!
//! Only the items on which the arms differ carry evidence in a paired sign
//! test. Of `n` items a share `d` are discordant, so the number of discordant
//! pairs is binomial; of those, the candidate wins a share `w`, so its wins are
//! binomial in turn; the test is significant when the wins reach the critical
//! count for that many pairs. The power is the chance of that, summed over the
//! numbers of discordant pairs there may be. The examination draws the same
//! number of items for each of its four kinds of difficulty, so the size is
//! found in steps of four.

use serde::{Deserialize, Serialize};

use super::stats::{binomial_pmf, critical_wins};

/// The kinds of difficulty the examination balances its items over.
pub const CELLS: usize = 4;

/// How many consecutive sizes must reach the target before a size is taken:
/// with discrete counts the power is not monotone in the number of items, and
/// a size that reaches the target by luck of the critical count is not
/// reliable.
const STABLE_SIZES: usize = 3;

/// The largest `per_cell` the search tries before it gives up.
const MAX_PER_CELL: usize = 5_000;

/// Terms of the sum over discordant pairs smaller than this are left out.
const NEGLIGIBLE: f64 = 1e-14;

/// What a comparison came to in an earlier examination, from which the
/// discordance and the win share of the next are taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observed {
    /// Items both arms were asked.
    pub items: usize,
    /// Items on which exactly one of them was right.
    pub discordant: usize,
    /// Of those, the items the candidate got right.
    pub candidate_wins: usize,
}

/// The power of the one-sided exact sign test over `items` paired items on
/// which a share `discordance` are discordant and the candidate wins a share
/// `win_share` of those, at level `alpha`.
#[must_use]
pub fn power(items: usize, discordance: f64, win_share: f64, alpha: f64) -> f64 {
    let pairs = binomial_pmf(items, discordance);
    let mut total = 0.0;
    for (d, &p_d) in pairs.iter().enumerate() {
        if p_d < NEGLIGIBLE {
            continue;
        }
        let Some(critical) = critical_wins(d, alpha) else {
            continue;
        };
        let wins = binomial_pmf(d, win_share);
        total += p_d * wins[critical..].iter().sum::<f64>();
    }
    total.min(1.0)
}

/// How the examination was sized, with everything the size follows from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Derivation {
    /// What the discordance and the win share were taken from.
    pub observed: Observed,
    /// Where the observation comes from, in words.
    pub source: String,
    /// The share of items on which the arms differ.
    pub discordance: f64,
    /// The share of those that go to the candidate.
    pub win_share: f64,
    /// The level of the one-sided test.
    pub alpha: f64,
    /// The power asked for.
    pub target_power: f64,
    /// The items of each kind of difficulty.
    pub per_cell: usize,
    /// The items in all.
    pub items: usize,
    /// The power at that size.
    pub achieved_power: f64,
    /// The discordant pairs expected at that size.
    pub expected_discordant: f64,
}

/// The smallest `per_cell` at which a one-sided sign test at `alpha` has at
/// least `target_power` against the discordance and win share `observed`
/// shows, and at the next sizes too.
///
/// # Errors
/// The observation has no discordant pair or no candidate win, so there is no
/// difference to find, or the target is not reached at any size tried.
pub fn derive(
    observed: Observed,
    source: &str,
    alpha: f64,
    target_power: f64,
) -> anyhow::Result<Derivation> {
    anyhow::ensure!(
        observed.items > 0
            && observed.discordant > 0
            && observed.discordant <= observed.items
            && observed.candidate_wins <= observed.discordant,
        "the observation {observed:?} is not a count of discordant pairs"
    );
    let discordance = observed.discordant as f64 / observed.items as f64;
    let win_share = observed.candidate_wins as f64 / observed.discordant as f64;
    anyhow::ensure!(
        win_share > 0.5,
        "the candidate wins {} of {} discordant pairs: that is no difference in its favour to size an examination for",
        observed.candidate_wins,
        observed.discordant
    );
    anyhow::ensure!(
        alpha > 0.0 && alpha < 1.0 && target_power > 0.0 && target_power < 1.0,
        "alpha {alpha} and power {target_power} must lie strictly between 0 and 1"
    );
    let reaches = |per_cell: usize| power(per_cell * CELLS, discordance, win_share, alpha);
    let per_cell = (1..=MAX_PER_CELL)
        .find(|&p| (p..p + STABLE_SIZES).all(|q| reaches(q) >= target_power))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "power {target_power} is not reached with {MAX_PER_CELL} items per kind"
            )
        })?;
    let items = per_cell * CELLS;
    Ok(Derivation {
        observed,
        source: source.to_string(),
        discordance,
        win_share,
        alpha,
        target_power,
        per_cell,
        items,
        achieved_power: reaches(per_cell),
        expected_discordant: discordance * items as f64,
    })
}
