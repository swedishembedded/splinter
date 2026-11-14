// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for self-improvement loops that
// keep only the changes the evidence supports, for its clients. If your team
// needs expertise in anytime-valid testing and false-discovery control for
// model and agent gates you can procure our services by sending an email to
// info@swedishembedded.com.

//! The evidence gate: a per-round noninferiority test whose false acceptances
//! are controlled across rounds, instead of accepting any net dev gain.
//!
//! The construction is TRUST (Gao, Qi and Cui, "Trustworthy recursive
//! self-improvement with statistical guarantees", arXiv 2611.10787v1,
//! sections 3-4 and appendices B, C and G). Round `t` tests the candidate
//! against the incumbent:
//!
//! * **Hypotheses.** `H0: delta <= -eps` against `H1: delta > -eps`, where
//!   `delta` is the candidate's true score minus the incumbent's and `eps` the
//!   tolerated loss (equation 5). Rejecting `H0` accepts the candidate.
//! * **Observations.** A paired outcome `D` in `[-1, 1]` per dev item
//!   (candidate pass minus incumbent pass), rescaled to `Y = (1 + D) / 2`;
//!   the null boundary of `Y` is `p_eps = (1 - eps) / 2`.
//! * **Overlap correction.** Draw `i` has exposure probability `zeta_i` (the
//!   share of the items it could be that earlier rounds already used,
//!   [`super::dev_pool`]), and the corrected boundary is
//!   `u_i = p_eps + beta * zeta_i`: `beta` is the optimism allowed on a reused
//!   item (Assumption 1), so reused evidence must be stronger.
//! * **E-process.** `M_m = prod_{i<=m} [(1 - eta) + eta * Y_i / u_i]`
//!   (equation 10), as the equal-weight mixture over the fixed betting
//!   fractions [`BETTING_FRACTIONS`] (appendix G.1); a convex mixture of
//!   nonnegative supermartingales is one (Corollary 1), so the mixture is a
//!   valid e-value at any stopping time.
//! * **Decision.** Accept at the first `m` with `M_m >= 1 / alpha_t`, else
//!   keep the incumbent. Because the test is anytime-valid the caller may
//!   grade in batches and stop as soon as it crosses.
//! * **Level allocation.** `B_1 = q`; `alpha_t = min(alpha_max, rho_t * B_t *
//!   (R_{t-1} + 1))`, charge `c_t = alpha_t / (R_{t-1} + 1)`, `B_{t+1} = B_t -
//!   c_t` (equations 6-7), with `rho_t = clip(1 / (T - t + 1), rho_min,
//!   rho_max)` (equation 8 with the model multiplier `w_t = 1`, the paper's
//!   "TRUST without LLM" arm). Under Assumption 1 the false evolution rate,
//!   `E[accepted rounds with delta <= -eps / max(accepted, 1)]`, is at most
//!   `q` (Theorem 2), for any stopping of the loop too.
//!
//! The paper leaves `alpha_max`, `rho_min` and `rho_max` unstated; here they
//! are parameters with conservative defaults ([`EvidenceGate::new`]): no
//! round tests above the whole budget (`alpha_max = q`), and no round spends
//! more than half of what is left ([`DEFAULT_RHO_MAX`]), so a loop that runs
//! past its planned horizon still has budget. Rounds past the horizon test at
//! level zero (no validation, the incumbent stays), as in the paper's proof
//! of optional stopping.

use serde::{Deserialize, Serialize};

/// The fixed betting fractions the e-process mixes with equal weight
/// (appendix G.1 of the paper).
pub const BETTING_FRACTIONS: [f64; 4] = [0.10, 0.25, 0.50, 0.75];

/// The smallest share of the remaining budget a round spends. The paper
/// requires `0 < rho_min` and states no value; with the default allocation
/// `1 / (T - t + 1)` is never below `1 / T`, so this bound only matters past
/// a horizon of a hundred rounds.
pub const DEFAULT_RHO_MIN: f64 = 0.01;

/// The largest share of the remaining budget a round spends. The paper
/// requires `rho_max < 1` and states no value; half keeps budget for rounds
/// beyond the planned horizon, at the cost of a smaller level in the last
/// planned round.
pub const DEFAULT_RHO_MAX: f64 = 0.5;

/// A setting of the evidence gate that cannot be used.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum EvidenceError {
    /// A parameter is out of its range.
    #[error("evidence gate: {0}")]
    Parameter(String),
    /// An observation is out of its range.
    #[error("evidence gate observation: {0}")]
    Observation(String),
}

/// The evidence gate's parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceGate {
    /// The tolerated loss `eps`, in `[0, 1)` on the paired difference scale
    /// `[-1, 1]`: a candidate whose true pass rate is lower by `eps` or more
    /// is a false evolution. Zero makes ties count against the candidate and
    /// the gate accept almost nothing (the paper's experiment).
    pub eps: f64,
    /// The false evolution rate budget `q`, in `(0, 1)`.
    pub q: f64,
    /// The overlap correction strength `beta`, in `[0, (1 + eps) / 2]`: the
    /// optimism allowed on a reused dev item.
    pub beta: f64,
    /// The highest level any round tests at, in `(0, 1)`.
    pub alpha_max: f64,
    /// The smallest share of the remaining budget a round spends.
    pub rho_min: f64,
    /// The largest share of the remaining budget a round spends.
    pub rho_max: f64,
}

impl EvidenceGate {
    /// A gate with tolerance `eps`, budget `q` and correction `beta`, and the
    /// conservative defaults for what the paper leaves unstated:
    /// `alpha_max = q`, [`DEFAULT_RHO_MIN`], [`DEFAULT_RHO_MAX`].
    ///
    /// # Errors
    /// A parameter is out of range ([`Self::validate`]).
    pub fn new(eps: f64, q: f64, beta: f64) -> Result<Self, EvidenceError> {
        let gate = Self {
            eps,
            q,
            beta,
            alpha_max: q,
            rho_min: DEFAULT_RHO_MIN,
            rho_max: DEFAULT_RHO_MAX,
        };
        gate.validate()?;
        Ok(gate)
    }

    /// Refuses parameters outside the ranges the guarantee is proved for.
    ///
    /// # Errors
    /// The first parameter out of range, named.
    pub fn validate(&self) -> Result<(), EvidenceError> {
        let refuse = |why: String| Err(EvidenceError::Parameter(why));
        if !(self.eps >= 0.0 && self.eps < 1.0) {
            return refuse(format!("eps is {}, not in [0, 1)", self.eps));
        }
        if !(self.q > 0.0 && self.q < 1.0) {
            return refuse(format!("q is {}, not in (0, 1)", self.q));
        }
        let beta_max = 1.0 - self.null_boundary();
        if !(self.beta >= 0.0 && self.beta <= beta_max) {
            return refuse(format!(
                "beta is {}, not in [0, {beta_max}] (1 - (1 - eps) / 2)",
                self.beta
            ));
        }
        if !(self.alpha_max > 0.0 && self.alpha_max < 1.0) {
            return refuse(format!("alpha_max is {}, not in (0, 1)", self.alpha_max));
        }
        if !(self.rho_min > 0.0 && self.rho_min < self.rho_max && self.rho_max < 1.0) {
            return refuse(format!(
                "rho bounds are {} and {}; 0 < rho_min < rho_max < 1 is required",
                self.rho_min, self.rho_max
            ));
        }
        Ok(())
    }

    /// The null boundary of `Y` before correction, `(1 - eps) / 2`.
    #[must_use]
    pub fn null_boundary(&self) -> f64 {
        (1.0 - self.eps) / 2.0
    }

    /// The corrected boundary `p_eps + beta * zeta` of a draw with exposure
    /// probability `zeta`.
    #[must_use]
    pub fn boundary(&self, zeta: f64) -> f64 {
        self.null_boundary() + self.beta * zeta
    }
}

/// The level of one round, fixed before any of its dev items is graded.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Allocation {
    /// The round, from 1.
    pub round: u32,
    /// The rounds the budget was planned over.
    pub horizon: u32,
    /// Accepted rounds before this one.
    pub accepted_before: u32,
    /// The share of the remaining budget this round spends.
    pub rho: f64,
    /// The round's level; zero skips the round.
    pub alpha: f64,
    /// What the round takes from the budget, `alpha / (accepted_before + 1)`.
    pub charge: f64,
    /// The budget before the round.
    pub budget_before: f64,
    /// The budget after it.
    pub budget_after: f64,
}

/// The false evolution budget across rounds: what is left, and how many
/// rounds ran and were accepted. A caller keeps it durably (or rebuilds it
/// from its decision records with [`Budget::settle`]) so a restarted loop
/// spends the same levels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// The budget left, `B_t`.
    pub remaining: f64,
    /// Rounds settled.
    pub rounds: u32,
    /// Rounds accepted, `R_{t-1}`.
    pub accepted: u32,
}

impl Budget {
    /// The budget before the first round: all of `q`.
    #[must_use]
    pub fn start(gate: &EvidenceGate) -> Self {
        Self {
            remaining: gate.q,
            rounds: 0,
            accepted: 0,
        }
    }

    /// The level of the next round of a loop planned over `horizon` rounds.
    /// Past the horizon the level is zero.
    #[must_use]
    pub fn allocate(&self, gate: &EvidenceGate, horizon: u32) -> Allocation {
        let round = self.rounds + 1;
        let scale = f64::from(self.accepted) + 1.0;
        let (rho, alpha) = if round > horizon || self.remaining <= 0.0 {
            (0.0, 0.0)
        } else {
            let even = 1.0 / f64::from(horizon - round + 1);
            let rho = even.clamp(gate.rho_min, gate.rho_max);
            (rho, gate.alpha_max.min(rho * self.remaining * scale))
        };
        let charge = alpha / scale;
        Allocation {
            round,
            horizon,
            accepted_before: self.accepted,
            rho,
            alpha,
            charge,
            budget_before: self.remaining,
            budget_after: (self.remaining - charge).max(0.0),
        }
    }

    /// Records the outcome of the round `allocation` was made for.
    pub fn settle(&mut self, allocation: &Allocation, accepted: bool) {
        self.remaining = allocation.budget_after;
        self.rounds = allocation.round;
        self.accepted += u32::from(accepted);
    }

    /// The budget spent so far out of `gate.q`.
    #[must_use]
    pub fn spent(&self, gate: &EvidenceGate) -> f64 {
        gate.q - self.remaining
    }
}

/// The evidence of one round, accumulated observation by observation.
#[derive(Clone, Debug, PartialEq)]
pub struct EProcess {
    gate: EvidenceGate,
    alpha: f64,
    capital: [f64; BETTING_FRACTIONS.len()],
    observed: usize,
    crossed_at: Option<usize>,
}

impl EProcess {
    /// A round at level `alpha` ([`Allocation::alpha`]); at level zero it
    /// never accepts.
    #[must_use]
    pub fn new(gate: &EvidenceGate, alpha: f64) -> Self {
        Self {
            gate: *gate,
            alpha,
            capital: [1.0; BETTING_FRACTIONS.len()],
            observed: 0,
            crossed_at: None,
        }
    }

    /// The threshold the mixture must reach, `1 / alpha` (infinite at level
    /// zero).
    #[must_use]
    pub fn threshold(&self) -> f64 {
        if self.alpha > 0.0 {
            1.0 / self.alpha
        } else {
            f64::INFINITY
        }
    }

    /// Adds the paired difference `difference` in `[-1, 1]` of a draw with
    /// exposure probability `zeta` in `[0, 1]`; whether the round has crossed
    /// (then later observations are ignored: the round stopped there).
    ///
    /// # Errors
    /// An observation is out of range.
    pub fn observe(&mut self, difference: f64, zeta: f64) -> Result<bool, EvidenceError> {
        if !(-1.0..=1.0).contains(&difference) {
            return Err(EvidenceError::Observation(format!(
                "a paired difference of {difference} is outside [-1, 1]"
            )));
        }
        if !(0.0..=1.0).contains(&zeta) {
            return Err(EvidenceError::Observation(format!(
                "an exposure probability of {zeta} is outside [0, 1]"
            )));
        }
        if self.crossed_at.is_some() {
            return Ok(true);
        }
        let y = (1.0 + difference) / 2.0;
        let u = self.gate.boundary(zeta);
        for (capital, eta) in self.capital.iter_mut().zip(BETTING_FRACTIONS) {
            *capital *= (1.0 - eta) + eta * y / u;
        }
        self.observed += 1;
        if self.value() >= self.threshold() {
            self.crossed_at = Some(self.observed);
        }
        Ok(self.crossed_at.is_some())
    }

    /// The mixture's e-value, at the crossing once crossed.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.capital.iter().sum::<f64>() / self.capital.len() as f64
    }

    /// Observations taken (up to and including the crossing).
    #[must_use]
    pub fn observed(&self) -> usize {
        self.observed
    }

    /// The observation, from 1, at which the round crossed.
    #[must_use]
    pub fn crossed_at(&self) -> Option<usize> {
        self.crossed_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(eps: f64, q: f64, beta: f64) -> EvidenceGate {
        EvidenceGate::new(eps, q, beta).unwrap()
    }

    #[test]
    fn parameters_outside_the_proved_ranges_are_refused_by_name() {
        for (eps, q, beta, name) in [
            (1.0, 0.3, 0.0, "eps"),
            (-0.1, 0.3, 0.0, "eps"),
            (0.1, 0.0, 0.0, "q"),
            (0.1, 1.0, 0.0, "q"),
            (0.1, 0.3, -0.01, "beta"),
            (0.1, 0.3, 0.56, "beta"),
        ] {
            let refused = EvidenceGate::new(eps, q, beta).unwrap_err().to_string();
            assert!(refused.contains(name), "{refused}");
        }
        assert!(
            EvidenceGate::new(0.1, 0.3, 0.55).is_ok(),
            "beta at 1 - p_eps"
        );
        let mut bad = gate(0.1, 0.3, 0.01);
        bad.rho_max = 1.0;
        assert!(bad.validate().unwrap_err().to_string().contains("rho"));
        bad = gate(0.1, 0.3, 0.01);
        bad.alpha_max = 0.0;
        assert!(bad
            .validate()
            .unwrap_err()
            .to_string()
            .contains("alpha_max"));
    }

    #[test]
    fn the_boundary_is_the_noninferiority_null_raised_by_the_overlap_correction() {
        let g = gate(0.1, 0.3, 0.01);
        assert!((g.null_boundary() - 0.45).abs() < 1e-12);
        assert!((g.boundary(0.0) - 0.45).abs() < 1e-12);
        assert!((g.boundary(1.0) - 0.46).abs() < 1e-12);
        assert!((g.boundary(0.5) - 0.455).abs() < 1e-12);
    }

    #[test]
    fn the_e_process_is_the_equal_weight_mixture_of_the_fixed_fraction_products() {
        let g = gate(0.1, 0.3, 0.01);
        let mut process = EProcess::new(&g, 0.0);
        let observations = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.5), (1.0, 1.0)];
        for (d, zeta) in observations {
            assert!(!process.observe(d, zeta).unwrap());
        }
        let expected: f64 = BETTING_FRACTIONS
            .iter()
            .map(|eta| {
                observations
                    .iter()
                    .map(|&(d, zeta)| {
                        let y = (1.0 + d) / 2.0;
                        (1.0 - eta) + eta * y / (0.45 + 0.01 * zeta)
                    })
                    .product::<f64>()
            })
            .sum::<f64>()
            / 4.0;
        assert!((process.value() - expected).abs() < 1e-12);
        assert_eq!(process.observed(), 4);
        assert_eq!(process.crossed_at(), None, "level zero never accepts");
    }

    #[test]
    fn the_round_stops_at_the_first_crossing_and_ignores_later_observations() {
        let g = gate(0.1, 0.3, 0.0);
        let mut process = EProcess::new(&g, 0.2);
        let mut crossed = 0;
        for i in 1..=20 {
            if process.observe(1.0, 0.0).unwrap() {
                crossed = i;
                break;
            }
        }
        assert!(crossed > 0, "twenty straight wins cross 1 / 0.2");
        let at = process.value();
        assert!(at >= 5.0);
        assert!(process.observe(-1.0, 0.0).unwrap());
        assert_eq!(
            process.value(),
            at,
            "the stopped value is the crossing value"
        );
        assert_eq!(process.crossed_at(), Some(crossed));
        assert!(process.observe(2.0, 0.0).is_err());
        assert!(process.observe(0.0, 1.5).is_err());
    }

    #[test]
    fn levels_follow_the_budget_the_acceptances_and_the_horizon() {
        let mut g = gate(0.1, 0.3, 0.01);
        g.alpha_max = 0.9;
        let mut budget = Budget::start(&g);
        // Round 1 of 4: rho = clip(1/4) = 0.25, alpha = 0.25 * 0.3 * 1.
        let first = budget.allocate(&g, 4);
        assert!((first.rho - 0.25).abs() < 1e-12);
        assert!((first.alpha - 0.075).abs() < 1e-12);
        assert!((first.charge - 0.075).abs() < 1e-12);
        budget.settle(&first, true);
        assert!((budget.remaining - 0.225).abs() < 1e-12);
        // Round 2 of 4 after one acceptance: rho = 1/3, alpha = 1/3 * 0.225 * 2,
        // charged at half.
        let second = budget.allocate(&g, 4);
        assert!((second.alpha - 0.15).abs() < 1e-12);
        assert!((second.charge - 0.075).abs() < 1e-12);
        budget.settle(&second, false);
        // Round 4 of 4: rho = clip(1) = rho_max.
        let third = budget.allocate(&g, 4);
        budget.settle(&third, false);
        let last = budget.allocate(&g, 4);
        assert!((last.rho - DEFAULT_RHO_MAX).abs() < 1e-12);
        budget.settle(&last, false);
        assert!(budget.spent(&g) <= g.q + 1e-12);
        let beyond = budget.allocate(&g, 4);
        assert_eq!((beyond.alpha, beyond.charge), (0.0, 0.0));
        assert_eq!(budget.accepted, 1);
    }

    #[test]
    fn the_default_cap_keeps_every_level_within_the_budget() {
        let g = gate(0.1, 0.3, 0.01);
        let mut budget = Budget::start(&g);
        for _ in 0..12 {
            let allocation = budget.allocate(&g, 12);
            assert!(allocation.alpha <= g.q);
            assert!(allocation.charge <= allocation.rho * allocation.budget_before + 1e-15);
            budget.settle(&allocation, true);
        }
        assert!(budget.remaining >= 0.0 && budget.spent(&g) <= g.q);
    }
}
