// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents that stop at
// their spending limits instead of after them. If your team needs expertise
// in agent cost control, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Per-attempt spending limits and the rule that decides one has been spent.
//!
//! Wall-clock time and tool rounds are raced by the runner directly; this
//! module owns the limits that are read off the usage the kernel reports:
//! generated tokens and billed cost. Unmeasured is never free: a cost cap on
//! a provider whose usage reports carry no price is exhausted on the first
//! such report, because an attempt that cannot see its spend cannot honour
//! a cap on it.

use crate::outcome::Usage;

/// The configured usage limits of one attempt. `None` is "no limit".
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Budget {
    /// Model output tokens, summed over the attempt's usage reports.
    pub max_output_tokens: Option<u64>,
    /// Provider-billed USD. Only meaningful for a remote model; local
    /// inference is not billed per token.
    pub max_cost_usd: Option<f64>,
}

/// Why `usage` has spent `budget`, or `None` while it has not. `reports` is
/// how many usage reports the provider has emitted so far, which is what
/// separates "no price reported yet" from "nothing spent yet".
#[must_use]
pub fn exhausted(budget: &Budget, usage: &Usage, reports: u64) -> Option<String> {
    if let Some(cap) = budget.max_output_tokens {
        if usage.output_tokens >= cap {
            return Some(format!(
                "output-token budget spent: {} of {cap}",
                usage.output_tokens
            ));
        }
    }
    if let Some(cap) = budget.max_cost_usd {
        match usage.cost_usd {
            Some(cost) if cost >= cap => {
                return Some(format!("cost budget spent: ${cost:.4} of ${cap:.4}"));
            }
            None if reports > 0 => {
                return Some(format!(
                    "cost budget of ${cap:.4} cannot be enforced: the provider reported usage \
                     without a price"
                ));
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(output_tokens: u64, cost_usd: Option<f64>) -> Usage {
        Usage {
            output_tokens,
            cost_usd,
            ..Usage::default()
        }
    }

    #[test]
    fn an_unlimited_budget_is_never_spent() {
        assert_eq!(
            exhausted(&Budget::default(), &usage(1 << 40, Some(1e9)), 7),
            None
        );
    }

    #[test]
    fn the_output_token_cap_is_spent_on_reaching_it() {
        let budget = Budget {
            max_output_tokens: Some(1000),
            ..Budget::default()
        };
        assert_eq!(exhausted(&budget, &usage(999, None), 3), None);
        assert!(exhausted(&budget, &usage(1000, None), 4)
            .unwrap()
            .contains("output-token"));
    }

    #[test]
    fn the_cost_cap_is_spent_on_reaching_it() {
        let budget = Budget {
            max_cost_usd: Some(0.5),
            ..Budget::default()
        };
        assert_eq!(exhausted(&budget, &usage(10, Some(0.49)), 2), None);
        assert!(exhausted(&budget, &usage(10, Some(0.5)), 3)
            .unwrap()
            .contains("cost budget spent"));
    }

    /// Unmeasured is not free: once the provider has reported usage without
    /// a price, a cost cap cannot be honoured and the attempt must stop.
    #[test]
    fn a_cost_cap_over_unpriced_usage_is_spent_not_ignored() {
        let budget = Budget {
            max_cost_usd: Some(0.5),
            ..Budget::default()
        };
        assert_eq!(
            exhausted(&budget, &usage(0, None), 0),
            None,
            "nothing reported yet is not a violation"
        );
        assert!(exhausted(&budget, &usage(10, None), 1)
            .unwrap()
            .contains("cannot be enforced"));
    }
}
