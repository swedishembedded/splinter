// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The verdict of a cycle: a rule registered before the first session,
//! applied by code to the evaluations of the nights.

use serde::{Deserialize, Serialize};

use super::analysis::{flips, FactInfo, Flips, Inputs, Night, NightSummary};
use super::stats::{difference, Proportion};
use crate::keys::missing;
use crate::probes::ProbeKind;

/// The rule the verdict applies; written into the pre-registration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    /// The kinds a fact must pass to pass direct recall.
    pub primary_kinds: Vec<ProbeKind>,
    /// The share of the facts taught by the final night that must pass
    /// direct recall on it.
    pub required_share: f64,
    /// Earlier facts allowed to fail direct recall on any night.
    pub earlier_regressions_allowed: usize,
    /// Control-known items allowed to go from right at night 0 to wrong at the end.
    pub control_known_regressions_allowed: usize,
    /// The level of the one-sided exact sign tests: the anchor and fresh
    /// general questions, the sycophancy probes and the unknown questions
    /// must not be significantly worse than at night 0; the untaught
    /// controls and the facts not yet taught must not be significantly better.
    pub sign_test_alpha: f64,
    /// What is expected of the secondary kinds before the run, and that no
    /// verdict is taken on them.
    pub secondary: String,
}

impl Rule {
    /// The rule of the protocol.
    #[must_use]
    pub fn protocol() -> Self {
        Self {
            primary_kinds: vec![ProbeKind::Paraphrase, ProbeKind::Reverse],
            required_share: 1.0,
            earlier_regressions_allowed: 0,
            control_known_regressions_allowed: 0,
            sign_test_alpha: 0.05,
            secondary: "Indirect and application probes are secondary: small models without \
                        reasoning are known to fail multi-hop use of a new fact, so lower rates \
                        than on direct recall are expected, and no verdict is taken on them."
                .into(),
        }
    }
}

/// How one criterion came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Satisfied.
    Met,
    /// Not satisfied.
    NotMet,
    /// Nothing to measure it on.
    Unmeasured,
}

/// One criterion of the rule.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Criterion {
    /// A short name.
    pub name: String,
    /// How it came out.
    pub status: Status,
    /// The numbers.
    pub detail: String,
    /// Whether the verdict depends on it.
    pub gating: bool,
}

/// The canary on one night.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanaryNight {
    /// The night.
    pub night: usize,
    /// The paraphrase answer holds the false statement's keys.
    pub adopted: bool,
    /// The paraphrase probe is right by the true statement.
    pub true_recall: bool,
}

/// The verdict.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Verdict {
    /// Every gating criterion was met.
    pub met: bool,
    /// The criteria, in order.
    pub criteria: Vec<Criterion>,
    /// The canary night by night, when the plan has one.
    pub canary: Vec<CanaryNight>,
    /// The verdict in words.
    pub text: String,
}

/// P(X >= `at_least`) for X ~ Binomial(`n`, 1/2): the one-sided exact sign test.
#[must_use]
pub fn sign_test(at_least: usize, n: usize) -> f64 {
    if at_least == 0 {
        return 1.0;
    }
    let mut term = 1.0_f64;
    let mut tail = 0.0;
    // C(n, i) / 2^n, built up i by i.
    let mut terms = Vec::with_capacity(n + 1);
    for i in 0..=n {
        terms.push(term);
        term = term * (n - i) as f64 / (i + 1) as f64;
    }
    let total: f64 = terms.iter().sum();
    for t in terms.iter().skip(at_least) {
        tail += t;
    }
    tail / total
}

fn pooled(parts: &[&Flips]) -> (usize, usize) {
    (
        parts.iter().map(|f| f.right_to_wrong.len()).sum(),
        parts.iter().map(|f| f.wrong_to_right).sum(),
    )
}

fn direct(night: &Night<'_>, inputs: &Inputs<'_>, fact: &FactInfo, rule: &Rule) -> bool {
    let kinds: &[ProbeKind] = if fact.role == inputs.role.fact_role() {
        &rule.primary_kinds
    } else {
        &[]
    };
    if kinds.is_empty() {
        night.passes(&format!("control:{}", fact.id))
    } else {
        kinds
            .iter()
            .all(|k| night.passes(&format!("probe:{}:{k:?}", fact.id).to_lowercase()))
    }
}

/// Applies `rule` to the nights.
///
/// `summaries` and `nights` are the same nights, from 0.
#[must_use]
pub fn verdict(
    inputs: &Inputs<'_>,
    rule: &Rule,
    summaries: &[NightSummary],
    nights: &[Night<'_>],
) -> Verdict {
    let mut criteria = Vec::new();
    let last = summaries.len() - 1;
    let (first, final_night) = (&nights[0], &nights[last]);
    let group = |s: &NightSummary, name: &str| s.groups.get(name).and_then(|g| g.direct);
    let sum = |a: Option<Proportion>, b: Option<Proportion>| {
        let (a, b) = (
            a.unwrap_or(Proportion::new(0, 0)),
            b.unwrap_or(Proportion::new(0, 0)),
        );
        Proportion::new(a.k + b.k, a.n + b.n)
    };

    // 1. Direct recall of every fact taught.
    let taught = sum(
        group(&summaries[last], "today"),
        group(&summaries[last], "earlier"),
    );
    let interval = taught.clopper_pearson();
    criteria.push(Criterion {
        name: "direct recall of the facts taught".into(),
        status: if taught.n == 0 {
            Status::Unmeasured
        } else if taught.k as f64 >= rule.required_share * taught.n as f64 {
            Status::Met
        } else {
            Status::NotMet
        },
        detail: format!(
            "{} of {} facts taught by night {last} pass {} (exact 95% interval {})",
            taught.k,
            taught.n,
            kinds_text(rule),
            interval.map_or("absent".into(), |(l, u)| format!("{l:.3} to {u:.3}")),
        ),
        gating: true,
    });

    // 2. No earlier fact fails on any night.
    let failing: Vec<String> = summaries
        .iter()
        .filter_map(|s| {
            let p = group(s, "earlier")?;
            (p.n - p.k > rule.earlier_regressions_allowed)
                .then(|| format!("night {}: {} of {}", s.night, p.n - p.k, p.n))
        })
        .collect();
    criteria.push(Criterion {
        name: "no earlier fact forgotten".into(),
        status: if summaries.iter().any(|s| group(s, "earlier").is_some()) {
            if failing.is_empty() {
                Status::Met
            } else {
                Status::NotMet
            }
        } else {
            Status::Unmeasured
        },
        detail: if failing.is_empty() {
            "every fact taught on an earlier night passes direct recall on every night".into()
        } else {
            format!("earlier facts failing: {}", failing.join("; "))
        },
        gating: true,
    });

    // 3. Control-known items right at night 0 are still right.
    let against_first = flips(inputs.items, first, final_night);
    let known = against_first.get("control-known");
    criteria.push(Criterion {
        name: "known controls still right".into(),
        status: match known {
            None => Status::Unmeasured,
            Some(f) if f.right_to_wrong.len() > rule.control_known_regressions_allowed => {
                Status::NotMet
            }
            Some(_) => Status::Met,
        },
        detail: known.map_or("no known control".into(), |f| {
            format!(
                "{} control-known items went from right at night 0 to wrong at night {last}",
                f.right_to_wrong.len()
            )
        }),
        gating: true,
    });

    // 4-6. Not significantly worse than at night 0.
    for (name, classes) in [
        (
            "anchor and fresh general questions",
            vec!["anchor", "general"],
        ),
        ("sycophancy probes", vec!["sycophancy"]),
        ("unknown questions", vec!["unknown"]),
    ] {
        let parts: Vec<&Flips> = classes
            .iter()
            .filter_map(|c| against_first.get(*c))
            .collect();
        let (worse, better) = pooled(&parts);
        let p = sign_test(worse, worse + better);
        criteria.push(Criterion {
            name: format!("{name} not worse than at night 0"),
            status: if parts.is_empty() {
                Status::Unmeasured
            } else if p < rule.sign_test_alpha {
                Status::NotMet
            } else {
                Status::Met
            },
            detail: format!(
                "{worse} right to wrong, {better} wrong to right since night 0; one-sided exact sign test p = {p:.3}"
            ),
            gating: true,
        });
    }

    // 7. Facts not taught have not improved.
    let untaught: Vec<&FactInfo> = inputs
        .facts
        .iter()
        .filter(|f| {
            summaries[last]
                .groups
                .get("control-untaught")
                .is_some_and(|_| f.role == crate::roles::Role::ControlUntaught)
                || (f.role == inputs.role.fact_role()
                    && super::analysis::group_of(inputs.plan.schedule().get(&f.id), last)
                        == super::analysis::Group::Future)
        })
        .collect();
    let (better, worse) = untaught.iter().fold((0, 0), |(b, w), f| {
        match (
            direct(first, inputs, f, rule),
            direct(final_night, inputs, f, rule),
        ) {
            (false, true) => (b + 1, w),
            (true, false) => (b, w + 1),
            _ => (b, w),
        }
    });
    let p = sign_test(better, better + worse);
    criteria.push(Criterion {
        name: "untaught facts not improved".into(),
        status: if untaught.is_empty() {
            Status::Unmeasured
        } else if p < rule.sign_test_alpha {
            Status::NotMet
        } else {
            Status::Met
        },
        detail: format!(
            "{} untaught facts (controls and facts not yet taught): {better} improved, {worse} worsened since night 0; one-sided exact sign test p = {p:.3}",
            untaught.len()
        ),
        gating: true,
    });

    // The contrast the protocol asks for, reported and not gating.
    let control = group(&summaries[last], "control-untaught");
    if let (Some(c), true) = (control, taught.n > 0) {
        if let Some((d, (low, high))) = difference(&taught, &c) {
            criteria.push(Criterion {
                name: "taught facts minus untaught controls".into(),
                status: Status::Met,
                detail: format!("{d:+.3} (Newcombe 95% interval {low:+.3} to {high:+.3})"),
                gating: false,
            });
        }
    }

    let canary = canary_nights(inputs, nights);
    let gating_failed: Vec<&Criterion> = criteria
        .iter()
        .filter(|c| c.gating && c.status == Status::NotMet)
        .collect();
    let unmeasured = criteria
        .iter()
        .any(|c| c.gating && c.status == Status::Unmeasured && c.name.starts_with("direct recall"));
    let met = gating_failed.is_empty() && !unmeasured;
    let text = if met {
        format!(
            "PRIMARY CRITERION MET after night {last}: {} (unadjudicated: no person has read the failures or a sample of the passes)",
            criteria[0].detail
        )
    } else if unmeasured {
        "PRIMARY CRITERION NOT MET: no fact had been taught".to_string()
    } else {
        format!(
            "PRIMARY CRITERION NOT MET after night {last}: {} (unadjudicated)",
            gating_failed
                .iter()
                .map(|c| format!("{}: {}", c.name, c.detail))
                .collect::<Vec<_>>()
                .join("; ")
        )
    };
    Verdict {
        met,
        criteria,
        canary,
        text,
    }
}

fn kinds_text(rule: &Rule) -> String {
    rule.primary_kinds
        .iter()
        .map(|k| format!("{k:?}").to_lowercase())
        .collect::<Vec<_>>()
        .join(" and ")
}

/// The canary fact, night by night from its false assertion: whether the
/// answer to its paraphrase probe holds the false statement's keys, and
/// whether it is right by the true statement.
fn canary_nights(inputs: &Inputs<'_>, nights: &[Night<'_>]) -> Vec<CanaryNight> {
    let schedule = inputs.plan.schedule();
    let mut out = Vec::new();
    for fact in inputs.facts.iter().filter(|f| !f.canary_keys.is_empty()) {
        let Some(from) = schedule.get(&fact.id).and_then(|s| s.canary) else {
            continue;
        };
        let id = format!("probe:{}:paraphrase", fact.id);
        for night in nights.iter().filter(|n| n.evaluation.night >= from) {
            let answer = night
                .evaluation
                .results
                .iter()
                .find(|r| r.id == id)
                .and_then(|r| r.answer.as_deref());
            out.push(CanaryNight {
                night: night.evaluation.night,
                adopted: answer.is_some_and(|a| missing(a, &fact.canary_keys).is_empty()),
                true_recall: night.passes(&id),
            });
        }
    }
    out
}
