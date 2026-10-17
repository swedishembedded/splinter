// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that projects
// verified agent experience into supervised datasets, for its clients. If
// your team needs expertise in dataset curation for fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The names a `dataset build` is asked in: its views, strip policies and
//! minimum strengths, and how they are parsed from the command line.

use std::str::FromStr;

use serde::Serialize;
use splinter_core::annotation::Strength;
use splinter_core::experience::PrivilegedKind;
use splinter_core::rng::Seed;
use splinter_data::{Fraction, Strip};

use splinter_orchestrator::error::OrchestratorError;

/// The weakest decision a view counts when a command names none:
/// consistency admits critiques verified by their retry's outcome and
/// agreement among answers, and leaves judged verdicts to be asked for.
pub const DEFAULT_MIN_STRENGTH: Strength = Strength::Consistency;

/// The views a dataset can be built through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ViewName {
    /// A passed experience's instruction and final answer.
    SftFinal,
    /// Each action of a trajectory, in the context before it.
    SftStep,
    /// A task and candidate answer, and a verified critique of it.
    Critic,
    /// A chosen and a rejected answer to one task.
    Preference,
    /// A task and candidate answer, and `pass` or `fail`.
    Verifier,
    /// The passing action where a passing and a failed attempt part.
    Decision,
    /// An instruction and the source span it is grounded in.
    Retrieval,
    /// A whole trajectory and its derived reward.
    Outcome,
    /// A denoise task's corrupted passage and its original.
    Denoise,
    /// The raw text of source parts.
    Cpt,
    /// The writer's own text as the answer, in the shape the policy is
    /// asked in.
    Voice,
    /// A model's own answers as given, unless a verifier decided one wrong:
    /// what a run rehearses so the new data does not move it off them.
    Rehearsal,
}

impl ViewName {
    /// Every view, in the order `--help` lists them.
    pub const ALL: [Self; 12] = [
        Self::SftFinal,
        Self::SftStep,
        Self::Critic,
        Self::Preference,
        Self::Verifier,
        Self::Decision,
        Self::Retrieval,
        Self::Outcome,
        Self::Denoise,
        Self::Cpt,
        Self::Voice,
        Self::Rehearsal,
    ];

    /// The name a command line uses.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SftFinal => "sft-final",
            Self::SftStep => "sft-step",
            Self::Critic => "critic",
            Self::Preference => "preference",
            Self::Verifier => "verifier",
            Self::Decision => "decision",
            Self::Retrieval => "retrieval",
            Self::Outcome => "outcome",
            Self::Denoise => "denoise",
            Self::Cpt => "cpt",
            Self::Voice => "voice",
            Self::Rehearsal => "rehearsal",
        }
    }

    pub(crate) fn takes_strip(self) -> bool {
        !matches!(self, Self::Cpt | Self::Voice)
    }

    pub(crate) fn takes_min_strength(self) -> bool {
        !matches!(
            self,
            Self::Cpt | Self::Denoise | Self::Voice | Self::Rehearsal
        )
    }
}

impl FromStr for ViewName {
    type Err = OrchestratorError;

    fn from_str(text: &str) -> Result<Self, OrchestratorError> {
        Self::ALL
            .into_iter()
            .find(|view| view.as_str() == text)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|v| v.as_str()).collect();
                OrchestratorError::Refused(format!(
                    "unknown view {text:?}; the views are {}",
                    names.join(", ")
                ))
            })
    }
}

/// A `--strip` policy as written, before the seed `mix:F` needs is settled.
///
/// The text names a fraction of subjects that keep their privileged context;
/// which subjects is decided by a seed the run states
/// ([`StripPolicy::with_seed`]), so that the same dataset can be built again.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum StripPolicy {
    /// `all` or `keep:K,..`: nothing is left to draw.
    Fixed(Strip),
    /// `mix:F`: the subjects that keep their context are drawn by a seed.
    Mix {
        /// The share of subjects that keep their privileged context.
        keep_fraction: Fraction,
    },
}

impl StripPolicy {
    /// The strip to build with. A `mix:F` policy asks `seed` for the seed
    /// that picks its subjects; the other policies never ask for one.
    pub fn with_seed<E>(self, seed: impl FnOnce() -> Result<Seed, E>) -> Result<Strip, E> {
        match self {
            Self::Fixed(strip) => Ok(strip),
            Self::Mix { keep_fraction } => Ok(Strip::Mix {
                keep_fraction,
                seed: seed()?.value(),
            }),
        }
    }
}

/// A `--strip` policy: `all`, `keep:K,..` or `mix:F`.
pub fn parse_strip(text: &str) -> Result<StripPolicy, OrchestratorError> {
    if text == "all" {
        return Ok(StripPolicy::Fixed(Strip::All));
    }
    if let Some(kinds) = text.strip_prefix("keep:") {
        let kinds = kinds
            .split(',')
            .map(privileged_kind)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(StripPolicy::Fixed(Strip::Keep(kinds)));
    }
    if let Some(fraction) = text.strip_prefix("mix:") {
        let value: f64 = fraction.parse().map_err(|_| {
            OrchestratorError::Refused(format!("mix:{fraction} needs a fraction in [0, 1]"))
        })?;
        let keep_fraction = Fraction::new(value)
            .map_err(|e| OrchestratorError::Refused(format!("mix:{fraction}: {e}")))?;
        return Ok(StripPolicy::Mix { keep_fraction });
    }
    Err(OrchestratorError::Refused(format!(
        "{text:?} is not a strip policy: all, keep:<kind>,.. or mix:<fraction>"
    )))
}

/// A privileged kind as `keep:` names it: `passage`, `hint`, `critique`,
/// `reference`, `oracle`, or `other:<name>`.
fn privileged_kind(name: &str) -> Result<PrivilegedKind, OrchestratorError> {
    Ok(match name {
        "passage" => PrivilegedKind::Passage,
        "hint" => PrivilegedKind::Hint,
        "critique" => PrivilegedKind::Critique,
        "reference" => PrivilegedKind::Reference,
        "oracle" => PrivilegedKind::Oracle,
        other => match other.strip_prefix("other:").filter(|n| !n.is_empty()) {
            Some(name) => PrivilegedKind::Other(name.to_string()),
            None => {
                return Err(OrchestratorError::Refused(format!(
                    "unknown privileged kind {name:?}: passage, hint, critique, reference, \
                     oracle or other:<name>"
                )))
            }
        },
    })
}

/// A `--min-strength`: `executable`, `formal`, `consistency` or `judged`.
pub fn parse_strength(text: &str) -> Result<Strength, OrchestratorError> {
    Ok(match text {
        "executable" => Strength::Executable,
        "formal" => Strength::Formal,
        "consistency" => Strength::Consistency,
        "judged" => Strength::Judged,
        _ => {
            return Err(OrchestratorError::Refused(format!(
                "{text:?} is not a strength: executable, formal, consistency or judged"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_policies_and_strengths_parse_or_are_refused() {
        let never = || -> Result<Seed, ()> { Err(()) };
        assert_eq!(
            parse_strip("all").ok().map(|p| p.with_seed(never)),
            Some(Ok(Strip::All))
        );
        assert_eq!(
            parse_strip("keep:hint,other:executable-check")
                .ok()
                .map(|p| p.with_seed(never)),
            Some(Ok(Strip::Keep(vec![
                PrivilegedKind::Hint,
                PrivilegedKind::Other("executable-check".into())
            ])))
        );
        // `mix:F` takes the seed it is given, and only it asks for one.
        let given = splinter_model::seed::draw();
        let mixed = parse_strip("mix:0.25")
            .unwrap()
            .with_seed::<()>(|| Ok(given));
        assert!(matches!(mixed, Ok(Strip::Mix { seed, .. }) if seed == given.value()));
        for refused in ["none", "keep:hnt", "mix:2", "mix:x", "keep:other:"] {
            assert!(parse_strip(refused).is_err(), "{refused}");
        }
        assert_eq!(parse_strength("formal").ok(), Some(Strength::Formal));
        assert!(parse_strength("strong").is_err());
        for view in ViewName::ALL {
            assert_eq!(view.as_str().parse::<ViewName>().ok(), Some(view));
        }
        assert!("sft".parse::<ViewName>().is_err());
    }
}
