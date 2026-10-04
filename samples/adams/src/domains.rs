// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on
// varied situations so a learned method does not collapse into a few settings
// for its clients. If your team needs expertise in building diverse,
// verifiable training scenarios then you can procure our services by sending
// an email to info@swedishembedded.com.

//! Where scenarios are set, and which are made for which principle.
//!
//! Left to choose, the helper puts nearly every situation in the same three
//! fields, and a model trained on that learns "this field, this layout" and
//! not the method. So code assigns each scenario its field from a fixed list,
//! checks the draft is in it, and holds a few fields out of training entirely:
//! what the model does there is transfer to a setting it never saw.

use crate::scenario::Case;

/// A field a scenario may be set in.
#[derive(Debug, PartialEq, Eq)]
pub struct Domain {
    /// What the helper is told.
    pub name: &'static str,
    /// Words of the field: a draft in it uses at least one.
    pub keywords: &'static [&'static str],
    /// Whether the field is held out of training, for the out-of-domain slice
    /// of the benchmark.
    pub held_out: bool,
}

const fn domain(name: &'static str, keywords: &'static [&'static str], held_out: bool) -> Domain {
    Domain {
        name,
        keywords,
        held_out,
    }
}

/// The fields, in a fixed order: a scenario's field is the one its hash picks.
pub static DOMAINS: [Domain; 18] = [
    domain(
        "a city council",
        &["council", "mayor", "ordinance", "city"],
        false,
    ),
    domain(
        "a labor union local",
        &["union", "strike", "members", "contract"],
        false,
    ),
    domain(
        "a tenants' cooperative",
        &["tenants", "landlord", "rent", "cooperative"],
        false,
    ),
    domain(
        "a regional newsroom",
        &["newsroom", "editor", "reporters", "story"],
        false,
    ),
    domain(
        "a church congregation",
        &["congregation", "pastor", "parish", "church"],
        false,
    ),
    domain(
        "a community nonprofit",
        &["donors", "grant", "nonprofit", "volunteers"],
        false,
    ),
    domain(
        "a youth sports league",
        &["league", "coach", "parents", "season"],
        false,
    ),
    domain(
        "a university department",
        &["faculty", "dean", "students", "department"],
        false,
    ),
    domain(
        "a manufacturing plant",
        &["plant", "shift", "foreman", "production"],
        false,
    ),
    domain(
        "a farmers' cooperative",
        &["farmers", "harvest", "crop", "cooperative"],
        false,
    ),
    domain(
        "a public library board",
        &["library", "librarian", "trustees", "board"],
        false,
    ),
    domain(
        "a neighborhood association",
        &["neighborhood", "residents", "association", "block"],
        false,
    ),
    domain(
        "a family restaurant",
        &["restaurant", "chef", "kitchen", "owner"],
        false,
    ),
    domain(
        "a shipping company",
        &["ships", "captain", "port", "crew"],
        false,
    ),
    domain(
        "a legal aid clinic",
        &["clients", "attorneys", "court", "clinic"],
        false,
    ),
    domain(
        "a symphony orchestra",
        &["orchestra", "musicians", "conductor", "symphony"],
        true,
    ),
    domain(
        "a volunteer fire department",
        &["firefighters", "station", "chief", "fire"],
        true,
    ),
    domain(
        "an open-source project",
        &["maintainers", "contributors", "repository", "project"],
        true,
    ),
];

/// Scenarios made for one principle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Job {
    /// Which scenario of the principle this is: 0 is the first, which the
    /// helper set in whatever field it chose.
    pub variant: u8,
    /// What the scenario is designed to be.
    pub case: Case,
    /// The field it is set in; none for the first scenario of a principle.
    pub domain: Option<&'static Domain>,
}

fn byte(principle_id: &str, variant: u8, salt: &str) -> u8 {
    blake3::hash(format!("{principle_id}\n{variant}\n{salt}").as_bytes()).as_bytes()[0]
}

/// The scenarios to make for `principle_id`: the planned first one, and
/// `variants - 1` more. The second is of the opposite kind to the first (a
/// non-fit where the first fits, a fit where it did not), so a principle is
/// seen both holding and not holding; later ones are chosen by hash. Every one
/// after the first is set in a field chosen by hash from [`DOMAINS`].
#[must_use]
pub fn jobs(principle_id: &str, first: Case, variants: u8) -> Vec<Job> {
    (0..variants.max(1))
        .map(|variant| {
            if variant == 0 {
                return Job {
                    variant,
                    case: first,
                    domain: None,
                };
            }
            let case = match (variant, first.applies()) {
                (1, true) => {
                    if byte(principle_id, variant, "case").is_multiple_of(2) {
                        Case::MissingPrecondition
                    } else {
                        Case::SurfaceAnalogy
                    }
                }
                (1, false) => Case::Clear,
                _ => match byte(principle_id, variant, "case") % 10 {
                    0..=5 => Case::Clear,
                    6 => Case::Weak,
                    7 | 8 => Case::MissingPrecondition,
                    _ => Case::SurfaceAnalogy,
                },
            };
            let at = usize::from(byte(principle_id, variant, "domain")) % DOMAINS.len();
            Job {
                variant,
                case,
                domain: Some(&DOMAINS[at]),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_scenario_is_the_planned_one_and_the_rest_are_set_in_listed_fields() {
        let jobs = jobs("P-1", Case::Clear, 3);
        assert_eq!(jobs.len(), 3);
        assert_eq!(
            (jobs[0].variant, jobs[0].case, jobs[0].domain),
            (0, Case::Clear, None)
        );
        assert!(jobs[1..].iter().all(|j| j.domain.is_some()));
        assert_eq!(
            super::jobs("P-1", Case::Clear, 3),
            jobs,
            "the plan does not depend on anything but its inputs"
        );
        assert_eq!(
            super::jobs("P-1", Case::Weak, 0).len(),
            1,
            "there is always a first scenario"
        );
    }

    #[test]
    fn a_principle_is_seen_both_holding_and_not_holding() {
        for n in 0..200 {
            let id = format!("P-{n}");
            for first in [
                Case::Clear,
                Case::Weak,
                Case::MissingPrecondition,
                Case::SurfaceAnalogy,
            ] {
                let planned = jobs(&id, first, 2);
                assert_ne!(
                    planned[0].case.applies(),
                    planned[1].case.applies(),
                    "{id} {first:?}"
                );
            }
        }
    }

    #[test]
    fn the_fields_are_used_about_evenly_and_some_are_held_out() {
        let mut counts = [0usize; DOMAINS.len()];
        for n in 0..3600 {
            let id = format!("P-{n}");
            let job = jobs(&id, Case::Clear, 2)[1];
            let at = DOMAINS
                .iter()
                .position(|d| std::ptr::eq(d, job.domain.unwrap()))
                .unwrap();
            counts[at] += 1;
        }
        assert!(counts.iter().all(|&c| c > 100 && c < 320), "{counts:?}");
        let held_out = DOMAINS.iter().filter(|d| d.held_out).count();
        assert!((2..=4).contains(&held_out), "{held_out} fields held out");
        assert!(DOMAINS.iter().all(|d| d.keywords.len() >= 3));
    }
}
