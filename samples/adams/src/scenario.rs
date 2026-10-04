// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in building test situations that show where a historical method applies and
// where it does not, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Modern situations built from a principle, never from a topic.
//!
//! A scenario is designed from a verified principle: when it applies, what he
//! did, what limits it. Most are fits of different strength; some are
//! deliberate non-fits, where a precondition is missing or the resemblance is
//! only on the surface, because a model that applies his method everywhere has
//! learned a caricature. The helper drafts; code decides whether the draft may
//! be used: it must be a modern problem, carry none of the source's wording,
//! and state conditions that match the case it was designed as.

use std::sync::Arc;

use splinter_sdk::agent::schemars::JsonSchema;
use splinter_sdk::agent::typed::TypedCall;
use splinter_sdk::measure::verifiers::quotation::{words, TextIndex};

use crate::helper::Helper;
use crate::principles::Principle;

/// How a scenario relates to the principle it was designed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Case {
    /// Every condition holds; the method applies plainly.
    Clear,
    /// It applies, but a qualification matters.
    Weak,
    /// A condition the principle needs is missing; the method should not be
    /// applied until it is known.
    MissingPrecondition,
    /// It looks like the principle's situation and is not.
    SurfaceAnalogy,
}

impl Case {
    /// Whether the method applies.
    pub fn applies(self) -> bool {
        matches!(self, Case::Clear | Case::Weak)
    }

    /// What the designer is told to build.
    fn brief(self) -> &'static str {
        match self {
            Case::Clear => "Design a situation in which every trigger condition holds and nothing in the qualifications applies, so the method plainly fits.",
            Case::Weak => "Design a situation in which the trigger conditions hold but one qualification matters, so the method fits only with care.",
            Case::MissingPrecondition => "Design a situation that resembles the principle's but in which at least one trigger condition is missing or unknown, so the method should not be applied until it is established.",
            Case::SurfaceAnalogy => "Design a situation that shares the principle's vocabulary or look but not its structure: the conditions that make the method work are absent, and the resemblance is only on the surface.",
        }
    }
}

/// What the helper drafts.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
pub struct Drafted {
    /// The field the situation is in: software teams, a hospital, a school board.
    pub domain: String,
    /// The situation, in the present day, concrete, with its people and numbers.
    pub situation: String,
    /// Facts about the situation that the person asking would know and state.
    pub observations: Vec<String>,
    /// What the person asks for.
    pub request: String,
    /// The principle's conditions that hold here.
    pub conditions_present: Vec<String>,
    /// The principle's conditions that are missing or unknown here.
    pub conditions_missing: Vec<String>,
}

/// A scenario that passed the gate.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Scenario {
    pub id: String,
    pub principle_id: String,
    pub case: Case,
    pub domain: String,
    pub situation: String,
    pub observations: Vec<String>,
    pub request: String,
    pub conditions_present: Vec<String>,
    pub conditions_missing: Vec<String>,
}

/// Words a situation must run to be a problem and not a sentence, and a
/// request to be a request.
const MIN_SITUATION_WORDS: usize = 40;
const MIN_REQUEST_WORDS: usize = 5;
/// How many facts the asker states, and the fewest words of each.
const OBSERVATIONS: std::ops::RangeInclusive<usize> = 2..=8;
const MIN_OBSERVATION_WORDS: usize = 4;
/// A run this long that is in the source is the source's wording.
const SOURCE_RUN_WORDS: usize = 8;
/// Years that put a situation in his century, not ours.
const HIS_CENTURY: std::ops::RangeInclusive<u16> = 1700..=1850;

fn names_him_or_his_century(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("samuel adams")
        || words(text)
            .iter()
            .any(|w| w == "adams" || w.parse::<u16>().is_ok_and(|y| HIS_CENTURY.contains(&y)))
}

fn copies_source(text: &str, corpus: &TextIndex) -> bool {
    words(text)
        .windows(SOURCE_RUN_WORDS)
        .any(|run| corpus.contains(&run.join(" ")))
}

/// Refuse a draft that cannot be used, naming why.
pub fn gate(drafted: &Drafted, case: Case, corpus: &TextIndex) -> Result<(), String> {
    if words(&drafted.situation).len() < MIN_SITUATION_WORDS {
        return Err(format!("the situation is too short to be a problem: write at least {MIN_SITUATION_WORDS} words with its people and numbers"));
    }
    if words(&drafted.request).len() < MIN_REQUEST_WORDS {
        return Err("the request is too short".to_string());
    }
    if !OBSERVATIONS.contains(&drafted.observations.len())
        || drafted
            .observations
            .iter()
            .any(|o| words(o).len() < MIN_OBSERVATION_WORDS)
    {
        return Err(format!(
            "give {} to {} observations of at least {MIN_OBSERVATION_WORDS} words each",
            OBSERVATIONS.start(),
            OBSERVATIONS.end()
        ));
    }
    let texts = std::iter::once(&drafted.situation)
        .chain(&drafted.observations)
        .chain(std::iter::once(&drafted.request));
    for text in texts {
        if names_him_or_his_century(text) {
            return Err("the scenario names Samuel Adams or sits in the eighteenth century: it must be a present-day problem".to_string());
        }
        if copies_source(text, corpus) {
            return Err("the scenario repeats the source's own wording: describe the situation in your own words".to_string());
        }
    }
    let (present, missing) = (
        !drafted.conditions_present.is_empty(),
        !drafted.conditions_missing.is_empty(),
    );
    match case {
        Case::Clear if missing => {
            Err("a clear fit has no missing conditions: leave conditions_missing empty".to_string())
        }
        Case::Clear if !present => Err("name the principle's conditions that hold".to_string()),
        Case::Weak if !(present && missing) => Err(
            "a weak fit has a condition that holds and one that is missing or unknown".to_string(),
        ),
        Case::MissingPrecondition | Case::SurfaceAnalogy if !missing => {
            Err("this case must name what is missing in conditions_missing".to_string())
        }
        _ => Ok(()),
    }
}

/// The role the designer is given.
const ROLE: &str = "a designer of realistic present-day situations for testing whether a method from the past applies";

/// The task the designer is given.
const TASK: &str = "You are given a principle that describes how a historical figure worked: when it applies, what he did, and what limits it. Design one present-day situation, in a field far from his, that tests the principle as the case below directs. Make it concrete: people, numbers, constraints. Do not name the figure or his period, and do not use any wording from the principle's own statement. State which of the principle's conditions hold in your situation and which are missing or unknown, and the facts the person asking would know and say.";

/// What the designer is shown of a principle: the rule and its limits, never
/// the quotations that support it.
#[derive(serde::Serialize)]
struct Brief<'a> {
    case: &'static str,
    description: &'a str,
    trigger_conditions: &'a [String],
    expected_behavior: &'a [String],
    qualifications: &'a [String],
}

/// What the designer may write across every attempt, repairs included.
const OUTPUT_TOKENS: u64 = 3000;
const REPAIRS: u32 = 2;

/// Draft a scenario of `case` from `principle` and gate it.
///
/// # Errors
/// The helper could not be reached, or no draft passed the gate after the
/// permitted corrections.
pub fn design(
    helper: &Helper,
    principle: &Principle,
    case: Case,
    corpus: &Arc<TextIndex>,
) -> anyhow::Result<Scenario> {
    let checked = Arc::clone(corpus);
    let call =
        TypedCall::<Drafted>::new("design_scenario", TASK, ROLE, crate::helper::CALL_DEADLINE)
            .max_output_tokens(OUTPUT_TOKENS)
            .repairs(REPAIRS)
            .postcondition(move |drafted| gate(drafted, case, &checked));
    let brief = Brief {
        case: case.brief(),
        description: &principle.description,
        trigger_conditions: &principle.trigger_conditions,
        expected_behavior: &principle.expected_behavior,
        qualifications: &principle.qualifications,
    };
    let d = helper.call(call, &brief)?;
    let id = format!(
        "scenario-{}",
        &blake3::hash(format!("{}\n{case:?}\n{}", principle.id, d.situation).as_bytes()).to_hex()
            [..12]
    );
    Ok(Scenario {
        id,
        principle_id: principle.id.clone(),
        case,
        domain: d.domain,
        situation: d.situation,
        observations: d.observations,
        request: d.request,
        conditions_present: d.conditions_present,
        conditions_missing: d.conditions_missing,
    })
}

/// Which case each principle is designed as: most fit, some deliberately do
/// not, chosen by hash so the plan does not depend on order. Of every ten, six
/// are clear fits, one weak, two a missing precondition, one a surface analogy.
pub fn plan(principle_ids: &[String]) -> Vec<(String, Case)> {
    principle_ids
        .iter()
        .map(|id| {
            let bucket = blake3::hash(id.as_bytes()).as_bytes()[0] % 10;
            let case = match bucket {
                0..=5 => Case::Clear,
                6 => Case::Weak,
                7 | 8 => Case::MissingPrecondition,
                _ => Case::SurfaceAnalogy,
            };
            (id.clone(), case)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Period;
    use crate::principles::{Found, Principle, Status};
    use crate::testkit::{helper, words as distinct};

    const SOURCE: &str = "Let the Committee write to every Town, that the Sense of the People may be known before the Assembly meets again.";

    fn corpus() -> Arc<TextIndex> {
        Arc::new(TextIndex::new([SOURCE]))
    }

    fn good() -> Drafted {
        Drafted {
            domain: "a software company".into(),
            situation: format!("A company of 800 engineers is moving its services to a new platform and the five regional teams each want a different schedule. The platform group has announced a single date. {}", distinct("ctx", 20)),
            observations: vec!["There are five regional teams.".into(), "The platform group announced one migration date.".into()],
            request: "Advise the head of engineering on how to get the teams to agree.".into(),
            conditions_present: vec!["a grievance unanswered by the centre".into()],
            conditions_missing: Vec::new(),
        }
    }

    fn principle() -> Principle {
        Principle {
            id: "P-abc".into(),
            description:
                "When the ministry will not act, he had the towns state the claim together.".into(),
            trigger_conditions: vec!["a grievance the centre has not answered".into()],
            expected_behavior: vec!["write to the towns and collect their resolutions".into()],
            qualifications: vec!["not when the towns are divided".into()],
            status: Status::SourceSupported,
            support: vec![Found {
                doc_id: "d1".into(),
                quote: SOURCE.into(),
                period: Period::PreRevolution,
                recipient: None,
            }],
            dropped: Vec::new(),
        }
    }

    fn drafted_json(d: &Drafted) -> String {
        serde_json::to_string(d).unwrap()
    }

    #[test]
    fn a_concrete_modern_draft_that_matches_its_case_passes() {
        assert_eq!(gate(&good(), Case::Clear, &corpus()), Ok(()));
    }

    #[test]
    fn a_situation_too_thin_to_be_a_problem_is_refused() {
        let mut d = good();
        d.situation = "A company has teams.".into();
        assert!(gate(&d, Case::Clear, &corpus())
            .unwrap_err()
            .contains("short"));
    }

    #[test]
    fn the_sources_own_wording_in_a_draft_is_refused() {
        let mut d = good();
        d.situation.push_str(
            " Let the Committee write to every Town, that the Sense of the People may be known.",
        );
        assert!(gate(&d, Case::Clear, &corpus())
            .unwrap_err()
            .contains("source"));
        let mut d = good();
        d.request = "Let the Committee write to every Town, that the Sense of the People".into();
        assert!(gate(&d, Case::Clear, &corpus())
            .unwrap_err()
            .contains("source"));
    }

    #[test]
    fn a_draft_that_names_him_or_sits_in_his_century_is_refused() {
        let mut d = good();
        d.situation.push_str(" What would Samuel Adams do?");
        assert!(gate(&d, Case::Clear, &corpus()).is_err());
        let mut d = good();
        d.situation.push_str(" The board met in 1774 to decide.");
        assert!(gate(&d, Case::Clear, &corpus()).is_err());
    }

    #[test]
    fn the_stated_conditions_must_match_the_case() {
        let mut missing = good();
        missing.conditions_missing = vec!["whether the teams agree among themselves".into()];
        assert!(
            gate(&missing, Case::Clear, &corpus())
                .unwrap_err()
                .contains("missing"),
            "a clear fit has nothing missing"
        );
        assert_eq!(gate(&missing, Case::MissingPrecondition, &corpus()), Ok(()));
        assert!(
            gate(&good(), Case::MissingPrecondition, &corpus()).is_err(),
            "a missing precondition must name what is missing"
        );
        assert!(gate(&good(), Case::SurfaceAnalogy, &corpus()).is_err());
        let mut weak = missing.clone();
        weak.conditions_present = Vec::new();
        assert!(
            gate(&weak, Case::Weak, &corpus()).is_err(),
            "a weak fit still has a condition that holds"
        );
    }

    #[test]
    fn the_facts_the_asker_states_are_two_to_eight_and_substantial() {
        let mut d = good();
        d.observations.truncate(1);
        assert!(gate(&d, Case::Clear, &corpus())
            .unwrap_err()
            .contains("observations"));
        let mut d = good();
        d.observations = (0..9)
            .map(|i| format!("Fact number {i} about the situation."))
            .collect();
        assert!(gate(&d, Case::Clear, &corpus()).is_err());
        let mut d = good();
        d.observations[0] = "Five.".into();
        assert!(gate(&d, Case::Clear, &corpus()).is_err());
    }

    #[test]
    fn a_scenario_is_designed_from_the_principles_rule_not_its_quotations() {
        let (h, provider) = helper(&[&drafted_json(&good())]);
        let s = design(&h, &principle(), Case::Clear, &corpus()).unwrap();
        assert_eq!((s.principle_id.as_str(), s.case), ("P-abc", Case::Clear));
        assert!(s.id.starts_with("scenario-"));
        let sent = provider.sent();
        assert!(
            sent.contains("a grievance the centre has not answered")
                && sent.contains("not when the towns are divided")
        );
        assert!(
            !sent.contains("Sense of the People"),
            "the designer is never handed his wording to copy"
        );
    }

    #[test]
    fn a_draft_that_fails_the_gate_is_sent_back_and_a_corrected_one_is_kept() {
        let mut leaky = good();
        leaky.request =
            "Let the Committee write to every Town, that the Sense of the People may be known."
                .into();
        let (h, provider) = helper(&[&drafted_json(&leaky), &drafted_json(&good())]);
        let s = design(&h, &principle(), Case::Clear, &corpus()).unwrap();
        assert_eq!(s.request, good().request);
        assert!(provider.requests() >= 2);
    }

    #[test]
    fn a_helper_that_never_produces_a_usable_draft_is_an_error() {
        let mut leaky = good();
        leaky.request =
            "Let the Committee write to every Town, that the Sense of the People may be known."
                .into();
        let (h, _) = helper(&[&drafted_json(&leaky)]);
        assert!(design(&h, &principle(), Case::Clear, &corpus()).is_err());
    }

    #[test]
    fn the_plan_gives_every_principle_a_case_and_designs_a_share_as_non_fits() {
        let ids: Vec<String> = (0..200).map(|i| format!("P-{i}")).collect();
        let planned = plan(&ids);
        assert_eq!(planned.len(), 200);
        let non_fits = planned.iter().filter(|(_, c)| !c.applies()).count();
        assert!((30..=70).contains(&non_fits), "{non_fits} of 200");
        let mut reversed = ids.clone();
        reversed.reverse();
        let again: std::collections::HashMap<_, _> = plan(&reversed).into_iter().collect();
        assert!(
            planned.iter().all(|(id, case)| again[id] == *case),
            "the plan does not depend on order"
        );
    }
}
