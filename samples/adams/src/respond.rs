// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in making a model say what its answer rests on and in checking that it does,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The answer to a scenario: how the student is asked, what an answer looks
//! like, and how a helper's draft of one is accepted.
//!
//! The student is asked under one system message and shown the situation, the
//! facts it was given and, in retrieval mode, passages from his papers with
//! their ids. A good answer begins by saying whether his method applies, gives
//! practical advice or asks for what it cannot decide without, and ends in a
//! grounding block. Code checks every part: the applicability agrees with the
//! case the scenario was designed as, a non-fit that says it needs information
//! asks for it, and the grounding keeps its word ([`crate::grounding`]).

use std::collections::HashSet;
use std::sync::Arc;

use splinter_sdk::agent::schemars::JsonSchema;
use splinter_sdk::agent::typed::TypedCall;

use crate::curate::Document;
use crate::grounding::{self, Violation};
use crate::helper::Helper;
use crate::principles::{Found, Principle};
use crate::scenario::{Case, Scenario};

/// The system message the student is asked under: the persona, the format and
/// the rule that what it cannot source it says it cannot.
pub const SYSTEM: &str = "You are Samuel Adams (1722-1803), applying the method of your own papers to a present-day problem you never met. Answer in the first person. Begin with one line, `Applicability:`, then APPLIES, PARTLY, DOES_NOT_APPLY or NEEDS_INFORMATION. Give practical advice, or ask for what you cannot responsibly decide without. End with a `Grounding:` block, one line per claim, each starting with its label: SOURCE_DIRECT (your own words, quoted, then the document id in square brackets), SOURCE_INFERRED (what your papers show), MODERN_OBSERVATION (a fact you were given), PERSONA_TRANSFER (your method carried to this problem), SPECULATION. Quote only what is in the passages you are shown. Say plainly what the record does not show.";

/// Whether his method applies, as an answer says it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Applicability {
    Applies,
    Partly,
    DoesNotApply,
    NeedsInformation,
}

impl Applicability {
    /// The applicability an answer states on its first line, if it states one.
    pub fn of(answer: &str) -> Option<Applicability> {
        let first = answer.lines().map(str::trim).find(|l| !l.is_empty())?;
        let (label, value) = first.split_once(':')?;
        if !label.trim().eq_ignore_ascii_case("applicability") {
            return None;
        }
        match value.trim().to_uppercase().as_str() {
            "APPLIES" => Some(Applicability::Applies),
            "PARTLY" => Some(Applicability::Partly),
            "DOES_NOT_APPLY" => Some(Applicability::DoesNotApply),
            "NEEDS_INFORMATION" => Some(Applicability::NeedsInformation),
            _ => None,
        }
    }

    /// Whether this is the right thing to say about a scenario of `case`.
    pub fn fits(self, case: Case) -> bool {
        match self {
            Applicability::Applies | Applicability::Partly => case.applies(),
            Applicability::DoesNotApply | Applicability::NeedsInformation => !case.applies(),
        }
    }
}

/// How the student is shown a scenario. In retrieval mode `evidence` is the
/// passages from his papers, each with the id of its document; in internalized
/// mode there are none and the student answers from what it has learned. The
/// case the scenario was designed as is never shown.
pub fn student_prompt(scenario: &Scenario, evidence: &[Found]) -> String {
    let mut prompt = format!(
        "Situation: {}\n\nFacts you were given:\n",
        scenario.situation
    );
    for fact in &scenario.observations {
        prompt.push_str(&format!("- {fact}\n"));
    }
    prompt.push_str(&format!("\nRequest: {}\n", scenario.request));
    if !evidence.is_empty() {
        prompt.push_str("\nPassages from your papers (cite them by id):\n");
        for passage in evidence {
            prompt.push_str(&format!("[{}] \"{}\"\n", passage.doc_id, passage.quote));
        }
    }
    prompt
}

/// Say what is wrong with an answer, in words the drafter can act on, or
/// nothing if it keeps every word.
fn describe(violation: &Violation) -> String {
    match violation {
        Violation::MissingBlock => "end the answer with a `Grounding:` block".to_string(),
        Violation::UnknownLabel(label) => format!("{label:?} is not a label: use SOURCE_DIRECT, SOURCE_INFERRED, MODERN_OBSERVATION, PERSONA_TRANSFER or SPECULATION"),
        Violation::UncitedSource(line) => format!("a SOURCE_DIRECT line needs a quotation of at least six words and a document id in square brackets: {line}"),
        Violation::FabricatedQuote(quote) => format!("this quotation is not in the document it cites or in the passages you were shown, so it must go: {quote}"),
        Violation::UnsupportedFact(fact) => format!("this MODERN_OBSERVATION is not among the facts you were given: {fact}"),
        Violation::Anachronism(word) => format!("a line labelled as his own uses {word:?}, which did not exist in his lifetime"),
    }
}

/// What a draft answer must be, or why not: the applicability agrees with the
/// case, a non-fit that needs information asks for it, a fit quotes him when
/// there are passages to quote, and the grounding keeps its word against the
/// documents it may cite and the facts it was given.
pub fn check(text: &str, scenario: &Scenario, docs: &[&Document]) -> Result<(), String> {
    let Some(stated) = Applicability::of(text) else {
        return Err("the first line must be `Applicability:` followed by APPLIES, PARTLY, DOES_NOT_APPLY or NEEDS_INFORMATION".to_string());
    };
    if !stated.fits(scenario.case) {
        return Err(format!("the applicability {stated:?} does not suit this situation: decide it from whether the principle's conditions hold here"));
    }
    let grounded = grounding::parse(text);
    if stated == Applicability::NeedsInformation && !grounded.body.contains('?') {
        return Err("an answer that needs information must ask for it: put the question in the answer, ending in a question mark".to_string());
    }
    let violations = grounding::verify(&grounded, docs, &scenario.observations);
    if !violations.is_empty() {
        return Err(violations
            .iter()
            .map(describe)
            .collect::<Vec<_>>()
            .join("; "));
    }
    let quotes_him = grounded
        .items
        .iter()
        .flatten()
        .any(|i| i.label == Some(grounding::Label::SourceDirect));
    if scenario.case.applies() && !docs.is_empty() && !quotes_him {
        return Err(
            "a fit must rest on his words: add a SOURCE_DIRECT line quoting one of the passages"
                .to_string(),
        );
    }
    Ok(())
}

/// What the helper drafts.
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
struct Drafted {
    /// The whole answer, beginning with the Applicability line and ending in the Grounding block.
    answer: String,
}

/// An answer accepted for a scenario.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Answer {
    pub scenario_id: String,
    pub text: String,
}

/// The role the drafter is given.
const ROLE: &str = "Samuel Adams, answering a present-day problem with the method of his own papers and saying plainly what they do not show";

/// What the drafter is shown: the rules of the answer, what it should conclude
/// about whether his method applies, and exactly what the student will see.
#[derive(serde::Serialize)]
struct Brief<'a> {
    rules: &'static str,
    conclusion: &'static str,
    the_question: &'a str,
}

const OUTPUT_TOKENS: u64 = 1400;
const REPAIRS: u32 = 2;

/// Draft an answer to `scenario` as he would give it, citing only `evidence`,
/// and check it.
///
/// # Errors
/// The helper could not be reached, or no draft passed the check after the
/// permitted corrections.
pub fn draft(
    helper: &Helper,
    scenario: &Scenario,
    principle: &Principle,
    docs: &Arc<Vec<Document>>,
) -> anyhow::Result<Answer> {
    let cited: HashSet<&str> = principle
        .support
        .iter()
        .map(|f| f.doc_id.as_str())
        .collect();
    let allowed: Vec<Document> = docs
        .iter()
        .filter(|d| cited.contains(d.id.as_str()))
        .cloned()
        .collect();
    let checked = scenario.clone();
    let call = TypedCall::<Drafted>::new(
        "answer_as_adams",
        "Write the answer to the question below. Follow the rules exactly.",
        ROLE,
        crate::helper::CALL_DEADLINE,
    )
    .max_output_tokens(OUTPUT_TOKENS)
    .repairs(REPAIRS)
    .postcondition(move |drafted| {
        let refs: Vec<&Document> = allowed.iter().collect();
        check(&drafted.answer, &checked, &refs)
    });
    let prompt = student_prompt(scenario, &principle.support);
    let conclusion = if scenario.case.applies() {
        "His method applies here: say APPLIES or PARTLY, advise accordingly, and rest the advice on a quoted passage."
    } else {
        "His method should not be applied yet or here: say DOES_NOT_APPLY or NEEDS_INFORMATION, explain which condition is missing or unknown, and ask for what you need."
    };
    let text = helper
        .call(
            call,
            &Brief {
                rules: SYSTEM,
                conclusion,
                the_question: &prompt,
            },
        )?
        .answer;
    Ok(Answer {
        scenario_id: scenario.id.clone(),
        text: text.trim().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Period;
    use crate::principles::Status;
    use crate::testkit::{doc, helper};

    const WORDS_OF_HIS: &str = "Let the Committee write to every Town, that the Sense of the People may be known before the Assembly meets again.";

    fn scenario(case: Case) -> Scenario {
        Scenario {
            id: "scenario-1".into(),
            principle_id: "P-abc".into(),
            case,
            domain: "software".into(),
            situation: "Five regional teams want different migration dates.".into(),
            observations: vec![
                "There are five regional teams.".into(),
                "The platform group announced one date.".into(),
            ],
            request: "Advise the head of engineering.".into(),
            conditions_present: vec!["a grievance unanswered".into()],
            conditions_missing: if case.applies() {
                Vec::new()
            } else {
                vec!["whether the teams agree".into()]
            },
        }
    }

    fn principle() -> Principle {
        Principle {
            id: "P-abc".into(),
            description: "He had the towns state the claim together.".into(),
            trigger_conditions: vec!["a grievance unanswered".into()],
            expected_behavior: vec!["write to the towns".into()],
            qualifications: vec!["not when divided".into()],
            status: Status::SourceSupported,
            support: vec![Found {
                doc_id: "d1".into(),
                quote: "Let the Committee write to every Town".into(),
                period: Period::PreRevolution,
                recipient: None,
            }],
            dropped: Vec::new(),
        }
    }

    fn docs() -> Arc<Vec<Document>> {
        Arc::new(vec![doc("d1", 1770, "A B", WORDS_OF_HIS)])
    }

    const FIT: &str = "Applicability: APPLIES\n\nI would have each team put its own case in writing, then gather them for the platform group.\n\nGrounding:\n- SOURCE_DIRECT: \"Let the Committee write to every Town\" [d1]\n- MODERN_OBSERVATION: There are five regional teams\n- PERSONA_TRANSFER: the committee method carried to the teams\n";

    const NON_FIT: &str = "Applicability: NEEDS_INFORMATION\n\nI cannot say yet. Do the five teams agree among themselves, and who owns the schedule?\n\nGrounding:\n- SOURCE_INFERRED: my papers show I gathered opinion before acting\n- MODERN_OBSERVATION: There are five regional teams\n";

    fn refs(d: &Arc<Vec<Document>>) -> Vec<&Document> {
        d.iter().collect()
    }

    #[test]
    fn the_applicability_is_read_from_the_first_line_whatever_its_case_and_spacing() {
        assert_eq!(
            Applicability::of("Applicability: APPLIES\nrest"),
            Some(Applicability::Applies)
        );
        assert_eq!(
            Applicability::of("  applicability:  partly \nrest"),
            Some(Applicability::Partly)
        );
        assert_eq!(
            Applicability::of("Applicability: DOES_NOT_APPLY"),
            Some(Applicability::DoesNotApply)
        );
        assert_eq!(
            Applicability::of("Applicability: NEEDS_INFORMATION"),
            Some(Applicability::NeedsInformation)
        );
        assert_eq!(
            Applicability::of("It applies.\nApplicability: APPLIES"),
            None,
            "it must be the first line"
        );
        assert_eq!(Applicability::of("Applicability: MAYBE"), None);
        assert_eq!(Applicability::of(""), None);
    }

    #[test]
    fn what_is_said_about_applicability_must_fit_the_case() {
        assert!(
            Applicability::Applies.fits(Case::Clear) && Applicability::Partly.fits(Case::Clear)
        );
        assert!(Applicability::Partly.fits(Case::Weak) && Applicability::Applies.fits(Case::Weak));
        for case in [Case::MissingPrecondition, Case::SurfaceAnalogy] {
            assert!(
                Applicability::DoesNotApply.fits(case)
                    && Applicability::NeedsInformation.fits(case)
            );
            assert!(!Applicability::Applies.fits(case) && !Applicability::Partly.fits(case));
        }
        assert!(!Applicability::DoesNotApply.fits(Case::Clear));
    }

    #[test]
    fn the_student_is_shown_the_situation_the_facts_and_the_request_and_in_retrieval_mode_the_passages(
    ) {
        let evidence = principle().support;
        let retrieval = student_prompt(&scenario(Case::Clear), &evidence);
        for want in [
            "Five regional teams",
            "There are five regional teams.",
            "Advise the head of engineering.",
            "[d1]",
            "Let the Committee write to every Town",
        ] {
            assert!(retrieval.contains(want), "{want} in {retrieval}");
        }
        let internalized = student_prompt(&scenario(Case::Clear), &[]);
        assert!(
            !internalized.contains("[d1]") && !internalized.contains("Passages from your papers"),
            "{internalized}"
        );
        assert!(internalized.contains("Five regional teams"));
    }

    #[test]
    fn the_prompt_does_not_reveal_which_case_it_was_designed_as() {
        let p = student_prompt(&scenario(Case::SurfaceAnalogy), &[]);
        assert!(
            !p.contains("whether the teams agree")
                && !p.to_lowercase().contains("surface")
                && !p.contains("conditions"),
            "{p}"
        );
    }

    #[test]
    fn a_grounded_answer_that_fits_its_case_is_accepted() {
        let d = docs();
        assert_eq!(check(FIT, &scenario(Case::Clear), &refs(&d)), Ok(()));
        assert_eq!(
            check(NON_FIT, &scenario(Case::MissingPrecondition), &refs(&d)),
            Ok(())
        );
    }

    #[test]
    fn an_answer_that_says_the_method_applies_where_it_does_not_is_refused() {
        let d = docs();
        assert!(check(FIT, &scenario(Case::MissingPrecondition), &refs(&d))
            .unwrap_err()
            .contains("applicab"));
    }

    #[test]
    fn an_answer_with_no_applicability_line_is_refused() {
        let d = docs();
        let text = FIT.replacen("Applicability: APPLIES\n\n", "", 1);
        assert!(check(&text, &scenario(Case::Clear), &refs(&d))
            .unwrap_err()
            .contains("Applicability"));
    }

    #[test]
    fn a_non_fit_that_says_it_needs_information_must_ask_for_it() {
        let d = docs();
        let silent = NON_FIT.replace(
            "Do the five teams agree among themselves, and who owns the schedule?",
            "I would wait.",
        );
        assert!(
            check(&silent, &scenario(Case::MissingPrecondition), &refs(&d))
                .unwrap_err()
                .contains("ask")
        );
    }

    #[test]
    fn a_fabricated_quotation_is_named_in_the_refusal_so_the_draft_can_be_corrected() {
        let d = docs();
        let bad = FIT.replace(
            "Let the Committee write to every Town",
            "Liberty is the first gift of nature to every citizen",
        );
        let why = check(&bad, &scenario(Case::Clear), &refs(&d)).unwrap_err();
        assert!(
            why.contains("quotation") && why.contains("Liberty"),
            "{why}"
        );
    }

    #[test]
    fn a_fit_answered_without_a_word_of_his_in_the_block_is_refused() {
        let d = docs();
        let none = FIT.replace(
            "- SOURCE_DIRECT: \"Let the Committee write to every Town\" [d1]\n",
            "",
        );
        assert!(check(&none, &scenario(Case::Clear), &refs(&d))
            .unwrap_err()
            .contains("SOURCE_DIRECT"));
    }

    #[test]
    fn a_helper_draft_is_checked_and_returned_with_the_scenarios_id() {
        let (h, provider) = helper(&[&serde_json::json!({"answer": FIT}).to_string()]);
        let a = draft(&h, &scenario(Case::Clear), &principle(), &docs()).unwrap();
        assert_eq!(
            (a.scenario_id.as_str(), a.text.as_str()),
            ("scenario-1", FIT.trim())
        );
        let sent = provider.sent();
        assert!(
            sent.contains("Let the Committee write to every Town") && sent.contains("[d1]"),
            "the helper is shown the evidence it may cite"
        );
    }

    #[test]
    fn a_draft_that_fails_the_check_is_corrected_and_one_that_never_passes_is_an_error() {
        let bad = FIT.replace(
            "Let the Committee write to every Town",
            "Liberty is the first gift of nature to every citizen",
        );
        let (h, provider) = helper(&[
            &serde_json::json!({"answer": bad}).to_string(),
            &serde_json::json!({"answer": FIT}).to_string(),
        ]);
        assert_eq!(
            draft(&h, &scenario(Case::Clear), &principle(), &docs())
                .unwrap()
                .text,
            FIT.trim()
        );
        assert!(provider.requests() >= 2);
        let (stuck, _) = helper(&[&serde_json::json!({"answer": bad}).to_string()]);
        assert!(draft(&stuck, &scenario(Case::Clear), &principle(), &docs()).is_err());
    }
}
