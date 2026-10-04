// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in turning verified behavior of a historical figure into present-day
// situations and checked answers, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Transfer: from each verified principle a present-day scenario of a planned
//! case, and a checked answer to it.
//!
//! Two helper calls per principle, each gated in code, each result appended as
//! it finishes. A principle whose scenario or answer cannot be made to pass
//! its checks is recorded with the reason and given up on after two attempts;
//! the run goes on.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use splinter_sdk::measure::verifiers::quotation::TextIndex;

use crate::curate::Document;
use crate::domains::{self, Job};
use crate::helper::Helper;
use crate::principles::Principle;
use crate::respond::{self, Answer};
use crate::scenario::{self, Case, Scenario};

/// Attempts a principle gets before the run stops asking about it.
pub const MAX_ATTEMPTS: usize = 2;

/// What one principle came to, one line of the results file.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Result {
    pub principle_id: String,
    /// Which scenario of the principle this is; 0 is the first.
    #[serde(default)]
    pub variant: u8,
    pub case: Case,
    pub scenario: Option<Scenario>,
    pub answer: Option<Answer>,
    /// Set when the helper failed or no draft passed its checks.
    pub error: Option<String>,
}

/// The results already in `path`; none when there is no file yet.
///
/// # Errors
/// The file exists and cannot be read, or a line is not a result.
pub fn read_results(path: &Path) -> anyhow::Result<Vec<Result>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

/// The planned (principle, scenario) pairs not yet done, and not failed too
/// often. A scenario is done or given up on by itself: finishing a principle's
/// first does not finish its others.
pub fn pending<'a>(
    planned: &[(&'a Principle, Job)],
    results: &[Result],
) -> Vec<(&'a Principle, Job)> {
    planned
        .iter()
        .filter(|(p, job)| {
            let mine = results
                .iter()
                .filter(|r| r.principle_id == p.id && r.variant == job.variant);
            let done = mine.clone().any(|r| r.error.is_none());
            let failures = mine.filter(|r| r.error.is_some()).count();
            !done && failures < MAX_ATTEMPTS
        })
        .copied()
        .collect()
}

/// A scenario and an answer for one principle, or what went wrong, with the
/// scenario kept when it was made.
fn make(
    helper: &Helper,
    principle: &Principle,
    job: Job,
    docs: &Arc<Vec<Document>>,
    corpus: &Arc<TextIndex>,
) -> Result {
    let (case, variant) = (job.case, job.variant);
    let failed = |scenario: Option<Scenario>, e: anyhow::Error| Result {
        principle_id: principle.id.clone(),
        variant,
        case,
        scenario,
        answer: None,
        error: Some(format!("{e:#}")),
    };
    let scenario = match scenario::design(helper, principle, case, job.domain, variant, corpus) {
        Ok(s) => s,
        Err(e) => return failed(None, e),
    };
    match respond::draft(helper, &scenario, principle, docs) {
        Ok(answer) => Result {
            principle_id: principle.id.clone(),
            variant,
            case,
            scenario: Some(scenario),
            answer: Some(answer),
            error: None,
        },
        Err(e) => failed(Some(scenario), e),
    }
}

/// Make a scenario and an answer for every pending principle, at most
/// `limit`, appending each result as it finishes. Returns how many were asked.
///
/// # Errors
/// `out` cannot be read or written.
pub fn run_all(
    helper: &Helper,
    principles: &[Principle],
    docs: &Arc<Vec<Document>>,
    corpus: &Arc<TextIndex>,
    variants: u8,
    out: &Path,
    limit: Option<usize>,
) -> anyhow::Result<usize> {
    let ids: Vec<String> = principles.iter().map(|p| p.id.clone()).collect();
    let cases = scenario::plan(&ids);
    let planned: Vec<(&Principle, Job)> = principles
        .iter()
        .zip(&cases)
        .flat_map(|(p, (_, case))| {
            domains::jobs(&p.id, *case, variants)
                .into_iter()
                .map(move |job| (p, job))
        })
        .collect();
    let todo = pending(&planned, &read_results(out)?);
    let todo = &todo[..limit.map_or(todo.len(), |n| n.min(todo.len()))];
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)?;
    for (n, (principle, job)) in todo.iter().enumerate() {
        let result = make(helper, principle, *job, docs, corpus);
        writeln!(file, "{}", serde_json::to_string(&result)?)?;
        file.flush()?;
        eprintln!(
            "[{}/{}] {}#{} ({:?}){}",
            n + 1,
            todo.len(),
            principle.id,
            job.variant,
            job.case,
            result
                .error
                .as_deref()
                .map_or(String::new(), |e| format!(" FAILED: {e}"))
        );
    }
    Ok(todo.len())
}

/// The corpus the gates look source wording up in: every document's text.
pub fn corpus_index(docs: &[Document]) -> TextIndex {
    TextIndex::new(docs.iter().map(|d| d.body.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Period;
    use crate::principles::{Found, Status};
    use crate::testkit::{doc, helper, responding, words};

    const WORDS_OF_HIS: &str = "Let the Committee write to every Town, that the Sense of the People may be known before the Assembly meets again.";

    fn principle(n: usize) -> Principle {
        Principle {
            id: format!("P-{n}"),
            description: format!("When the centre will not act, he had the towns state the claim together, variant {n}."),
            trigger_conditions: vec!["a grievance unanswered".into()],
            expected_behavior: vec!["write to the towns".into()],
            qualifications: vec!["not when divided".into()],
            status: Status::SourceSupported,
            support: vec![Found { doc_id: "d1".into(), quote: "Let the Committee write to every Town".into(), period: Period::PreRevolution, recipient: None }],
            dropped: Vec::new(),
        }
    }

    fn docs() -> Arc<Vec<Document>> {
        Arc::new(vec![doc("d1", 1770, "A B", WORDS_OF_HIS)])
    }

    fn scenario_json(case: Case) -> String {
        let (present, missing) = if case.applies() {
            (vec!["a grievance unanswered"], vec![])
        } else {
            (
                vec!["a grievance unanswered"],
                vec!["whether the teams agree"],
            )
        };
        let (present, missing) = if case == Case::Weak {
            (present, vec!["whether the teams agree"])
        } else {
            (present, missing)
        };
        serde_json::json!({
            "domain": "software",
            "situation": format!("A company of 800 engineers is moving its services to a new platform and the five regional teams each want a different schedule while the platform group has announced one date. {}", words("ctx", 20)),
            "observations": ["There are five regional teams.", "The platform group announced one date."],
            "request": "Advise the head of engineering on getting the teams to agree.",
            "conditions_present": present,
            "conditions_missing": missing,
        })
        .to_string()
    }

    fn fit_answer() -> String {
        serde_json::json!({"answer": "Applicability: APPLIES\n\nI would have each team put its case in writing and gather them for the platform group.\n\nGrounding:\n- SOURCE_DIRECT: \"Let the Committee write to every Town\" [d1]\n- MODERN_OBSERVATION: There are five regional teams\n- PERSONA_TRANSFER: the committee method carried to the teams\n"}).to_string()
    }

    fn clear_only() -> Vec<Principle> {
        // Principles whose planned case is Clear, so one scripted pair of replies suits them all.
        (0..200)
            .map(principle)
            .filter(|p| scenario::plan(std::slice::from_ref(&p.id))[0].1 == Case::Clear)
            .take(3)
            .collect()
    }

    /// A helper that designs a clear-fit scenario when asked to and answers when asked
    /// to; for a principle whose text contains `failing`, its scenario is not JSON.
    fn working(failing: &'static str) -> (Helper, Arc<crate::testkit::Scripted>) {
        let (scenario, answer) = (scenario_json(Case::Clear), fit_answer());
        responding(move |asked| {
            if asked.contains("Design a situation") {
                if !failing.is_empty() && asked.contains(failing) {
                    "not json".to_string()
                } else {
                    scenario.clone()
                }
            } else {
                answer.clone()
            }
        })
    }

    fn out() -> std::path::PathBuf {
        tempfile::tempdir().unwrap().keep().join("transfer.jsonl")
    }

    #[test]
    fn a_principle_yields_a_scenario_of_its_planned_case_and_a_checked_answer() {
        let corpus = Arc::new(corpus_index(&docs()));
        let ps = clear_only();
        let (h, _) = working("");
        let path = out();
        assert_eq!(
            run_all(&h, &ps[..1], &docs(), &corpus, 1, &path, None).unwrap(),
            1
        );
        let r = &read_results(&path).unwrap()[0];
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(
            (r.principle_id.as_str(), r.case),
            (ps[0].id.as_str(), Case::Clear)
        );
        assert_eq!(r.scenario.as_ref().unwrap().principle_id, ps[0].id);
        assert_eq!(
            r.answer.as_ref().unwrap().scenario_id,
            r.scenario.as_ref().unwrap().id
        );
    }

    #[test]
    fn a_principle_the_helper_fails_on_is_recorded_and_the_others_are_still_made() {
        let corpus = Arc::new(corpus_index(&docs()));
        let ps = clear_only();
        // The first scenario draft is not the shape asked for, however often it is sent back.
        let (h, _) = helper(&[
            "not json",
            "not json",
            "not json",
            &scenario_json(Case::Clear),
            &fit_answer(),
        ]);
        let path = out();
        run_all(&h, &ps[..2], &docs(), &corpus, 1, &path, None).unwrap();
        let results = read_results(&path).unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results[0].error.is_some() && results[0].scenario.is_none(),
            "{:?}",
            results[0]
        );
    }

    #[test]
    fn a_finished_principle_is_not_asked_again_and_a_given_up_one_is_not_retried() {
        let ps = clear_only();
        let planned: Vec<(&Principle, Job)> = ps
            .iter()
            .map(|p| (p, domains::jobs(&p.id, Case::Clear, 1)[0]))
            .collect();
        let done = |id: &str, failed: bool| Result {
            principle_id: id.into(),
            variant: 0,
            case: Case::Clear,
            scenario: None,
            answer: None,
            error: failed.then(|| "x".into()),
        };
        let results = vec![
            done(&ps[0].id, false),
            done(&ps[1].id, true),
            done(&ps[2].id, true),
            done(&ps[2].id, true),
        ];
        let left: Vec<&str> = pending(&planned, &results)
            .iter()
            .map(|(p, _)| p.id.as_str())
            .collect();
        assert_eq!(left, [ps[1].id.as_str()]);
    }

    #[test]
    fn a_principles_first_scenario_being_done_does_not_finish_its_others() {
        let ps = clear_only();
        let planned: Vec<(&Principle, Job)> = domains::jobs(&ps[0].id, Case::Clear, 3)
            .into_iter()
            .map(|job| (&ps[0], job))
            .collect();
        let finished = |variant: u8, failed: bool| Result {
            principle_id: ps[0].id.clone(),
            variant,
            case: Case::Clear,
            scenario: None,
            answer: None,
            error: failed.then(|| "x".into()),
        };
        let left = |results: &[Result]| -> Vec<u8> {
            pending(&planned, results)
                .iter()
                .map(|(_, j)| j.variant)
                .collect()
        };
        assert_eq!(left(&[]), [0, 1, 2]);
        assert_eq!(left(&[finished(0, false)]), [1, 2]);
        assert_eq!(
            left(&[finished(0, false), finished(1, true), finished(1, true)]),
            [2],
            "a scenario given up on is not retried"
        );
    }

    #[test]
    fn nothing_is_asked_twice_across_runs() {
        let corpus = Arc::new(corpus_index(&docs()));
        let ps = clear_only();
        let (h, _) = working("");
        let path = out();
        run_all(&h, &ps, &docs(), &corpus, 1, &path, None).unwrap();
        assert_eq!(
            run_all(&h, &ps, &docs(), &corpus, 1, &path, None).unwrap(),
            0
        );
        assert_eq!(read_results(&path).unwrap().len(), ps.len());
    }

    #[test]
    fn the_limit_bounds_how_many_are_asked_in_one_run() {
        let corpus = Arc::new(corpus_index(&docs()));
        let ps = clear_only();
        let (h, _) = working("");
        let path = out();
        assert_eq!(
            run_all(&h, &ps, &docs(), &corpus, 1, &path, Some(1)).unwrap(),
            1
        );
    }
}
