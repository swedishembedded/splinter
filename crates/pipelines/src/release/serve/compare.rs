// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases proven to run on the
// plain serving stack before they ship, for its clients. If your team
// needs expertise in model deployment, you can procure our services by
// sending an email to info@swedishembedded.com.

//! How a served run is compared with the same run in-process: by answer
//! (text, or what it says) for a task answered in one reply, by verdict for a
//! task an agent worked on.

use crate::release::meaning;
use crate::release::probe::Probe;
use splinter_orchestrator::context::Context;

/// Why the served answers cannot be compared, when the server gave no
/// answer to a task the in-process model answered: that is the server
/// failing, and calling it a disagreement would blame the served model for
/// it. Each such task is named.
pub(super) fn unanswered(
    tasks: &[String],
    served: &[Probe],
    in_process: &[Probe],
) -> Option<String> {
    let named: Vec<&str> = tasks
        .iter()
        .zip(served.iter().zip(in_process))
        .filter(|(_, (s, l))| s.answer.is_none() && l.answer.is_some())
        .map(|(task, _)| task.as_str())
        .collect();
    (!named.is_empty()).then(|| {
        format!(
            "the server gave no answer to {} of {} task(s) the in-process model answered: {}",
            named.len(),
            served.len(),
            named.join(", ")
        )
    })
}

/// Whether two probes' verdicts allow their answers to be one: equal, or
/// one of them undecided - a judge may abstain on an answer, and an answer
/// it declines to grade is compared by what it says.
fn verdicts_compatible(a: &Probe, b: &Probe) -> bool {
    a.verdict == b.verdict
        || (a.verdict.is_none() != b.verdict.is_none() && a.answer.is_some() && b.answer.is_some())
}

/// Per task, whether the served answer is the in-process one: [`alike`] as
/// text, or - the same verdict, but worded differently - saying the same
/// thing ([`meaning::says_the_same`]). The embedding model is loaded only
/// when some task is not alike as text.
pub(super) fn agreements(
    ctx: &Context,
    served: &[Probe],
    in_process: &[Probe],
) -> Result<Vec<bool>, String> {
    let mut agree: Vec<bool> = served
        .iter()
        .zip(in_process)
        .map(|(s, l)| alike(s, l))
        .collect();
    // Tasks answered on both sides with the same verdict, not yet agreed.
    let reworded: Vec<usize> = (0..agree.len())
        .filter(|&i| {
            !agree[i]
                && verdicts_compatible(&served[i], &in_process[i])
                && served[i].answer.is_some()
                && in_process[i].answer.is_some()
        })
        .collect();
    if reworded.is_empty() {
        return Ok(agree);
    }
    // The control is every task answered on both sides, so an answer must
    // be nearer its own counterpart than to the other tasks' answers.
    let answered: Vec<usize> = (0..agree.len())
        .filter(|&i| served[i].answer.is_some() && in_process[i].answer.is_some())
        .collect();
    let texts = |probes: &[Probe]| -> Vec<String> {
        answered
            .iter()
            .filter_map(|&i| probes[i].answer.clone())
            .collect()
    };
    // The judge that graded the answers is a base on the device still; the
    // embedding model loads beside nothing.
    ctx.release_bases();
    let embedder = ctx
        .embedder()
        .map_err(|e| format!("comparing what the answers say: {e}"))?;
    let embed = |texts: &[String]| {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        embedder
            .embed(&refs)
            .map_err(|e| format!("comparing what the answers say: {e}"))
    };
    let same = meaning::says_the_same(&embed(&texts(served))?, &embed(&texts(in_process))?);
    for (&task, same) in answered.iter().zip(same) {
        if reworded.contains(&task) {
            agree[task] = same;
        }
    }
    Ok(agree)
}

/// How much of the shorter of two answers the longer must begin with, in
/// percent, for a served answer to count as the in-process one. Two
/// processes decode the same weights with kernels that sum in a different
/// order (a prompt prefilled in chunks of another size, say), so a long
/// greedy continuation may part ways near its end on a near-tie between two
/// tokens; an adapter bound wrongly parts ways at the start.
const ALIKE_PREFIX_PERCENT: usize = 90;

/// Whether two probes of one task answered alike: the same verdict, and
/// final answers - every run of whitespace one space, none at either end -
/// that are equal or share a beginning of at least
/// [`ALIKE_PREFIX_PERCENT`] percent of the shorter one. Comparing verdicts
/// alone would prove nothing where both sides fail: two different wrong
/// answers grade the same.
fn alike(a: &Probe, b: &Probe) -> bool {
    let words = |answer: &Option<String>| {
        answer
            .as_deref()
            .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
    };
    if !verdicts_compatible(a, b) {
        return false;
    }
    match (words(&a.answer), words(&b.answer)) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            let shared = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            let shorter = a.chars().count().min(b.chars().count());
            shared * 100 >= shorter * ALIKE_PREFIX_PERCENT
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_alike_are_the_same_text_whitespace_aside_and_graded_the_same() {
        let probe = |answer: Option<&str>, verdict: Option<bool>| {
            Probe::graded(answer.map(str::to_string), verdict)
        };
        let wrong = Some(false);
        assert!(alike(
            &probe(Some("115200  baud\n"), wrong),
            &probe(Some(" 115200 baud"), wrong)
        ));
        assert!(
            !alike(
                &probe(Some("9600 baud"), wrong),
                &probe(Some("I do not know."), wrong)
            ),
            "two wrong answers are not the same answer"
        );
        assert!(!alike(
            &probe(Some("115200 baud"), Some(true)),
            &probe(Some("115200 baud"), Some(false))
        ));
        assert!(alike(&probe(None, None), &probe(None, None)));
        assert!(!alike(&probe(None, wrong), &probe(Some(""), wrong)));
    }

    #[test]
    fn a_long_answer_that_parts_ways_only_at_its_end_is_still_the_same_answer() {
        let probe = |answer: &str| Probe::graded(Some(answer.to_string()), Some(false));
        let in_process = "The default location for models if `brain-data-dir` is not \
                          specified is `<home>/brain`.";
        let served = "The default location for models if `brain-data-dir` is not \
                      specified is `<home>/brain/models`.";
        assert!(alike(&probe(in_process), &probe(served)));
        assert!(
            !alike(
                &probe(in_process),
                &probe("The models are kept in the current directory unless told otherwise.")
            ),
            "answers that part ways early are different answers"
        );
    }

    #[test]
    fn a_task_the_server_did_not_answer_is_an_error_and_an_abstaining_judge_is_not() {
        let probe =
            |answer: Option<&str>, verdict| Probe::graded(answer.map(str::to_string), verdict);
        let ids = ["t1".to_string(), "t2".to_string()];
        let local = [probe(Some("a"), Some(true)), probe(Some("b"), Some(false))];
        let served = [probe(Some("a"), None), probe(None, None)];
        let why = unanswered(&ids, &served, &local).unwrap();
        assert!(
            why.contains("1 of 2") && why.contains("t2") && !why.contains("t1"),
            "{why}"
        );
        assert_eq!(unanswered(&ids, &local, &local), None);
        assert!(
            alike(&served[0], &local[0]),
            "an undecided verdict compares by text"
        );
        assert!(!alike(&probe(Some("a"), Some(false)), &local[0]));
    }
}
