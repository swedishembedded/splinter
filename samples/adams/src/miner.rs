// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in mining a historical record for behavior that each claim can be traced
// back to, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Mining principles: a helper reads a few of his documents and proposes how
//! he worked; code decides how much of that the documents bear out.
//!
//! The helper sees only documents a model may learn from, never a held-out
//! one, and is asked for rules that name when they apply, what he did and what
//! limits them, each with the words that show it. What it returns is a
//! proposal until [`crate::principles::verify`] has looked every quotation up.

use std::collections::HashSet;

use splinter_sdk::agent::schemars::JsonSchema;
use splinter_sdk::agent::typed::TypedCall;

use crate::curate::Document;
use crate::helper::Helper;
use crate::principles::{self, Principle, Proposal, Rejection};

/// Documents a helper reads in one call, and the words of each it is shown.
pub const DOCUMENTS_PER_BUNDLE: usize = 4;
pub const WORDS_PER_DOCUMENT: usize = 700;
/// Principles a helper is asked for per bundle, at most.
const MAX_PRINCIPLES: usize = 3;

/// One document as the helper is shown it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Shown {
    pub id: String,
    pub year: u16,
    pub recipient: Option<String>,
    /// The first words of the document, cut at a word boundary.
    pub text: String,
    /// Whether the document runs on past what is shown.
    pub truncated: bool,
}

/// The documents one call reads.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Bundle {
    pub id: String,
    pub documents: Vec<Shown>,
}

/// What a helper returns.
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
struct Mined {
    /// Up to three principles the documents show.
    principles: Vec<Proposal>,
}

/// Group the documents a model may learn from into bundles of at most
/// `per_bundle`, in a fixed order (by recipient, then date) so a bundle's id
/// and contents do not depend on file order. Each document is in exactly one
/// bundle, cut to `words` words at a word boundary.
pub fn bundles(
    docs: &[Document],
    allowed: &HashSet<String>,
    per_bundle: usize,
    words: usize,
) -> Vec<Bundle> {
    let mut chosen: Vec<&Document> = docs.iter().filter(|d| allowed.contains(&d.id)).collect();
    chosen.sort_by(|a, b| {
        let key = |d: &Document| {
            (
                d.recipient.clone().unwrap_or_default(),
                d.date.year,
                d.date.month,
                d.date.day,
                d.id.clone(),
            )
        };
        key(a).cmp(&key(b))
    });
    chosen
        .chunks(per_bundle.max(1))
        .map(|group| {
            let documents: Vec<Shown> = group
                .iter()
                .map(|d| {
                    let all: Vec<&str> = d.body.split_whitespace().collect();
                    Shown {
                        id: d.id.clone(),
                        year: d.date.year,
                        recipient: d.recipient.clone(),
                        text: all[..words.min(all.len())].join(" "),
                        truncated: all.len() > words,
                    }
                })
                .collect();
            let names: Vec<&str> = documents.iter().map(|d| d.id.as_str()).collect();
            Bundle {
                id: format!(
                    "bundle-{}",
                    &blake3::hash(names.join("\n").as_bytes()).to_hex()[..12]
                ),
                documents,
            }
        })
        .collect()
}

/// What the helper is told: the task, and what a principle is.
const TASK: &str = "You read documents Samuel Adams wrote and state how he worked, as principles. A principle is a rule that can be tested: when a stated condition holds, he did a stated thing, with its limits. Give at most three. Every principle must cite passages from the documents you were shown: the document's id and the words, copied exactly, at least six words long. Do not state anything the documents do not show. Say when a principle would not apply.";

/// The role the helper is given.
const ROLE: &str = "an historian of the American Revolution who reads primary sources closely and states nothing they do not show";

/// How long the helper may take, and how much it may write, for one bundle.
const OUTPUT_TOKENS: u64 = 1800;
const REPAIRS: u32 = 2;

/// The helper's proposals for one bundle.
///
/// # Errors
/// The helper could not be reached or did not return the shape asked for.
pub fn mine(helper: &Helper, bundle: &Bundle) -> anyhow::Result<Vec<Proposal>> {
    let shown: HashSet<String> = bundle.documents.iter().map(|d| d.id.clone()).collect();
    let call = TypedCall::<Mined>::new("mine_principles", TASK, ROLE, crate::helper::CALL_DEADLINE)
        .max_output_tokens(OUTPUT_TOKENS)
        .repairs(REPAIRS)
        .postcondition(move |mined| {
            if mined.principles.len() > MAX_PRINCIPLES {
                return Err(format!("give at most {MAX_PRINCIPLES} principles"));
            }
            for principle in &mined.principles {
                if principle.support.is_empty() {
                    return Err("every principle must cite at least one passage".to_string());
                }
                if let Some(bad) = principle
                    .support
                    .iter()
                    .find(|s| !shown.contains(&s.doc_id))
                {
                    return Err(format!(
                        "cite only the ids you were shown, not {:?}",
                        bad.doc_id
                    ));
                }
            }
            Ok(())
        });
    Ok(helper.call(call, bundle)?.principles)
}

/// The proposals of a bundle, each checked against the documents: the
/// principles that stand and the proposals that were rejected, with why.
///
/// # Errors
/// The helper could not be reached or did not return the shape asked for.
pub fn mine_and_verify(
    helper: &Helper,
    bundle: &Bundle,
    docs: &[Document],
    allowed: &HashSet<String>,
) -> anyhow::Result<(Vec<Principle>, Vec<Rejection>)> {
    let (mut stand, mut rejected) = (Vec::new(), Vec::new());
    for proposal in mine(helper, bundle)? {
        match principles::verify(&proposal, docs, allowed) {
            Ok(principle) => stand.push(principle),
            Err(rejection) => rejected.push(rejection),
        }
    }
    Ok((stand, rejected))
}

/// Attempts a bundle gets before the run stops asking about it.
pub const MAX_ATTEMPTS: usize = 2;

/// What one bundle came to, one line of the results file.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BundleResult {
    pub bundle_id: String,
    pub principles: Vec<Principle>,
    pub rejected: Vec<Rejection>,
    /// Set when the helper failed; the bundle is retried until it has failed
    /// [`MAX_ATTEMPTS`] times.
    pub error: Option<String>,
}

/// The results already in `path`; none when there is no file yet.
///
/// # Errors
/// The file exists and cannot be read, or a line is not a result.
pub fn read_results(path: &std::path::Path) -> anyhow::Result<Vec<BundleResult>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

/// The bundles still to ask about: not answered, and not failed too often.
pub fn pending<'a>(bundles: &'a [Bundle], results: &[BundleResult]) -> Vec<&'a Bundle> {
    bundles
        .iter()
        .filter(|b| {
            let mine = results.iter().filter(|r| r.bundle_id == b.id);
            let answered = mine.clone().any(|r| r.error.is_none());
            let failures = mine.filter(|r| r.error.is_some()).count();
            !answered && failures < MAX_ATTEMPTS
        })
        .collect()
}

/// Ask about every pending bundle, at most `limit`, appending one result per
/// bundle to `out` as it finishes so an interruption loses at most the bundle
/// in flight. A bundle the helper fails on is recorded and the run goes on.
/// Returns how many bundles were asked.
///
/// # Errors
/// `out` cannot be read or written.
pub fn mine_all(
    helper: &Helper,
    bundles: &[Bundle],
    docs: &[Document],
    allowed: &HashSet<String>,
    out: &std::path::Path,
    limit: Option<usize>,
) -> anyhow::Result<usize> {
    use std::io::Write;
    let todo = pending(bundles, &read_results(out)?);
    let todo = &todo[..limit.map_or(todo.len(), |n| n.min(todo.len()))];
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)?;
    for (n, bundle) in todo.iter().enumerate() {
        let result = match mine_and_verify(helper, bundle, docs, allowed) {
            Ok((principles, rejected)) => BundleResult {
                bundle_id: bundle.id.clone(),
                principles,
                rejected,
                error: None,
            },
            Err(e) => BundleResult {
                bundle_id: bundle.id.clone(),
                principles: Vec::new(),
                rejected: Vec::new(),
                error: Some(format!("{e:#}")),
            },
        };
        writeln!(file, "{}", serde_json::to_string(&result)?)?;
        file.flush()?;
        eprintln!(
            "[{}/{}] {}: {} principles stand, {} rejected{}",
            n + 1,
            todo.len(),
            bundle.id,
            result.principles.len(),
            result.rejected.len(),
            result
                .error
                .as_deref()
                .map_or(String::new(), |e| format!(", FAILED: {e}"))
        );
    }
    Ok(todo.len())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use splinter_sdk::agent::solve::Model;
    use splinter_sdk::agent::sven::model::{
        CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
    };

    use super::*;
    use crate::document::{Authorship, Date, Period};

    fn doc(id: &str, year: u16, recipient: &str, body: &str) -> Document {
        Document {
            schema: crate::curate::SCHEMA,
            id: id.into(),
            source_id: "cushing-1".into(),
            heading: format!("TO {}.", recipient.to_uppercase()),
            recipient: Some(recipient.into()),
            note: "[MS.]".into(),
            date: Date {
                year,
                month: None,
                day: None,
            },
            period: Period::of(year),
            authorship: Authorship::DraftInHand,
            authorship_confidence: 0.8,
            authorship_basis: "test".into(),
            temporal_holdout: false,
            body: body.into(),
        }
    }

    fn words(prefix: &str, n: usize) -> String {
        (0..n)
            .map(|i| format!("{prefix}{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn corpus(n: usize) -> (Vec<Document>, HashSet<String>) {
        let docs: Vec<Document> = (0..n)
            .map(|i| {
                doc(
                    &format!("d{i:02}"),
                    1765 + (i as u16 % 30),
                    if i % 3 == 0 {
                        "James Otis"
                    } else {
                        "Arthur Lee"
                    },
                    &words(&format!("t{i}x"), 900),
                )
            })
            .collect();
        let allowed = docs
            .iter()
            .filter(|d| d.id != "d03")
            .map(|d| d.id.clone())
            .collect();
        (docs, allowed)
    }

    /// A model that answers each request with the next reply of its script,
    /// the last one repeating, and keeps what it was sent.
    struct Scripted {
        replies: Mutex<VecDeque<String>>,
        seen: Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }
        fn model_name(&self) -> &str {
            "scripted-1"
        }
        async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
            self.seen
                .lock()
                .unwrap()
                .push(format!("{:?}", req.messages));
            let mut replies = self.replies.lock().unwrap();
            let reply = if replies.len() > 1 {
                replies.pop_front()
            } else {
                replies.front().cloned()
            }
            .unwrap_or_default();
            Ok(Box::pin(futures::stream::iter(vec![
                Ok(ResponseEvent::TextDelta(reply)),
                Ok(ResponseEvent::Done),
            ])))
        }
    }

    /// The bundle that holds document `id`: bundles are ordered by recipient.
    fn holding<'a>(made: &'a [Bundle], id: &str) -> &'a Bundle {
        made.iter()
            .find(|b| b.documents.iter().any(|d| d.id == id))
            .unwrap()
    }

    fn helper(replies: &[&str]) -> (Helper, Arc<Scripted>) {
        let provider = Arc::new(Scripted {
            replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            seen: Mutex::new(Vec::new()),
        });
        (
            Helper::with_model(Model::new(provider.clone(), "scripted")),
            provider,
        )
    }

    #[test]
    fn every_allowed_document_is_in_exactly_one_bundle_and_no_other_is() {
        let (docs, allowed) = corpus(23);
        let made = bundles(&docs, &allowed, 4, 100);
        let mut ids: Vec<&str> = made
            .iter()
            .flat_map(|b| b.documents.iter().map(|d| d.id.as_str()))
            .collect();
        ids.sort_unstable();
        let mut want: Vec<&str> = allowed.iter().map(String::as_str).collect();
        want.sort_unstable();
        assert_eq!(ids, want);
        assert!(made
            .iter()
            .all(|b| b.documents.len() <= 4 && !b.documents.is_empty()));
        assert!(!ids.contains(&"d03"), "a held-out document is never shown");
    }

    #[test]
    fn bundles_do_not_depend_on_the_order_the_documents_arrive_in() {
        let (docs, allowed) = corpus(15);
        let mut reversed = docs.clone();
        reversed.reverse();
        assert_eq!(
            bundles(&docs, &allowed, 4, 100),
            bundles(&reversed, &allowed, 4, 100)
        );
    }

    #[test]
    fn a_long_document_is_cut_at_a_word_boundary_and_a_short_one_is_whole() {
        let mut docs = vec![
            doc("long", 1770, "A B", &words("l", 500)),
            doc("short", 1770, "A B", "just a few words here"),
        ];
        docs[1].recipient = Some("A B".into());
        let allowed: HashSet<String> = ["long", "short"].iter().map(|s| s.to_string()).collect();
        let made = bundles(&docs, &allowed, 4, 50);
        let shown = |id: &str| {
            made.iter()
                .flat_map(|b| &b.documents)
                .find(|d| d.id == id)
                .unwrap()
                .text
                .clone()
        };
        assert_eq!(shown("long").split_whitespace().count(), 50);
        assert_eq!(shown("short"), "just a few words here");
    }

    #[test]
    fn a_bundle_id_names_its_documents() {
        let (docs, allowed) = corpus(9);
        let made = bundles(&docs, &allowed, 4, 100);
        let ids: HashSet<&str> = made.iter().map(|b| b.id.as_str()).collect();
        assert_eq!(ids.len(), made.len());
    }

    const REPLY: &str = r#"{"principles":[{"description":"When the ministry will not act he had the towns act together.","trigger_conditions":["a grievance unanswered"],"expected_behavior":["write to the towns"],"qualifications":["not when divided"],"support":[{"doc_id":"d00","quote":"t0x0 t0x1 t0x2 t0x3 t0x4 t0x5 t0x6"},{"doc_id":"d00","quote":"words that were never written anywhere at all"}]}]}"#;

    #[test]
    fn the_helper_is_shown_the_documents_ids_and_text_and_its_proposals_are_returned() {
        let (docs, allowed) = corpus(8);
        let made = bundles(&docs, &allowed, 4, 100);
        let (h, provider) = helper(&[REPLY]);
        let proposals = mine(&h, holding(&made, "d00")).unwrap();
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].support.len(), 2);
        let sent = provider.seen.lock().unwrap().join(" ");
        assert!(
            sent.contains("d00") && sent.contains("t0x5"),
            "the helper must see ids and text"
        );
        assert!(!sent.contains("t3x"), "a held-out document is never sent");
    }

    #[test]
    fn what_the_helper_proposes_is_checked_against_the_documents() {
        let (docs, allowed) = corpus(8);
        let made = bundles(&docs, &allowed, 4, 100);
        let (h, _) = helper(&[REPLY]);
        let (principles, rejected) =
            mine_and_verify(&h, holding(&made, "d00"), &docs, &allowed).unwrap();
        assert!(rejected.is_empty());
        assert_eq!(principles.len(), 1);
        assert_eq!(principles[0].support.len(), 1, "the real quotation stands");
        assert!(
            principles[0].dropped[0].contains("not in"),
            "the invented one is dropped with its reason"
        );
    }

    #[test]
    fn a_proposal_none_of_whose_words_are_his_comes_back_as_a_rejection() {
        let reply = r#"{"principles":[{"description":"A principle with nothing behind it.","trigger_conditions":["x"],"expected_behavior":["y"],"qualifications":[],"support":[{"doc_id":"d00","quote":"invented words that appear in no document"}]}]}"#;
        let (docs, allowed) = corpus(8);
        let made = bundles(&docs, &allowed, 4, 100);
        let (h, _) = helper(&[reply]);
        let (principles, rejected) =
            mine_and_verify(&h, holding(&made, "d00"), &docs, &allowed).unwrap();
        assert!(principles.is_empty());
        assert_eq!(rejected.len(), 1);
    }

    #[test]
    fn a_reply_that_is_not_the_shape_is_an_error_not_an_empty_success() {
        let (docs, allowed) = corpus(8);
        let made = bundles(&docs, &allowed, 4, 100);
        let (h, _) = helper(&["not json at all", "still not", "nor this", "nor this one"]);
        assert!(mine(&h, holding(&made, "d00")).is_err());
    }

    fn result(id: &str, failed: bool) -> BundleResult {
        BundleResult {
            bundle_id: id.into(),
            principles: Vec::new(),
            rejected: Vec::new(),
            error: failed.then(|| "x".to_string()),
        }
    }

    #[test]
    fn an_answered_bundle_is_not_asked_again_and_a_failed_one_is_retried_a_bounded_number_of_times()
    {
        let (docs, allowed) = corpus(12);
        let made = bundles(&docs, &allowed, 4, 50);
        let (a, b, c) = (&made[0].id, &made[1].id, &made[2].id);
        let done = vec![
            result(a, false),
            result(b, true),
            result(c, true),
            result(c, true),
        ];
        let left: Vec<&str> = pending(&made, &done)
            .iter()
            .map(|x| x.id.as_str())
            .collect();
        assert_eq!(
            left,
            [b.as_str()],
            "b failed once and is retried; c failed twice and is given up on; a is done"
        );
    }

    #[test]
    fn a_run_writes_one_line_per_bundle_resumes_and_survives_a_failing_bundle() {
        let (docs, allowed) = corpus(12);
        let made = bundles(&docs, &allowed, 4, 50);
        let out = tempfile::tempdir().unwrap().keep().join("results.jsonl");
        // The first bundle's reply is not the shape asked for, however often it is sent back.
        let (h, _) = helper(&["not json"]);
        assert_eq!(
            mine_all(&h, &made, &docs, &allowed, &out, Some(1)).unwrap(),
            1
        );
        let first = read_results(&out).unwrap();
        assert_eq!(first.len(), 1);
        assert!(
            first[0].error.is_some(),
            "the failure is recorded, not hidden"
        );
        // A later run, with a working helper, asks about the rest and retries the failed one.
        let (good, _) = helper(&[r#"{"principles":[]}"#]);
        let asked = mine_all(&good, &made, &docs, &allowed, &out, None).unwrap();
        assert_eq!(asked, made.len());
        let all = read_results(&out).unwrap();
        assert!(all.iter().filter(|r| r.error.is_none()).count() == made.len());
        assert_eq!(
            mine_all(&good, &made, &docs, &allowed, &out, None).unwrap(),
            0,
            "nothing is asked twice"
        );
    }
}
