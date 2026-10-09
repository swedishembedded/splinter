// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Extraction: claims proposed per session with their quotes; a reply that
//! stays unusable is reported with the session it was for.

use serde_json::json;
use splinter_core::model_ref::ModelRef;
use splinter_knowledge::claims::extract::ExtractionPolicy;
use splinter_orchestrator::runs;
use splinter_pipelines::claims::{extract, ExtractRequest};
use splinter_pipelines::sessions::{intake, IntakeRequest, DEFAULT_MAX_SESSION_BYTES};

use crate::fixtures::{
    deploy, port_correction, with_secret, world, Outcome, World, API_KEY, WHERE,
};

fn reply(claims: Vec<serde_json::Value>) -> String {
    json!({ "claims": claims }).to_string()
}

pub fn port_claim() -> serde_json::Value {
    json!({
        "kind": "correction",
        "statement": "The Tessera dashboard listens on port 9090.",
        "question": WHERE,
        "quotes": [{"step": 1, "quote": "the Tessera dashboard"},
                   {"step": 3, "quote": "It listens on port 9090"}],
        "said_wrong": "listens on port 8080",
        "subject": "world",
    })
}

fn deploy_claim() -> serde_json::Value {
    json!({
        "kind": "procedure",
        "statement": "To deploy the Brindle service to staging, run `brindle deploy --env staging`.",
        "question": "How do I deploy the Brindle service to staging?",
        "quotes": [{"step": 1, "quote": "Deploy the Brindle service to staging"}],
        "observations": [{"step": 2, "quote": "deployed brindle 1.4.2"}],
        "calls": [2],
        "subject": "world",
    })
}

/// The extractor's script: a claim for each kind of session, and an
/// unusable reply for the one holding a secret.
pub fn script(prompt: &str) -> String {
    if prompt.contains("Tessera") {
        reply(vec![port_claim()])
    } else if prompt.contains("Brindle") {
        reply(vec![deploy_claim()])
    } else {
        "I could not find anything.".into()
    }
}

pub fn taken(
    w: &World,
    paths: &[std::path::PathBuf],
) -> anyhow::Result<Vec<splinter_core::source::SourceId>> {
    let report = intake(
        &w.ctx,
        &IntakeRequest {
            paths,
            max_bytes: DEFAULT_MAX_SESSION_BYTES,
        },
    )?;
    Ok(report.sessions.into_iter().map(|s| s.source).collect())
}

pub fn request<'a>(
    sessions: &'a [splinter_core::source::SourceId],
    extractor: &'a ModelRef,
) -> ExtractRequest<'a> {
    ExtractRequest {
        sessions,
        extractor,
        policy: ExtractionPolicy {
            repairs: 0,
            ..ExtractionPolicy::default()
        },
        passes: 1,
        deadline: None,
        cancel: splinter_agent::CancelToken::new(),
    }
}

#[test]
fn claims_are_proposed_per_session_with_their_quotes() -> Outcome {
    let w = world(script)?;
    let paths = [
        w.write("a.atif.json", &port_correction())?,
        w.write("b.atif.json", &deploy())?,
    ];
    let sessions = taken(&w, &paths)?;
    let extractor = ModelRef::policy_default();
    let done = runs::record(&w.ctx, "claims extract", &sessions, |_| {
        extract(&w.ctx, &request(&sessions, &extractor))
    })?;
    let report = &done.report;
    assert_eq!((report.sessions, report.proposals), (2, 2), "{report:#?}");
    assert_eq!(report.by_kind.get("correction"), Some(&1));
    assert_eq!(report.by_kind.get("procedure"), Some(&1));
    assert!(report.failed.is_empty());

    let set = w.ctx.claims().get_set(&report.claim_set)?;
    assert_eq!(set.extractor, "scripted/extractor-1");
    let first = &set.sessions[0].proposals[0];
    assert_eq!(first.quotes[1].step, 3);
    assert_eq!(first.quotes[1].text, "It listens on port 9090");
    assert!(
        w.ctx.claims().entries()?.is_empty(),
        "extraction rules on nothing"
    );
    assert!(runs::list(&w.ctx)?
        .runs
        .iter()
        .any(|r| r.command == "claims extract"));
    Ok(())
}

#[test]
fn a_reply_that_stays_unusable_is_reported_against_its_session() -> Outcome {
    let w = world(script)?;
    let paths = [
        w.write("a.atif.json", &port_correction())?,
        w.write("secret.atif.json", &with_secret())?,
    ];
    let sessions = taken(&w, &paths)?;
    let extractor = ModelRef::policy_default();
    let report = extract(&w.ctx, &request(&sessions, &extractor))?;
    assert_eq!(report.proposals, 1);
    assert_eq!(report.failed.len(), 1, "{report:#?}");
    assert_eq!(report.failed[0].session, sessions[1]);
    assert!(!report.failed[0].reason.is_empty());

    let set = w.ctx.claims().get_set(&report.claim_set)?;
    assert_eq!(
        set.sessions[1].failure.as_deref(),
        Some(report.failed[0].reason.as_str())
    );
    assert!(set.sessions[1].proposals.is_empty());
    assert!(
        !w.shown().contains(API_KEY),
        "nothing passed to the model holds the secret"
    );
    Ok(())
}

/// The second pass reads the session from the last step back.
const SECOND_PASS: &str = "from the last step back to the first";

fn tls_claim() -> serde_json::Value {
    json!({
        "kind": "fact",
        "statement": "The Tessera dashboard has TLS enabled.",
        "question": "Does the Tessera dashboard use TLS?",
        "quotes": [{"step": 3, "quote": "the March move"}],
        "subject": "world",
    })
}

/// The first pass finds two claims, the second only one of them, worded
/// differently.
fn two_readings(prompt: &str) -> String {
    if prompt.contains(SECOND_PASS) {
        let mut same = port_claim();
        same["statement"] = json!("Port 9090 is where the Tessera dashboard listens.");
        same["quotes"] = json!([{"step": 3, "quote": "port 9090"}]);
        reply(vec![same])
    } else {
        reply(vec![port_claim(), tls_claim()])
    }
}

#[test]
fn a_claim_only_one_pass_produced_is_listed_as_unconfirmed_and_not_ruled_on() -> Outcome {
    let w = world(two_readings)?;
    let paths = [w.write("a.atif.json", &port_correction())?];
    let sessions = taken(&w, &paths)?;
    let extractor = ModelRef::policy_default();
    let report = extract(
        &w.ctx,
        &ExtractRequest {
            passes: 2,
            ..request(&sessions, &extractor)
        },
    )?;
    assert_eq!(
        (report.passes, report.proposals, report.disagreements),
        (2, 1, 1),
        "{report:#?}"
    );
    assert_eq!(
        report.unconfirmed[0].statement,
        "The Tessera dashboard has TLS enabled."
    );
    let set = w.ctx.claims().get_set(&report.claim_set)?;
    assert_eq!(set.passes, 2);
    assert_eq!(set.sessions[0].proposals.len(), 1);
    assert_eq!(set.sessions[0].unconfirmed.len(), 1, "kept, never ruled on");
    assert_eq!(
        w.prompts
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .len(),
        2,
        "one call per pass"
    );

    // One pass keeps both.
    let one = extract(&w.ctx, &request(&sessions, &extractor))?;
    assert_eq!((one.proposals, one.disagreements), (2, 0));
    Ok(())
}
