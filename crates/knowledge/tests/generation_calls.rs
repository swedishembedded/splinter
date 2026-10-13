// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any body of writing into verifiable training tasks, for its clients. If
// your team needs expertise in scaling synthetic data generation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the generator asks the model for each kind of task as its own
//! request, up to a stated number at once, and keeps the proposals in the
//! order of the kinds however the answers arrive. It records every request -
//! the kind, a digest of what the model was shown, how long it took and what
//! came of it - so a run can be reconstructed from the report.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use splinter_core::clock::FixedClock;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_knowledge::tasks::propose::{ProposalRequest, ProposeError, Proposed, TaskProposer};
use splinter_knowledge::tasks::{
    CallOutcome, Catalogue, GenerationPolicy, ModelTaskGenerator, Rejection, SourceText,
};
use splinter_store::sources::SourceStore;
use splinter_store::StateRoot;

const LETTER: &str = "I have received your favour of the tenth and enclose the bill of lading for the hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month if the wind holds fair.\n\nI am of opinion that the public debt is a curse upon a nation, and no generation has a right to bind another by loans it cannot repay within the term of its own life.";

/// A model that never has anything to say, and counts how many requests it
/// is answering at once.
struct Probe {
    now: AtomicUsize,
    peak: AtomicUsize,
}

#[async_trait]
impl TaskProposer for Probe {
    fn identity(&self) -> &str {
        "probe"
    }

    async fn propose(&self, _: &ProposalRequest<'_>) -> Result<Proposed, ProposeError> {
        let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(40)).await;
        self.now.fetch_sub(1, Ordering::SeqCst);
        Ok(Proposed::Declined {
            rejection: Rejection::NoReply,
            detail: "nothing to say".into(),
        })
    }
}

fn source(test: &str) -> (SourceStore, SourceText) {
    let dir = std::env::temp_dir().join(format!("splinter-calls-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = SourceStore::new(&splinter_store::workspace::Workspace::at(&StateRoot::new(
        dir,
    )));
    let captured = CapturedSource::new(
        Origin::Repository {
            path: "/letters".into(),
            revision: None,
            skipped: Vec::new(),
        },
        vec![PartContent {
            name: "letter.txt".into(),
            media_type: "text/plain".into(),
            bytes: LETTER.as_bytes().to_vec(),
        }],
        &FixedClock::new("2026-10-01T00:00:00.000Z"),
    )
    .unwrap();
    let id = store.put_source(&captured).unwrap();
    let text = SourceText::load(&store, &id, "letter.txt").unwrap();
    (store, text)
}

async fn run(
    test: &str,
    in_flight: usize,
) -> (Arc<Probe>, splinter_knowledge::tasks::GenerationReport) {
    let (store, text) = source(test);
    let probe = Arc::new(Probe {
        now: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
    });
    let catalogue = Catalogue::builtin();
    let kinds: Vec<_> = ["recall", "explain", "advise"]
        .iter()
        .map(|name| catalogue.get(name).unwrap())
        .collect();
    let policy = GenerationPolicy {
        requests_in_flight: in_flight,
        ..GenerationPolicy::default()
    };
    let report = ModelTaskGenerator::new(probe.clone(), store)
        .with_policy(policy)
        .generate(&text, &kinds)
        .await
        .unwrap();
    (probe, report)
}

#[tokio::test]
async fn requests_for_kinds_overlap_up_to_the_stated_limit() {
    let (probe, _) = run("overlap", 3).await;
    assert_eq!(probe.peak.load(Ordering::SeqCst), 3);
    let (probe, _) = run("serial", 1).await;
    assert_eq!(probe.peak.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn every_request_is_recorded_in_the_order_of_the_kinds() {
    let (_, report) = run("record", 3).await;
    let kinds: Vec<_> = report.calls.iter().map(|c| c.kind.as_str()).collect();
    assert_eq!(kinds, ["recall", "explain", "advise"]);
    for call in &report.calls {
        assert!(call.elapsed >= Duration::from_millis(40), "{call:?}");
        assert!(call.prompt_bytes > 0);
        assert_eq!(
            call.outcome,
            CallOutcome::Declined {
                rejection: Rejection::NoReply
            }
        );
    }
    let digests: std::collections::BTreeSet<_> =
        report.calls.iter().map(|c| c.prompt.clone()).collect();
    assert_eq!(digests.len(), 3, "each kind is shown its own brief");
}

#[tokio::test]
async fn a_policy_that_allows_no_request_is_refused() {
    let (store, text) = source("zero");
    let probe = Arc::new(Probe {
        now: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
    });
    let kind = Catalogue::builtin().get("recall").unwrap().clone();
    let policy = GenerationPolicy {
        requests_in_flight: 0,
        ..GenerationPolicy::default()
    };
    let refused = ModelTaskGenerator::new(probe, store)
        .with_policy(policy)
        .generate(&text, &[&kind])
        .await;
    assert!(refused.is_err());
}
