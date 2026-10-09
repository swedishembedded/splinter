// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates: proposals ruled on by code, every ruling kept, a later claim that
//! contradicts an earlier one about the same thing superseding it, a fact
//! said again reinforcing the live claim instead of being refused, and a
//! judge deciding the pairs when one is named.

use serde_json::json;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::runs;
use splinter_pipelines::claims::{extract, ledger, GateRequest};

use crate::extract::{port_claim, request, taken};
use crate::fixtures::{deploy, port_changed, port_correction, world, Outcome, World, WHERE};

fn gate(
    ctx: &splinter_orchestrator::Context,
    set: &splinter_core::digest::Digest,
) -> Result<splinter_pipelines::claims::ClaimsGated, splinter_orchestrator::OrchestratorError> {
    splinter_pipelines::claims::gate(ctx, &GateRequest::new(set))
}

fn reply(claims: Vec<serde_json::Value>) -> String {
    json!({ "claims": claims }).to_string()
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

/// Monday's reply holds a good claim, one with an invented port, and one
/// resting on the agent's own sentence; Tuesday's changes the answer.
fn script(prompt: &str) -> String {
    if prompt.contains("March move") {
        let mut invented = port_claim();
        invented["statement"] = json!("The Tessera dashboard listens on port 9091.");
        let mut agent_words = port_claim();
        agent_words["quotes"] = json!([{"step": 2, "quote": "listens on port 8080"}]);
        reply(vec![port_claim(), invented, agent_words])
    } else if prompt.contains("TLS change") {
        reply(vec![json!({
            "kind": "fact",
            "statement": "The Tessera dashboard listens on port 9443.",
            "question": WHERE,
            "quotes": [{"step": 3, "quote": "The Tessera dashboard listens on port 9443"}],
            "subject": "world",
        })])
    } else if prompt.contains("Brindle") {
        reply(vec![deploy_claim()])
    } else {
        reply(vec![])
    }
}

fn extracted(
    w: &World,
    trajectories: &[(&str, &atif::Trajectory)],
) -> anyhow::Result<splinter_core::digest::Digest> {
    let mut paths = Vec::new();
    for (name, trajectory) in trajectories {
        paths.push(w.write(name, trajectory)?);
    }
    let sessions = taken(w, &paths)?;
    let extractor = ModelRef::policy_default();
    Ok(extract(&w.ctx, &request(&sessions, &extractor))?.claim_set)
}

#[test]
fn proposals_are_ruled_on_by_code_and_every_refusal_is_reported_with_its_reason() -> Outcome {
    let w = world(script)?;
    let set = extracted(
        &w,
        &[
            ("a.atif.json", &port_correction()),
            ("b.atif.json", &deploy()),
        ],
    )?;

    let done = runs::record(&w.ctx, "claims gate", &set, |_| gate(&w.ctx, &set))?;
    let report = &done.report;
    assert_eq!(report.proposals, 4);
    assert_eq!(
        (report.ruled, report.admitted, report.already_ruled),
        (4, 2, 0)
    );
    assert_eq!(
        report.refused.get("unsupported_term"),
        Some(&1),
        "{report:#?}"
    );
    assert_eq!(
        report.refused.get("assistant_evidence"),
        Some(&1),
        "{report:#?}"
    );
    assert_eq!(report.rulings.len(), 4, "nothing is silently dropped");
    let invented = report
        .rulings
        .iter()
        .find(|r| r.statement.contains("9091"))
        .map(|r| r.reason.clone());
    assert!(invented.flatten().is_some_and(|r| r.contains("9091")));
    assert!(runs::list(&w.ctx)?
        .runs
        .iter()
        .any(|r| r.command == "claims gate"));

    // Ruling the same set again rules on nothing.
    let again = gate(&w.ctx, &set)?;
    assert_eq!((again.ruled, again.already_ruled), (0, 4));
    assert_eq!(w.ctx.claims().entries()?.len(), 4);
    Ok(())
}

#[test]
fn a_later_claim_supersedes_an_earlier_one_and_a_repeat_is_reinforced() -> Outcome {
    let w = world(script)?;
    let monday = extracted(&w, &[("a.atif.json", &port_correction())])?;
    gate(&w.ctx, &monday)?;
    let before = ledger(&w.ctx)?;
    assert_eq!(before.live.len(), 1);
    let old = before.live[0].claim.clone();

    let tuesday_session = port_changed();
    let tuesday = extracted(&w, &[("b.atif.json", &tuesday_session)])?;
    let report = gate(&w.ctx, &tuesday)?;
    assert_eq!((report.admitted, report.superseded), (1, 1), "{report:#?}");
    assert_eq!(report.rulings[0].supersedes, vec![old.clone()]);

    let now = ledger(&w.ctx)?;
    assert_eq!(now.live.len(), 1);
    assert!(now.live[0].statement.contains("9443"));
    assert_eq!(
        now.superseded.len(),
        1,
        "the earlier claim stays, with the relation"
    );
    assert_eq!(now.superseded[0].claim, old);
    assert_eq!(now.superseded[0].by, now.live[0].claim);
    assert_eq!(now.refused.len(), 2, "monday's refusals stay too");

    // The same thing said in another session is not new, and not refused:
    // the live claim is reinforced.
    let mut wednesday_session = port_changed();
    wednesday_session.session_id = Some("wednesday".into());
    let wednesday = extracted(&w, &[("c.atif.json", &wednesday_session)])?;
    let repeat = gate(&w.ctx, &wednesday)?;
    assert_eq!(
        (
            repeat.admitted,
            repeat.reinforced,
            repeat.refused.get("duplicate")
        ),
        (0, 1, None),
        "{repeat:#?}"
    );
    assert_eq!(repeat.rulings[0].outcome, "reinforced");
    assert_eq!(
        repeat.rulings[0].reinforces.as_ref(),
        Some(&now.live[0].claim)
    );
    let after = ledger(&w.ctx)?;
    assert_eq!(after.live.len(), 1);
    assert_eq!(after.live[0].reinforced, 1, "the ledger counts it");
    Ok(())
}

/// A judge that calls every pair separate.
fn separating(prompt: &str) -> String {
    if prompt.contains("`earlier` and a `later` claim") {
        json!({"verdict": "separate", "reason": "two aspects"}).to_string()
    } else {
        script(prompt)
    }
}

#[test]
fn a_named_judge_decides_the_pairs() -> Outcome {
    let w = world(separating)?;
    let monday = extracted(&w, &[("a.atif.json", &port_correction())])?;
    gate(&w.ctx, &monday)?;
    let tuesday = extracted(&w, &[("b.atif.json", &port_changed())])?;
    let judge = ModelRef::policy_default();
    let report = splinter_pipelines::claims::gate(
        &w.ctx,
        &GateRequest {
            judge: Some(&judge),
            ..GateRequest::new(&tuesday)
        },
    )?;
    assert_eq!((report.admitted, report.superseded), (1, 0), "{report:#?}");
    assert_eq!(ledger(&w.ctx)?.live.len(), 2, "the judge kept both live");
    Ok(())
}
