// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the night also teaches the model to be selective. After the
//! supervised candidate is trained, pairs derived from tonight's live state
//! train it further by preference: hold a fact it knows when the person
//! pushes a false value, accept a statement without inventing support for
//! it, decline to invent the specifics of what it cannot know. Every pair is
//! verified by code, a fifth of each source is never trained on and is what
//! the gate measures against the champion, and a true correction must still
//! be taken.

use splinter_core::training::Regime;
use splinter_pipelines::absorb::stance::{
    StanceRequest, REFUSED_CHOSEN, REFUSED_NO_REJECTED, REFUSED_SEALED,
};
use splinter_pipelines::absorb::{absorb, Absorbed};
use splinter_pipelines::claims::{forget, ledger};

use crate::behave::{Behaviour, GUESS};
use crate::night::{serve_release, statement, Learner, Night, Outcome};
use crate::stance_world::{
    facts, pairs_in, stance_night, stance_request, PairLine, DECLINE_WRITTEN,
};

fn learner(supervised: Behaviour, preference: Behaviour) -> Learner {
    Learner {
        supervised,
        preference,
        ..Learner::default()
    }
}

fn night_of(w: &Night, learner: &Learner, pairs: usize) -> anyhow::Result<Absorbed> {
    let dir = w.sessions("s", &[0, 1, 2])?;
    Ok(absorb(&w.ctx, &stance_request(w, dir, pairs), learner)?.report)
}

fn stance_of(
    report: &Absorbed,
) -> anyhow::Result<&splinter_pipelines::absorb::stance::StanceBuilt> {
    report
        .stance
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no stance stage: {report:#?}"))
}

fn pairs(report: &Absorbed) -> anyhow::Result<Vec<PairLine>> {
    let built = stance_of(report)?;
    let path = built
        .path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no pair dataset: {built:#?}"))?;
    Ok(pairs_in(path))
}

fn is_hold(pair: &PairLine) -> bool {
    pair.prompt.len() == 3
}

fn is_accept(pair: &PairLine) -> bool {
    pair.prompt.len() == 1 && pair.prompt[0].contains("Svc")
}

fn is_decline(pair: &PairLine) -> bool {
    pair.prompt.len() == 1 && !pair.prompt[0].contains("Svc")
}

#[test]
fn a_night_derives_three_kinds_of_pairs_and_trains_them_by_preference_from_the_supervised_candidate(
) -> Outcome {
    let w = stance_night("stance-pairs", Behaviour::Pushover);
    let learner = learner(Behaviour::Pushover, Behaviour::Firm);
    let dir = w.sessions("s", &[0, 1, 2])?;
    let mut request = stance_request(&w, dir, 30);
    request.stance.beta = Some(0.25);
    let r = absorb(&w.ctx, &request, &learner)?.report;
    assert_eq!(r.stopped, None, "{r:#?}");
    let built = stance_of(&r)?;

    // Composition: supervised first, then a preference pass that starts from
    // that adapter and trains on the pairs alone.
    let plans = learner
        .plans
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?;
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].regime, Regime::Sft);
    assert_eq!(plans[1].regime, Regime::Dpo);
    let supervised = r
        .supervised
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no supervised candidate"))?;
    assert_eq!(plans[1].continue_from.as_ref(), Some(&supervised.adapter));
    assert_eq!(plans[1].datasets.len(), 1);
    assert_eq!(
        Some(&plans[1].datasets[0].id),
        built.dataset.as_ref(),
        "the pairs alone"
    );
    assert!((plans[1].beta - 0.25).abs() < f32::EPSILON);
    assert!(plans[1].rehearsal.is_none() && plans[1].replay_file.is_none());

    // The pairs, each verified by code.
    let lines = pairs(&r)?;
    let (hold, accept, decline): (Vec<_>, Vec<_>, Vec<_>) = (
        lines.iter().filter(|p| is_hold(p)).collect(),
        lines.iter().filter(|p| is_accept(p)).collect(),
        lines.iter().filter(|p| is_decline(p)).collect(),
    );
    assert!(!hold.is_empty() && !accept.is_empty() && !decline.is_empty());
    assert_eq!(
        (hold.len(), accept.len(), decline.len()),
        (built.hold.pairs, built.accept.pairs, built.decline.pairs)
    );
    let pool = facts();
    for pair in &hold {
        let fact = pool
            .iter()
            .find(|f| f.question == pair.prompt[0])
            .ok_or_else(|| anyhow::anyhow!("a hold pair about no fact: {:?}", pair.prompt))?;
        assert!(fact.known && !fact.poisoned, "{}", fact.statement);
        assert_eq!(pair.prompt[1], fact.statement, "the policy's own answer");
        assert!(pair.chosen.contains(&fact.statement), "{}", pair.chosen);
        assert!(
            !pair.rejected.contains(&fact.statement) && pair.rejected.starts_with("You are right"),
            "{}",
            pair.rejected
        );
    }
    for pair in &accept {
        assert_eq!(pair.chosen, "Noted, thank you.");
        assert!(pair.rejected.contains("Quillan"), "{}", pair.rejected);
    }
    for pair in &decline {
        assert_eq!(pair.chosen, DECLINE_WRITTEN);
        assert_eq!(pair.rejected, GUESS);
    }
    assert!(
        built.hold.refused.get(REFUSED_CHOSEN).copied().unwrap_or(0) >= 1,
        "a teacher that gives in makes no pair: {built:#?}"
    );
    assert_eq!(
        built.hold.written_rejections, 0,
        "on-policy samples adopted"
    );

    // A fifth of each source is never trained on, whatever the pairs say.
    let all: String = std::fs::read_to_string(built.path.as_ref().unwrap())?;
    assert_eq!(built.hold.held_out, 2);
    assert_eq!(built.accept.held_out, 1, "a fifth of nine sayings");
    assert_eq!(built.decline.held_out, 2);
    for held in &built.held_out {
        assert!(!all.contains(held.as_str()), "{held} was trained on");
    }

    // The final candidate is the preference one; it was judged by the gate on
    // the held-out items against the champion, and a true correction is still
    // taken: every claim is newly answered.
    let candidate = r
        .candidate
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no candidate"))?;
    assert_eq!(candidate.regime, Regime::Dpo);
    assert!(candidate.datasets.len() >= 2, "{:?}", candidate.datasets);
    let gate = r.gate.as_ref().ok_or_else(|| anyhow::anyhow!("no gate"))?;
    assert!(gate.passed, "{gate:#?}");
    assert_eq!(gate.gained.len(), 3, "true corrections are still adopted");
    let rates = r
        .stance_rates
        .ok_or_else(|| anyhow::anyhow!("no stance rates: {gate:#?}"))?;
    for line in [rates.hold, rates.accept, rates.decline] {
        let line = line.ok_or_else(|| anyhow::anyhow!("a rate is missing: {rates:#?}"))?;
        assert!(line.champion < 0.01 && line.candidate > 0.99, "{line:?}");
    }
    // The ledger shows the rates the release was let through with.
    let release = r
        .release
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no release"))?;
    let shown = ledger(&w.ctx)?.stance;
    assert_eq!(shown.len(), 1);
    assert_eq!(shown[0].release, release);
    assert!(shown[0].rates.hold.is_some() && shown[0].rates.decline.is_some());
    Ok(())
}

#[test]
fn a_hold_pair_without_an_adopting_sample_uses_a_written_capitulation_and_counts_it() -> Outcome {
    let w = stance_night("stance-written", Behaviour::Pushover);
    // A supervised model that never gives in, never invents: nothing on
    // policy to reject.
    let r = night_of(&w, &learner(Behaviour::Firm, Behaviour::Firm), 30)?;
    let built = stance_of(&r)?;
    assert!(built.hold.pairs > 0, "{built:#?}");
    assert_eq!(built.hold.written_rejections, built.hold.pairs);
    assert!(pairs(&r)?
        .iter()
        .filter(|p| is_hold(p))
        .all(|p| p.rejected.starts_with("You are right")));
    // Accepting and declining have no fallback: no sample invented, no pair.
    assert_eq!((built.accept.pairs, built.decline.pairs), (0, 0));
    assert!(
        built
            .accept
            .refused
            .get(REFUSED_NO_REJECTED)
            .copied()
            .unwrap_or(0)
            > 0
    );
    assert!(
        built
            .decline
            .refused
            .get(REFUSED_NO_REJECTED)
            .copied()
            .unwrap_or(0)
            > 0
    );
    Ok(())
}

#[test]
fn a_stance_rate_that_falls_below_the_champions_refuses_the_candidate() -> Outcome {
    // The champion holds, accepts briefly and declines; the preference pass
    // leaves a model that gives in.
    let w = stance_night("stance-fell", Behaviour::Firm);
    let r = night_of(&w, &learner(Behaviour::Pushover, Behaviour::Pushover), 30)?;
    let gate = r.gate.as_ref().ok_or_else(|| anyhow::anyhow!("no gate"))?;
    let check = gate
        .stance
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no stance check: {gate:#?}"))?;
    assert!(!check.passed && !gate.passed, "{gate:#?}");
    let reason = check.reason.as_deref().unwrap_or_default();
    assert!(
        reason.contains("hold") && reason.contains("decline"),
        "{reason}"
    );
    assert!(gate.improvement.passed && gate.retention.passed);
    assert!(r.release.is_none() && w.ctx.releases().alias("default")?.is_none());
    assert!(r
        .stopped
        .as_deref()
        .is_some_and(|s| s.contains("stays in use")));
    Ok(())
}

#[test]
fn a_stance_pass_that_stops_the_model_taking_a_true_correction_is_refused_by_improvement() -> Outcome
{
    let w = stance_night("stance-stubborn", Behaviour::Pushover);
    let r = night_of(&w, &learner(Behaviour::Pushover, Behaviour::Stubborn), 30)?;
    let gate = r.gate.as_ref().ok_or_else(|| anyhow::anyhow!("no gate"))?;
    // It holds and declines better than the champion, and answers no claim.
    assert!(gate.stance.as_ref().is_some_and(|s| s.passed), "{gate:#?}");
    assert!(
        gate.gained.is_empty() && !gate.improvement.passed,
        "{gate:#?}"
    );
    assert!(r.release.is_none());
    Ok(())
}

#[test]
fn without_stance_the_night_is_supervised_alone() -> Outcome {
    let w = stance_night("stance-off", Behaviour::Pushover);
    let learner = learner(Behaviour::Plain, Behaviour::Firm);
    let dir = w.sessions("s", &[0, 1])?;
    let request = splinter_pipelines::absorb::AbsorbRequest {
        stance: StanceRequest::off(),
        ..stance_request(&w, dir, 30)
    };
    let r = absorb(&w.ctx, &request, &learner)?.report;
    assert!(r.stance.is_none() && r.supervised.is_none() && r.stance_rates.is_none());
    let plans = learner
        .plans
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?;
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].regime, Regime::Sft);
    assert!(r.gate.as_ref().is_some_and(|g| g.stance.is_none()));
    Ok(())
}

#[test]
fn the_pairs_are_a_bounded_share_of_the_nights_records() -> Outcome {
    let w = stance_night("stance-cap", Behaviour::Pushover);
    let r = night_of(&w, &learner(Behaviour::Pushover, Behaviour::Firm), 10_000)?;
    let built = stance_of(&r)?;
    let trained = r.dataset.as_ref().map_or(0, |d| d.trained);
    assert_eq!(trained, 54);
    assert_eq!(built.cap, 27, "half of the records trained on");
    assert!(pairs(&r)?.len() <= built.cap);
    Ok(())
}

#[test]
fn a_sealed_probe_in_a_pair_is_a_hard_error_naming_the_probe() -> Outcome {
    let w = stance_night("stance-sealed", Behaviour::Pushover);
    let file = w.scratch.0.join("probes.jsonl");
    // The probe's answer shares a run of words with what the decline writer
    // says; nothing in the items themselves does.
    let probe = serde_json::json!({
        "name": "dock-probe",
        "question": "A sealed question about the dock?",
        "reference": "I have no record of that matter and cannot say what was paid",
    });
    std::fs::write(&file, format!("{probe}\n"))?;
    let dir = w.sessions("s", &[0, 1, 2])?;
    let mut request = stance_request(&w, dir, 30);
    request.sealed_probes = vec![file];
    let err = absorb(
        &w.ctx,
        &request,
        &learner(Behaviour::Pushover, Behaviour::Firm),
    )
    .err()
    .ok_or_else(|| anyhow::anyhow!("a pair that contains a probe was trained on"))?;
    assert!(err.to_string().contains("dock-probe"), "{err}");
    Ok(())
}

#[test]
fn an_item_that_is_a_sealed_probe_is_left_out_and_counted() -> Outcome {
    let w = stance_night("stance-sealed-item", Behaviour::Pushover);
    let file = w.scratch.0.join("probes.jsonl");
    let probe = serde_json::json!({ "name": "canal-probe", "question": facts()[1].question });
    std::fs::write(&file, format!("{probe}\n"))?;
    let dir = w.sessions("s", &[0, 1, 2])?;
    let mut request = stance_request(&w, dir, 30);
    request.sealed_probes = vec![file];
    let r = absorb(
        &w.ctx,
        &request,
        &learner(Behaviour::Pushover, Behaviour::Firm),
    )?
    .report;
    let built = stance_of(&r)?;
    assert_eq!(
        built.hold.refused.get(REFUSED_SEALED),
        Some(&1),
        "{built:#?}"
    );
    let seen = facts()[1].question.clone();
    assert!(!built.held_out.contains(&seen));
    assert!(pairs(&r)?.iter().all(|p| p.prompt[0] != seen));
    Ok(())
}

#[test]
fn the_pairs_are_derived_again_from_the_live_claims_each_night() -> Outcome {
    let w = stance_night("stance-rederive", Behaviour::Pushover);
    let learner = learner(Behaviour::Pushover, Behaviour::Firm);
    let dir = w.sessions("s", &[0, 1, 2])?;
    let first = absorb(&w.ctx, &stance_request(&w, dir.clone(), 30), &learner)?.report;
    let release = first
        .release
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no release: {first:#?}"))?;
    assert!(pairs(&first)?.iter().any(|p| p.prompt[0].contains("Svc0")));
    serve_release(&w.ctx, &release);

    let target = ledger(&w.ctx)?
        .live
        .into_iter()
        .find(|l| l.statement == statement(0))
        .ok_or_else(|| anyhow::anyhow!("no claim for Svc0"))?;
    forget(&w.ctx, &target.claim.to_string())?;
    let second = absorb(&w.ctx, &stance_request(&w, dir, 30), &learner)?.report;
    assert_eq!(second.stopped, None, "{second:#?}");
    let lines = pairs(&second)?;
    assert!(!lines.is_empty());
    assert!(
        lines.iter().all(|p| !p.prompt.concat().contains("Svc0")),
        "a forgotten claim is in no pair"
    );
    assert!(lines
        .iter()
        .any(|p| p.prompt[0].contains("Svc1") || p.prompt[0].contains("Svc2")));
    Ok(())
}
