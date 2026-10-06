// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: a candidate becomes the policy only through the release gate.
//!
//! * It is released when it beats the champion on the new data's held-out
//!   tasks - the records training held out, and the variants (the same
//!   fact in other words) of the tasks it trained on - by a significant
//!   sign test, keeps every earlier release's
//!   held-out tasks, holds the anchor suite, and serves on plain brain with
//!   its own digest and the same answers, graded the same; the release is
//!   immutable, its manifest records every number, and `default` points at
//!   it.
//! * A served candidate that answers in other words than in-process is not
//!   released, however its answers are graded: each disagreement is
//!   reported with both answers.
//! * Each check that fails - or cannot be measured - blocks the release
//!   and says why.
//! * A held-out task the candidate was trained on - the same question, or
//!   a near duplicate of it, among its training records - measures nothing
//!   and is left out of the gate's suites, counted as leaked.
//! * The variants are counted: how many were measured, and how many were
//!   left out and why. A variant of a task that was held out, not trained
//!   on, is not a measure of what was learned.
//! * The next candidate trains from the champion, replaying a seeded
//!   sample of the earlier release's training records.
//! * `rollback` moves an alias back along its lineage, and refuses with no
//!   previous release.
//!
//! Models are scripted, training is a test double whose adapter file says
//! what its candidate knows, and `brain` is a stand-in script that serves
//! such an adapter (see the fixtures); no weights are needed.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{
    anchor_file, candidate, candidate_on, dataset, dataset_of, decide, gate_context, policy,
    question, released, variant_asking, variant_set, variant_set_of, Brain, FakeTrainer, ANCHOR,
    BASE_BYTES, FACTS, NOW,
};
use common::Scratch;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_model::ModelSelection;
use splinter_orchestrator::releases::{ArtifactKind, ReleasedArtifact};
use splinter_orchestrator::Context;
use splinter_pipelines::eval::{eval, EvalRequest, SuiteChoice};
use splinter_pipelines::release::{anchor, list, release, rollback, ReleaseRequest, Released};
use splinter_pipelines::train::{
    train, Candidate, TrainRequest, Tuning, DEFAULT_REPLAY_FRACTION, DEFAULT_REPLAY_SHARE,
};

fn freeze_anchor(scratch: &Scratch, ctx: &Context) {
    anchor::freeze(ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
}

#[test]
fn a_candidate_that_passes_every_check_is_released() {
    let (scratch, ctx) = gate_context("release-pass", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    let (candidate, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);

    let decided = decide(&ctx, &candidate);
    let gate = &decided.gate;
    assert!(gate.passed, "{gate:#?}");
    assert_eq!(
        decided.champion, None,
        "before any release the base is the champion"
    );

    let improvement = gate.improvement.measured.as_ref().unwrap();
    assert_eq!(
        improvement.suite.tasks, 8,
        "a tenth of {FACTS}, grown to the units a paired test needs, held out"
    );
    assert_eq!(improvement.comparison.candidate_wins, 8);
    assert_eq!(improvement.sign_test.discordant, 8);
    assert!((improvement.sign_test.p_value - 1.0 / 256.0).abs() < 1e-12);
    assert_eq!(improvement.alpha, 0.05);
    let retention = gate.retention.measured.as_ref().unwrap();
    assert!(
        retention.suites.is_empty() && gate.retention.passed,
        "nothing to retain yet"
    );
    let anchor = gate.anchor.measured.as_ref().unwrap();
    assert_eq!((anchor.version, anchor.drop), (1, Some(0.0)));
    assert_eq!(
        anchor.drop_interval.map(|i| (i.low, i.high)),
        Some((0.0, 0.0))
    );
    let serve = gate.serve.measured.as_ref().unwrap();
    assert_eq!(serve.served_digest, candidate.adapter_digest);
    assert!(
        serve.startup_line.starts_with("brain serve: "),
        "{}",
        serve.startup_line
    );
    assert_eq!((serve.sampled, serve.agreed), (8, 8));

    // The release: its id is its manifest's digest, and the manifest says
    // what it is and why it was released.
    let id = decided.release.clone().unwrap();
    let store = ctx.releases();
    let stored = store.get(&id).unwrap();
    assert_eq!(
        id.0,
        Digest::of(&splinter_core::digest::canonical_json(&stored.manifest).unwrap()),
        "the id is the digest of the manifest"
    );
    let manifest = &stored.manifest;
    let ReleasedArtifact::Adapter {
        base_model,
        base_digest,
        adapter_digest,
        ..
    } = &manifest.artifact
    else {
        panic!("a trained candidate is released as an adapter");
    };
    assert_eq!(*base_digest, Digest::sha256_of(BASE_BYTES));
    assert_eq!(base_model, "Qwen/Qwen3-0.6B");
    assert_eq!(adapter_digest.as_str(), candidate.adapter_digest);
    assert_eq!(manifest.artifact.kind(), ArtifactKind::Adapter);
    assert_eq!(manifest.parent, None);
    assert_eq!(manifest.candidate, candidate.candidate);
    assert_eq!(manifest.datasets, candidate.datasets);
    assert_eq!(manifest.training.record["trainer"], "fake");

    // The experience database traces the release back through the run that
    // made it to the dataset it read, which pinned the database as it was.
    let trace = ctx.workspace().trace_release(&id.0).unwrap().unwrap();
    assert_eq!(trace.candidates, vec![candidate.candidate.clone()]);
    assert_eq!(
        trace.datasets,
        candidate
            .datasets
            .iter()
            .map(|d| d.0.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(manifest.gate.llm(), Some(gate));
    assert_eq!(manifest.created_at, NOW);
    assert_eq!(
        stored.artifact, candidate.adapter,
        "the release names the file its candidate was kept as: one file, not a copy"
    );

    // `default` points at it, and the policy is now base plus its adapter.
    assert_eq!(store.alias("default").unwrap(), Some(id.clone()));
    let ModelSelection::Local(weights) = ctx.selection(&policy()).unwrap() else {
        panic!("the policy is local");
    };
    assert_eq!(weights.adapter.as_deref(), Some(stored.artifact.as_path()));

    // Immutable: the same release is never written twice, and its adapter file
    // is read-only.
    let again = store.put(manifest).unwrap();
    assert_eq!(
        again.id, id,
        "putting the same manifest again is the same release"
    );
    assert!(std::fs::metadata(&stored.artifact)
        .unwrap()
        .permissions()
        .readonly());
}

/// Releasing a candidate is idempotent: asking again for one that is already
/// released changes nothing, and a release that was made official but whose
/// alias never moved (the process died between the two) is completed by the
/// next ask, without grading the candidate again or storing a second release.
#[test]
fn releasing_a_candidate_again_completes_or_repeats_but_never_duplicates() {
    let (scratch, ctx) = gate_context("release-idempotent", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    let (candidate, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let first = decide(&ctx, &candidate);
    let id = first.release.clone().unwrap();
    let store = ctx.releases();

    let again = decide(&ctx, &candidate);
    assert_eq!(again.release, Some(id.clone()), "the same release");
    assert_eq!(
        again.gate, first.gate,
        "the recorded gate, not a new grading"
    );
    assert_eq!(store.list().unwrap(), vec![id.clone()]);
    assert_eq!(store.alias_history("default").unwrap().len(), 1);

    // The process died after the release was committed and before the alias
    // moved: no alias points anywhere.
    std::fs::remove_dir_all(ctx.root().expdb().join("signals/pointer/alias-default")).unwrap();
    assert_eq!(store.alias("default").unwrap(), None);
    let completed = decide(&ctx, &candidate);
    assert_eq!(completed.release, Some(id.clone()));
    assert_eq!(store.alias("default").unwrap(), Some(id.clone()));
    assert_eq!(store.list().unwrap(), vec![id]);
}

/// One blocked release: the reason the gate gives, and nothing released.
fn blocked(ctx: &Context, candidate: &Candidate) -> Released {
    let decided = decide(ctx, candidate);
    assert!(!decided.gate.passed, "{:#?}", decided.gate);
    assert_eq!(decided.release, None);
    assert_eq!(
        ctx.releases().alias("default").unwrap(),
        decided.champion,
        "a blocked release moves no alias"
    );
    decided
}

#[test]
fn each_failing_check_blocks_the_release_and_says_why() {
    // No improvement: the candidate knows no more than the base, so no
    // held-out task is discordant and the sign test is not significant.
    let (scratch, ctx) = gate_context("release-no-gain", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    let (same, _) = candidate(&ctx, "alpha", &[ANCHOR]);
    let decided = blocked(&ctx, &same);
    let gate = &decided.gate;
    assert!(!gate.improvement.passed);
    let improvement = gate.improvement.measured.as_ref().unwrap();
    assert_eq!(improvement.sign_test.discordant, 0);
    assert_eq!(improvement.comparison.both_wrong, 8, "ties are counted");
    let why = gate.improvement.reason.as_deref().unwrap();
    assert!(why.contains("no significant improvement"), "{why}");
    assert!(
        gate.retention.passed && gate.anchor.passed && gate.serve.passed,
        "{gate:#?}"
    );

    // Anchor regression: it learned the new facts and forgot the anchor.
    let (scratch, ctx) = gate_context("release-anchor", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    let (forgetful, _) = candidate(&ctx, "alpha", &["alpha"]);
    let gate = blocked(&ctx, &forgetful).gate;
    assert!(gate.improvement.passed && gate.serve.passed, "{gate:#?}");
    assert!(!gate.anchor.passed);
    let measured = gate.anchor.measured.as_ref().unwrap();
    assert_eq!(measured.drop, Some(1.0));
    // Four items, all lost: the interval around the drop is the drop itself.
    // Every one of four items discordant: about fifteen thousand would be
    // needed to see two points at that discordance.
    assert_eq!(measured.items_to_see_two_points, Some(15450));
    let interval = measured.drop_interval.unwrap();
    assert_eq!((interval.low, interval.high), (1.0, 1.0));
    assert!(gate.anchor.reason.as_deref().unwrap().contains("anchor"));

    // No anchor suite frozen: unmeasured, so blocked.
    let (_scratch, ctx) = gate_context("release-no-anchor", Brain::Honest);
    let (good, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let gate = blocked(&ctx, &good).gate;
    assert!(gate.anchor.measured.is_none());
    assert!(gate
        .anchor
        .reason
        .as_deref()
        .unwrap()
        .contains("not measured"));

    // Served under another digest than the candidate's.
    let (scratch, ctx) = gate_context("release-digest", Brain::WrongDigest);
    freeze_anchor(&scratch, &ctx);
    let (good, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let gate = blocked(&ctx, &good).gate;
    assert!(gate.improvement.passed && gate.anchor.passed, "{gate:#?}");
    let serve = gate.serve.measured.as_ref().unwrap();
    assert_ne!(serve.served_digest, serve.expected_digest);
    assert_eq!(serve.sampled, 0, "a wrong adapter is not asked anything");
    assert!(gate
        .serve
        .reason
        .as_deref()
        .unwrap()
        .contains("reported adapter"));

    // No brain binary: the serve check is unmeasured, which blocks.
    let (scratch, ctx) = gate_context("release-no-brain", Brain::Missing);
    freeze_anchor(&scratch, &ctx);
    let (good, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let gate = blocked(&ctx, &good).gate;
    assert!(gate.improvement.passed && gate.anchor.passed, "{gate:#?}");
    assert!(gate.serve.measured.is_none());
    let why = gate.serve.reason.as_deref().unwrap();
    assert!(
        why.contains("not measured") && why.contains("no brain binary"),
        "{why}"
    );
}

#[test]
fn a_served_candidate_that_says_the_same_in_other_words_is_released() {
    // A server that samples, or sums in another order, words one answer
    // otherwise; what serving must keep is what the answer says.
    let (scratch, ctx) = gate_context("release-divergent", Brain::Divergent);
    freeze_anchor(&scratch, &ctx);
    let (good, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);

    let decided = decide(&ctx, &good);
    let gate = &decided.gate;
    assert!(gate.passed && decided.release.is_some(), "{gate:#?}");
    let serve = gate.serve.measured.as_ref().unwrap();
    assert_eq!(serve.served_digest, good.adapter_digest);
    assert_eq!((serve.sampled, serve.agreed), (8, 8), "{serve:#?}");
}

#[test]
fn a_server_that_dies_while_asked_is_not_measured_and_says_what_it_wrote() {
    let (scratch, ctx) = gate_context("release-crashing", Brain::Crashing);
    freeze_anchor(&scratch, &ctx);
    let (good, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);

    let decided = decide(&ctx, &good);
    let gate = &decided.gate;
    assert!(!gate.serve.passed && decided.release.is_none(), "{gate:#?}");
    assert!(gate.serve.measured.is_none(), "{gate:#?}");
    let why = gate.serve.reason.as_deref().unwrap();
    assert!(
        why.contains("not measured") && why.contains("wgpu error: Out of Memory"),
        "{why}"
    );
}

#[test]
fn a_served_candidate_that_answers_something_else_is_not_released() {
    let (scratch, ctx) = gate_context("release-different", Brain::Different);
    freeze_anchor(&scratch, &ctx);
    let (good, _) = candidate(&ctx, "alpha", &[ANCHOR, "alpha"]);

    let decided = decide(&ctx, &good);
    let gate = &decided.gate;
    assert!(
        gate.improvement.passed && gate.retention.passed && gate.anchor.passed,
        "{gate:#?}"
    );
    assert!(!gate.serve.passed && decided.release.is_none(), "{gate:#?}");
    let serve = gate.serve.measured.as_ref().unwrap();
    assert_eq!((serve.sampled, serve.agreed), (8, 0));
    let first = &serve.disagreed[0];
    assert_eq!(
        (first.in_process.as_deref(), first.served.as_deref()),
        (Some("alpha-52"), Some("The weather is mild in spring."))
    );
    let why = gate.serve.reason.as_deref().unwrap();
    assert!(why.contains("answered differently"), "{why}");
}

#[test]
fn a_held_out_task_trained_on_is_left_out_of_the_gate() {
    let (scratch, ctx) = gate_context("release-leak", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    // The newest fact is held out, and a copy of it is trained on first.
    let mut facts = vec![FACTS - 1];
    facts.extend(0..FACTS);
    let leaky = dataset_of(&ctx, "alpha", &facts);
    let candidate = train(
        &ctx,
        &TrainRequest {
            datasets: vec![leaky.to_string()],
            rehearsal: None,
            from: policy(),
            replay_fraction: DEFAULT_REPLAY_FRACTION,
            steps: Some(1),
            rank: 4,
            beta: None,
            tuning: Tuning::default(),
        },
        &FakeTrainer::knowing(&[ANCHOR, "alpha"]),
        &CancelToken::new(),
    )
    .unwrap();

    let gate = decide(&ctx, &candidate).gate;
    let improvement = gate.improvement.measured.as_ref().unwrap();
    assert_eq!(
        improvement.suite.excluded.get("leaked"),
        Some(&1),
        "{improvement:#?}"
    );
    assert_eq!(
        improvement.suite.tasks, 7,
        "eight held out, one of them leaked"
    );
    assert_eq!(improvement.comparison.candidate_wins, 7);
    assert!(gate.passed, "the rest still decides: {gate:#?}");
}

#[test]
fn a_retention_drop_beyond_the_bound_blocks_the_release() {
    let (scratch, ctx) = gate_context("release-retention", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    let first = released(&ctx, "alpha", &[ANCHOR, "alpha"]);

    // Beta learned, alpha forgotten.
    let (forgetful, _) = candidate(&ctx, "beta", &[ANCHOR, "beta"]);
    let decided = blocked(&ctx, &forgetful);
    assert_eq!(decided.champion, Some(first.clone()));
    let gate = &decided.gate;
    assert!(
        gate.improvement.passed && gate.anchor.passed && gate.serve.passed,
        "{gate:#?}"
    );
    assert!(!gate.retention.passed);
    let retention = gate.retention.measured.as_ref().unwrap();
    assert_eq!(retention.bound, 0.05);
    let [suite] = retention.suites.as_slice() else {
        panic!("one earlier release: {retention:#?}");
    };
    assert_eq!(suite.release, first);
    assert_eq!(suite.drop, Some(1.0));
    assert!(!suite.passed);
}

#[test]
fn the_next_candidate_continues_the_champion_and_replays_its_data() {
    let (scratch, ctx) = gate_context("release-continue", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    let first = released(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let champion = ctx.releases().get(&first).unwrap();

    let (next, trainer) = candidate(&ctx, "beta", &[ANCHOR, "alpha", "beta"]);
    let plans = trainer.plans.lock().unwrap();
    let plan = &plans[0];
    assert_eq!(plan.parent, Some(first.clone()));
    assert_eq!(
        plan.continue_from.as_deref(),
        Some(champion.artifact.as_path()),
        "the champion's adapter is continued, not the base"
    );
    // A quarter of alpha's 52 trained-on records; never one it held out.
    let replay = next.replay.as_ref().unwrap();
    assert_eq!(replay.fraction, DEFAULT_REPLAY_FRACTION);
    assert_eq!(replay.records, 13);
    // The replay is a fixed share of the draws, not a plain union: a large
    // replay set would otherwise take most steps from the new data.
    assert_eq!(plan.replay_share, Some(DEFAULT_REPLAY_SHARE));
    assert!(plan.replay_file.is_some());
    assert_eq!(replay.sources[0].release, first);
    assert_eq!(
        (replay.sources[0].available, replay.sources[0].sampled),
        (52, 13)
    );
    let file = std::fs::read_to_string(
        ctx.artifacts()
            .path(replay.digest.as_ref().unwrap())
            .expect("the replayed records are kept"),
    )
    .unwrap();
    assert_eq!(replay.digest, Some(Digest::of(file.as_bytes())));
    assert_eq!(file.lines().count(), 13);
    assert!(file.lines().all(|l| l.contains("alpha-")), "{file}");
    for held_out in 52..FACTS {
        assert!(
            !file.contains(&question("alpha", held_out)),
            "{held_out} was held out"
        );
    }
    drop(plans);

    // It keeps alpha, so it is released on top of the first.
    let decided = decide(&ctx, &next);
    assert!(decided.gate.passed, "{:#?}", decided.gate);
    let retention = decided.gate.retention.measured.as_ref().unwrap();
    assert_eq!(retention.suites.len(), 1);
    assert_eq!(retention.suites[0].drop, Some(0.0));
    let second = ctx
        .releases()
        .get(decided.release.as_ref().unwrap())
        .unwrap();
    assert_eq!(second.manifest.parent, Some(first));
    assert_eq!(second.manifest.replay.as_ref(), Some(replay));

    // Trained from the base while a champion exists: refused, never gated.
    let data = dataset(&ctx, "gamma", FACTS);
    let base = ModelRef::Local {
        checkpoint: ctx.config().policy_base.display().to_string(),
        adapter: None,
        context_tokens: None,
    };
    let from_base = train(
        &ctx,
        &TrainRequest {
            datasets: vec![data.to_string()],
            rehearsal: None,
            from: base,
            replay_fraction: DEFAULT_REPLAY_FRACTION,
            steps: Some(1),
            rank: 4,
            beta: None,
            tuning: Tuning::default(),
        },
        &FakeTrainer::knowing(&[ANCHOR, "alpha", "beta", "gamma"]),
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(
        (from_base.parent.as_ref(), from_base.replay.as_ref()),
        (None, None)
    );
    let request = ReleaseRequest::new(from_base.candidate);
    let refused = release(&ctx, &request, &CancelToken::new()).unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
    assert!(
        refused.to_string().contains("trained from the base alone"),
        "{refused}"
    );
}

#[test]
fn rollback_moves_the_alias_back_and_refuses_without_a_previous_release() {
    let (scratch, ctx) = gate_context("release-rollback", Brain::Honest);
    let refused = rollback(&ctx, "default").unwrap_err();
    assert!(
        refused.to_string().contains("points at no release"),
        "{refused}"
    );

    freeze_anchor(&scratch, &ctx);
    let first = released(&ctx, "alpha", &[ANCHOR, "alpha"]);
    let refused = rollback(&ctx, "default").unwrap_err();
    assert!(
        refused.to_string().contains("no previous release"),
        "{refused}"
    );

    let second = released(&ctx, "beta", &[ANCHOR, "alpha", "beta"]);
    let listed = list(&ctx).unwrap();
    let ids: Vec<&ReleaseId> = listed.releases.iter().map(|r| &r.id).collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&&first) && ids.contains(&&second));
    let line = listed.releases.iter().find(|r| r.id == second).unwrap();
    assert_eq!(
        (line.aliases.as_slice(), line.parent.as_ref()),
        (&["default".to_string()][..], Some(&first))
    );

    let rolled = rollback(&ctx, "default").unwrap();
    assert_eq!((rolled.from, rolled.to.clone()), (second, first.clone()));
    let store = ctx.releases();
    assert_eq!(store.alias("default").unwrap(), Some(first.clone()));
    let ModelSelection::Local(weights) = ctx.selection(&policy()).unwrap() else {
        panic!("local");
    };
    assert_eq!(weights.adapter, Some(store.get(&first).unwrap().artifact));
    assert!(
        rollback(&ctx, "default").is_err(),
        "the first release has no previous one"
    );
}

#[test]
fn eval_freezes_the_anchor_suite_and_scores_one_model_on_a_suite() {
    let (scratch, ctx) = gate_context("release-eval", Brain::Honest);
    let file = anchor_file(&scratch.0, 4);
    let freeze = |file: &std::path::Path| {
        eval(
            &ctx,
            &EvalRequest {
                model: None,
                suite: SuiteChoice::Anchor,
                freeze: vec![file.to_path_buf()],
            },
            &CancelToken::new(),
        )
        .unwrap()
        .anchor
        .unwrap()
    };
    let first = freeze(&file);
    assert_eq!((first.version, first.tasks), (1, 4));
    assert_eq!(
        freeze(&file).digest,
        first.digest,
        "the same tasks are the same version"
    );
    let other = scratch.0.join("other");
    std::fs::create_dir_all(&other).unwrap();
    let second = freeze(&anchor_file(&other, 5));
    assert_eq!((second.version, second.tasks), (2, 5));

    let (candidate, _) = candidate(&ctx, "alpha", &["alpha"]);
    let score = |suite: SuiteChoice| {
        eval(
            &ctx,
            &EvalRequest {
                model: Some(candidate.candidate.clone()),
                suite,
                freeze: Vec::new(),
            },
            &CancelToken::new(),
        )
        .unwrap()
        .scores
    };
    let held_out = score(SuiteChoice::HeldOut);
    assert_eq!((held_out[0].graded, held_out[0].accuracy), (8, Some(1.0)));
    assert!(held_out[0].missed.is_empty());
    let anchor = score(SuiteChoice::Anchor);
    assert_eq!((anchor[0].graded, anchor[0].accuracy), (5, Some(0.0)));
    // What it got wrong is named, with the answer it gave.
    assert_eq!(anchor[0].missed.len(), 5);
    assert!(anchor[0]
        .missed
        .iter()
        .all(|m| m.kind == "recall" && m.answer.as_deref() == Some("I do not know.")));
    let refused = eval(
        &ctx,
        &EvalRequest {
            model: Some("policy:default".into()),
            suite: SuiteChoice::HeldOut,
            freeze: Vec::new(),
        },
        &CancelToken::new(),
    )
    .unwrap_err();
    assert!(
        refused.is_refusal(),
        "no release, so no held-out suite: {refused}"
    );
}

#[test]
fn variants_of_the_trained_facts_give_the_sign_test_enough_tasks() {
    let (scratch, ctx) = gate_context("release-variants", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    // Twelve facts: three held out (a quarter is the most), so a sign test
    // over the records alone can never reach significance (three wins is
    // p = 0.125).
    let (candidate, _) = candidate_on(&ctx, dataset(&ctx, "alpha", 12), &[ANCHOR, "alpha"]);
    let alone = blocked(&ctx, &candidate).gate;
    assert!(!alone.improvement.passed, "{alone:#?}");
    let measured = alone.improvement.measured.as_ref().unwrap();
    assert_eq!(measured.suite.tasks, 3);
    assert_eq!(measured.variants, None, "no variant was written");

    // Nine variants of three trained facts; a variant of a held-out fact
    // (not learned from); and a variant whose wording is a question the
    // candidate trained on.
    variant_set(&ctx, "alpha", &[0, 1, 2], 3);
    variant_set(&ctx, "alpha", &[11], 3);
    variant_set_of(
        &ctx,
        "alpha",
        vec![(3, variant_asking(&ctx, "alpha", 3, &question("alpha", 4)))],
    );

    let decided = decide(&ctx, &candidate);
    let gate = &decided.gate;
    assert!(gate.passed, "{gate:#?}");
    let improvement = gate.improvement.measured.as_ref().unwrap();
    assert_eq!(
        improvement.suite.tasks,
        3 + 9,
        "held-out records and variants"
    );
    assert_eq!(improvement.comparison.candidate_wins, 12);
    assert!(improvement.sign_test.p_value < improvement.alpha);
    // The two kinds of task are reported apart: generalisation to the
    // held-out records, recall of trained facts under paraphrase.
    let generalisation = improvement.generalisation.as_ref().unwrap();
    assert_eq!(generalisation.suite.tasks, 3);
    assert_eq!(generalisation.comparison.candidate_wins, 3);
    assert!(
        generalisation.sign_test.p_value > improvement.alpha,
        "three records alone are not significant: {generalisation:?}"
    );
    // What the tasks say beside the family-level decision: twelve wins, no
    // losses, over the held-out records and the variants together.
    let by_task = improvement.task_level.unwrap();
    assert!(by_task.p_value < 0.001, "{by_task:?}");
    assert!(by_task.low <= by_task.difference && by_task.difference <= by_task.high);
    assert!(improvement
        .generalisation
        .as_ref()
        .unwrap()
        .task_level
        .is_some());
    let recall = improvement.recall.as_ref().unwrap();
    assert_eq!(recall.suite.tasks, 9);
    assert_eq!(recall.comparison.candidate_wins, 9);
    let variants = improvement.variants.as_ref().unwrap();
    assert_eq!(variants.tasks, 9, "{variants:#?}");
    assert_eq!(
        variants.excluded.iter().collect::<Vec<_>>(),
        [(&"leaked".to_string(), &1)],
        "the question trained on measures nothing"
    );
    assert!(decided.release.is_some());
}

#[test]
fn variants_of_what_the_candidate_did_not_learn_do_not_count_as_evidence() {
    let (scratch, ctx) = gate_context("release-variants-unlearned", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    // It learned nothing of "alpha": its variants stay wrong, like its
    // held-out records, so the extra tasks add ties, not wins.
    let (candidate, _) = candidate_on(&ctx, dataset(&ctx, "alpha", 12), &[ANCHOR]);
    variant_set(&ctx, "alpha", &[0, 1, 2], 3);

    let gate = blocked(&ctx, &candidate).gate;
    let improvement = gate.improvement.measured.as_ref().unwrap();
    assert_eq!(improvement.variants.as_ref().unwrap().tasks, 9);
    assert_eq!(improvement.sign_test.discordant, 0);
    assert_eq!(improvement.recall.as_ref().unwrap().comparison.paired, 9);
    assert_eq!(
        improvement
            .generalisation
            .as_ref()
            .unwrap()
            .comparison
            .paired,
        3
    );
    assert!(!gate.improvement.passed);
}
