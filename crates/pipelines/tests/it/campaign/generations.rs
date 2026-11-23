// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate and
// rollback, for its clients. If your team needs expertise in self-improving
// training loops you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec of `splinter campaign`'s controller, against a scripted brain service
//! and scripted examiner and practice (`super::world`):
//!
//! * a generation is rollout, ingest, curate, train, evaluate, gate and
//!   promote, each recorded in the ledger before the next starts;
//! * killed at a random point mid-rollout or mid-training, a campaign resumes
//!   to the same dataset manifest hashes and the same ledger records, and
//!   brain trains each candidate once (its job reused by `resume_key`);
//! * a stage delivered again is a no-op;
//! * a forged or mismatched verdict, or retention not shown non-inferior,
//!   rejects the candidate and the champion stays;
//! * a rollback moves the champion back with its reason, appended to its
//!   history;
//! * a budget stops the campaign before the stage that would exceed it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use serde_json::Value;
use sha2::Digest as _;
use splinter_eval::promotion::{Reason, Rejection, Role};
use splinter_pipelines::campaign::{MoveKind, Stopped, CAMPAIGN_UNIT, STAGES};
use splinter_pipelines::served::ServedError;
use splinter_store::stage_ledger::Commit;

use super::world::{generation_of, Doubles, Rig, Says, Spoil, GOOD};

/// A retention bound below the -5 point margin.
const FORGETS: Says = Says {
    dev: [0.05, 0.25],
    retention_lower: -0.08,
};

/// A ledger record as a reader compares it: unit, key, outputs, summary.
type Record = (String, String, Value, Value);

/// The ledger's records as a reader compares them: unit, key, outputs and
/// summary. Attempt numbers count interruptions and are left out.
fn records(commits: &[Commit]) -> Vec<Record> {
    commits
        .iter()
        .map(|c| {
            (
                c.unit.clone(),
                c.key.to_string(),
                serde_json::to_value(&c.outputs).unwrap(),
                c.summary.clone(),
            )
        })
        .collect()
}

/// The dataset each generation curated, by its manifest's digest.
fn datasets(commits: &[Commit]) -> Vec<Value> {
    commits
        .iter()
        .filter(|c| c.unit.ends_with("/curate"))
        .map(|c| c.summary["dataset"].clone())
        .collect()
}

#[test]
fn two_generations_run_every_stage_in_order_promote_one_candidate_and_keep_the_other() {
    let rig = Rig::new("campaign-two");
    let mut doubles = Doubles::new(&rig, &[GOOD, FORGETS]);
    let seed = splinter_model::seed::draw().value();
    let report = {
        let mut controller = doubles.open(&rig, rig.spec("two", 2, seed)).unwrap();
        controller.run().unwrap()
    };

    assert_eq!(report.stopped, Some(Stopped::Done));
    let units: Vec<String> = rig_commits(&rig, "two")
        .into_iter()
        .map(|c| c.unit)
        .collect();
    let mut expected = vec![CAMPAIGN_UNIT.to_string()];
    for generation in [1, 2] {
        expected.extend(STAGES.iter().map(|s| format!("g{generation:04}/{s}")));
    }
    assert_eq!(units, expected, "every stage recorded, in order");

    let [first, second] = &report.generations[..] else {
        panic!("{:?}", report.generations)
    };
    assert_eq!(first.moved.kind, MoveKind::Promote);
    assert_eq!(second.moved.kind, MoveKind::Keep);
    let reasons = &second.promotion.as_ref().unwrap().reasons;
    assert!(
        matches!(reasons[..], [Reason::Inferior { lower, margin }] if lower == -0.08 && margin == 0.05),
        "a non-inferiority failure rejects: {reasons:?}"
    );
    assert_eq!(
        generation_of(&report.champion),
        1,
        "the first candidate stays the champion"
    );
    assert_eq!(
        *doubles.practice.champions.borrow(),
        [0, 1],
        "each generation practised under its champion"
    );

    // Each candidate trained from the base, once, as a served job under its key.
    let created = rig.brain.jobs().created();
    assert_eq!(rig.brain.jobs().trained(), 2);
    assert!(created.iter().all(|p| p["resume_key"]
        .as_str()
        .is_some_and(|k| k.starts_with("two-g000"))));
    assert!(created.iter().all(|p| p["base"] == "Qwen/Qwen3-0.6B"));
    // The second dataset is curated over both generations' experience.
    let commits = rig_commits(&rig, "two");
    let sets: Vec<usize> = commits
        .iter()
        .filter(|c| c.unit.ends_with("/curate"))
        .map(|c| c.summary["sets"].as_array().unwrap().len())
        .collect();
    assert_eq!(sets, [1, 2]);
    assert_eq!(report.spent.rollouts, 16);
    assert!(report.spent.gpu_hours > 0.0);
}

fn rig_commits(rig: &Rig, name: &str) -> Vec<Commit> {
    let dir = splinter_pipelines::campaign::campaign_dir(&rig.ctx, name);
    // The ledger's pack, read without taking the campaign's lock.
    std::fs::read_to_string(dir.join("ledger.jsonl"))
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["record"] == "commit")
        .map(|mut v| {
            v.as_object_mut().unwrap().remove("record");
            serde_json::from_value(v).unwrap()
        })
        .collect()
}

/// A three-generation campaign that retires a task in every generation, so
/// that a resumed generation must draw its tasks from the rollouts before it
/// and not from its own.
fn kill_spec(rig: &Rig, seed: u64) -> splinter_pipelines::campaign::CampaignSpec {
    let mut spec = rig.spec("kill", 3, seed);
    spec.rollout.retire_after = 2;
    spec
}

/// The ledger and datasets of an uninterrupted run of a three-generation
/// campaign seeded `seed`, under the scratch name `name`.
fn uninterrupted(name: &str, seed: u64) -> (Vec<Record>, Vec<Value>) {
    let rig = Rig::new(name);
    let mut doubles = Doubles::new(&rig, &[GOOD, GOOD, GOOD]);
    doubles
        .open(&rig, kill_spec(&rig, seed))
        .unwrap()
        .run()
        .unwrap();
    let commits = rig_commits(&rig, "kill");
    (records(&commits), datasets(&commits))
}

#[test]
fn a_campaign_killed_mid_rollout_and_mid_training_resumes_to_the_same_datasets_and_ledger() {
    let seed = splinter_model::seed::draw().value();
    // The kill points are drawn from the seed, so every run of the test tries
    // other ones; printed to reproduce a failure.
    let rollout_generation = 1 + u32::try_from(seed % 3).unwrap();
    let rollout_trial = usize::try_from((seed >> 8) % 8).unwrap();
    let train_generation = 1 + u32::try_from((seed >> 16) % 3).unwrap();
    eprintln!(
        "seed {seed}: kill rollout of g{rollout_generation} after {rollout_trial} trials, \
         training of g{train_generation}"
    );
    let (reference, reference_datasets) = uninterrupted("campaign-kill", seed);

    // The same scratch path again: the same absolute paths in every record.
    let rig = Rig::new("campaign-kill");
    let mut doubles = Doubles::new(&rig, &[GOOD, GOOD, GOOD]);
    doubles
        .practice
        .kill_at
        .set(Some((rollout_generation, rollout_trial)));
    doubles
        .trainer
        .kill
        .set(Some(format!("kill-g{train_generation:04}")));
    let mut kills = 0;
    loop {
        let ran = catch_unwind(AssertUnwindSafe(|| {
            let mut controller = doubles.open(&rig, kill_spec(&rig, seed)).unwrap();
            controller.run().unwrap()
        }));
        match ran {
            Ok(report) => {
                assert_eq!(report.stopped, Some(Stopped::Done));
                break;
            }
            Err(_) => kills += 1,
        }
        assert!(kills <= 2, "two kills at most");
    }
    assert_eq!(kills, 2, "killed mid-rollout and mid-training");
    let commits = rig_commits(&rig, "kill");
    assert_eq!(
        datasets(&commits),
        reference_datasets,
        "the same dataset manifest hashes"
    );
    assert_eq!(records(&commits), reference, "the same ledger records");
    assert_eq!(
        rig.brain.jobs().trained(),
        3,
        "brain trained each candidate once: the killed one's job was reused by its resume_key"
    );
}

#[test]
fn a_stage_delivered_again_is_a_no_op() {
    let rig = Rig::new("campaign-again");
    let mut doubles = Doubles::new(&rig, &[GOOD]);
    let seed = splinter_model::seed::draw().value();
    let first = doubles
        .open(&rig, rig.spec("again", 1, seed))
        .unwrap()
        .run()
        .unwrap();
    let before = rig_commits(&rig, "again");
    let (rolled, examined) = (doubles.practice.ran.get(), doubles.exam.ran.get());

    let again = doubles
        .open(&rig, rig.spec("again", 1, seed))
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(
        records(&rig_commits(&rig, "again")),
        records(&before),
        "nothing recorded"
    );
    assert_eq!(
        (doubles.practice.ran.get(), doubles.exam.ran.get()),
        (rolled, examined)
    );
    assert_eq!(rig.brain.jobs().trained(), 1, "nothing trained again");
    assert_eq!(again.champion, first.champion);
}

#[test]
fn a_forged_or_mismatched_verdict_rejects_the_candidate_and_the_champion_stays() {
    for (spoil, expected) in [
        (Spoil::Forged, (Role::Dev, Rejection::BadSignature)),
        (
            Spoil::OtherCandidate,
            (
                Role::Dev,
                Rejection::Candidate {
                    expected: None,
                    found: Some("sha256:another".into()),
                },
            ),
        ),
    ] {
        let rig = Rig::new(&format!("campaign-spoil-{spoil:?}"));
        let mut doubles = Doubles::new(&rig, &[GOOD]);
        doubles.exam.spoil = Some((1, spoil));
        let seed = splinter_model::seed::draw().value();
        let report = doubles
            .open(&rig, rig.spec("spoil", 1, seed))
            .unwrap()
            .run()
            .unwrap();
        let generation = &report.generations[0];
        assert_eq!(generation.moved.kind, MoveKind::Keep);
        assert_eq!(
            generation_of(&report.champion),
            0,
            "the base stays the champion"
        );
        let reasons = &generation.promotion.as_ref().unwrap().reasons;
        let found = reasons
            .iter()
            .find_map(|r| match r {
                Reason::Verdict { role, rejection } => Some((*role, rejection.clone())),
                _ => None,
            })
            .unwrap();
        match (&found.1, &expected.1) {
            (Rejection::Candidate { found: f, .. }, Rejection::Candidate { found: e, .. }) => {
                assert_eq!((found.0, f), (expected.0, e));
            }
            _ => assert_eq!(found, expected),
        }
    }
}

#[test]
fn a_rollback_moves_the_champion_back_with_its_reason_and_the_next_generation_starts_from_it() {
    let rig = Rig::new("campaign-rollback");
    let mut doubles = Doubles::new(&rig, &[GOOD, GOOD]);
    let seed = splinter_model::seed::draw().value();
    {
        let mut controller = doubles.open(&rig, rig.spec("back", 1, seed)).unwrap();
        let report = controller.run().unwrap();
        assert_eq!(generation_of(&report.champion), 1);

        let refused = controller.rollback(2, "no such champion").unwrap_err();
        assert!(matches!(refused, ServedError::Config(_)), "{refused}");
        assert!(
            controller.rollback(0, " ").is_err(),
            "a rollback needs its reason"
        );
        let moved = controller.rollback(0, "regressed in production").unwrap();
        assert_eq!(
            (moved.kind, generation_of(&moved.to)),
            (MoveKind::Rollback, 0)
        );
        let again = controller.rollback(0, "regressed in production").unwrap();
        assert_eq!(
            again, moved,
            "the same rollback asked again records nothing more"
        );

        let report = controller.report().unwrap();
        assert_eq!(generation_of(&report.champion), 0);
        let kinds: Vec<MoveKind> = report.history.iter().map(|m| m.kind).collect();
        assert_eq!(
            kinds,
            [MoveKind::Promote, MoveKind::Rollback],
            "the history is appended to"
        );
        assert_eq!(report.history[1].reason, "regressed in production");
    }
    let report = doubles
        .open(&rig, rig.spec("back", 2, seed))
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(
        *doubles.practice.champions.borrow(),
        [0, 0],
        "generation 2 practised under the base"
    );
    assert_eq!(report.generations.len(), 2);
}

#[test]
fn a_budget_stops_the_campaign_before_the_stage_that_would_exceed_it_and_a_raised_one_continues() {
    let rig = Rig::new("campaign-budget");
    let mut doubles = Doubles::new(&rig, &[GOOD, GOOD]);
    let seed = splinter_model::seed::draw().value();
    let mut spec = rig.spec("budget", 2, seed);
    spec.budget.rollouts = Some(10);
    let report = doubles.open(&rig, spec.clone()).unwrap().run().unwrap();
    assert!(
        matches!(&report.stopped, Some(Stopped::Budget { limit, .. }) if limit == "rollouts"),
        "{:?}",
        report.stopped
    );
    assert_eq!(
        report.generations.len(),
        1,
        "the second rollout of 8 would exceed 10"
    );
    spec.budget.rollouts = Some(16);
    let report = doubles.open(&rig, spec).unwrap().run().unwrap();
    assert_eq!(report.stopped, Some(Stopped::Done));
    assert_eq!(report.generations.len(), 2);
}

#[test]
fn a_campaign_is_continued_as_it_began() {
    let rig = Rig::new("campaign-identity");
    let mut doubles = Doubles::new(&rig, &[GOOD]);
    let seed = splinter_model::seed::draw().value();
    doubles.open(&rig, rig.spec("same", 1, seed)).unwrap();
    let mut changed = rig.spec("same", 1, seed);
    changed.train.rank = 8;
    assert!(matches!(
        doubles.open(&rig, changed),
        Err(ServedError::Config(_))
    ));
    let reseeded = rig.spec("same", 1, seed.wrapping_add(1));
    assert!(matches!(
        doubles.open(&rig, reseeded),
        Err(ServedError::Config(_))
    ));
}

#[test]
fn a_campaign_begun_from_an_adapter_champion_practises_it_trains_from_the_base_and_rolls_back_to_it(
) {
    let rig = Rig::new("campaign-adapter-champion");
    let adapter = rig.scratch.0.join("champion.safetensors");
    std::fs::write(&adapter, b"an adapter trained before the campaign").unwrap();
    let digest = format!(
        "sha256:{:x}",
        sha2::Sha256::digest(std::fs::read(&adapter).unwrap())
    );
    let mut doubles = Doubles::new(&rig, &[GOOD, GOOD]);
    let seed = splinter_model::seed::draw().value();
    let mut spec = rig.spec("adapter", 1, seed);
    spec.champion = Some(format!("{}+{}", spec.base, adapter.display()));
    let champion = spec.champion.clone().unwrap();

    let report = doubles.open(&rig, spec.clone()).unwrap().run().unwrap();
    assert_eq!(
        *doubles.practice.models.borrow(),
        std::slice::from_ref(&champion),
        "generation 1 practised under the named champion, not the base"
    );
    let first = &report.generations[0];
    assert_eq!(first.moved.from.adapter.as_deref(), Some(digest.as_str()));
    assert_eq!(first.moved.kind, MoveKind::Promote);
    let created = rig.brain.jobs().created();
    assert!(
        created.iter().all(|p| p["base"] == "Qwen/Qwen3-0.6B"),
        "the candidate is trained from the base: {created:?}"
    );

    let moved = doubles
        .open(&rig, spec.clone())
        .unwrap()
        .rollback(0, "the adapter champion served better")
        .unwrap();
    assert_eq!(
        moved.to.model, champion,
        "generation 0 is the named champion"
    );
    assert_eq!(moved.to.adapter.as_deref(), Some(digest.as_str()));

    let mut other = spec;
    other.champion = None;
    assert!(
        matches!(doubles.open(&rig, other), Err(ServedError::Config(_))),
        "a campaign is continued from the champion it began with"
    );
    let mut wrong = rig.spec("wrong", 1, seed);
    wrong.champion = Some("local:/elsewhere/Qwen3+/a.safetensors".into());
    assert!(
        wrong.validate().is_err(),
        "the champion is an adapter on the base"
    );
}
