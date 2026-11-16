// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The run's contract, on the smoke profile against a model service double:
//! a fresh run goes end to end; a second run trains nothing and repeats no
//! episode; a run killed in a stage resumes it; a corrupt checkpoint is
//! caught; a changed seed is a new run; a changed setting recomputes only
//! what reads it; and nothing is promoted without a verified Promote.

#![allow(clippy::unwrap_used)]

mod common;

use std::path::Path;
use std::sync::atomic::Ordering;

use common::{FakeBrain, FakeExaminer};
use splinter_minimind::config::{Config, Profile};
use splinter_minimind::stages::{run_all, Revisions, Run};
use splinter_sdk::bench::examine::Examiner;
use splinter_sdk::store::stage_ledger::{Commit, StageLedger};

fn revisions() -> Revisions {
    let mut r = Revisions::default();
    for name in ["splinter", "sven", "brain", "bench"] {
        r.0.insert(name.into(), "r1".into());
    }
    r
}

fn go(
    dir: &Path,
    config: Config,
    seed: u64,
    brain: &FakeBrain,
    examiner: Option<&dyn Examiner>,
) -> anyhow::Result<String> {
    let mut out = Vec::new();
    let result = {
        let mut run = Run::open(dir, config, seed, revisions(), brain, examiner, &mut out)?;
        run_all(&mut run)
    };
    let shown = String::from_utf8(out).unwrap();
    result.map(|()| shown)
}

fn commits(dir: &Path) -> Vec<Commit> {
    StageLedger::open(dir).unwrap().commits().unwrap()
}

fn commit<'a>(commits: &'a [Commit], unit: &str) -> &'a Commit {
    commits
        .iter()
        .rev()
        .find(|c| c.unit == unit)
        .unwrap_or_else(|| panic!("no commit of {unit}"))
}

fn begins(dir: &Path) -> usize {
    std::fs::read_to_string(dir.join("ledger.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| l.contains("\"record\":\"begin\""))
        .count()
}

fn smoke() -> Config {
    Config::of(Profile::Smoke)
}

// seed: the specs compare runs of one seed with each other; any seed works.
const SEED: u64 = 17;

#[test]
fn a_fresh_run_goes_end_to_end_and_a_second_run_trains_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let brain = FakeBrain::new(dir.path());
    let examiner = FakeExaminer {
        verdict: "Promote",
        verified: true,
    };
    let shown = go(dir.path(), smoke(), SEED, &brain, Some(&examiner)).unwrap();
    let all = commits(dir.path());
    assert_eq!(
        commit(&all, "demonstrate").summary["accepted"],
        8,
        "the expert solves every training instance through sven"
    );
    assert_eq!(commit(&all, "probe/ouro/init").summary["passed"], 0);
    assert_eq!(commit(&all, "probe/ouro/sft").summary["passed"], 4);
    assert!(commit(&all, "ouro/experience/examine").summary["promotes"]
        .as_bool()
        .unwrap());
    assert!(shown.contains("probe ouro/sft: 4/4"), "{shown}");
    assert!(dir.path().join("report.md").exists());
    let trained = brain.trained();
    let started = begins(dir.path());

    let again = go(dir.path(), smoke(), SEED, &brain, Some(&examiner)).unwrap();
    assert_eq!(brain.trained(), trained, "a finished run trains nothing");
    assert_eq!(
        begins(dir.path()),
        started,
        "and starts no unit, no episode among them"
    );
    assert!(
        again.contains("probe ouro/sft: 4/4"),
        "the probes are shown from the ledger: {again}"
    );
}

#[test]
fn a_run_killed_in_fine_tuning_resumes_it_and_reuses_what_came_before() {
    let dir = tempfile::tempdir().unwrap();
    let brain = FakeBrain::new(dir.path());
    *brain.fail_once.lock().unwrap() = Some("ouro-sft".into());
    let err = go(dir.path(), smoke(), SEED, &brain, None).unwrap_err();
    assert!(format!("{err:#}").contains("killed during"), "{err:#}");
    let before = brain.trained();
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let all = commits(dir.path());
    assert_eq!(
        commit(&all, "ouro/sft").attempt,
        2,
        "the fine-tune resumed under a new attempt"
    );
    assert_eq!(
        commit(&all, "ouro/pretrain").attempt,
        1,
        "pretraining was not redone"
    );
    assert!(brain.trained() > before);
}

#[test]
fn an_episode_interrupted_by_a_failing_server_runs_again_under_a_new_attempt_and_counts_once() {
    let dir = tempfile::tempdir().unwrap();
    let brain = FakeBrain::new(dir.path());
    brain.down.store(true, Ordering::SeqCst);
    *brain.down_for.lock().unwrap() = Some("sft".into());
    assert!(
        go(dir.path(), smoke(), SEED, &brain, None).is_err(),
        "a model server that is down stops the run"
    );
    brain.down.store(false, Ordering::SeqCst);
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let all = commits(dir.path());
    let probe = commit(&all, "probe/ouro/sft");
    assert_eq!(probe.summary["total"], 4, "every episode counted once");
    let retried: Vec<&Commit> = all
        .iter()
        .filter(|c| c.unit.starts_with("probe/ouro/sft/") && c.attempt > 1)
        .collect();
    assert_eq!(
        retried.len(),
        1,
        "the interrupted episode ran again under a new attempt id"
    );
}

#[test]
fn a_corrupt_checkpoint_is_caught_and_only_it_is_recomputed() {
    let dir = tempfile::tempdir().unwrap();
    let brain = FakeBrain::new(dir.path());
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let trained = brain.trained();
    std::fs::write(
        dir.path().join("checkpoints/ouro/sft/weights.bin"),
        "flipped",
    )
    .unwrap();
    let shown = go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    assert!(shown.contains("corrupt"), "{shown}");
    assert_eq!(
        brain.trained(),
        trained + 1,
        "the corrupt checkpoint, and nothing else, is trained again"
    );
}

#[test]
fn a_changed_seed_is_a_new_run_and_a_changed_setting_recomputes_only_what_reads_it() {
    let dir = tempfile::tempdir().unwrap();
    let brain = FakeBrain::new(dir.path());
    go(dir.path(), smoke(), SEED, &brain, None).unwrap();
    let trained = brain.trained();

    let mut longer = smoke();
    longer.sft.steps += 5;
    go(dir.path(), longer, SEED, &brain, None).unwrap();
    let all = commits(dir.path());
    assert_eq!(
        brain.trained(),
        trained + 2,
        "the fine-tune and the training after it, nothing before"
    );
    assert_eq!(commit(&all, "ouro/pretrain").attempt, 1);
    let demonstrations = all.iter().filter(|c| c.unit == "demonstrate").count();
    assert_eq!(demonstrations, 1, "the demonstrations are reused");

    let other = tempfile::tempdir().unwrap();
    let fresh = FakeBrain::new(other.path());
    go(other.path(), smoke(), SEED + 1, &fresh, None).unwrap();
    let a = commit(&commits(dir.path()), "world").key.clone();
    let b = commit(&commits(other.path()), "world").key.clone();
    assert_ne!(a, b, "a changed seed is a new identity");
}

#[test]
fn no_promotion_without_a_verified_promote() {
    for (verdict, verified) in [("Reject", true), ("Inconclusive", true), ("Promote", false)] {
        let dir = tempfile::tempdir().unwrap();
        let brain = FakeBrain::new(dir.path());
        let examiner = FakeExaminer { verdict, verified };
        go(dir.path(), smoke(), SEED, &brain, Some(&examiner)).unwrap();
        let all = commits(dir.path());
        assert!(
            !commit(&all, "ouro/experience/examine").summary["promotes"]
                .as_bool()
                .unwrap(),
            "{verdict} {verified}"
        );
        let report = std::fs::read_to_string(dir.path().join("report.md")).unwrap();
        assert!(report.contains("not promoted"), "{report}");
    }
}

#[test]
fn a_fetched_corpus_is_checked_against_its_pin_and_joins_the_world() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("corpus.txt");
    std::fs::write(&source, "The river runs north.\n\nThe road runs south.\n").unwrap();
    let sha = splinter_sdk::vocabulary::digest::Digest::sha256_of(&std::fs::read(&source).unwrap())
        .hex()
        .to_string();
    let corpus = |sha256: &str| splinter_minimind::config::FetchedCorpus {
        name: "local".into(),
        url: format!("file://{}", source.display()),
        sha256: sha256.into(),
        license: "CC0-1.0".into(),
        slice_bytes: 1000,
    };
    let run_dir = dir.path().join("run");
    let brain = FakeBrain::new(&run_dir);
    let mut config = smoke();
    config.fetch = vec![corpus(&"0".repeat(64))];
    let refused = go(&run_dir, config.clone(), SEED, &brain, None).unwrap_err();
    assert!(
        format!("{refused:#}").contains("not the pinned"),
        "{refused:#}"
    );
    config.fetch = vec![corpus(&sha)];
    go(&run_dir, config, SEED, &brain, None).unwrap();
    let all = commits(&run_dir);
    assert_eq!(commit(&all, "world").summary["kinds"]["fetched"], 2);
    assert_eq!(commit(&all, "fetch/local").summary["license"], "CC0-1.0");
}
