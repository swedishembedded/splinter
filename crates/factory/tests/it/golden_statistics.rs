// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning factories whose reported numbers
// are recomputed from the raw trials, never only blessed, for its clients. If
// your team needs expertise in trustworthy evaluation of fine-tuned agents,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec (6B, G-FS-17: the factory's product statistics): the numbers a run
//! reports are what its trials imply, recomputed here from how each trial
//! ended (the scripted backend's record) and the plan's documented rule,
//! never from the factory's own summaries. Each evaluation pairs the baseline
//! and the candidate slot by slot: its pairs, wins, losses, pass-rate
//! difference, exact one-sided sign-test p (`sum_{k >= wins} C(n, k) / 2^n`
//! over the `n` discordant pairs) and decision at the plan's alpha divided
//! among its comparisons; the dataset's waterfall starts at every collected
//! trial and keeps the passes, every other trial excluded as no success. The
//! reference run's pinned golden products are held to these as well, so the
//! golden guards correctness and not only change.

#![allow(clippy::unwrap_used)] // a panic is the failure report

use std::collections::BTreeMap;

use serde_json::Value;
use splinter_factory::run::dataset::dataset_of;
use splinter_factory::run::eval::result_of;

use crate::factory_run::{decisive, resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{Repo, RECIPE};

/// The plan's alpha (`eval/dev.yaml`) and the comparisons the recipe makes.
const ALPHA: f64 = 0.05;
const COMPARISONS: f64 = 2.0;

/// The exact one-sided tail `P(Binomial(n, 1/2) >= wins)`.
fn tail(wins: usize, n: usize) -> f64 {
    let choose = |n: usize, k: usize| (0..k).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64);
    (wins..=n).map(|k| choose(n, k)).sum::<f64>() / 2f64.powi(n as i32)
}

/// What an evaluation of `candidate` should report, from the trials.
fn expected(
    returned: &[(String, String, u32, Outcome)],
    candidate: &str,
) -> (usize, usize, usize, f64, f64, &'static str) {
    let arm = |pass: &dyn Fn(&str) -> bool| -> BTreeMap<(String, u32), bool> {
        returned
            .iter()
            .filter(|(p, ..)| pass(p))
            .map(|(_, task, slot, outcome)| ((task.clone(), *slot), *outcome == Outcome::Pass))
            .collect()
    };
    let baseline = arm(&|p| p.starts_with("baseline-"));
    let tried = arm(&|p| p == format!("eval-{candidate}.candidate"));
    let (mut wins, mut losses, mut base, mut cand) = (0, 0, 0, 0);
    for (slot, &c) in &tried {
        let b = baseline[slot];
        wins += usize::from(c && !b);
        losses += usize::from(b && !c);
        base += usize::from(b);
        cand += usize::from(c);
    }
    let pairs = tried.len();
    let difference = (cand as f64 - base as f64) / pairs as f64;
    let p = tail(wins, wins + losses);
    let decision = if p < ALPHA / COMPARISONS && difference > 0.0 {
        "improved"
    } else if difference <= 0.0 {
        "not_improved"
    } else {
        "inconclusive"
    };
    (pairs, wins, losses, difference, p, decision)
}

fn check(dir: &std::path::Path, backend: &FakeBackend) {
    let returned = backend.returned.lock().unwrap().clone();
    for candidate in ["lora-r8", "lora-r16"] {
        let result = result_of(&dir.join(format!("eval-{candidate}"))).unwrap();
        let stats = &result.statistics;
        let (pairs, wins, losses, difference, p, decision) = expected(&returned, candidate);
        let context = format!("{candidate}: {stats}");
        assert_eq!(stats["pairs"], pairs, "{context}");
        assert_eq!(stats["wins"], wins, "{context}");
        assert_eq!(stats["losses"], losses, "{context}");
        assert!(
            (stats["difference"].as_f64().unwrap() - difference).abs() < 1e-12,
            "{context}"
        );
        assert!(
            (stats["p"].as_f64().unwrap() - p).abs() < 1e-12,
            "{context}: p {p}"
        );
        assert!(
            (stats["alpha"].as_f64().unwrap() - ALPHA / COMPARISONS).abs() < 1e-15,
            "{context}"
        );
        assert_eq!(
            serde_json::to_value(result.decision).unwrap(),
            decision,
            "{context}"
        );
    }
    let collected: Vec<_> = returned
        .iter()
        .filter(|(pass, ..)| pass == "local" || pass == "remote")
        .collect();
    let passes = collected
        .iter()
        .filter(|(.., o)| *o == Outcome::Pass)
        .count();
    let dataset = dataset_of(&dir.join("sft")).unwrap();
    let waterfall: BTreeMap<String, u64> = serde_json::to_value(&dataset.waterfall)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["step"].as_str().unwrap().to_string(),
                s["remaining"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(waterfall["raw_trials"], collected.len() as u64);
    assert_eq!(waterfall["verified_successes"], passes as u64);
    let excluded: u64 = serde_json::to_value(&dataset.exclusions)
        .unwrap()
        .as_object()
        .unwrap()
        .values()
        .map(|n| n.as_u64().unwrap())
        .sum();
    assert_eq!(excluded, (collected.len() - passes) as u64);
}

fn scenario(backend: &FakeBackend) -> tempfile::TempDir {
    let repo = Repo::new();
    decisive(&repo);
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, RECIPE),
        dir.path(),
        backend,
        &FakeTrainer::default(),
    )
    .unwrap();
    dir
}

#[test]
fn the_reported_statistics_are_what_the_trials_imply() {
    // The reference run, whose products are pinned: every baseline slot
    // fails and every candidate slot passes.
    let reference = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    reference.script("train/001-first", &[Outcome::Pass, Outcome::Pass]);
    let dir = scenario(&reference);
    check(dir.path(), &reference);
    let pinned: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/it/fixtures/golden-products.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let returned = reference.returned.lock().unwrap().clone();
    for candidate in ["lora-r8", "lora-r16"] {
        let (pairs, wins, losses, _, p, decision) = expected(&returned, candidate);
        let golden = &pinned["evaluations"][candidate];
        assert_eq!(golden["statistics"]["pairs"], pairs);
        assert_eq!(golden["statistics"]["wins"], wins);
        assert_eq!(golden["statistics"]["losses"], losses);
        assert!((golden["statistics"]["p"].as_f64().unwrap() - p).abs() < 1e-12);
        assert_eq!(golden["decision"], decision);
    }

    // Mixed slots: wins, losses and ties, and a candidate not good enough.
    let mixed = FakeBackend {
        candidate_slots: [
            (0, Outcome::Pass),
            (1, Outcome::Pass),
            (2, Outcome::Pass),
            (3, Outcome::Fail),
            (4, Outcome::Pass),
            (5, Outcome::Fail),
            (6, Outcome::Pass),
            (7, Outcome::Pass),
        ]
        .into_iter()
        .map(|(slot, o)| (format!("eval/001-held#{slot}"), o))
        .collect(),
        ..FakeBackend::with(Outcome::Fail)
    };
    for (slot, outcome) in [
        Outcome::Fail,
        Outcome::Pass,
        Outcome::Fail,
        Outcome::Pass,
        Outcome::Fail,
        Outcome::Fail,
        Outcome::Fail,
        Outcome::Pass,
    ]
    .into_iter()
    .enumerate()
    {
        mixed.script(&format!("eval/001-held#{slot}"), &[outcome]);
    }
    mixed.script("train/001-first", &[Outcome::Pass, Outcome::Fail]);
    mixed.script("train/002-second", &[Outcome::Fail, Outcome::Pass]);
    let dir = scenario(&mixed);
    check(dir.path(), &mixed);
    let (_, wins, losses, ..) = expected(&mixed.returned.lock().unwrap(), "lora-r8");
    assert!(
        wins > 0 && losses > 0,
        "the mixed scenario has both: {wins} {losses}"
    );
}
