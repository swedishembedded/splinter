// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's practice
// where its measured capability gaps are, for its clients. If your team needs
// expertise in adaptive curricula for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec of competence per capability and the curriculum it steers:
//!
//! * the examiner's capability ids reach the competence from a real practice
//!   manifest (`bench task practice` output), from the export's own labels,
//!   and, for material staged before the examiner named them, from a
//!   relabelled staging matched by source digest;
//! * each capability's pass rate carries a 95% interval, and an attempt the
//!   infrastructure stopped is never a graded 0;
//! * the selection prefers the capabilities the policy fails on the
//!   frontier, never draws a retired task or one the infrastructure stopped
//!   every time, and the same seed draws the same tasks;
//! * nothing sealed is counted, and nothing outside the practice material is
//!   ever selected.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use splinter_core::outcome::FailureClass;
use splinter_core::rng::Seed;
use splinter_eval::campaign::pass_rate::PassRates;
use splinter_pipelines::bench::artifact::sha256_hex;
use splinter_pipelines::bench::capabilities::{capability_report, CapabilityRequest};
use splinter_pipelines::curriculum::capability::{
    competence, select, Attempt, CapabilityCompetence, SelectRequest, UNLABELLED,
};

use crate::common::bench::{export, Trial};
use crate::common::Scratch;

/// A practice manifest `bench task practice tasks/compiled --count 40 --seed
/// 52` wrote, its tasks labelled by the examiner's taxonomy.
const PRACTICE: &str = include_str!("../fixtures/practice-compiled-s52.json");

/// Three staged tasks of it and what the examiner labels each with.
const ABSTAIN: &str = "abstain-withheld-ekbygg-6200028";
const DEADLINE: &str = "deadline-single-backdata-6200018";
const APPLY: &str = "apply-plain-granbygg-6200001";

/// The practice directory holding `manifest`; returns it.
fn practice_dir(root: &Path, manifest: &str) -> PathBuf {
    let dir = root.join("practice").join("compiled");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("practice.json"), manifest).unwrap();
    dir
}

/// A practice run's directory over `material`: its report naming the policy,
/// the material by sha256 and each trial's staged task, and its export.
fn practice_run(root: &Path, material: &Path, trials: Vec<(Trial, &str)>) -> PathBuf {
    let run = root.join("run");
    let staged: Vec<Value> = trials
        .iter()
        .map(|(t, staged)| json!({"trial": t.name, "task": staged}))
        .collect();
    let bench: Vec<Trial> = trials.into_iter().map(|(t, _)| t.scored()).collect();
    export(&run.join("export"), &bench);
    let report = json!({
        "policy": {"model": "brain/Qwen/Qwen3-4B", "adapter": "blake3:a2"},
        "provenance": {"dir": material,
                       "practice_sha256": sha256_hex(&fs::read(material.join("practice.json")).unwrap())},
        "trials": staged,
    });
    fs::write(run.join("report.json"), report.to_string()).unwrap();
    run
}

fn of<'a>(capabilities: &'a [CapabilityCompetence], id: &str) -> &'a CapabilityCompetence {
    capabilities
        .iter()
        .find(|c| c.capability == id)
        .unwrap_or_else(|| panic!("no capability {id}"))
}

/// The practice run every labelling test reads: abstain passes, the
/// deadline fails once and passes once, apply passes.
fn run_trials() -> Vec<(Trial, &'static str)> {
    vec![
        (
            Trial::new("t1", "org/compiled-abstain", Some("train"), true),
            ABSTAIN,
        ),
        (
            Trial::new("t2", "org/compiled-deadline", Some("train"), false),
            DEADLINE,
        ),
        (
            Trial::new("t3", "org/compiled-deadline", Some("train"), true),
            DEADLINE,
        ),
        (
            Trial::new("t4", "org/compiled-apply", Some("train"), true),
            APPLY,
        ),
    ]
}

#[test]
fn capability_ids_flow_from_a_real_practice_manifest_into_competence_per_policy() {
    let scratch = Scratch::new("capabilities-material");
    let material = practice_dir(&scratch.0, PRACTICE);
    let run = practice_run(&scratch.0, &material, run_trials());
    let report = capability_report(&CapabilityRequest {
        dirs: &[run],
        labels: &[],
        policy: None,
    });
    assert!(report.refused.is_empty(), "{:?}", report.refused);
    assert_eq!(report.labelled.material, 4);
    let [policy] = report.policies.as_slice() else {
        panic!("one policy: {:?}", report.policies)
    };
    assert_eq!(policy.policy, "brain/Qwen/Qwen3-4B+blake3:a2");
    let ids: Vec<&str> = policy
        .capabilities
        .iter()
        .map(|c| c.capability.as_str())
        .collect();
    assert_eq!(
        ids,
        [
            "abstention",
            "apply-rules",
            "exact-arithmetic",
            "locate-in-workspace",
            "structured-answer"
        ],
        "the examiner's ids, as data"
    );
    let arithmetic = of(&policy.capabilities, "exact-arithmetic");
    assert_eq!(
        (arithmetic.tasks, arithmetic.graded, arithmetic.passes),
        (1, 2, 1)
    );
    assert_eq!(of(&policy.capabilities, "apply-rules").graded, 4);
    let source = &report.sources[0];
    assert!(source.practice_sha256.is_some() && source.index_sha256.is_some());
}

#[test]
fn an_exports_own_labels_win_and_old_material_is_labelled_by_source_digest_only() {
    let scratch = Scratch::new("capabilities-relabelled");
    // The run's material as staged before the examiner named capabilities.
    let mut old: Value = serde_json::from_str(PRACTICE).unwrap();
    for task in old["tasks"].as_array_mut().unwrap() {
        task.as_object_mut().unwrap().remove("capabilities");
    }
    let material = practice_dir(&scratch.0.join("old"), &old.to_string());
    let mut trials = run_trials();
    trials[0].0 =
        Trial::new("t1", "org/compiled-abstain", Some("train"), true).capabilities(&["abstention"]);
    let run = practice_run(&scratch.0, &material, trials);
    let labels = practice_dir(&scratch.0.join("relabelled"), PRACTICE);
    let report = capability_report(&CapabilityRequest {
        dirs: std::slice::from_ref(&run),
        labels: &[labels],
        policy: None,
    });
    assert_eq!(
        (
            report.labelled.export,
            report.labelled.labels,
            report.labelled.none
        ),
        (1, 3, 0)
    );
    // Without the relabelled staging the old material names nothing.
    let bare = capability_report(&CapabilityRequest {
        dirs: &[run],
        labels: &[],
        policy: None,
    });
    assert_eq!((bare.labelled.export, bare.labelled.none), (1, 3));
    assert_eq!(of(&bare.policies[0].capabilities, UNLABELLED).attempts, 3);
}

fn attempt(
    task: &str,
    capability: &str,
    passed: Option<bool>,
    class: Option<FailureClass>,
) -> Attempt {
    Attempt {
        task: task.into(),
        capabilities: vec![capability.into()],
        passed,
        class,
    }
}

fn rates_of(attempts: &[Attempt]) -> PassRates {
    let mut by_task: BTreeMap<&str, Vec<Option<bool>>> = BTreeMap::new();
    for a in attempts {
        by_task.entry(&a.task).or_default().push(a.passed);
    }
    let mut rates = PassRates::default();
    for (task, outcomes) in by_task {
        rates.observe(task, 0, &outcomes);
    }
    rates
}

#[test]
fn competence_has_an_interval_and_an_infrastructure_stop_is_never_a_zero() {
    let wrong = Some(FailureClass::WrongContent);
    let gateway = Some(FailureClass::Gateway);
    let attempts = vec![
        attempt("a", "x", Some(true), None),
        attempt("a", "x", Some(true), None),
        attempt("b", "x", Some(true), None),
        attempt("b", "x", Some(false), wrong),
        attempt("c", "x", None, gateway),
        attempt("c", "x", None, gateway),
    ];
    let measured = competence(&attempts, &rates_of(&attempts));
    let x = of(&measured, "x");
    assert_eq!((x.attempts, x.graded, x.passes, x.ungraded), (6, 4, 3, 2));
    let share = x.pass_rate.unwrap();
    assert!((share.value - 0.75).abs() < 1e-12);
    assert!(share.low < 0.75 && 0.75 < share.high && share.low > 0.0 && share.high <= 1.0);
    assert_eq!(x.failures.get("gateway"), Some(&2));
    assert_eq!(
        (x.model_failures, x.failures.get("wrong_content")),
        (1, Some(&1))
    );
    assert_eq!(x.bands.unmeasured, 1, "task c has no graded attempt");
}

/// A pool where `weak` is failed half the time for the model's own reasons,
/// `strong` is mastered, `hard` never solved, and one task of `weak` was
/// only ever stopped by the infrastructure.
fn pool() -> (Vec<String>, BTreeMap<String, Vec<String>>, Vec<Attempt>) {
    let wrong = Some(FailureClass::WrongContent);
    let mut attempts = Vec::new();
    for task in ["w1", "w2"] {
        for i in 0..8 {
            attempts.push(attempt(
                task,
                "weak",
                Some(i % 2 == 0),
                (i % 2 == 1).then_some(wrong).flatten(),
            ));
        }
    }
    for task in ["s1", "s2"] {
        attempts.extend((0..8).map(|_| attempt(task, "strong", Some(true), None)));
    }
    attempts.extend((0..8).map(|_| attempt("h1", "hard", Some(false), wrong)));
    attempts.extend((0..4).map(|_| attempt("i1", "weak", None, Some(FailureClass::ModelServer))));
    let names = ["w1", "w2", "w3", "s1", "s2", "h1", "i1"];
    let labels = names
        .iter()
        .map(|t| {
            let capability = match &t[..1] {
                "w" | "i" => "weak",
                "s" => "strong",
                _ => "hard",
            };
            ((*t).to_string(), vec![capability.to_string()])
        })
        .collect();
    (names.map(String::from).to_vec(), labels, attempts)
}

#[test]
fn selection_prefers_the_frontier_capability_and_never_draws_retired_or_infrastructure_failed_tasks(
) {
    let (tasks, labels, attempts) = pool();
    let rates = rates_of(&attempts);
    let measured = competence(&attempts, &rates);
    assert!(of(&measured, "weak").weight > of(&measured, "hard").weight);
    assert!(of(&measured, "weak").weight > of(&measured, "strong").weight);
    let request = |budget, seed| SelectRequest {
        pool: &tasks,
        labels: &labels,
        rates: &rates,
        competence: &measured,
        budget,
        transfer: 1,
        seed: Seed::new(seed),
    };
    let seed = splinter_model::seed::draw().value();
    let everything = select(&request(None, seed));
    let mut drawn = everything.tasks.clone();
    drawn.sort();
    assert_eq!(
        drawn,
        ["h1", "w1", "w2", "w3"],
        "retired s1, s2 and the stopped i1 never"
    );
    assert_eq!(everything.infrastructure, ["i1"]);
    assert_eq!(everything.transfer, ["weak"]);
    // Over many draws of two, the frontier capability's tasks come first far
    // more often than the never-solved one's.
    let (mut weak, mut hard) = (0, 0);
    for offset in 0..400 {
        for task in select(&request(Some(2), seed.wrapping_add(offset))).tasks {
            match &task[..1] {
                "w" => weak += 1,
                "h" => hard += 1,
                other => panic!("drew {other}"),
            }
        }
    }
    assert!(weak > 6 * hard, "weak {weak}, hard {hard}");
}

#[test]
fn the_same_seed_draws_the_same_tasks() {
    let (tasks, labels, attempts) = pool();
    let rates = rates_of(&attempts);
    let measured = competence(&attempts, &rates);
    let draw = |seed| {
        select(&SelectRequest {
            pool: &tasks,
            labels: &labels,
            rates: &rates,
            competence: &measured,
            budget: Some(3),
            transfer: 0,
            seed: Seed::new(seed),
        })
    };
    let seed = splinter_model::seed::draw().value();
    assert_eq!(draw(seed), draw(seed));
    let orders: std::collections::BTreeSet<Vec<String>> =
        (0..32).map(|s| draw(seed.wrapping_add(s)).tasks).collect();
    assert!(orders.len() > 1, "the seed decides the draw");
}

#[test]
fn nothing_sealed_is_counted_and_nothing_outside_the_material_is_selected() {
    let scratch = Scratch::new("capabilities-sealed");
    let sealed = export(
        &scratch.0.join("sealed"),
        &[Trial::new("e1", "org/exam", Some("train"), true).capabilities(&["abstention"])],
    );
    fs::write(sealed.join("sealed-export.json"), "{}").unwrap();
    let report = capability_report(&CapabilityRequest {
        dirs: &[sealed],
        labels: &[],
        policy: Some("p"),
    });
    assert!(report
        .refused
        .iter()
        .any(|r| r.reason.contains("sealed-export.json")));
    assert!(report.policies.iter().all(|p| p.trials == 0));

    // History and labels name tasks the material does not hold: they are
    // counted as competence, never drawn.
    let (tasks, mut labels, mut attempts) = pool();
    attempts.extend((0..8).map(|i| attempt("exam-item", "weak", Some(i < 4), None)));
    labels.insert("exam-item".into(), vec!["weak".into()]);
    let rates = rates_of(&attempts);
    let measured = competence(&attempts, &rates);
    let seed = splinter_model::seed::draw().value();
    for offset in 0..64 {
        let selection = select(&SelectRequest {
            pool: &tasks,
            labels: &labels,
            rates: &rates,
            competence: &measured,
            budget: None,
            transfer: 1,
            seed: Seed::new(seed.wrapping_add(offset)),
        });
        assert!(selection.tasks.iter().all(|t| tasks.contains(t)));
        assert!(!selection.weights.contains_key("exam-item"));
    }
}
