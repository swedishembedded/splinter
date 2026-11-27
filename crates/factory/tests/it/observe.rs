// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every generation can be
// explained from its own records, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A generation's state read back from a finished run's records alone: whose
//! records its dataset holds and from which side of the target's frontier,
//! how hard its tasks were for each policy, the passes over the data its
//! candidate trained, every evaluation with its purpose, task set and
//! interval, the release's reasons, how each attempt stage's failures divide,
//! and which tasks a reference solution proved solvable.

use serde_json::json;
use splinter_factory::observe::conduct::FailureClass;
use splinter_factory::observe::data::Policy;
use splinter_factory::observe::probe::{probe, Verdict};
use splinter_factory::observe::{observe, Options};

use crate::factory_run::{decisive, resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{Repo, RECIPE};

/// The reference recipe with a reference-solution stage, promotion on its
/// target evaluations, and a second held-out task.
fn recipe(repo: &Repo) -> String {
    repo.write(
        "tasks/vim/eval/002-held/task.toml",
        "[task]\nname = \"org/eval/002-held\"\n[metadata]\nbench_family = \"course\"\n",
    );
    repo.write(
        "tasks/vim/eval/002-held/instruction.md",
        "Do eval/002-held.\n",
    );
    repo.write("tasks/vim/eval/002-held/tests/test.sh", "exit 0\n");
    // seed: a recipe states every stage's seed; the fake backend draws nothing from it.
    RECIPE
        .replace(
            "  - id: local\n",
            "  - id: oracle\n    uses: attempt.collect\n    inputs: { tasks: $stages.tasks.tasks }\n    \
             with: { agent: oracle, environment: environments/vim.yaml, seed: 9 }\n  - id: local\n",
        )
        .replace("with: { promote: false }", "with: { promote: true, requires: [target] }")
}

#[test]
fn a_generations_state_is_read_back_from_its_records_alone() {
    let repo = Repo::new();
    decisive(&repo);
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        reference_on: [
            ("train/001-first".to_owned(), Outcome::Pass),
            ("train/002-second".to_owned(), Outcome::Fail),
        ]
        .into(),
        ..FakeBackend::with(Outcome::Fail)
    };
    // The student passes the first task once of two and never the second;
    // the teacher, sent only what the student never solved, passes it.
    backend.script("train/001-first#0", &[Outcome::Pass]);
    backend.script(
        "train/002-second",
        &[Outcome::Fail, Outcome::Fail, Outcome::Pass, Outcome::Pass],
    );
    let trainer = FakeTrainer {
        curve: Some(json!({
            "points": [
                {"step": 1, "monitor_loss": 1.0},
                {"step": 2, "monitor_loss": 0.5},
                {"step": 3, "monitor_loss": 0.6}
            ],
            "selected_step": 2, "steps_completed": 3, "stopped_early": true,
            "speed": {"row_tokens": 900, "tokens_per_second": 300.0}
        })),
        ..FakeTrainer::default()
    };
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &recipe(&repo)),
        dir.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let before = splinter_factory::tree::tree_digest(dir.path()).unwrap();

    // seed: the interval's seed; with every task won the interval is the point whatever it is.
    let observed = observe(
        dir.path(),
        &Options {
            seed: 3,
            utilisation: None,
        },
    )
    .unwrap();
    assert_eq!(
        splinter_factory::tree::tree_digest(dir.path()).unwrap(),
        before,
        "observing changes nothing"
    );
    assert_eq!(
        observed.generations.len(),
        1,
        "one release stage, one generation"
    );
    let generation = &observed.generations[0];
    assert_eq!(
        generation.candidates.len(),
        2,
        "both trained candidates are evaluated for the release"
    );

    // Data: whose records, from which side of the student's frontier.
    let data = generation.candidates[0]
        .data
        .as_ref()
        .expect("the dataset is read");
    let records = data.train.records + data.validation.records;
    assert_eq!(records, 3, "the student's one pass and the teacher's two");
    let own: usize = data
        .sources
        .iter()
        .filter(|s| s.policy == Policy::Own)
        .map(|s| s.train.records + s.validation.records)
        .sum();
    assert_eq!(own, 1);
    let local = data.sources.iter().find(|s| s.stage == "local").unwrap();
    assert_eq!(local.policy, Policy::Own);
    assert_eq!(
        local.pass_rate.map(|p| (p.successes, p.total)),
        Some((1, 4))
    );
    let remote = data.sources.iter().find(|s| s.stage == "remote").unwrap();
    assert_eq!(
        (remote.policy, remote.mean_task_pass_rate),
        (Policy::Other, Some(1.0))
    );
    let frontier = data
        .frontier
        .as_ref()
        .expect("the student attempted the tasks");
    assert_eq!(
        (
            frontier.failed_tasks,
            frontier.sometimes_tasks,
            frontier.passed_tasks
        ),
        (1, 1, 0)
    );
    assert_eq!(frontier.records_failed.get(&Policy::Other), Some(&2));
    assert_eq!(frontier.records_sometimes.get(&Policy::Own), Some(&1));
    assert_eq!(frontier.frontier_share, Some(1.0));

    // Training: the passes over the data at the selected step.
    let training = generation.candidates[0]
        .training
        .as_ref()
        .expect("the training is read");
    assert_eq!(
        (training.selected_step, training.stopped_early),
        (Some(2), Some(true))
    );
    assert_eq!(
        training.records_per_step,
        Some(1),
        "no grad_accum, one rank"
    );
    let draws = data.train.records as f64;
    assert_eq!(training.draws_per_epoch, Some(draws));
    assert_eq!(training.epochs_at_selected, Some(2.0 / draws));
    assert_eq!(
        (
            training.validation_loss_first,
            training.validation_loss_selected,
            training.validation_loss_last
        ),
        (Some(1.0), Some(0.5), Some(0.6))
    );

    // Capability: each evaluation, its purpose, tasks and interval; the release's reasons.
    assert_eq!(generation.evaluations.len(), 2);
    for e in &generation.evaluations {
        assert_eq!(e.purpose.as_deref(), Some("target"));
        assert_eq!(
            (e.task_set.label.as_str(), e.task_set.tasks),
            ("the plan's tasks", 2)
        );
        assert_eq!(
            (e.wins, e.losses),
            (Some(16), Some(0)),
            "eight attempts at each of two tasks"
        );
        let interval = e.interval.expect("two tasks give an interval");
        assert_eq!(
            (interval.low, interval.high, e.interval_tasks),
            (1.0, 1.0, 2)
        );
        assert_eq!(
            e.task_set.solvability.unknown, 2,
            "no reference ran on the held-out tasks"
        );
        let baseline = e
            .baseline
            .conduct
            .as_ref()
            .expect("the baseline arm's trials are read");
        assert_eq!(baseline.failed, 16);
    }
    assert!(
        generation.release.promoted.is_some(),
        "{:?}",
        generation.release
    );
    assert_eq!(generation.release.why.len(), 2);

    // Conduct and solvability: the reference solution proved one task and failed the other.
    let oracle = observed
        .attempts
        .iter()
        .find(|a| a.stage == "oracle")
        .unwrap();
    assert!(oracle.reference);
    let local = observed
        .attempts
        .iter()
        .find(|a| a.stage == "local")
        .unwrap();
    assert_eq!(
        (local.solvability.witnessed, local.solvability.refuted),
        (1, 1)
    );
    assert_eq!(local.conduct.failed, 3);
    assert_eq!(
        local
            .conduct
            .attribution
            .get(&FailureClass::UnreachableGoal),
        Some(&2),
        "the student's failures at the refuted task are the interface's"
    );
    assert_eq!(
        local.conduct.attribution.values().sum::<usize>(),
        local.conduct.failed
    );
    let sv = &local.conduct.self_verification;
    assert_eq!(sv.judged, 4);
    assert_eq!(sv.claimed_passed + sv.unclaimed_passed, 1);
    assert_eq!(local.pace.trials, 4);
}

/// The reference recipe with a step budget on `lora-r8` and two knobs that
/// both claim to move the passes over the data: the budget, and the rank.
fn knobbed() -> String {
    format!(
        "{}knobs:\n  epochs:\n    set: lora-r8.params.epochs\n    clears: [params.steps]\n    \
         moves: training.epochs_at_selected\n  rank:\n    set: lora-r8.params.rank\n    \
         moves: training.epochs_at_selected\n",
        RECIPE
            .replace("params: { rank: 8 }", "params: { rank: 8, steps: 2 }")
            .replace(
                "with: { promote: false }",
                "with: { promote: true, requires: [target] }"
            )
    )
}

#[test]
fn a_probe_turns_one_knob_and_reports_a_knob_that_does_not_move_its_variable_as_a_defect() {
    let repo = Repo::new();
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Pass)
    };
    let trainer = FakeTrainer {
        curve_to_the_last_step: true,
        ..FakeTrainer::default()
    };
    let control = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &knobbed()),
        control.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    // seed: the interval's seed; every arm passes, so the interval is the point whatever it is.
    let options = Options {
        seed: 5,
        utilisation: None,
    };
    let resolve = |root: &std::path::Path, text: &str| {
        splinter_factory::resolve::resolve(root, text, crate::repo::connections())
            .map_err(|e| e.0.join("; "))
    };
    let execute = |resolved: &splinter_factory::resolve::Resolved, out: &std::path::Path| {
        run(resolved, out, &backend, &trainer)
            .map(|_| ())
            .map_err(|e| e.to_string())
    };

    let epochs = tempfile::tempdir().unwrap();
    let report = probe(
        control.path(),
        epochs.path(),
        "epochs",
        &json!(3),
        &options,
        resolve,
        execute,
    )
    .unwrap();
    assert_eq!(report.generation, 1);
    assert_eq!(report.verdict, Verdict::Moved, "{report:#?}");
    assert!(!report.defect);
    assert!(
        (report.probed.unwrap() - 3.0).abs() < 1e-9,
        "three passes at the last step: {report:#?}"
    );
    assert!(report.control.unwrap() < report.probed.unwrap());
    assert_eq!(
        report.capability.len(),
        2,
        "both evaluations of the generation"
    );
    for c in &report.capability {
        assert_eq!(
            (c.control.unwrap().value, c.probed.unwrap().value),
            (1.0, 1.0)
        );
        assert_ne!(
            c.followed,
            Some(true),
            "every arm passed before and after: {c:?}"
        );
    }

    let rank = tempfile::tempdir().unwrap();
    let resolve = |root: &std::path::Path, text: &str| {
        splinter_factory::resolve::resolve(root, text, crate::repo::connections())
            .map_err(|e| e.0.join("; "))
    };
    let execute = |resolved: &splinter_factory::resolve::Resolved, out: &std::path::Path| {
        run(resolved, out, &backend, &trainer)
            .map(|_| ())
            .map_err(|e| e.to_string())
    };
    let report = probe(
        control.path(),
        rank.path(),
        "rank",
        &json!(16),
        &options,
        resolve,
        execute,
    )
    .unwrap();
    assert_eq!(report.verdict, Verdict::DidNotMove, "{report:#?}");
    assert!(
        report.defect,
        "the rank does not move the passes over the data"
    );
    assert_eq!(report.change, Some(0.0));

    let refused = probe(
        control.path(),
        tempfile::tempdir().unwrap().path(),
        "nope",
        &json!(1),
        &options,
        |_, _| Ok(()),
        |_: &(), _| Ok(()),
    );
    assert!(refused.unwrap_err().to_string().contains("no knob"));
}
