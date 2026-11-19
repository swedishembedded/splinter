// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's practice
// where its measured capability gaps are, for its clients. If your team needs
// expertise in adaptive curricula for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec of the capability curriculum in a running campaign
//! (`[rollout] curriculum = "capability"`), over the scripted world: the
//! champion fails the `compute` tasks half the time for the model's own
//! reasons and always solves the `lookup` ones, so generation 2 practises
//! more on `compute` than generation 1 did, weighs every `compute` task above
//! every `lookup` one, and asks the examiner for transfer exercises of
//! `compute`.

#![allow(clippy::unwrap_used)]

use std::cell::RefCell;
use std::collections::BTreeMap;

use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_core::outcome::FailureClass;
use splinter_orchestrator::Context;
use splinter_pipelines::campaign::{
    CampaignController, CampaignError, Parts, Rolled, RolloutCurriculum, RolloutRequest, Rollouts,
    TransferSpec, Trial,
};
use splinter_store::stage_ledger::Commit;

use super::world::{Dying, Exam, Rig, Ticks, GOOD};
use crate::common::bench::{export, Trial as BenchTrial};

/// The practice tasks and the capability the examiner labels each with.
const TASKS: [(&str, &str); 6] = [
    ("c1", "compute"),
    ("c2", "compute"),
    ("c3", "compute"),
    ("l1", "lookup"),
    ("l2", "lookup"),
    ("l3", "lookup"),
];

fn capability_of(task: &str) -> &'static str {
    if task.starts_with('c') || task.starts_with('x') {
        "compute"
    } else {
        "lookup"
    }
}

/// What one rollout was asked for.
#[derive(Clone, Debug)]
struct Asked {
    tasks: Vec<String>,
    transfer: Option<(Vec<String>, usize)>,
}

/// Practice whose champion passes a `compute` task on its first attempt
/// only, with wrong content after, and every `lookup` attempt; transfer
/// exercises are tasks `x<n>` of the capability asked for.
#[derive(Default)]
struct Labelled {
    asked: RefCell<Vec<Asked>>,
}

impl Rollouts for Labelled {
    fn identity(&self) -> Value {
        json!({"practice": "labelled", "tasks": TASKS})
    }

    fn tasks(&self) -> Result<Vec<String>, CampaignError> {
        Ok(TASKS.iter().map(|(t, _)| (*t).to_string()).collect())
    }

    fn capabilities(&self) -> Result<BTreeMap<String, Vec<String>>, CampaignError> {
        Ok(TASKS
            .iter()
            .map(|(t, c)| ((*t).to_string(), vec![(*c).to_string()]))
            .collect())
    }

    fn rollout(
        &mut self,
        _ctx: &Context,
        request: &RolloutRequest<'_>,
        _cancel: &CancelToken,
    ) -> Result<Rolled, CampaignError> {
        let transfer = request.transfer.map(|t| (t.capabilities.to_vec(), t.count));
        self.asked.borrow_mut().push(Asked {
            tasks: request.tasks.to_vec(),
            transfer: transfer.clone(),
        });
        let mut tasks = request.tasks.to_vec();
        if let Some((_, count)) = &transfer {
            tasks.extend((1..=*count).map(|n| format!("x{}-{n}", request.generation)));
        }
        let mut trials = Vec::new();
        let mut bench = Vec::new();
        for task in &tasks {
            for attempt in 1..=request.attempts {
                let pass = capability_of(task) == "lookup" || attempt == 1;
                let name: &'static str =
                    Box::leak(format!("g{}-{task}-{attempt}", request.generation).into_boxed_str());
                let id: &'static str = Box::leak(format!("org/{task}").into_boxed_str());
                bench.push(BenchTrial::new(name, id, Some("train"), pass).scored());
                trials.push(Trial {
                    task: task.clone(),
                    reward: Some(if pass { 1.0 } else { 0.0 }),
                    passed: Some(pass),
                    class: (!pass).then_some(FailureClass::WrongContent),
                    capabilities: vec![capability_of(task).to_string()],
                    correctness: None,
                    cost: None,
                    completion_tokens: None,
                });
            }
        }
        let dir = request.dir.join("export");
        let _ = std::fs::remove_dir_all(&dir);
        export(&dir, &bench);
        Ok(Rolled {
            export: dir,
            tokens: Some(15 * trials.len() as u64),
            trials,
        })
    }
}

/// The share of `tasks` that exercise `compute`.
fn compute_share(tasks: &[String]) -> f64 {
    tasks
        .iter()
        .filter(|t| capability_of(t) == "compute")
        .count() as f64
        / tasks.len() as f64
}

/// The rollout records of the ledger, by generation.
fn rollout_records(commits: &[Commit]) -> Vec<Value> {
    commits
        .iter()
        .filter(|c| c.unit.ends_with("/rollout"))
        .map(|c| c.summary.clone())
        .collect()
}

#[test]
fn generation_two_practises_more_on_the_capability_generation_one_failed() {
    let rig = Rig::new("campaign-capabilities");
    // seed: the test asserts one recorded draw of the campaign's tasks.
    let mut spec = rig.spec("capabilities", 2, 20_261_119);
    spec.rollout.curriculum = RolloutCurriculum::Capability;
    spec.rollout.tasks = Some(4);
    spec.rollout.transfer = Some(TransferSpec {
        capabilities: 1,
        count: 2,
    });
    let mut practice = Labelled::default();
    let mut exam = Exam::new(&rig.scratch.0.join("bench-private"), &[GOOD, GOOD]);
    let trainer = Dying::default();
    let ticks = Ticks::default();
    let commits = {
        let mut controller = CampaignController::open(
            &rig.ctx,
            spec,
            Parts {
                rollouts: &mut practice,
                examiner: &mut exam,
                trainer: &trainer,
                stopwatch: &ticks,
                gpus: 1,
                cancel: CancelToken::new(),
                progress: None,
            },
        )
        .unwrap();
        controller.run().unwrap();
        controller.ledger().commits().unwrap()
    };
    let asked = practice.asked.borrow().clone();
    assert_eq!(asked.len(), 2, "{asked:?}");
    let (first, second) = (&asked[0], &asked[1]);
    assert_eq!(first.tasks.len(), 4);
    assert!(
        first.transfer.is_none(),
        "nothing is measured before generation 1"
    );
    assert_eq!(
        second.transfer,
        Some((vec!["compute".to_string()], 2)),
        "the failed capability gets fresh exercises"
    );
    let practised = |a: &Asked| {
        let extra = a.transfer.as_ref().map_or(0, |(_, n)| *n);
        let compute = a
            .tasks
            .iter()
            .filter(|t| capability_of(t) == "compute")
            .count()
            + extra;
        compute as f64 / (a.tasks.len() + extra) as f64
    };
    assert!(
        practised(second) > practised(first),
        "generation 1 practised compute {:.2}, generation 2 {:.2}",
        practised(first),
        practised(second)
    );
    assert!(compute_share(&second.tasks) >= compute_share(&first.tasks));
    // The record says why: every compute task outweighs every lookup task.
    let records = rollout_records(&commits);
    let weights = &records[1]["selection"]["weights"];
    for (compute, lookup) in [("c1", "l1"), ("c2", "l2"), ("c3", "l3")] {
        assert!(
            weights[compute].as_f64().unwrap() > weights[lookup].as_f64().unwrap(),
            "{weights}"
        );
    }
    assert!(records[0]["selection"]["transfer"]
        .as_array()
        .unwrap()
        .is_empty());
}
