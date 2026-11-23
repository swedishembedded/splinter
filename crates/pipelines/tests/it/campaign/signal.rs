// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a reinforcement-learning
// budget on the tasks that teach the most per token, for its clients. If your
// team needs expertise in curricula for policy-gradient training on agent
// rollouts, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec of the variance-per-cost curriculum in a running campaign
//! (`[rollout] curriculum = "signal"`), over the scripted world: the champion
//! always solves the `easy` tasks, solves the `split` and `dear` tasks on
//! every other attempt, and a `dear` attempt costs ten times a `split` one.
//! Generation 2 never practises an `easy` task (its attempts all earned the
//! same), and weighs every `split` task above every `dear` one; the record
//! says so.

#![allow(clippy::unwrap_used)]

use std::cell::RefCell;

use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_orchestrator::Context;
use splinter_pipelines::campaign::{
    CampaignController, Parts, Rolled, RolloutCurriculum, RolloutRequest, Rollouts, Trial,
};
use splinter_pipelines::served::ServedError;

use super::world::{Dying, Exam, Rig, Ticks, GOOD};
use crate::common::bench::{export, Trial as BenchTrial};

const TASKS: [&str; 6] = ["easy1", "easy2", "split1", "split2", "dear1", "dear2"];

/// Practice of the module documentation; records each rollout's tasks.
#[derive(Default)]
struct Costed {
    asked: RefCell<Vec<Vec<String>>>,
}

impl Rollouts for Costed {
    fn identity(&self) -> Value {
        json!({"practice": "costed", "tasks": TASKS})
    }

    fn tasks(&self) -> Result<Vec<String>, ServedError> {
        Ok(TASKS.iter().map(|t| (*t).to_string()).collect())
    }

    fn rollout(
        &mut self,
        _ctx: &Context,
        request: &RolloutRequest<'_>,
        _cancel: &CancelToken,
    ) -> Result<Rolled, ServedError> {
        self.asked.borrow_mut().push(request.tasks.to_vec());
        let mut trials = Vec::new();
        let mut bench = Vec::new();
        for task in request.tasks {
            for attempt in 1..=request.attempts {
                let pass = task.starts_with("easy") || attempt % 2 == 1;
                let cost = if task.starts_with("dear") { 1000 } else { 100 };
                let name: &'static str =
                    Box::leak(format!("g{}-{task}-{attempt}", request.generation).into_boxed_str());
                let id: &'static str = Box::leak(format!("org/{task}").into_boxed_str());
                bench.push(BenchTrial::new(name, id, Some("train"), pass).scored());
                trials.push(Trial {
                    task: task.clone(),
                    reward: Some(if pass { 1.0 } else { 0.0 }),
                    passed: Some(pass),
                    class: None,
                    capabilities: Vec::new(),
                    correctness: Some(if pass { 1.0 } else { 0.5 }),
                    cost: Some(cost),
                    completion_tokens: Some(cost / 10),
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

#[test]
fn generation_two_drops_the_flat_tasks_and_prefers_the_cheaper_spread_ones() {
    let rig = Rig::new("campaign-signal");
    // seed: the test asserts one recorded draw of the campaign's tasks.
    let mut spec = rig.spec("signal", 2, 20_261_119);
    spec.rollout.curriculum = RolloutCurriculum::Signal;
    let mut practice = Costed::default();
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
    assert_eq!(
        asked[0].len(),
        TASKS.len(),
        "nothing is measured before generation 1"
    );
    assert!(
        asked[1].iter().all(|t| !t.starts_with("easy")),
        "{:?}",
        asked[1]
    );
    let records: Vec<&Value> = commits
        .iter()
        .filter(|c| c.unit.ends_with("/rollout"))
        .map(|c| &c.summary)
        .collect();
    let signal = &records[1]["signal"];
    assert_eq!(signal["uninformative"], json!(["easy1", "easy2"]));
    let weight = |t: &str| signal["weights"][t].as_f64().unwrap();
    for (split, dear) in [("split1", "dear1"), ("split2", "dear2")] {
        assert!(weight(split) > weight(dear), "{signal}");
    }
    assert_eq!(signal["reward"], "verdict");
}
