// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a pipeline runs its stages in order and records each as the run's
//! own, whatever the stages do.
//!
//! The engine, not the stage, checks for a cancel and for the budget before
//! a stage, skips a stage that does not apply, stops where a stage says to,
//! and records every stage that ran - with how long it took, and why it
//! failed when it did - so lineage of a run is a property of running it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde_json::json;
use splinter_core::clock::FixedClock;
use splinter_orchestrator::pipeline::{Flow, Pipeline, Stage, StageEnd};
use splinter_orchestrator::runs::{record, Recorder};
use splinter_orchestrator::{Config, Context, OrchestratorError};
use splinter_store::runs::read_run;
use splinter_store::StateRoot;

/// A scratch state root removed when dropped.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(test: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("splinter-pipeline-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn context(scratch: &Scratch) -> Context {
    let config = Config {
        state_root: StateRoot::new(scratch.0.join("state")),
        model_store: scratch.0.join("models"),
        policy_base: scratch.0.join("models/base"),
        openrouter_api_key: None,
        brain_api_key: None,
        allow_remote: false,
        command_env: BTreeMap::new(),
        working_dir: scratch.0.clone(),
        brain_binary: None,
        front_door_model: None,
        assistant_model: None,
        bf16_base: false,
        default_budget: None,
    };
    Context::new(config, false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-09-30T08:00:00.000Z")))
}

/// A stage that notes it ran in the state, and does what it is told.
struct Step {
    name: &'static str,
    enabled: bool,
    budgeted: bool,
    stop: Option<&'static str>,
    cancel: bool,
    fail: bool,
}

impl Step {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            enabled: true,
            budgeted: true,
            stop: None,
            cancel: false,
            fail: false,
        }
    }
}

impl Stage<Vec<String>> for Step {
    fn name(&self) -> &'static str {
        self.name
    }

    fn enabled(&self, _: &Vec<String>) -> bool {
        self.enabled
    }

    fn honors_budget(&self) -> bool {
        self.budgeted
    }

    fn run(
        &self,
        _: &Context,
        run: &mut Recorder<'_>,
        state: &mut Vec<String>,
    ) -> Result<StageEnd, OrchestratorError> {
        state.push(self.name.to_string());
        if self.cancel {
            run.cancel_token().cancel();
        }
        if self.fail {
            return Err(OrchestratorError::Refused(format!("{} broke", self.name)));
        }
        Ok(StageEnd {
            summary: json!({ "stage": self.name }),
            flow: self
                .stop
                .map_or(Flow::Continue, |why| Flow::Stop(why.into())),
        })
    }
}

fn pipeline(steps: Vec<Step>) -> Pipeline<Vec<String>> {
    steps
        .into_iter()
        .fold(Pipeline::new(), |pipeline, step| pipeline.then(step))
}

/// How a pipeline ended: the stages that ran and why it stopped, if it did.
type Ended = Result<(Vec<String>, Option<String>), OrchestratorError>;

/// Runs `pipeline` as a recorded run; how it ended, and the recorded run.
fn execute(
    test: &str,
    pipeline: &Pipeline<Vec<String>>,
    deadline: Option<Instant>,
) -> (Ended, splinter_store::runs::Run) {
    let scratch = Scratch::new(test);
    let ctx = context(&scratch);
    let mut ran = Vec::new();
    let mut run_id = String::new();
    let outcome = record(&ctx, "test", &json!({}), |run| {
        run_id = run.id().to_string();
        let stopped = pipeline.run(&ctx, run, &mut ran, deadline)?;
        Ok(json!({ "stopped": stopped }))
    })
    .map(|_| ());
    let run = read_run(ctx.workspace(), &run_id).unwrap();
    let result = outcome.map(|()| {
        let stopped = run.outputs["stopped"].as_str().map(str::to_string);
        (ran.clone(), stopped)
    });
    (result, run)
}

#[test]
fn stages_run_in_order_and_each_is_recorded_with_its_summary_and_time() {
    let line = pipeline(vec![Step::new("a"), Step::new("b"), Step::new("c")]);
    assert_eq!(line.names(), ["a", "b", "c"]);
    let (result, run) = execute("order", &line, None);
    let (ran, stopped) = result.unwrap();
    assert_eq!(ran, ["a", "b", "c"]);
    assert_eq!(stopped, None);
    let recorded: Vec<(&str, &serde_json::Value)> = run
        .stages
        .iter()
        .map(|s| (s.stage.as_str(), &s.summary))
        .collect();
    assert_eq!(
        recorded,
        [
            ("a", &json!({ "stage": "a" })),
            ("b", &json!({ "stage": "b" })),
            ("c", &json!({ "stage": "c" })),
        ]
    );
    assert!(run
        .stages
        .iter()
        .all(|s| s.duration_ms.is_some() && s.error.is_none()));
}

#[test]
fn a_stage_that_does_not_apply_is_skipped_and_not_recorded() {
    let mut skipped = Step::new("b");
    skipped.enabled = false;
    let line = pipeline(vec![Step::new("a"), skipped, Step::new("c")]);
    let (result, run) = execute("skip", &line, None);
    assert_eq!(result.unwrap().0, ["a", "c"]);
    let names: Vec<&str> = run.stages.iter().map(|s| s.stage.as_str()).collect();
    assert_eq!(names, ["a", "c"]);
}

#[test]
fn a_stage_can_stop_the_pipeline_and_say_why() {
    let mut gate = Step::new("b");
    gate.stop = Some("nothing to train on");
    let line = pipeline(vec![Step::new("a"), gate, Step::new("c")]);
    let (result, run) = execute("stop", &line, None);
    let (ran, stopped) = result.unwrap();
    assert_eq!(ran, ["a", "b"], "no stage runs after the stop");
    assert_eq!(stopped.as_deref(), Some("nothing to train on"));
    let names: Vec<&str> = run.stages.iter().map(|s| s.stage.as_str()).collect();
    assert_eq!(names, ["a", "b"], "the stopping stage is still recorded");
}

#[test]
fn a_cancel_stops_the_run_before_the_next_stage() {
    let mut cancelling = Step::new("a");
    cancelling.cancel = true;
    let line = pipeline(vec![cancelling, Step::new("b")]);
    let (result, run) = execute("cancel", &line, None);
    assert!(matches!(result, Err(OrchestratorError::Cancelled)));
    assert_eq!(run.status, splinter_store::runs::RunStatus::Cancelled);
    let names: Vec<&str> = run.stages.iter().map(|s| s.stage.as_str()).collect();
    assert_eq!(names, ["a"]);
}

#[test]
fn a_spent_budget_stops_the_run_before_a_stage_that_honors_it() {
    let mut free = Step::new("setup");
    free.budgeted = false;
    let line = pipeline(vec![free, Step::new("work"), Step::new("more")]);
    let spent = Instant::now().checked_sub(Duration::from_secs(1));
    let (result, run) = execute("budget", &line, spent);
    let (ran, stopped) = result.unwrap();
    assert_eq!(ran, ["setup"], "a stage that ignores the budget still runs");
    assert_eq!(
        stopped.as_deref(),
        Some("the budget was spent before the work stage")
    );
    assert_eq!(run.stages.len(), 1);
}

#[test]
fn a_failing_stage_is_recorded_with_why_and_the_failure_propagates() {
    let mut broken = Step::new("b");
    broken.fail = true;
    let line = pipeline(vec![Step::new("a"), broken, Step::new("c")]);
    let (result, run) = execute("failure", &line, None);
    assert!(matches!(result, Err(OrchestratorError::Refused(_))));
    assert_eq!(run.status, splinter_store::runs::RunStatus::Failed);
    let last = run.stages.last().unwrap();
    assert_eq!(last.stage, "b");
    assert_eq!(last.error.as_deref(), Some("b broke"));
    assert!(run.stages[0].error.is_none());
}
