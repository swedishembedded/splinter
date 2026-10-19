// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: `harness optimise` runs candidate sven harnesses on a frozen task
//! set, verifies their attempts with the tasks' own verifiers, compares each
//! with the baseline's and promotes only a verified improvement with no
//! regressed family.
//!
//! The tasks are the faults `practice` makes of a tiny Python project. The
//! model is scripted to mend the module a request names only when the prompt
//! it is sent says `MEND`, so a harness whose agent step carries that word
//! solves every task and one without it solves none: the harness, not the
//! model, is what the candidates differ in.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::gate::{put_base, FakeTrainer};
use common::repo::{call, module_of, project, request_of, tool_calls};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::harness_opt::gate::{PromotionRule, Verdict};
use splinter_pipelines::harness_opt::space::MutationSpace;
use splinter_pipelines::harness_opt::{
    optimise, OptimiseRequest, ProposerChoice, CHAMPION_POINTER,
};
use splinter_pipelines::practice::{practice, PracticeRequest};
use splinter_store::tasks::TaskSetId;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};
use sven_sdk::search::Strategy;

/// Mends the module a request names, but only when the first user turn
/// contains `MEND`; otherwise looks and does nothing.
struct Prompted;

#[async_trait::async_trait]
impl ModelProvider for Prompted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "prompted-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let request = request_of(&req);
        let events = match (tool_calls(&req).len(), module_of(&request)) {
            (0, Some(n)) if request.contains("MEND") => call(
                "c-fix",
                "write_file",
                serde_json::json!({
                    "path": format!("shop/tier{n}.py"),
                    "text": common::repo::module(n),
                    "append": false
                }),
            ),
            _ => vec![ResponseEvent::TextDelta("done".into()), ResponseEvent::Done],
        };
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

fn editor_ref() -> ModelRef {
    "local:scripted/editor".parse().unwrap()
}

/// Answers each request with the next reply (the last one again after that).
struct Replies {
    replies: Vec<String>,
    next: std::sync::atomic::AtomicUsize,
}

impl Replies {
    fn new(replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            replies: replies.iter().map(ToString::to_string).collect(),
            next: 0.into(),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Replies {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "editor-1"
    }
    async fn complete(&self, _req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let at = self
            .next
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .min(self.replies.len().saturating_sub(1));
        let text = self.replies.get(at).cloned().unwrap_or_default();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(text)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

const PLAIN: &str = "{{ inputs.task }}";
const MENDING: &str = "MEND {{ inputs.task }}";

fn harness(prompt: &str) -> String {
    format!(
        r#"
api_version: "1"
name: solver
inputs: {{ task: string }}
capabilities: [ReadFile, WriteFile, ExecuteShell]
limits: {{ max_steps: 10, max_model_turns: 6, max_tool_calls: 6 }}
entry: work
steps:
  work: {{ agent: {{ prompt: "{prompt}", then: done }} }}
  done: {{ return: {{ value: {{ finished: true }} }} }}
"#
    )
}

fn space() -> MutationSpace {
    MutationSpace::parse(&format!(
        "knobs:\n  - path: [steps, work, agent, prompt]\n    choices: [\"{PLAIN}\", \"{MENDING}\"]\n"
    ))
    .unwrap()
}

/// A context whose `policy:default` is the prompt-sensitive model, over a
/// state root holding the tasks `practice` made of a tiny project.
fn context_with_tasks(test: &str) -> (Scratch, Context, TaskSetId) {
    context_with_editor(test, Replies::new(&[]))
}

/// As [`context_with_tasks`], with `editor` serving the `local:scripted/editor` model.
fn context_with_editor(test: &str, editor: Arc<Replies>) -> (Scratch, Context, TaskSetId) {
    let scratch = Scratch::new(test);
    let build = |model: Arc<dyn ModelProvider>| {
        Context::new(config(&scratch), false)
            .unwrap()
            .with_clock(Box::new(FixedClock::new("2026-10-18T08:00:00.000Z")))
            .with_model(
                ModelRef::policy_default(),
                Model::new(model, "scripted/prompted-1"),
            )
    };
    let making = build(common::repo::Mender::new(true));
    put_base(&making);
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let recorded = practice(
        &making,
        &PracticeRequest {
            repository: repo,
            revision: "HEAD".into(),
            max_tasks: Some(32),
            mutants_per_file: Some(1),
            exam_families: Some(1),
            pass_at_k: PassAtK::new(2, None, None),
            no_release: true,
            ..PracticeRequest::default()
        },
        &FakeTrainer::knowing(&[]),
    )
    .unwrap();
    let tasks = recorded.report.tasks.unwrap().task_set.unwrap();
    drop(making);
    let ctx =
        build(Arc::new(Prompted)).with_model(editor_ref(), Model::new(editor, "scripted/editor-1"));
    put_base(&ctx);
    (scratch, ctx, tasks)
}

fn request<'a>(
    tasks: &'a TaskSetId,
    solver: &'a ModelRef,
    baseline: &'a str,
) -> OptimiseRequest<'a> {
    OptimiseRequest {
        task_set: tasks,
        solver,
        baseline,
        space: Some(space()),
        proposer: splinter_pipelines::harness_opt::ProposerChoice::Mutation,
        strategy: Strategy::BestOfN,
        evaluations: 3,
        parallelism: 1,
        attempts: 1,
        rule: PromotionRule {
            alpha: 0.1,
            min_paired: 3,
        },
        // seed: the search seed fixes which candidates are drawn, and the
        // seed test asserts the same seed draws the same ones; the interval
        // seed only has to be some seed.
        search_seed: 7,
        interval_seed: Seed::new(1),
        cancel: splinter_agent::CancelToken::new(),
    }
}

#[test]
fn a_candidate_that_verifiably_solves_what_the_baseline_does_not_is_promoted() {
    let (_scratch, ctx, tasks) = context_with_tasks("harness-opt-promote");
    let solver = ModelRef::policy_default();
    let baseline = harness(PLAIN);

    let report = optimise(&ctx, &request(&tasks, &solver, &baseline)).unwrap();

    // The first evaluation is the baseline against its own arm: no evidence.
    assert_eq!(report.candidates[0].decision.verdict, Verdict::Inconclusive);
    let promoted = report.promoted.as_ref().expect("a verified improvement");
    let line = &report.candidates[promoted.index as usize];
    assert_eq!(line.decision.verdict, Verdict::Promote, "{line:#?}");
    assert!(line.evidence.regressed_families.is_empty());
    assert!(line.evidence.candidate_right > line.evidence.baseline_right);
    assert!(report.metrics.solved);

    // The pointer names the stored champion, whose document is the candidate.
    let (version, value) = ctx.workspace().pointer(CHAMPION_POINTER).unwrap().unwrap();
    assert_eq!((version, value.as_str()), (1, promoted.document.as_str()));
    let champion: serde_json::Value = ctx
        .workspace()
        .get_document("harness_champion", &promoted.document)
        .unwrap()
        .unwrap();
    assert!(champion["document"].as_str().unwrap().contains("MEND"));
}

#[test]
fn a_candidate_that_regresses_the_baseline_is_rejected_and_nothing_is_promoted() {
    let (_scratch, ctx, tasks) = context_with_tasks("harness-opt-reject");
    let solver = ModelRef::policy_default();
    let baseline = harness(MENDING);

    let report = optimise(&ctx, &request(&tasks, &solver, &baseline)).unwrap();

    assert!(report.promoted.is_none(), "{report:#?}");
    assert!(report.candidates.len() > 1);
    assert!(
        report.candidates[1..]
            .iter()
            .all(|c| c.decision.verdict == Verdict::Reject
                && !c.evidence.regressed_families.is_empty()),
        "{:#?}",
        report.candidates
    );
    assert!(!report.metrics.solved, "partial credit is not success");
    assert_eq!(ctx.workspace().pointer(CHAMPION_POINTER).unwrap(), None);
}

#[test]
fn the_same_search_seed_draws_the_same_candidates() {
    let (_scratch, ctx, tasks) = context_with_tasks("harness-opt-seed");
    let solver = ModelRef::policy_default();
    let baseline = harness(PLAIN);
    let first = optimise(&ctx, &request(&tasks, &solver, &baseline)).unwrap();
    let second = optimise(&ctx, &request(&tasks, &solver, &baseline)).unwrap();
    let documents = |r: &splinter_pipelines::harness_opt::Optimised| {
        r.candidates
            .iter()
            .map(|c| c.document.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(documents(&first), documents(&second));

    // Candidates evaluated two at a time are the same candidates.
    let parallel = optimise(
        &ctx,
        &OptimiseRequest {
            parallelism: 2,
            ..request(&tasks, &solver, &baseline)
        },
    )
    .unwrap();
    assert_eq!(documents(&first), documents(&parallel));
}

const BREAKS: &str = r#"{"rationale": "r", "edits": [{"path": ["entry"], "value": "nowhere"}]}"#;
const MENDS: &str = r#"{"rationale": "ask it to mend", "edits": [{"path": ["steps", "work", "agent", "prompt"], "value": "MEND {{ inputs.task }}"}]}"#;

fn by_model<'a>(
    tasks: &'a TaskSetId,
    solver: &'a ModelRef,
    baseline: &'a str,
    space: Option<MutationSpace>,
) -> OptimiseRequest<'a> {
    OptimiseRequest {
        space,
        proposer: ProposerChoice::Model {
            model: editor_ref(),
            deadline: std::time::Duration::from_secs(60),
            repairs: 2,
        },
        evaluations: 2,
        ..request(tasks, solver, baseline)
    }
}

#[test]
fn a_model_edit_the_compiler_refuses_is_corrected_before_it_counts_and_then_promoted() {
    let editor = Replies::new(&[BREAKS, MENDS]);
    let (_scratch, ctx, tasks) = context_with_editor("harness-opt-model", editor);
    let solver = ModelRef::policy_default();
    let baseline = harness(PLAIN);

    let report = optimise(&ctx, &by_model(&tasks, &solver, &baseline, None)).unwrap();

    let stats = report.proposals.expect("a model proposed");
    assert_eq!(
        (
            stats.model_proposals,
            stats.model_failures,
            stats.mutation_fallbacks
        ),
        (1, 0, 0)
    );
    let promoted = report
        .promoted
        .as_ref()
        .expect("the corrected edit mends the tasks");
    assert_eq!(
        report.candidates[promoted.index as usize].decision.verdict,
        Verdict::Promote
    );
    // Only the document that compiled was ever evaluated.
    assert_eq!(report.candidates.len(), 2);
}

#[test]
fn a_model_that_never_makes_a_compiling_edit_is_replaced_by_a_mutation_or_stops_the_run() {
    let (_scratch, ctx, tasks) =
        context_with_editor("harness-opt-fallback", Replies::new(&[BREAKS]));
    let solver = ModelRef::policy_default();
    let baseline = harness(PLAIN);

    let with_space = optimise(&ctx, &by_model(&tasks, &solver, &baseline, Some(space()))).unwrap();
    let stats = with_space.proposals.unwrap();
    assert_eq!(
        (
            stats.model_proposals,
            stats.model_failures,
            stats.mutation_fallbacks
        ),
        (0, 1, 1)
    );
    assert!(
        with_space.promoted.is_some(),
        "the mutation of the space mends the tasks"
    );

    let without = optimise(&ctx, &by_model(&tasks, &solver, &baseline, None));
    let message = without.unwrap_err().to_string();
    assert!(message.contains("no edit that compiles"), "{message}");
}
