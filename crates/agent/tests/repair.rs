// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements critique-and-retry loops that turn failed
// agent attempts into verified revisions, for its clients. If your team
// needs expertise in agent self-improvement or reward modelling, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a failed experience becomes a critique, a retry with the critique
//! produces a revision, and the chain is recorded as relations in the
//! experience store that the critic, decision and preference views read.
//!
//! * The critic sees the task's instruction, the failed answer and the
//!   verifiers' evidence summaries - never the checks' code or the
//!   reference.
//! * A critique is verified by outcome: pass when the retry that received
//!   it passes where the critiqued attempt failed, fail when it still fails.
//! * The revision's instruction stays the task's; the critique is teacher-
//!   only material a student view strips.
//! * The repair loop is bounded, never retries a pass, and stops on an
//!   environment error.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::critic::{Critic, CRITIQUE_TASK_KIND};
use splinter_agent::repair::{Repair, RepairBudget, RepairError, RepairReport, Stop};
use splinter_agent::solve::{solve, Model, SolveOptions, SYSTEM_PROMPT};
use splinter_lab::verifiers::executable::{ExecutableCheck, ExecutableVerifier, Expectation};
use splinter_lab::verifiers::{verify_and_annotate, Strongest};
use splinter_record::annotation::{decide, AnnotationBody, RelationKind, Strength};
use splinter_record::clock::Clock;
use splinter_record::experience::{ExperienceId, Privileged, PrivilegedKind, Provenance, Task};
use splinter_record::experiences::ExperienceStore;
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeRegistry,
};
use splinter_views::{
    Corpus, Critic as CriticView, DecisionView, Exclusion, Preference, RecordBody, SftFinal, Strip,
    View,
};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};

const SOLVER: &str = "scripted/solver-1";
const CRITIC: &str = "scripted/critic-1";

const INSTRUCTION: &str =
    "Write a Python function add(a, b) that returns the sum of a and b. Reply with the code.";
const WRONG: &str = "```python\ndef add(a, b):\n    return a - b\n```\n";
const CORRECT: &str = "```python\ndef add(a, b):\n    return a + b\n```\n";
/// The reference solution: never shown to the critic.
const REFERENCE: &str = "def add(a, b):\n    return a + b  # REFERENCE-SOLUTION\n";
/// The check's code: never shown to the critic.
const CHECK_CODE: &str = "assert add(2, 3) == 5, 'SECRET-CHECK'\n";
/// A critique that quotes the instruction, as critiques do.
const CRITIQUE: &str = "The task asks for a function that returns the sum of a and b, but \
    this one subtracts b from a, so every call with a nonzero b is wrong.";

/// A model whose reply is a function of the last user message, keeping
/// every such message it was sent.
struct Scripted {
    name: &'static str,
    reply: fn(&str) -> String,
    prompts: Mutex<Vec<String>>,
}

impl Scripted {
    fn new(name: &'static str, reply: fn(&str) -> String) -> Arc<Self> {
        Arc::new(Self {
            name,
            reply,
            prompts: Mutex::new(Vec::new()),
        })
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        self.name
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let prompt = req
            .messages
            .iter()
            .rev()
            .find_map(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let reply = (self.reply)(&prompt);
        self.prompts.lock().unwrap().push(prompt);
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// Answers correctly only once it has been told what was wrong.
fn learns_from_critique(prompt: &str) -> String {
    if prompt.contains(CRITIQUE) {
        CORRECT.into()
    } else {
        WRONG.into()
    }
}

fn always_wrong(_: &str) -> String {
    WRONG.into()
}

fn always_right(_: &str) -> String {
    CORRECT.into()
}

fn critiques(_: &str) -> String {
    CRITIQUE.into()
}

/// A clock that advances one second per reading, so every record it stamps
/// is a distinct experience.
#[derive(Default)]
struct Ticking(AtomicU64);

impl Clock for Ticking {
    fn utc_now(&self) -> String {
        let tick = self.0.fetch_add(1, Ordering::Relaxed);
        format!("2026-09-30T10:{:02}:{:02}.000Z", tick / 60, tick % 60)
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("splinter-repair-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(path.join("sandbox")).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Everything one spec works with: a python environment, a construct task
/// checked in it, a store, and the verifiers the task is graded by.
struct Bench {
    _scratch: Scratch,
    rt: tokio::runtime::Runtime,
    env: RuntimeEnvironment,
    task: Task,
    store: ExperienceStore,
    clock: Ticking,
    verifiers: Strongest,
}

impl Bench {
    fn new(test: &str) -> Self {
        let scratch = Scratch::new(test);
        let sandbox = ProcessSandbox::new(
            scratch.0.join("sandbox"),
            BTreeMap::new(),
            &[],
            Limits::default(),
        );
        let env =
            RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "python3", Arc::new(sandbox))
                .unwrap();
        let check = ExecutableCheck {
            code: CHECK_CODE.into(),
            stdin: None,
            expect: Expectation::exit_code(0),
            environment: None,
        };
        let task = Task::new(
            "construct",
            vec![],
            ResolvedEnvironment::Runtime(env.clone()).record().unwrap(),
            INSTRUCTION,
            vec![
                Privileged {
                    kind: PrivilegedKind::Reference,
                    content: REFERENCE.into(),
                    span: None,
                },
                check.as_check().unwrap(),
            ],
        )
        .unwrap();
        let store = ExperienceStore::new(&Workspace::at(&StateRoot::new(scratch.0.join("state"))));
        let verifiers = Strongest::new(vec![Box::new(ExecutableVerifier::new(vec![env.clone()]))]);
        Self {
            _scratch: scratch,
            rt: tokio::runtime::Runtime::new().unwrap(),
            env,
            task,
            store,
            clock: Ticking::default(),
            verifiers,
        }
    }

    fn environment(&self) -> ResolvedEnvironment {
        ResolvedEnvironment::Runtime(self.env.clone())
    }

    /// A first attempt by a model replying `reply`, stored and verified.
    fn attempt(&self, reply: fn(&str) -> String) -> ExperienceId {
        let solution = self
            .rt
            .block_on(solve(
                &self.task,
                &self.environment(),
                Scripted::new("solver-1", reply),
                SolveOptions::new(Duration::from_secs(60)),
            ))
            .unwrap();
        let experience = solution
            .into_experience(self.task.clone(), Provenance::new(SOLVER, &self.clock))
            .unwrap();
        let id = self.store.put(&experience).unwrap();
        let report =
            verify_and_annotate(&self.store, &self.verifiers, &self.task, &experience).unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        id
    }

    fn repair<'a>(
        &'a self,
        environment: &'a ResolvedEnvironment,
        solver: &Arc<Scripted>,
        critic: &Arc<Scripted>,
    ) -> Repair<'a> {
        Repair {
            store: &self.store,
            clock: &self.clock,
            runtime: self.rt.handle().clone(),
            environment,
            verifiers: &self.verifiers,
            solver: Model::new(solver.clone(), SOLVER),
            critic: Critic::new(Model::new(critic.clone(), CRITIC)),
        }
    }

    fn related(&self, id: &ExperienceId, kind: RelationKind) -> Vec<ExperienceId> {
        self.store
            .annotations(id)
            .unwrap()
            .annotations
            .into_iter()
            .filter_map(|note| match note.body {
                AnnotationBody::Relation { kind: k, other } if k == kind => Some(other),
                _ => None,
            })
            .collect()
    }

    fn passed(&self, id: &ExperienceId) -> Option<bool> {
        decide(&self.store.annotations(id).unwrap().annotations).map(|d| d.passed)
    }

    fn corpus(&self, ids: &[ExperienceId]) -> Corpus {
        Corpus::load(&self.store, ids).unwrap()
    }
}

fn budget(max_retries: usize) -> RepairBudget {
    RepairBudget::new(max_retries, Duration::from_secs(120))
}

/// A failed attempt repaired by a critique the solver learns from.
struct Chain {
    first: ExperienceId,
    critique: ExperienceId,
    revision: ExperienceId,
    report: RepairReport,
    critic_prompts: Vec<String>,
    solver_prompts: Vec<String>,
}

fn repaired(bench: &Bench) -> Chain {
    let first = bench.attempt(always_wrong);
    assert_eq!(bench.passed(&first), Some(false), "the first attempt fails");
    let solver = Scripted::new("solver-1", learns_from_critique);
    let critic = Scripted::new("critic-1", critiques);
    let environment = bench.environment();
    let report = bench
        .repair(&environment, &solver, &critic)
        .repair_loop(&bench.task, &first, &budget(3))
        .unwrap();
    assert!(matches!(report.stop, Stop::Passed), "{:?}", report.stop);
    assert_eq!(report.rounds.len(), 1);
    let round = &report.rounds[0];
    let retried = round.retried.as_ref().unwrap();
    Chain {
        first,
        critique: round.critique.id.clone(),
        revision: retried.revision.clone(),
        critic_prompts: critic.prompts(),
        solver_prompts: solver.prompts(),
        report,
    }
}

/// A chat record's turns after its first, which is the system turn every
/// solve in the chain ran under.
fn chat(body: &RecordBody) -> Vec<(String, String)> {
    match body {
        RecordBody::Chat { messages } => {
            let turns: Vec<(String, String)> = messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();
            assert_eq!(turns[0], ("system".into(), SYSTEM_PROMPT.into()));
            turns[1..].to_vec()
        }
        other => panic!("a chat record, got {other:?}"),
    }
}

#[test]
fn a_failed_construct_task_is_critiqued_retried_and_the_chain_feeds_the_views() {
    let bench = Bench::new("chain");
    let chain = repaired(&bench);

    // The critic saw the task, the answer and how the checks ended; not the
    // checks' code, the reference, or anything else only the teacher has.
    assert_eq!(chain.critic_prompts.len(), 1);
    let prompt = &chain.critic_prompts[0];
    assert!(prompt.contains(INSTRUCTION), "{prompt}");
    assert!(
        prompt.contains("return a - b"),
        "the failed answer: {prompt}"
    );
    assert!(
        prompt.contains("check 1: python") && prompt.contains("exit code 1"),
        "the check's evidence summary: {prompt}"
    );
    assert!(prompt.contains("failed checks: check 1"), "{prompt}");
    for hidden in [
        "SECRET-CHECK",
        "assert add",
        "REFERENCE-SOLUTION",
        "executable-check",
    ] {
        assert!(
            !prompt.contains(hidden),
            "{hidden:?} reached the critic: {prompt}"
        );
    }

    // The critique is an experience of its own, of the critic's closed-book
    // question, answered with the critique.
    let critique = bench.store.get(&chain.critique).unwrap();
    assert_eq!(critique.task.kind, CRITIQUE_TASK_KIND);
    assert_eq!(&critique.instruction, prompt);
    assert!(critique.privileged.is_empty());
    assert_eq!(critique.final_output.as_deref(), Some(CRITIQUE));
    assert_eq!(critique.provenance.solver, CRITIC);

    // The retry was told the critique; the revision records the task's own
    // instruction and carries the critique as teacher-only material.
    assert!(chain.solver_prompts.iter().any(|p| p.contains(CRITIQUE)));
    let revision = bench.store.get(&chain.revision).unwrap();
    assert_eq!(revision.instruction, INSTRUCTION);
    assert_eq!(revision.final_output.as_deref(), Some(CORRECT));
    assert!(revision
        .privileged
        .iter()
        .any(|p| p.kind == PrivilegedKind::Critique && p.content == CRITIQUE));

    // The chain, as relations, and the verdicts that close it.
    assert_eq!(
        bench.related(&chain.critique, RelationKind::CritiqueOf),
        std::slice::from_ref(&chain.first)
    );
    assert_eq!(
        bench.related(&chain.revision, RelationKind::RetryOf),
        std::slice::from_ref(&chain.first)
    );
    assert_eq!(
        bench.related(&chain.revision, RelationKind::PreferredOver),
        std::slice::from_ref(&chain.first)
    );
    assert_eq!(bench.passed(&chain.revision), Some(true));
    assert_eq!(
        bench.passed(&chain.critique),
        Some(true),
        "verified by outcome"
    );
    let retried = chain.report.rounds[0].retried.as_ref().unwrap();
    assert!(retried.preferred);
    assert_eq!(retried.decision.unwrap().strength, Strength::Executable);

    // Each view projects its record from the chain, through the store.
    let corpus = bench.corpus(&[
        chain.first.clone(),
        chain.critique.clone(),
        chain.revision.clone(),
    ]);
    let critic = CriticView::new(Strength::Consistency)
        .project(&corpus)
        .unwrap();
    assert_eq!(critic.records.len(), 1, "{:?}", critic.excluded);
    let turns = chat(&critic.records[0].body);
    assert!(turns[0].1.starts_with(INSTRUCTION) && turns[0].1.contains("return a - b"));
    assert_eq!(turns[1], ("assistant".into(), CRITIQUE.into()));

    let decision = DecisionView::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    assert_eq!(decision.records.len(), 1, "{:?}", decision.excluded);
    let turns = chat(&decision.records[0].body);
    assert_eq!(turns[0], ("user".into(), INSTRUCTION.into()));
    assert_eq!(turns.last().unwrap(), &("assistant".into(), CORRECT.into()));

    let preference = Preference::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    assert_eq!(preference.records.len(), 1, "{:?}", preference.excluded);
    match &preference.records[0].body {
        RecordBody::Preference {
            prompt,
            chosen,
            rejected,
        } => {
            assert_eq!(
                (prompt[0].content.as_str(), prompt[1].content.as_str()),
                (SYSTEM_PROMPT, INSTRUCTION)
            );
            assert_eq!(chosen.content, CORRECT);
            assert_eq!(rejected.content, WRONG);
        }
        other => panic!("a preference record, got {other:?}"),
    }
}

#[test]
fn the_revisions_student_view_does_not_contain_the_critique() {
    let bench = Bench::new("student");
    let chain = repaired(&bench);
    let corpus = bench.corpus(std::slice::from_ref(&chain.revision));

    let stripped = SftFinal::new(Strength::Executable)
        .with_strip(Strip::All)
        .project(&corpus)
        .unwrap();
    assert_eq!(stripped.records.len(), 1, "{:?}", stripped.excluded);
    let rendered = serde_json::to_string(&stripped.records[0].body).unwrap();
    assert!(!rendered.contains(CRITIQUE), "{rendered}");
    assert_eq!(chat(&stripped.records[0].body)[0].1, INSTRUCTION);

    let kept = SftFinal::new(Strength::Executable)
        .with_strip(Strip::Keep(vec![PrivilegedKind::Critique]))
        .project(&corpus)
        .unwrap();
    let rendered = serde_json::to_string(&kept.records[0].body).unwrap();
    assert!(
        rendered.contains(CRITIQUE),
        "a strip that keeps critiques shows it"
    );
}

#[test]
fn a_critique_whose_retry_still_fails_is_recorded_fail_and_the_critic_view_excludes_it() {
    let bench = Bench::new("still-fails");
    let first = bench.attempt(always_wrong);
    let solver = Scripted::new("solver-1", always_wrong);
    let critic = Scripted::new("critic-1", critiques);
    let environment = bench.environment();
    let report = bench
        .repair(&environment, &solver, &critic)
        .repair_loop(&bench.task, &first, &budget(1))
        .unwrap();
    assert!(matches!(report.stop, Stop::Retries), "{:?}", report.stop);
    let round = &report.rounds[0];
    let revision = &round.retried.as_ref().unwrap().revision;
    assert_eq!(bench.passed(revision), Some(false));
    assert_eq!(bench.passed(&round.critique.id), Some(false));
    assert!(bench
        .related(revision, RelationKind::PreferredOver)
        .is_empty());

    let corpus = bench.corpus(&[first, round.critique.id.clone(), revision.clone()]);
    let critic = CriticView::new(Strength::Judged).project(&corpus).unwrap();
    assert!(critic.records.is_empty());
    assert_eq!(critic.count(Exclusion::UnverifiedCritique), 1);
}

#[test]
fn the_loop_stops_at_its_budget_and_never_retries_a_pass() {
    let bench = Bench::new("budget");
    let environment = bench.environment();

    // Two retries, both failing: each round critiques the latest failure.
    let first = bench.attempt(always_wrong);
    let solver = Scripted::new("solver-1", always_wrong);
    let critic = Scripted::new("critic-1", critiques);
    let report = bench
        .repair(&environment, &solver, &critic)
        .repair_loop(&bench.task, &first, &budget(2))
        .unwrap();
    assert!(matches!(report.stop, Stop::Retries), "{:?}", report.stop);
    assert_eq!(report.rounds.len(), 2);
    assert_eq!(critic.prompts().len(), 2);
    let first_revision = report.rounds[0].retried.as_ref().unwrap().revision.clone();
    assert_eq!(report.rounds[1].critique.critiqued, first_revision);
    assert_eq!(
        bench.related(&report.rounds[1].critique.id, RelationKind::CritiqueOf),
        [first_revision]
    );

    // A spent deadline stops before anything runs.
    let critic = Scripted::new("critic-1", critiques);
    let report = bench
        .repair(&environment, &solver, &critic)
        .repair_loop(&bench.task, &first, &RepairBudget::new(3, Duration::ZERO))
        .unwrap();
    assert!(matches!(report.stop, Stop::Deadline), "{:?}", report.stop);
    assert!(report.rounds.is_empty() && critic.prompts().is_empty());

    // A pass is never retried, by the loop or directly.
    let passing = bench.attempt(always_right);
    let critic = Scripted::new("critic-1", critiques);
    let repair = bench.repair(&environment, &solver, &critic);
    let report = repair
        .repair_loop(&bench.task, &passing, &budget(3))
        .unwrap();
    assert!(
        matches!(report.stop, Stop::NotFailed(_)),
        "{:?}",
        report.stop
    );
    assert!(report.rounds.is_empty() && critic.prompts().is_empty());
    let critique = repair.critic.critique(
        &bench.store,
        &bench.clock,
        bench.rt.handle(),
        &passing,
        SolveOptions::new(Duration::from_secs(60)),
    );
    assert!(matches!(critique, Err(RepairError::NotFailed { .. })));
}

#[test]
fn the_loop_stops_early_when_the_task_cannot_run_in_the_offered_environment() {
    let bench = Bench::new("environment");
    let first = bench.attempt(always_wrong);
    let solver = Scripted::new("solver-1", always_right);
    let critic = Scripted::new("critic-1", critiques);
    let closed_book = ResolvedEnvironment::ClosedBook;
    let report = bench
        .repair(&closed_book, &solver, &critic)
        .repair_loop(&bench.task, &first, &budget(3))
        .unwrap();
    assert!(
        matches!(report.stop, Stop::Error(RepairError::Solve(_))),
        "{:?}",
        report.stop
    );
    assert_eq!(report.rounds.len(), 1, "the critique was written");
    assert!(report.rounds[0].retried.is_none());
    assert!(solver.prompts().is_empty(), "the solver never ran");
}
