// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that keeps a long
// verified trajectory in the dataset instead of dropping it, for its
// clients. If your team needs expertise in turning long agent trajectories
// into trainable windows then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: `dataset build` splits a record over the training block only when
//! it is asked to, and the unasked build keeps leaving it out.
//!
//! * With `split_over_block` false, a record longer than `--max-block` is
//!   still left out and counted as `OverBlock`.
//! * With it true, the same record is split into windows that each fit, and
//!   nothing is counted as `OverBlock`.
//! * When the policy base has a chat tokenizer, "fits" is the trainer's own
//!   count: every record the build writes passes the trainer's length check
//!   at the same `--max-block`, including a record whose JSON text fits a
//!   bound its rendered request does not. Without one, the build says it fell
//!   back to the floor.
//! * A build that names the base it is for has every record read by that
//!   base's trainer (brain's own check, through the service): a record the
//!   trainer refuses is left out under its own reason, never as over the
//!   block, so training never meets it; with no service to ask, such a build
//!   is refused.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use common::{config, scratch_context, Scratch, Scripted};
use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_core::model_ref::ModelRef;
use splinter_data::{chars_as_tokens, dataset_line, BlockMeasure, Exclusion, Projection, Record};
use splinter_model::capabilities::overlong_records;
use splinter_model::measure::RecordMeter;
use splinter_model::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::ArtifactStore;
use splinter_model::BrainEndpoints;
use splinter_orchestrator::Context;
use splinter_pipelines::datasets::{project_with, BuildRequest, ViewName, VoiceBuild};
use splinter_store::experiences::{ExperienceSet, SetId};

const INSTRUCTION: &str = "Make the failing test pass.";

/// An action that edits `path` and is answered with `ok`.
fn edit(step: u64, call: &str, path: &str) -> TraceStep {
    let mut step = TraceStep::new(step, StepOrigin::Agent, "");
    step.tool_calls =
        Some(vec![ToolInvocation::new(call, "edit")
            .with_arguments(json!({"path": path, "text": "fixed"}))]);
    step.observation = Some(StepObservation::single(ObservationEntry::for_call(
        call, "ok",
    )));
    step
}

/// Two actions and a final answer: a trajectory long enough that its packed
/// record can be split into more than one window.
fn attempt(ctx: &Context) -> Experience {
    let task = Task::new(
        WORKSPACE_CHANGE,
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({"snapshot": "tree-1"})),
        INSTRUCTION,
        Vec::new(),
    )
    .unwrap();
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    trajectory.steps = vec![
        TraceStep::new(1, StepOrigin::User, INSTRUCTION),
        edit(2, "c1", "a.py"),
        edit(3, "c2", "b.py"),
        TraceStep::new(4, StepOrigin::Agent, "Done."),
    ];
    let experience = Experience::new(
        task,
        trajectory,
        Some("PATCH".into()),
        Provenance::new(
            "scripted/solver-1",
            &FixedClock::new("2026-10-24T01:00:00.000Z"),
        ),
    )
    .unwrap();
    let id = ctx.experiences().put(&experience).unwrap();
    ctx.experiences()
        .annotate(&Annotation {
            experience: id,
            producer: Producer {
                name: "grader".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: Outcome::Pass,
                strength: Strength::Executable,
                evidence: json!({}),
            },
        })
        .unwrap();
    experience
}

fn set_of(ctx: &Context, experience: &Experience) -> SetId {
    ctx.experiences()
        .put_set(&ExperienceSet {
            name: "attempts".into(),
            members: vec![experience.id().unwrap()],
        })
        .unwrap()
}

fn request(set: SetId, max_block: Option<usize>, split_over_block: bool) -> BuildRequest {
    BuildRequest {
        sets: vec![set],
        view: ViewName::SftStep,
        strip: None,
        min_strength: Some(Strength::Executable),
        system_prompt: None,
        export_only: false,
        limit: None,
        voice: VoiceBuild::default(),
        max_block,
        pack: true,
        control: Default::default(),
        serialise_calls: false,
        only_actions: None,
        roles: None,
        images: false,
        rl: Default::default(),
        transition: Default::default(),
        practice: Vec::new(),
        split_seed: None,
        history: Default::default(),
        pass_curation: Default::default(),
        base: None,
        split_over_block,
    }
}

/// The bound the build measures a record by when no service can count it:
/// characters at the fallback ratio, plus the rows of its pictures.
fn measure(_ctx: &Context) -> impl Fn(&Record) -> usize {
    |record: &Record| {
        let (text, image_rows) = record.body.text_and_image_rows();
        chars_as_tokens(&text) + image_rows
    }
}

/// A context whose brain is a scripted service: it counts a record's
/// characters as its tokens, as the stand-in for the trainer's own count.
fn served_context(test: &str) -> (Scratch, Context, ScriptedEndpoint, BrainEndpoints) {
    served_context_with(test, |script| script)
}

/// [`served_context`] whose training service behaves as `script` makes it.
fn served_context_with(
    test: &str,
    script: impl FnOnce(JobScript) -> JobScript,
) -> (Scratch, Context, ScriptedEndpoint, BrainEndpoints) {
    let scratch = Scratch::new(test);
    let store = scratch.0.join("artifacts");
    std::fs::create_dir_all(&store).unwrap();
    let service =
        ScriptedEndpoint::start(vec![]).with_training(script(JobScript::over(store.clone())));
    let brain = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&store).unwrap());
    let mut config = config(&scratch);
    config.brain = Some(brain.clone());
    let ctx = Context::new(config, false).unwrap();
    (scratch, ctx, service, brain)
}

#[test]
fn a_build_splits_an_over_block_record_only_when_asked() {
    let (scratch, ctx) = scratch_context("dataset-split", Scripted::new(|_| String::new()), false);
    std::mem::forget(scratch);
    let experience = attempt(&ctx);
    let set = set_of(&ctx, &experience);

    let (unbounded, _) = project_with(&ctx, &request(set.clone(), None, false)).unwrap();
    assert_eq!(unbounded.records.len(), 1, "one record per trajectory");
    let measure = measure(&ctx);
    let bound = measure(&unbounded.records[0]) - 1;

    let (kept, _) = project_with(&ctx, &request(set.clone(), Some(bound), false)).unwrap();
    assert!(kept.records.is_empty(), "unasked, the record is left out");
    assert_eq!(kept.count(Exclusion::OverBlock), 1);

    let (split, _) = project_with(&ctx, &request(set, Some(bound), true)).unwrap();
    assert!(split.records.len() > 1, "asked, the record is split");
    assert_eq!(split.count(Exclusion::OverBlock), 0);
    for record in &split.records {
        assert!(measure(record) <= bound, "every window fits the bound");
    }
}

/// The dataset file of `projection`, as `dataset build` writes it.
fn written(scratch: &std::path::Path, projection: &Projection) -> std::path::PathBuf {
    let path = scratch.join("built.jsonl");
    let lines: Vec<String> = projection
        .records
        .iter()
        .map(|record| dataset_line(projection.objective, record).unwrap())
        .collect();
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    path
}

#[test]
fn a_build_bounds_records_by_the_count_the_trainer_checks_them_by() {
    let (scratch, ctx, _service, brain) = served_context("dataset-exact");
    let base = ctx.config().policy_base.clone();
    let experience = attempt(&ctx);
    let set = set_of(&ctx, &experience);

    let (unbounded, _) = project_with(&ctx, &request(set.clone(), None, false)).unwrap();
    assert_eq!(unbounded.block_measure, None, "no bound, no measure");
    let record = &unbounded.records[0];
    let meter = RecordMeter::for_model(&brain, &base, false, unbounded.packed);
    let rendered = meter
        .tokens(&dataset_line(unbounded.objective, record).unwrap())
        .unwrap();
    let floor = rendered - 1;

    // Left out at one token under the trainer's count, measured by the trainer.
    let (left_out, _) = project_with(&ctx, &request(set.clone(), Some(floor), false)).unwrap();
    assert!(left_out.records.is_empty());
    assert_eq!(left_out.count(Exclusion::OverBlock), 1);
    assert_eq!(
        left_out.block_measure,
        Some(BlockMeasure::Trainer {
            base: base.clone(),
            first_refusal: None
        })
    );

    // Kept whole at the trainer's own count.
    let (whole, _) = project_with(&ctx, &request(set.clone(), Some(rendered), false)).unwrap();
    assert_eq!(whole.records.len(), 1);

    // Split: every window passes the trainer's check at the same bound.
    let (split, _) = project_with(&ctx, &request(set, Some(floor), true)).unwrap();
    assert!(split.records.len() > 1, "asked, the record is split");
    let path = written(&scratch.0, &split);
    let overlong =
        overlong_records(&brain, &path, &base, false, split.packed, floor as u32).unwrap();
    assert!(overlong.is_empty(), "trainer refuses: {overlong:?}");
}

#[test]
fn a_build_without_a_brain_service_says_it_measured_by_the_floor() {
    let (scratch, ctx) = scratch_context("dataset-floor", Scripted::new(|_| String::new()), false);
    std::mem::forget(scratch);
    let experience = attempt(&ctx);
    let set = set_of(&ctx, &experience);
    let (bounded, _) = project_with(&ctx, &request(set, Some(1_000_000), false)).unwrap();
    let Some(BlockMeasure::Floor { reason }) = bounded.block_measure else {
        panic!("no service counts: {:?}", bounded.block_measure);
    };
    assert!(!reason.is_empty());
}

/// The records are counted for the model the training starts from, not for
/// the policy a build happens to run under: a campaign trains a base other
/// than the default policy, and counting by the policy's tokenizer and
/// template leaves in records the trainer then refuses (or leaves out ones it
/// reads).
#[test]
fn a_build_counts_records_for_the_base_the_training_starts_from() {
    let (scratch, ctx, service, _brain) = served_context("dataset-base");
    let trained = scratch.0.join("models/Org/Trained");
    let experience = attempt(&ctx);
    let set = set_of(&ctx, &experience);
    let for_trained = BuildRequest {
        base: Some(ModelRef::Local {
            checkpoint: trained.display().to_string(),
            adapter: None,
            context_tokens: None,
        }),
        ..request(set, Some(1_000_000), false)
    };
    let (kept, _) = project_with(&ctx, &for_trained).unwrap();
    assert_eq!(kept.records.len(), 1);
    assert_eq!(
        kept.block_measure,
        Some(BlockMeasure::Trainer {
            base: trained,
            first_refusal: None
        })
    );
    let checked = service.jobs().checks();
    assert_eq!(
        checked.last().unwrap()["base"],
        "Org/Trained",
        "the trained base reads the records"
    );
}

#[test]
fn a_stored_dataset_is_read_by_the_trainer_in_the_mode_a_run_reads_it_every_record_listed() {
    let (scratch, ctx, _service, _brain) = served_context("dataset-check");
    let dir = common::bench::export(
        &scratch.0.join("export"),
        &[
            common::bench::Trial::new("one", "org/a", Some("train"), true).scored(),
            common::bench::Trial::new("two", "org/b", Some("train"), true).scored(),
        ],
    );
    let ingested =
        splinter_pipelines::bench::ingest_bench_trials(&ctx, &[dir], common::bench::NO_OPTIONS)
            .unwrap();
    let mut request = common::bench::sft_request(ingested.train_set.unwrap());
    request.pack = true;
    let built = splinter_pipelines::datasets::build(&ctx, &request).unwrap();

    let checked =
        splinter_pipelines::datasets::check(&ctx, built.dataset.0.as_str(), None, None).unwrap();
    assert_eq!(checked.dataset, built.dataset);
    assert_eq!(
        checked.base, "Qwen/Qwen3-0.6B",
        "the policy's base reads it"
    );
    assert!(checked.pack_answers, "built packed, one example per record");
    assert!(checked.thinking, "the trajectories carry their reasoning");
    assert_eq!((checked.records, checked.valid_records), (2, 2));
    assert!(checked.refused.is_empty());
    assert_eq!(checked.longest.len(), 2, "every record is listed");
    assert!(checked.supervised_tokens.total > 0);
    assert!(checked.supervised_tokens.total < checked.tokens.total);
    assert_eq!(checked.examples_without_supervision, 0);
}

/// The base a build names, a checkpoint in the test's model store.
fn named_base(scratch: &Scratch) -> ModelRef {
    ModelRef::Local {
        checkpoint: scratch.0.join("models/Org/Trained").display().to_string(),
        adapter: None,
        context_tokens: None,
    }
}

#[test]
fn a_record_the_trainer_refuses_is_left_out_at_build_under_its_own_reason() {
    let (scratch, ctx, _service, _brain) =
        served_context_with("dataset-refused", |script| JobScript {
            refuse_records_containing: Some("b.py".into()),
            ..script
        });
    let experience = attempt(&ctx);
    let set = set_of(&ctx, &experience);
    // No bound asked: naming the base is enough for its trainer to read
    // every record.
    let named = BuildRequest {
        base: Some(named_base(&scratch)),
        ..request(set, None, false)
    };
    let (built, _) = project_with(&ctx, &named).unwrap();
    assert!(
        built.records.is_empty(),
        "the refused record is not written"
    );
    assert_eq!(built.count(Exclusion::TrainerRefused), 1);
    assert_eq!(
        built.count(Exclusion::OverBlock),
        0,
        "it is not over any block"
    );
    let Some(BlockMeasure::Trainer {
        first_refusal: Some(reason),
        ..
    }) = &built.block_measure
    else {
        panic!("the trainer read it: {:?}", built.block_measure);
    };
    assert!(reason.contains("scripted trainer refuses"), "{reason}");
}

#[test]
fn a_build_that_names_its_base_is_refused_without_a_trainer_to_check_it() {
    let (scratch, ctx) =
        scratch_context("dataset-unchecked", Scripted::new(|_| String::new()), false);
    let experience = attempt(&ctx);
    let set = set_of(&ctx, &experience);
    let named = BuildRequest {
        base: Some(named_base(&scratch)),
        ..request(set, Some(1_000_000), false)
    };
    let refused = project_with(&ctx, &named).unwrap_err().to_string();
    assert!(refused.contains("trainer"), "{refused}");
    std::mem::forget(scratch);
}

/// A service that cannot read the records for the base (here: an endpoint
/// without the artifact store a check is sent through) is no trainer check: a
/// build that names its base is refused, never bounded by the floor instead.
#[test]
fn a_build_that_names_its_base_is_refused_when_the_trainer_cannot_read_its_records() {
    let scratch = Scratch::new("dataset-unreadable");
    let service = ScriptedEndpoint::start(vec![]);
    let brain = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY).unwrap();
    let mut settings = config(&scratch);
    settings.brain = Some(brain);
    let ctx = Context::new(settings, false).unwrap();
    let experience = attempt(&ctx);
    let set = set_of(&ctx, &experience);
    let named = BuildRequest {
        base: Some(named_base(&scratch)),
        ..request(set, Some(1_000_000), false)
    };
    let refused = project_with(&ctx, &named).unwrap_err().to_string();
    assert!(refused.contains("trainer"), "{refused}");
}
