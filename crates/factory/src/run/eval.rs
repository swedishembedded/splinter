// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pre-registered model evaluations whose
// decision rule is fixed before a candidate is seen, for its clients. If
// your team needs expertise in trustworthy model evaluation, you can procure
// our services by sending an email to info@swedishembedded.com.

//! An evaluation stage: one candidate against the baseline, under the plan.
//!
//! **Development plans.** The plan's tasks are pinned once (`eval/_tasks`).
//! The baseline arm (the base model) and the candidate arm (the base with
//! the candidate served by the trainer) run the same agent, arguments,
//! environment, tasks, attempts and seeds, so their trials pair slot by
//! slot. Each arm is a product of its own under `eval/_arms/`; the baseline's
//! is shared by every
//! candidate: it is run once per plan. The decision is the exact one-sided
//! paired sign test over the slots both arms counted, at the plan's alpha
//! divided among the comparisons it makes (the plan's `comparisons`, or the
//! recipe's candidates): `improved` below it with the pass-rate difference at
//! least `min_difference`, `not_improved` when the candidate did not do
//! better, `inconclusive` otherwise.
//!
//! **Protected plans** name an evaluation authority instead of tasks: the
//! authority runs the sealed tasks and signs the verdict; this stage hands
//! it the candidate and keeps what it answers, checking its signature with
//! the plan's public key.
//!
//! Re-evaluating a committed candidate under an unchanged plan reuses the
//! product: sealed tasks are not consumed again.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use serde_json::json;
use splinter_core::digest::{digest_of, Digest};
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{ArmResult, Decision, EvalResult, TrialResult, EVAL_SCHEMA};
use splinter_core::factory::task::TaskRef;
use splinter_eval::promotion::{verify, Pair, PublicKey, Role};
use splinter_model::stats::one_sided_p;
use splinter_store::factory::products::Staging;

use super::pass::{RolloutPass, TaskSelector};
use super::rollout::{run_arm, trials_of, Arm};
use super::training::candidate_of;
use super::{io, read_json, stage_error, write_json, Factory, RunError};
use crate::eval_plan::EvalPlan;
use crate::ops::{names, EvaluationParams, TrainParams};
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;
use crate::tree::link_tree;

/// The operation of an evaluation's pinned tasks.
const EVAL_TASKS: &str = "tasks.pin@1";
/// The operation of an evaluation's arms.
const EVAL_ARM: &str = "attempt.collect@1";

/// The evaluation result in an eval product.
pub const RESULT_FILE: &str = "result.json";

/// An `evaluation` stage with its baseline resolved and its candidate named.
pub(crate) struct Evaluation {
    /// The evaluation plan's path.
    plan: String,
    agent: String,
    args: Vec<String>,
    baseline: ModelRef,
    comparisons: Option<u32>,
    /// The `model.train` stage whose candidate it evaluates.
    pub(crate) candidate: String,
}

fn section<'r>(
    factory: &'r Factory<'_>,
    stage: &PlannedStage,
) -> Result<(Evaluation, &'r EvalPlan), RunError> {
    let params: EvaluationParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let baseline = factory
        .resolved
        .model(&params.baseline)
        .cloned()
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                format!("model role {} is not bound", params.baseline),
            )
        })?;
    let (plan, _) = factory
        .resolved
        .eval_plans
        .get(&params.plan)
        .ok_or_else(|| stage_error(&stage.key, format!("{} is not resolved", params.plan)))?;
    let candidate = stage
        .source("candidate")
        .ok_or_else(|| stage_error(&stage.key, "evaluates no candidate"))?
        .to_owned();
    Ok((
        Evaluation {
            plan: params.plan,
            agent: params.agent,
            args: params.args,
            baseline,
            comparisons: params.comparisons,
            candidate,
        },
        plan,
    ))
}

/// The comparisons the plan's level is divided among: the plan's own
/// setting, else the stage's, else one.
fn comparisons(section: &Evaluation, plan: &EvalPlan) -> u32 {
    plan.decision
        .comparisons
        .or(section.comparisons)
        .unwrap_or(1)
        .max(1)
}

fn plan_digest(factory: &Factory<'_>, section: &Evaluation) -> Result<Digest, RunError> {
    factory
        .resolved
        .lock
        .files
        .get(&section.plan)
        .cloned()
        .ok_or_else(|| {
            stage_error(
                "eval",
                format!("{} is not pinned in the lock", section.plan),
            )
        })
}

fn tasks<'r>(
    factory: &'r Factory<'_>,
    section: &Evaluation,
    plan: &EvalPlan,
) -> Result<Vec<&'r TaskRef>, RunError> {
    let loaded = factory
        .resolved
        .eval_plans
        .get(&section.plan)
        .and_then(|(_, loaded)| loaded.as_ref())
        .ok_or_else(|| stage_error("eval", "a development plan names a collection"))?;
    Ok(loaded.split(plan.split.as_deref()))
}

/// The settings every arm of the evaluation runs under.
fn arm_pass(section: &Evaluation, plan: &EvalPlan, name: &str) -> Result<RolloutPass, RunError> {
    Ok(RolloutPass {
        name: name.to_owned(),
        agent: section.agent.clone(),
        args: section.args.clone(),
        files: Vec::new(),
        model: None,
        system_prompt: None,
        environment_variables: BTreeMap::new(),
        environment: plan
            .environment
            .clone()
            .ok_or_else(|| stage_error("eval", "a development plan names its environment"))?,
        tasks: TaskSelector::default(),
        attempts: plan.attempts,
        concurrency: plan.concurrency,
        budget: None,
        timeout_policy: splinter_core::factory::trial::TimeoutPolicy::Graded,
        retries: 2,
        seed: plan.seed,
        interactive: false,
        hint: None,
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let (section, plan) = section(factory, stage)?;
    let training = section.candidate.clone();
    let inputs = vec![(training.clone(), factory.product(&training)?.1.digest())];
    let comparisons = comparisons(&section, plan);
    let parameters = json!({
        "plan": plan_digest(factory, &section)?, "agent": section.agent, "args": section.args,
        "baseline": super::rollout::binding_of(factory, &stage.key, &section.baseline)?.0, "comparisons": comparisons,
    });
    let backend = if plan.protected() {
        String::new()
    } else {
        factory.services.backend.identity()
    };
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        parameters,
        inputs,
        &backend,
        Some(plan.seed),
    ))
}

/// Commit the plan's pinned tasks and return their view.
fn pinned_tasks(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    section: &Evaluation,
    plan: &EvalPlan,
) -> Result<std::path::PathBuf, RunError> {
    let loaded = factory
        .resolved
        .eval_plans
        .get(&section.plan)
        .and_then(|(_, loaded)| loaded.as_ref())
        .ok_or_else(|| stage_error("eval", "a development plan names a collection"))?;
    let selected = tasks(factory, section, plan)?;
    let identity = super::identity_of(
        EVAL_TASKS,
        "eval",
        json!({ "plan": plan_digest(factory, section)?, "tasks": selected.iter().map(|t| json!([t.id, t.digest])).collect::<Vec<_>>() }),
        Vec::new(),
        "",
        None,
    );
    let produce = |_: &Factory<'_>, staging: &Staging| -> Result<(), RunError> {
        for task in &selected {
            link_tree(&loaded.root.join(&task.id), &staging.path().join(&task.id))?;
        }
        write_json(&staging.path().join("tasks.json"), &selected)
    };
    let key = format!("{}/_tasks", stage.key);
    factory.commit(&key, EVAL_TASKS, &identity, &produce)?;
    Ok(factory.view(&key))
}

/// Commit one arm and return its trials and product digest.
fn arm_trials(
    factory: &Factory<'_>,
    key: &str,
    arm: &Arm<'_>,
    inputs: Vec<(String, Digest)>,
) -> Result<(Vec<TrialResult>, Digest), RunError> {
    let identity = super::identity_of(
        EVAL_ARM,
        &arm.name,
        arm.parameters(factory)?,
        inputs,
        &factory.services.backend.identity(),
        Some(arm.pass.seed),
    );
    let stage = identity.digest();
    let produce = |f: &Factory<'_>, staging: &Staging| run_arm(f, arm, &stage, staging);
    let (manifest, _) = factory.commit(key, EVAL_ARM, &identity, &produce)?;
    Ok((trials_of(&factory.view(key))?, manifest.digest()))
}

fn arm_result(arm: &str, model: &str, trials: &[TrialResult], product: Digest) -> ArmResult {
    let counted = trials.iter().filter(|t| t.grade.counted).count() as u64;
    let passed = trials.iter().filter(|t| t.verified_success()).count() as u64;
    ArmResult {
        arm: arm.into(),
        model: model.into(),
        trials: trials.len() as u64,
        counted,
        passed,
        pass_rate: (counted > 0).then(|| passed as f64 / counted as f64),
        trials_product: product,
    }
}

/// The paired sign test's decision over the slots both arms counted.
#[must_use]
pub fn decide(
    baseline: &[TrialResult],
    candidate: &[TrialResult],
    alpha: f64,
    min_difference: f64,
) -> (Decision, serde_json::Value) {
    let by_slot: BTreeMap<(&str, u32), &TrialResult> = baseline
        .iter()
        .map(|t| ((t.task.as_str(), t.slot.slot), t))
        .collect();
    let (mut wins, mut losses, mut pairs, mut base_pass, mut cand_pass) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for trial in candidate {
        let Some(base) = by_slot.get(&(trial.task.as_str(), trial.slot.slot)) else {
            continue;
        };
        if !(trial.grade.counted && base.grade.counted) {
            continue;
        }
        pairs += 1;
        let (b, c) = (base.verified_success(), trial.verified_success());
        base_pass += usize::from(b);
        cand_pass += usize::from(c);
        wins += usize::from(c && !b);
        losses += usize::from(b && !c);
    }
    let p = one_sided_p(wins, losses);
    let difference = if pairs == 0 {
        0.0
    } else {
        (cand_pass as f64 - base_pass as f64) / pairs as f64
    };
    let decision = if pairs == 0 {
        Decision::Inconclusive
    } else if p < alpha && difference >= min_difference && difference > 0.0 {
        Decision::Improved
    } else if difference <= 0.0 {
        Decision::NotImproved
    } else {
        Decision::Inconclusive
    };
    let statistics = json!({
        "test": "exact one-sided paired sign test", "pairs": pairs, "wins": wins, "losses": losses,
        "p": p, "alpha": alpha, "difference": difference, "min_difference": min_difference,
    });
    (decision, statistics)
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    _identity: &StageIdentity,
    staging: &Staging,
) -> Result<(), RunError> {
    let (section, plan) = section(factory, stage)?;
    let training_key = section.candidate.clone();
    let candidate = candidate_of(&factory.view(&training_key))?;
    if let Some(authority) = &plan.authority {
        return protected(
            factory, stage, &section, plan, authority, &candidate, staging,
        );
    }
    let trainer = factory.services.trainer.ok_or_else(|| {
        stage_error(
            &stage.key,
            "evaluation serves candidates through the trainer",
        )
    })?;
    let task_root = pinned_tasks(factory, stage, &section, plan)?;
    let tasks_digest = factory
        .products
        .manifest(&tasks_identity_id(factory, &section, plan)?)?
        .digest();
    let selected = tasks(factory, &section, plan)?;
    let (base_binding, champion) =
        super::rollout::binding_of(factory, &stage.key, &section.baseline)?;
    let comparisons = comparisons(&section, plan);
    // Named by what it is, so evaluations of one baseline under one plan
    // share it (one identity, run once), whichever candidates they judge.
    let baseline_name = format!(
        "baseline-{}",
        &digest_of(&(
            plan_digest(factory, &section)?,
            &section.agent,
            &section.args,
            &base_binding
        ))
        .hex()[..12]
    );
    let baseline_key = format!("{}/_arms/baseline", stage.key);
    let baseline = Arm {
        name: baseline_name.clone(),
        key: baseline_key.clone(),
        pass: arm_pass(&section, plan, &baseline_name)?,
        tasks: selected.clone(),
        task_root: task_root.clone(),
        model: Some(base_binding),
        champion,
    };
    let (base_trials, base_product) = arm_trials(
        factory,
        &baseline_key,
        &baseline,
        vec![("tasks".into(), tasks_digest.clone())],
    )?;
    // The candidate is served again here: the trainer's server may have
    // restarted since it was trained.
    let adapter = candidate
        .artifacts
        .iter()
        .find(|a| a.path.ends_with(".safetensors"))
        .map(|a| factory.view(&training_key).join(&a.path));
    // A candidate is served from the base it was trained from, which is the
    // baseline unless the baseline is a champion.
    let base = &training_base(factory, &section.candidate)?;
    let served = trainer
        .serve(base, adapter.as_deref())
        .map_err(|e| stage_error(&stage.key, e))?;
    let mut binding = factory
        .resolved
        .connections
        .bind(base)
        .map_err(|e| stage_error(&stage.key, e))?;
    binding.model = served.clone();
    let name = format!("{}.candidate", stage.key);
    let key = format!("{}/_arms/candidate", stage.key);
    let candidate_arm = Arm {
        name: name.clone(),
        key: key.clone(),
        pass: arm_pass(&section, plan, &name)?,
        tasks: selected,
        task_root,
        model: Some(binding),
        champion: None,
    };
    let inputs = vec![
        ("tasks".into(), tasks_digest),
        (
            training_key.clone(),
            factory.product(&training_key)?.1.digest(),
        ),
    ];
    let (cand_trials, cand_product) = arm_trials(factory, &key, &candidate_arm, inputs)?;
    let alpha = plan.decision.alpha / f64::from(comparisons);
    let (decision, statistics) = decide(
        &base_trials,
        &cand_trials,
        alpha,
        plan.decision.min_difference,
    );
    let result = EvalResult {
        schema: EVAL_SCHEMA.into(),
        plan: plan_digest(factory, &section)?,
        protected: false,
        name: section.candidate.clone(),
        baseline: Some(arm_result(
            "baseline",
            &section.baseline.to_string(),
            &base_trials,
            base_product,
        )),
        candidate: Some(arm_result(
            &section.candidate,
            &served,
            &cand_trials,
            cand_product,
        )),
        decision,
        statistics,
        signature: None,
    };
    write_json(&staging.path().join(RESULT_FILE), &result)?;
    std::fs::write(staging.path().join("report.md"), report(&result)).map_err(io(staging.path()))
}

/// The base model the `model.train` stage `training` trained from.
pub(crate) fn training_base<'r>(
    factory: &'r Factory<'_>,
    training: &str,
) -> Result<&'r ModelRef, RunError> {
    let stage = factory
        .resolved
        .plan
        .stage(training)
        .filter(|s| s.operation == names::TRAIN)
        .ok_or_else(|| stage_error(training, "is not a training stage"))?;
    let params: TrainParams = stage.params().map_err(|e| stage_error(training, e))?;
    factory
        .resolved
        .model(&params.base)
        .ok_or_else(|| stage_error(training, format!("model role {} is not bound", params.base)))
}

fn tasks_identity_id(
    factory: &Factory<'_>,
    section: &Evaluation,
    plan: &EvalPlan,
) -> Result<Digest, RunError> {
    let selected = tasks(factory, section, plan)?;
    Ok(super::identity_of(
        EVAL_TASKS,
        "eval",
        json!({ "plan": plan_digest(factory, section)?, "tasks": selected.iter().map(|t| json!([t.id, t.digest])).collect::<Vec<_>>() }),
        Vec::new(),
        "",
        None,
    )
    .digest())
}

/// Hand the candidate to the examiner and keep its signed verdict. The
/// examiner reads the candidate as it reads any (`adapter`, its
/// `adapter_digest`, the builds that made it); Splinter appends
/// `--candidate FILE --out FILE` to the plan's command. The verdict counts
/// only when the promotion gate's check passes: the examiner's signature
/// under the plan's public key, its schema, its own validity, that it is
/// about this candidate against the base, and its level. The examiner's
/// outcome decides: `Promote` is improved, `Reject` not improved, anything
/// else inconclusive. The signed document is kept verbatim as
/// `verdict.json`.
fn protected(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    section: &Evaluation,
    plan: &EvalPlan,
    authority: &crate::eval_plan::Authority,
    candidate: &splinter_core::factory::product::ModelCandidate,
    staging: &Staging,
) -> Result<(), RunError> {
    let Some((program, args)) = authority.command.split_first() else {
        return Err(stage_error(&stage.key, "the authority's command is empty"));
    };
    let training = factory.view(&section.candidate);
    let adapter = candidate
        .artifacts
        .iter()
        .find(|a| a.path.ends_with(".safetensors"))
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                "the examiner examines adapters; the candidate has none",
            )
        })?;
    let digest = adapter.digest.to_string();
    let examined = json!({
        "adapter": training.join(&adapter.path),
        "adapter_digest": digest,
        "series": candidate.name,
        "step": 0,
        "proposer_commit": Factory::implementation(""),
        "brain_commit": candidate.trainer,
    });
    let candidate_file = staging.path().join("examined-candidate.json");
    write_json(&candidate_file, &examined)?;
    // A champion baseline with an adapter is the examiner's predecessor; the
    // base model as it is otherwise.
    let champion = factory
        .resolved
        .champions
        .get(&section.baseline.to_string())
        .filter(|c| c.adapter.is_some());
    let mut command = Command::new(program);
    command.args(args).arg("--candidate").arg(&candidate_file);
    let mut predecessor_digest = None;
    if let Some(champion) = champion {
        let file = staging.path().join("examined-predecessor.json");
        let digest = champion
            .record
            .adapter_digest
            .as_ref()
            .map(ToString::to_string);
        write_json(
            &file,
            &json!({
                "adapter": champion.adapter, "adapter_digest": digest,
                "series": champion.record.candidate, "step": 0,
            }),
        )?;
        command.arg("--predecessor").arg(&file);
        predecessor_digest = digest;
    }
    let out = staging.path().join("verdict.json");
    let status = command
        .arg("--out")
        .arg(&out)
        .current_dir(&factory.resolved.repository)
        .status()
        .map_err(|e| stage_error(&stage.key, format!("{program}: {e}")))?;
    if !status.success() {
        return Err(stage_error(
            &stage.key,
            format!("the examiner exited with {status}"),
        ));
    }
    let document: serde_json::Value = read_json(&out)?;
    let key = PublicKey::read(&factory.resolved.repository.join(&authority.public_key))
        .map_err(|e| stage_error(&stage.key, format!("{}: {e}", authority.public_key)))?;
    let comparisons = plan
        .decision
        .comparisons
        .or(section.comparisons)
        .unwrap_or(1)
        .max(1);
    let alpha = plan.decision.alpha / f64::from(comparisons);
    let expected = Pair {
        candidate: Some(digest),
        predecessor: predecessor_digest,
    };
    let verified = verify(&document, &key, &expected, alpha, Role::Sealed).map_err(|e| {
        stage_error(
            &stage.key,
            format!("the examiner's verdict does not verify: {e:?}"),
        )
    })?;
    let decision = match document.get("verdict").and_then(serde_json::Value::as_str) {
        Some("Promote") => Decision::Improved,
        Some("Reject") => Decision::NotImproved,
        _ => Decision::Inconclusive,
    };
    let result = EvalResult {
        schema: EVAL_SCHEMA.into(),
        plan: plan_digest(factory, section)?,
        protected: true,
        name: section.candidate.clone(),
        baseline: None,
        candidate: None,
        decision,
        statistics: serde_json::to_value(verified.aggregate())
            .map_err(|e| stage_error(&stage.key, e))?,
        signature: document
            .pointer("/signature/value")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
    };
    write_json(&staging.path().join(RESULT_FILE), &result)?;
    std::fs::write(staging.path().join("report.md"), report(&result)).map_err(io(staging.path()))
}

/// The decision for a person.
fn report(result: &EvalResult) -> String {
    let mut out = format!("# Evaluation of {}\n\n", result.name);
    if let (Some(baseline), Some(candidate)) = (&result.baseline, &result.candidate) {
        out.push_str(
            "| arm | model | trials | counted | passed | pass rate |\n|---|---|---|---|---|---|\n",
        );
        for (arm, a) in [("baseline", baseline), ("candidate", candidate)] {
            let rate = a
                .pass_rate
                .map_or_else(|| "-".into(), |r| format!("{r:.3}"));
            out.push_str(&format!(
                "| {arm} | {} | {} | {} | {} | {rate} |\n",
                a.model, a.trials, a.counted, a.passed
            ));
        }
        out.push('\n');
    } else {
        out.push_str(
            "A protected evaluation: the examiner's signed verdict is `verdict.json`.\n\n",
        );
    }
    out.push_str(&format!(
        "**Decision: {:?}**\n\n```json\n{}\n```\n",
        result.decision,
        serde_json::to_string_pretty(&result.statistics).unwrap_or_default()
    ));
    out
}

/// The evaluation result of a committed eval view.
///
/// # Errors
/// It cannot be read.
pub fn result_of(view: &Path) -> Result<EvalResult, RunError> {
    read_json(&view.join(RESULT_FILE))
}
