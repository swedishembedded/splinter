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
use splinter_core::digest::Digest;
use splinter_core::factory::identity::{StageIdentity, StageKind};
use splinter_core::factory::product::{ArmResult, Decision, EvalResult, TrialResult, EVAL_SCHEMA};
use splinter_core::factory::task::TaskRef;
use splinter_eval::promotion::{verify, Pair, PublicKey, Role};
use splinter_model::stats::one_sided_p;
use splinter_store::factory::products::Staging;

use super::rollout::{run_arm, trials_of, Arm};
use super::training::candidate_of;
use super::{io, read_json, stage_error, write_json, Factory, RunError};
use crate::eval_plan::EvalPlan;
use crate::plan::{stage_key, PlannedStage};
use crate::recipe::{EvalSection, RolloutPass, TaskSelector};
use crate::tree::link_tree;

/// The evaluation result in an eval product.
pub const RESULT_FILE: &str = "result.json";

fn section<'r>(
    factory: &'r Factory<'_>,
    stage: &PlannedStage,
) -> Result<(&'r EvalSection, &'r EvalPlan), RunError> {
    let section = factory
        .resolved
        .recipe
        .eval
        .as_ref()
        .ok_or_else(|| stage_error(&stage.key, "the recipe evaluates nothing"))?;
    let (plan, _) = factory
        .resolved
        .eval
        .as_ref()
        .ok_or_else(|| stage_error(&stage.key, "the eval plan is not resolved"))?;
    Ok((section, plan))
}

fn plan_digest(factory: &Factory<'_>, section: &EvalSection) -> Result<Digest, RunError> {
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

fn tasks<'r>(factory: &'r Factory<'_>, plan: &EvalPlan) -> Result<Vec<&'r TaskRef>, RunError> {
    let (_, loaded) = factory
        .resolved
        .eval
        .as_ref()
        .ok_or_else(|| stage_error("eval", "no eval plan"))?;
    let loaded = loaded
        .as_ref()
        .ok_or_else(|| stage_error("eval", "a development plan names a collection"))?;
    Ok(loaded.split(plan.split.as_deref()))
}

/// The settings every arm of the evaluation runs under.
fn arm_pass(section: &EvalSection, plan: &EvalPlan, name: &str) -> Result<RolloutPass, RunError> {
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
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let (section, plan) = section(factory, stage)?;
    let training = stage_key(StageKind::Training, &stage.name);
    let inputs = vec![(training.clone(), factory.product(&training)?.1.digest())];
    let comparisons = plan
        .decision
        .comparisons
        .unwrap_or(section.candidates.len() as u32);
    let parameters = json!({
        "plan": plan_digest(factory, section)?, "agent": section.agent, "args": section.args,
        "baseline": section.baseline, "comparisons": comparisons,
    });
    let backend = if plan.protected() {
        String::new()
    } else {
        factory.services.backend.identity()
    };
    Ok(super::identity_of(
        StageKind::Eval,
        &stage.name,
        parameters,
        inputs,
        &backend,
        Some(plan.seed),
    ))
}

/// Commit the plan's pinned tasks and return their view.
fn pinned_tasks(
    factory: &Factory<'_>,
    section: &EvalSection,
    plan: &EvalPlan,
) -> Result<std::path::PathBuf, RunError> {
    let (_, loaded) = factory
        .resolved
        .eval
        .as_ref()
        .ok_or_else(|| stage_error("eval", "no eval plan"))?;
    let loaded = loaded
        .as_ref()
        .ok_or_else(|| stage_error("eval", "a development plan names a collection"))?;
    let selected = tasks(factory, plan)?;
    let identity = super::identity_of(
        StageKind::Tasks,
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
    factory.commit("eval/_tasks", StageKind::Tasks, &identity, &produce)?;
    Ok(factory.view("eval/_tasks"))
}

/// Commit one arm and return its trials and product digest.
fn arm_trials(
    factory: &Factory<'_>,
    key: &str,
    arm: &Arm<'_>,
    inputs: Vec<(String, Digest)>,
) -> Result<(Vec<TrialResult>, Digest), RunError> {
    let identity = super::identity_of(
        StageKind::Rollout,
        &arm.name,
        arm.parameters(factory)?,
        inputs,
        &factory.services.backend.identity(),
        Some(arm.pass.seed),
    );
    let produce = |f: &Factory<'_>, staging: &Staging| run_arm(f, arm, staging);
    let (manifest, _) = factory.commit(key, StageKind::Rollout, &identity, &produce)?;
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
    let training_key = stage_key(StageKind::Training, &stage.name);
    let candidate = candidate_of(&factory.view(&training_key))?;
    if let Some(authority) = &plan.authority {
        return protected(
            factory, stage, section, plan, authority, &candidate, staging,
        );
    }
    let trainer = factory.services.trainer.ok_or_else(|| {
        stage_error(
            &stage.key,
            "evaluation serves candidates through the trainer",
        )
    })?;
    let task_root = pinned_tasks(factory, section, plan)?;
    let tasks_digest = factory
        .products
        .manifest(&tasks_identity_id(factory, section, plan)?)?
        .digest();
    let selected = tasks(factory, plan)?;
    let base_binding = factory
        .resolved
        .connections
        .bind(&section.baseline)
        .map_err(|e| stage_error(&stage.key, e))?;
    let comparisons = plan
        .decision
        .comparisons
        .unwrap_or(section.candidates.len() as u32)
        .max(1);
    let baseline = Arm {
        name: "eval-baseline".into(),
        key: "eval/_arms/baseline".into(),
        pass: arm_pass(section, plan, "eval-baseline")?,
        tasks: selected.clone(),
        task_root: task_root.clone(),
        model: Some(base_binding.clone()),
    };
    let (base_trials, base_product) = arm_trials(
        factory,
        "eval/_arms/baseline",
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
    let served = trainer
        .serve(&section.baseline, adapter.as_deref())
        .map_err(|e| stage_error(&stage.key, e))?;
    let mut binding = base_binding;
    binding.model = served.clone();
    let name = format!("eval-{}", stage.name);
    let key = format!("eval/_arms/{}", stage.name);
    let candidate_arm = Arm {
        name: name.clone(),
        key: key.clone(),
        pass: arm_pass(section, plan, &name)?,
        tasks: selected,
        task_root,
        model: Some(binding),
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
        plan: plan_digest(factory, section)?,
        protected: false,
        name: stage.name.clone(),
        baseline: Some(arm_result(
            "baseline",
            &section.baseline.to_string(),
            &base_trials,
            base_product,
        )),
        candidate: Some(arm_result(&stage.name, &served, &cand_trials, cand_product)),
        decision,
        statistics,
        signature: None,
    };
    write_json(&staging.path().join(RESULT_FILE), &result)?;
    std::fs::write(staging.path().join("report.md"), report(&result)).map_err(io(staging.path()))
}

fn tasks_identity_id(
    factory: &Factory<'_>,
    section: &EvalSection,
    plan: &EvalPlan,
) -> Result<Digest, RunError> {
    let selected = tasks(factory, plan)?;
    Ok(super::identity_of(
        StageKind::Tasks,
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
    section: &EvalSection,
    plan: &EvalPlan,
    authority: &crate::eval_plan::Authority,
    candidate: &splinter_core::factory::product::ModelCandidate,
    staging: &Staging,
) -> Result<(), RunError> {
    let Some((program, args)) = authority.command.split_first() else {
        return Err(stage_error(&stage.key, "the authority's command is empty"));
    };
    let training = factory.view(&stage_key(StageKind::Training, &stage.name));
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
    let out = staging.path().join("verdict.json");
    let status = Command::new(program)
        .args(args)
        .arg("--candidate")
        .arg(&candidate_file)
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
        .unwrap_or(section.candidates.len() as u32)
        .max(1);
    let alpha = plan.decision.alpha / f64::from(comparisons);
    let expected = Pair {
        candidate: Some(digest),
        predecessor: None,
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
        name: stage.name.clone(),
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
