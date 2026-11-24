// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified evidence, for its clients. If your
// team needs expertise in measuring whether a harness change really helped,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! `harness.search`: the harness that verifiably beats a baseline, among
//! candidates drawn from the baseline's mutation space.
//!
//! The baseline harness runs on the pinned tasks as an arm of its own; then
//! each candidate, one mutation of the baseline (`best-of-n`) or of the best
//! harness found so far (`hill-climb`), drawn by the stage's seed and kept
//! only when Sven's compiler accepts it, runs as another arm with the same
//! agent, model, environment, attempts and attempt seeds, so the two arms'
//! trials pair slot by slot. A candidate is compared with the baseline by
//! the evaluation's own rule ([`super::eval::decide`]: the exact one-sided
//! paired sign test at `alpha`); one found `improved` with a larger
//! pass-rate difference than any before it is the winner. Every arm is a
//! committed product of its own, so a search that stops resumes without
//! running an arm again, and a candidate drawn twice runs once.
//!
//! The product holds `harness.yaml` (the winner, else the baseline),
//! `candidates/<sha>.yaml` (each candidate, named by its content) and
//! `search.json` (the baseline's pass rate, each candidate's decision and
//! statistics in the order drawn, and the winner).

use std::collections::BTreeMap;

use serde_json::json;
use splinter_agent::harness_space::MutationSpace;
use splinter_agent::sven::search::SearchRng;
use splinter_core::digest::Digest;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{Decision, TrialResult};
use splinter_store::factory::products::Staging;

use super::eval::{arm_trials, decide};
use super::models::binding_of;
use super::pass::{pin, RolloutPass, TaskSelector};
use super::rollout::Arm;
use super::{io, stage_error, write_json, Factory, RunError};
use crate::ops::{HarnessSearchParams, SearchStrategy};
use crate::plan::PlannedStage;

/// The stage's parameters and an arm of `harness` on its tasks: the
/// baseline's file, or a candidate the run made.
fn arm<'r>(
    factory: &'r Factory<'_>,
    stage: &PlannedStage,
    params: &HarnessSearchParams,
    name: &str,
    made: Option<std::path::PathBuf>,
) -> Result<Arm<'r>, RunError> {
    let model = factory
        .resolved
        .model(&params.model)
        .cloned()
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                format!("model role {} is not bound", params.model),
            )
        })?;
    let pin = pin(stage)?;
    let tasks = &factory
        .collections
        .get(pin)
        .ok_or_else(|| stage_error(&stage.key, format!("{pin} pins no tasks")))?
        .collection
        .tasks;
    let (binding, champion) = binding_of(factory, &stage.key, &model)?;
    Ok(Arm {
        name: format!("{}.{name}", stage.key),
        key: format!("{}/_arms/{name}", stage.key),
        pass: RolloutPass {
            name: format!("{}.{name}", stage.key),
            agent: params.agent.clone(),
            args: params.args.clone(),
            files: params.files.clone(),
            model: Some(model),
            system_prompt: None,
            environment_variables: BTreeMap::new(),
            environment: params.environment.clone(),
            tasks: TaskSelector::default(),
            attempts: params.attempts,
            concurrency: params.concurrency,
            budget: None,
            timeout_policy: splinter_core::factory::trial::TimeoutPolicy::Graded,
            retries: 2,
            seed: params.seed,
            interactive: false,
            hint: None,
            harness: Some(params.harness.clone()),
            made_harness: made,
        },
        tasks: tasks.iter().collect(),
        task_root: factory.view(pin),
        model: Some(binding),
        champion,
        briefs: None,
        continue_from: None,
    })
}

fn params(stage: &PlannedStage) -> Result<HarnessSearchParams, RunError> {
    stage.params().map_err(|e| stage_error(&stage.key, e))
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let params = params(stage)?;
    let baseline = arm(factory, stage, &params, "baseline", None)?;
    let space = factory
        .resolved
        .lock
        .files
        .get(&params.space)
        .ok_or_else(|| stage_error(&stage.key, format!("{} was not resolved", params.space)))?;
    let pin = pin(stage)?;
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        json!({
            "baseline": baseline.parameters(factory)?, "space": [params.space, space],
            "candidates": params.candidates, "strategy": params.strategy, "alpha": params.alpha,
        }),
        vec![(pin.to_owned(), factory.product(pin)?.1.digest())],
        &factory.services.backend.identity(),
        Some(params.seed),
    ))
}

/// The share of `trials` counted that passed.
fn pass_rate(trials: &[TrialResult]) -> Option<f64> {
    let counted = trials.iter().filter(|t| t.grade.counted).count();
    let passed = trials.iter().filter(|t| t.verified_success()).count();
    (counted > 0).then(|| passed as f64 / counted as f64)
}

/// The trials of `arm`, its product committed under its own key.
fn trials_of(
    factory: &Factory<'_>,
    arm: &Arm<'_>,
    pin: &str,
) -> Result<Vec<TrialResult>, RunError> {
    let inputs = vec![(pin.to_owned(), factory.product(pin)?.1.digest())];
    Ok(arm_trials(factory, &arm.key, arm, inputs)?.0)
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let fail = |why: String| stage_error(&stage.key, why);
    let params = params(stage)?;
    let pin = pin(stage)?;
    let repository = &factory.resolved.repository;
    let read = |file: &str| {
        let path = repository.join(file);
        std::fs::read_to_string(&path).map_err(io(&path))
    };
    let baseline_text = read(&params.harness)?;
    let space = MutationSpace::parse(&read(&params.space)?).map_err(|e| fail(e.to_string()))?;
    let baseline = trials_of(
        factory,
        &arm(factory, stage, &params, "baseline", None)?,
        pin,
    )?;
    let made = factory.work(&stage.key).join("candidates");
    std::fs::create_dir_all(&made).map_err(io(&made))?;
    let candidates_dir = staging.path().join("candidates");
    std::fs::create_dir_all(&candidates_dir).map_err(io(&candidates_dir))?;
    let mut rng = SearchRng::from_seed(params.seed);
    let mut candidates = Vec::new();
    // Every candidate is a comparison with the baseline; the best of them is
    // chosen, so each is tested at alpha / candidates (Bonferroni: valid
    // whatever the dependence between candidates, a hill climb's included)
    // and the chance that any useless candidate wins stays within alpha.
    let alpha_per_candidate = params.alpha / f64::from(params.candidates.max(1));
    // The winner: its difference over the baseline, its file and its text.
    let mut best: Option<(f64, String, String)> = None;
    for drawn in 0..params.candidates {
        let parent = match (params.strategy, &best) {
            (SearchStrategy::HillClimb, Some((_, _, text))) => text.clone(),
            _ => baseline_text.clone(),
        };
        let text = match space.mutate_compiling(&parent, &mut rng) {
            Ok(text) => text,
            Err(why) => {
                candidates.push(json!({ "drawn": drawn, "refused": why }));
                continue;
            }
        };
        if let Err(why) = splinter_agent::attempt_harness::check(&text) {
            candidates.push(json!({ "drawn": drawn, "refused": why.to_string() }));
            continue;
        }
        let name = Digest::sha256_of(text.as_bytes()).hex()[..12].to_owned();
        let file = format!("candidates/{name}.yaml");
        let path = made.join(format!("{name}.yaml"));
        std::fs::write(&path, &text).map_err(io(&path))?;
        std::fs::write(staging.path().join(&file), &text).map_err(io(staging.path()))?;
        let trials = trials_of(
            factory,
            &arm(factory, stage, &params, &name, Some(path))?,
            pin,
        )?;
        let (decision, statistics) = decide(&baseline, &trials, alpha_per_candidate, 0.0);
        let difference = statistics
            .get("difference")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        if decision == Decision::Improved && best.as_ref().is_none_or(|(b, _, _)| difference > *b) {
            best = Some((difference, file.clone(), text));
        }
        candidates.push(json!({
            "drawn": drawn, "harness": file, "pass_rate": pass_rate(&trials),
            "decision": decision, "statistics": statistics,
        }));
    }
    let found = best
        .as_ref()
        .map_or(baseline_text.as_str(), |(_, _, text)| text);
    std::fs::write(staging.path().join("harness.yaml"), found).map_err(io(staging.path()))?;
    let winner = best.map(|(difference, file, _)| {
        json!({ "harness": file, "difference": difference, "decision": Decision::Improved })
    });
    write_json(
        &staging.path().join("search.json"),
        &json!({
            "schema": "splinter.harness-search/1", "baseline": {
                "harness": params.harness, "pass_rate": pass_rate(&baseline),
            },
            "strategy": params.strategy, "alpha": params.alpha,
            "alpha_per_candidate": alpha_per_candidate,
            "candidates": candidates, "winner": winner,
        }),
    )
}
