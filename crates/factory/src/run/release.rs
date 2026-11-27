// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases promoted only on an
// independent, pre-registered verdict, for its clients. If your team needs
// expertise in safe model release processes, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The release stage: what the run decided about its candidates.
//!
//! Every candidate's evaluation is recorded, with the gates its candidate
//! met. Each evaluation measures its candidate for a purpose
//! ([`Purpose`]): `target` and `transfer` evaluations must find it
//! `improved` (a protected plan's only on a signed verdict); a `retention`
//! evaluation must find it retained: neither a loss the plan's sign test
//! finds significant nor a pass rate more than `retention_margin` below the
//! baseline's, measured on counted pairs with no more of its slots
//! uncounted than `max_unpaired_share` allows. A candidate is eligible when
//! every evaluation of it meets its gate and it was evaluated for every purpose the promotion
//! `requires` (all three unless the recipe narrows them). With
//! `promote: true` the eligible candidate with the largest target
//! difference is promoted; none eligible promotes nothing. With
//! `promote: false` the run is an experiment: nothing is promoted whatever
//! the verdicts.
//!
//! The release also names the champion the run leaves in use, which a
//! later generation's recipe names `champion:RUN` ([`crate::champion`]):
//! the promoted candidate; else the evaluation's baseline, kept (a champion
//! baseline stays the champion). A trained champion's adapter is copied
//! into the product, so the record is whole without the run's other
//! products.
//!
//! The product holds `release.json` and the champion's adapter.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_core::digest::Digest;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::Decision;
use splinter_core::verifiability::VerifierClass;
use splinter_knowledge::factory::alarm::{DivergenceAlarm, DivergencePolicy, Observation, Signal};
use splinter_store::factory::products::Staging;

use super::eval::result_of;
use super::training::candidate_of;
use super::{io, stage_error, write_json, Factory, RunError};
use crate::champion::{Champion, ChampionRecord, ADAPTER_FILE, RELEASE_SCHEMA};
use crate::ops::{EvaluationParams, PromotionParams, Purpose};
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;

pub use crate::champion::RELEASE_FILE;

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let mut inputs = Vec::new();
    for key in stage.sources("evaluations") {
        inputs.push((key.to_owned(), factory.product(key)?.1.digest()));
    }
    let promote = promote(stage)?;
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        gates_of(factory, stage, promote)?,
        inputs,
        "",
        None,
    ))
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let promote = promote(stage)?;
    let params: PromotionParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let mut evaluated = Vec::new();
    for key in stage.sources("evaluations") {
        evaluated.push((purpose_of(factory, key)?, result_of(&factory.view(key))?));
    }
    // Each candidate's gates, by purpose: the weakest evaluation decides.
    let mut gates: std::collections::BTreeMap<String, std::collections::BTreeMap<Purpose, Gate>> =
        std::collections::BTreeMap::new();
    for (purpose, result) in &evaluated {
        let gate = gate(
            *purpose,
            result,
            params.retention_margin,
            params.max_unpaired_share,
        );
        let held = gates.entry(result.name.clone()).or_default();
        let worst = held.get(purpose).map_or(gate, |g| (*g).min(gate));
        held.insert(*purpose, worst);
    }
    let lineage = lineage_of(kept_champion(factory, stage)?.1.as_ref())?;
    let generation = lineage.last().map_or(0, |p| p.generation + 1);
    let mut candidates = Vec::new();
    let mut best: Option<(f64, String)> = None;
    for (purpose, result) in &evaluated {
        let name = result.name.clone();
        let candidate = candidate_of(&factory.view(&name))?;
        let difference = result
            .statistics
            .get("difference")
            .and_then(serde_json::Value::as_f64);
        let held = &gates[&name];
        let point = point_of(&name, &evaluated, generation);
        let alarm = point
            .as_ref()
            .map(|p| divergence(&lineage, p, params.divergence_window));
        let diverging = matches!(alarm, Some(Signal::Diverging(_)));
        let eligible = held.values().all(|g| g.met())
            && params.required().iter().all(|p| held.contains_key(p))
            && !diverging;
        if eligible && *purpose == Purpose::Target {
            let gain = difference.unwrap_or(0.0);
            if best.as_ref().is_none_or(|(b, _)| gain > *b) {
                best = Some((gain, name.clone()));
            }
        }
        let shown: serde_json::Map<String, serde_json::Value> = Purpose::ALL
            .iter()
            .map(|p| {
                let state = held.get(p).map_or("missing", |g| g.name());
                (p.name().to_owned(), json!(state))
            })
            .collect();
        candidates.push(json!({
            "candidate": name, "purpose": purpose.name(), "decision": result.decision,
            "protected": result.protected, "signed": result.signature.is_some(),
            "difference": difference, "gates": shown, "eligible": eligible,
            "divergence": alarm.as_ref().map(signal_name), "point": point,
            "served_as": candidate.served_as, "artifacts": candidate.artifacts,
        }));
    }
    let promoted = if promote {
        best.map(|(_, name)| name)
    } else {
        None
    };
    // The lineage the run leaves: its champion's, and the promoted
    // candidate's point when one is promoted.
    let mut left = lineage.clone();
    if let Some(name) = &promoted {
        left.extend(point_of(name, &evaluated, generation));
    }
    let champion = match &promoted {
        Some(name) => {
            let view = factory.view(name);
            let candidate = candidate_of(&view)?;
            let adapter = candidate
                .artifacts
                .iter()
                .find(|a| a.path.ends_with(".safetensors"))
                .map(|a| (view.join(&a.path), a.digest.clone()));
            let base = super::eval::training_base(factory, name)?.clone();
            champion_record(staging, base, adapter, Some(name.clone()))?
        }
        None => match kept_champion(factory, stage)? {
            (_, Some(kept)) => champion_record(
                staging,
                kept.record.base.clone(),
                kept.adapter.clone().zip(kept.record.adapter_digest.clone()),
                kept.record.candidate.clone(),
            )?,
            (base, None) => champion_record(staging, base, None, None)?,
        },
    };
    write_json(
        &staging.path().join(RELEASE_FILE),
        &json!({
            "schema": RELEASE_SCHEMA, "promote": promote, "promoted": promoted,
            "candidates": candidates, "champion": champion, "lineage": left,
        }),
    )
}

/// One generation's point of the divergence alarm: its champion's pass rate
/// on what it was trained for (the reward a training run can push up) and
/// on tasks it was not trained on (what it cannot be trained to please).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// The generation, from 0.
    pub generation: u64,
    /// The candidate that became the champion.
    pub candidate: String,
    /// Its target evaluations' candidate pass rate, averaged.
    pub target: f64,
    /// Its transfer evaluations' candidate pass rate, averaged; `None`
    /// without one.
    pub transfer: Option<f64>,
}

/// The lineage the release of the run's champion recorded; empty for a
/// base model. The champion's digest covers its release record, so the
/// stage depends on it.
fn lineage_of(champion: Option<&Champion>) -> Result<Vec<Point>, RunError> {
    let Some(champion) = champion else {
        return Ok(Vec::new());
    };
    #[derive(Deserialize)]
    struct Recorded {
        #[serde(default)]
        lineage: Vec<Point>,
    }
    let path = champion.release.join(RELEASE_FILE);
    let bytes = std::fs::read(&path).map_err(io(&path))?;
    let recorded: Recorded = serde_json::from_slice(&bytes)
        .map_err(|e| stage_error("release", format!("{}: {e}", path.display())))?;
    Ok(recorded.lineage)
}

/// The point `candidate` adds at `generation`: the mean of its target and
/// of its transfer arms' pass rates; `None` without a target pass rate.
fn point_of(
    candidate: &str,
    evaluated: &[(Purpose, splinter_core::factory::product::EvalResult)],
    generation: u64,
) -> Option<Point> {
    let rate = |purpose: Purpose| {
        let rates: Vec<f64> = evaluated
            .iter()
            .filter(|(p, r)| *p == purpose && r.name == candidate)
            .filter_map(|(_, r)| r.candidate.as_ref().and_then(|a| a.pass_rate))
            .collect();
        (!rates.is_empty()).then(|| rates.iter().sum::<f64>() / rates.len() as f64)
    };
    Some(Point {
        generation,
        candidate: candidate.to_owned(),
        target: rate(Purpose::Target)?,
        transfer: rate(Purpose::Transfer),
    })
}

/// What the divergence alarm says of `point` after `lineage`, over the
/// last `window` generations.
pub(crate) fn divergence(lineage: &[Point], point: &Point, window: usize) -> Signal {
    let mut alarm = DivergenceAlarm::new(
        DivergencePolicy {
            window,
            ..DivergencePolicy::default()
        },
        VerifierClass::Objective,
    );
    let mut signal = Signal::Steady;
    for p in lineage.iter().chain(std::iter::once(point)) {
        signal = alarm.observe(Observation {
            step: p.generation,
            reward: p.target,
            heldout: p.transfer,
        });
    }
    signal
}

fn signal_name(signal: &Signal) -> &'static str {
    match signal {
        Signal::Steady => "steady",
        Signal::Diverging(_) => "diverging",
        Signal::Unconfirmed { .. } => "unconfirmed",
    }
}

/// Whether an evaluation met its purpose's gate, worst first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Gate {
    /// A protected plan's verdict without its signature.
    Unsigned,
    /// The candidate lost what the baseline could do.
    Regressed,
    /// It is not better.
    NotImproved,
    /// The evidence does not decide.
    Inconclusive,
    /// A retention evaluation found nothing lost.
    Retained,
    /// It is better.
    Improved,
}

impl Gate {
    fn met(self) -> bool {
        matches!(self, Self::Improved | Self::Retained)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Unsigned => "unsigned",
            Self::Regressed => "regressed",
            Self::NotImproved => "not-improved",
            Self::Inconclusive => "inconclusive",
            Self::Retained => "retained",
            Self::Improved => "improved",
        }
    }
}

/// The gate `result` meets for `purpose`.
fn gate(
    purpose: Purpose,
    result: &splinter_core::factory::product::EvalResult,
    margin: f64,
    max_unpaired_share: Option<f64>,
) -> Gate {
    if result.protected && result.signature.is_none() {
        return Gate::Unsigned;
    }
    if purpose == Purpose::Retention {
        return retention(&result.statistics, margin, max_unpaired_share);
    }
    match result.decision {
        Decision::Improved => Gate::Improved,
        Decision::NotImproved => Gate::NotImproved,
        Decision::Inconclusive => match max_unpaired_share {
            Some(share) if improved_within(&result.statistics, share) => Gate::Improved,
            _ => Gate::Inconclusive,
        },
    }
}

/// Whether a development plan's paired sign test that is inconclusive only
/// because some slots went uncounted is improved on the slots both arms
/// counted, with the uncounted share at most `share`.
fn improved_within(statistics: &serde_json::Value, share: f64) -> bool {
    let number = |key: &str| statistics.get(key).and_then(serde_json::Value::as_f64);
    let (Some(pairs), Some(unpaired), Some(p), Some(alpha), Some(difference), Some(min)) = (
        number("pairs"),
        number("unpaired_slots"),
        number("p"),
        number("alpha"),
        number("difference"),
        number("min_difference"),
    ) else {
        return false;
    };
    let slots = pairs + unpaired;
    pairs > 0.0
        && slots > 0.0
        && unpaired / slots <= share
        && p < alpha
        && difference >= min
        && difference > 0.0
}

/// The retention gate of an evaluation's `statistics`, whichever measured
/// them; a measurement that is not there retains nothing (inconclusive).
///
/// * A development plan's paired sign test (`wins`, `losses`, `difference`,
///   `alpha`, `pairs`, `unpaired_slots`): inconclusive without a counted
///   pair; regressed on a loss the sign test finds significant or a point
///   difference below `-margin`; else inconclusive when the share of slots
///   one arm did not count exceeds `max_unpaired_share` (any uncounted slot
///   when unset, the target rule); else retained.
/// * An examiner's aggregate (`gain`, `one_sided_lower`): retained only when
///   the one-sided lower bound clears `-margin` (non-inferiority);
///   regressed when the gain itself is past it; else inconclusive.
fn retention(statistics: &serde_json::Value, margin: f64, max_unpaired_share: Option<f64>) -> Gate {
    let number = |key: &str| statistics.get(key).and_then(serde_json::Value::as_f64);
    let count = |key: &str| {
        statistics
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
    };
    if let (Some(wins), Some(losses), Some(difference), Some(alpha)) = (
        count("wins"),
        count("losses"),
        number("difference"),
        number("alpha"),
    ) {
        // Nothing counted retains nothing: zero pairs read as no difference.
        let (Some(pairs), Some(unpaired)) = (count("pairs"), count("unpaired_slots")) else {
            return Gate::Inconclusive;
        };
        if pairs == 0 {
            return Gate::Inconclusive;
        }
        let lost = splinter_model::stats::one_sided_p(losses, wins) < alpha;
        if lost || difference < -margin {
            return Gate::Regressed;
        }
        let share = unpaired as f64 / (pairs + unpaired) as f64;
        return match max_unpaired_share {
            Some(bound) if share <= bound => Gate::Retained,
            None if unpaired == 0 => Gate::Retained,
            _ => Gate::Inconclusive,
        };
    }
    match (number("gain"), number("one_sided_lower")) {
        (_, Some(lower)) if lower > -margin => Gate::Retained,
        (Some(gain), Some(_)) if gain < -margin => Gate::Regressed,
        _ => Gate::Inconclusive,
    }
}

/// The purpose the evaluation stage `key` states: an evaluation the plan
/// does not hold, or whose parameters do not read, is an error, never
/// counted as a target evaluation.
fn purpose_of(factory: &Factory<'_>, key: &str) -> Result<Purpose, RunError> {
    let stage = factory.resolved.plan.stage(key).ok_or_else(|| {
        stage_error(
            key,
            "the promotion names an evaluation the plan does not hold",
        )
    })?;
    stage
        .params::<EvaluationParams>()
        .map(|p| p.purpose)
        .map_err(|e| stage_error(key, e))
}

/// The stage identity's parameters: whether it promotes, the champion it
/// would keep, and for a promotion what it requires of each evaluation.
fn gates_of(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    promote: bool,
) -> Result<serde_json::Value, RunError> {
    let kept = kept_champion(factory, stage)?;
    let mut parameters =
        json!({ "promote": promote, "kept": kept.0, "kept_digest": kept.1.map(|c| c.digest) });
    if promote {
        let params: PromotionParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
        let purposes = stage
            .sources("evaluations")
            .iter()
            .map(|key| purpose_of(factory, key).map(Purpose::name))
            .collect::<Result<Vec<_>, _>>()?;
        parameters["requires"] = json!(params.required());
        parameters["retention_margin"] = json!(params.retention_margin);
        if let Some(share) = params.max_unpaired_share {
            parameters["max_unpaired_share"] = json!(share);
        }
        parameters["divergence_window"] = json!(params.divergence_window);
        parameters["purposes"] = json!(purposes);
    }
    Ok(parameters)
}

/// Whether the stage promotes.
fn promote(stage: &PlannedStage) -> Result<bool, RunError> {
    stage
        .params::<PromotionParams>()
        .map(|p| p.promote)
        .map_err(|e| stage_error(&stage.key, e))
}

/// The model the run keeps when it promotes nothing: the baseline its
/// evaluations measured against (a champion, resolved).
fn kept_champion(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<(ModelRef, Option<Champion>), RunError> {
    let role = stage
        .sources("evaluations")
        .into_iter()
        .filter_map(|e| factory.resolved.plan.stage(e))
        .find_map(|e| e.params::<EvaluationParams>().ok())
        .map(|p| p.baseline)
        .ok_or_else(|| stage_error(&stage.key, "a release decides about evaluated candidates"))?;
    let model = factory
        .resolved
        .model(&role)
        .cloned()
        .ok_or_else(|| stage_error(&stage.key, format!("model role {role} is not bound")))?;
    let champion = factory.resolved.champions.get(&model.to_string()).cloned();
    Ok((model, champion))
}

/// A champion record, its adapter copied into the release product.
fn champion_record(
    staging: &Staging,
    base: ModelRef,
    adapter: Option<(PathBuf, Digest)>,
    candidate: Option<String>,
) -> Result<ChampionRecord, RunError> {
    let (adapter, adapter_digest) = match adapter {
        Some((source, digest)) => {
            let target = staging.path().join(ADAPTER_FILE);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(io(parent))?;
            }
            std::fs::copy(&source, &target).map_err(io(&source))?;
            (Some(ADAPTER_FILE.to_owned()), Some(digest))
        }
        None => (None, None),
    };
    Ok(ChampionRecord {
        base,
        adapter,
        adapter_digest,
        candidate,
    })
}

#[cfg(test)]
mod divergence_spec {
    use super::*;

    fn point(generation: u64, target: f64, transfer: Option<f64>) -> Point {
        Point {
            generation,
            candidate: format!("g{generation}"),
            target,
            transfer,
        }
    }

    /// A target that climbs while transfer stays flat is the alarm; one that
    /// transfer follows is not; and a single generation reads no trend.
    #[test]
    fn a_target_climbing_alone_over_the_window_diverges() {
        let flat = [point(0, 0.4, Some(0.5)), point(1, 0.6, Some(0.5))];
        let next = point(2, 0.8, Some(0.5));
        assert!(matches!(divergence(&flat, &next, 3), Signal::Diverging(_)));
        let following = [point(0, 0.4, Some(0.4)), point(1, 0.6, Some(0.6))];
        let next = point(2, 0.8, Some(0.8));
        assert_eq!(divergence(&following, &next, 3), Signal::Steady);
        assert_eq!(
            divergence(&[], &point(0, 0.9, Some(0.1)), 3),
            Signal::Steady
        );
        let unmeasured = [point(0, 0.4, None), point(1, 0.6, None)];
        assert!(matches!(
            divergence(&unmeasured, &point(2, 0.8, None), 3),
            Signal::Unconfirmed { .. }
        ));
    }
}

#[cfg(test)]
mod retention_spec {
    use super::*;
    use splinter_core::factory::product::{EvalResult, EVAL_SCHEMA};

    fn result(protected: bool, statistics: serde_json::Value) -> EvalResult {
        EvalResult {
            schema: EVAL_SCHEMA.into(),
            plan: Digest::of(b"plan"),
            protected,
            name: "lora".into(),
            baseline: None,
            candidate: None,
            decision: Decision::Inconclusive,
            statistics,
            signature: protected.then(|| "signed".into()),
        }
    }

    fn retention(protected: bool, statistics: serde_json::Value) -> Gate {
        gate(
            Purpose::Retention,
            &result(protected, statistics),
            0.05,
            None,
        )
    }

    /// A target evaluation left inconclusive only by uncounted slots is judged
    /// on the slots both arms counted when the uncounted share is within the
    /// run's bound, and stays inconclusive past it or without a bound.
    #[test]
    fn uncounted_slots_within_the_bound_do_not_block_a_clear_improvement() {
        let stats = json!({ "pairs": 200, "unpaired_slots": 4, "wins": 50, "losses": 1,
                            "p": 1e-12, "alpha": 0.05, "difference": 0.245, "min_difference": 0.0 });
        let target = |share| gate(Purpose::Target, &result(false, stats.clone()), 0.05, share);
        assert_eq!(target(None), Gate::Inconclusive);
        assert_eq!(target(Some(0.02)), Gate::Improved);
        assert_eq!(target(Some(0.01)), Gate::Inconclusive);
        let not_significant = json!({ "pairs": 200, "unpaired_slots": 1, "wins": 3, "losses": 2,
                                      "p": 0.5, "alpha": 0.05, "difference": 0.005, "min_difference": 0.0 });
        assert_eq!(
            gate(
                Purpose::Target,
                &result(false, not_significant),
                0.05,
                Some(0.02)
            ),
            Gate::Inconclusive
        );
    }

    /// A development plan's paired sign test over `pairs` counted pairs and
    /// `unpaired` slots one arm did not count.
    fn paired(
        wins: u64,
        losses: u64,
        pairs: u64,
        unpaired: u64,
        difference: f64,
    ) -> serde_json::Value {
        json!({
            "wins": wins, "losses": losses, "pairs": pairs, "unpaired_slots": unpaired,
            "difference": difference, "alpha": 0.05,
        })
    }

    fn retention_within(statistics: serde_json::Value, share: Option<f64>) -> Gate {
        gate(Purpose::Retention, &result(false, statistics), 0.05, share)
    }

    /// A development plan's retention: a significant loss or a point
    /// difference below the negative margin regresses; anything else measured
    /// on counted pairs is retained.
    #[test]
    fn a_measured_retention_is_retained_unless_it_lost() {
        assert_eq!(retention(false, paired(3, 2, 40, 0, -0.01)), Gate::Retained);
        assert_eq!(
            retention(false, paired(0, 9, 40, 0, -0.02)),
            Gate::Regressed
        );
        assert_eq!(
            retention(false, paired(2, 3, 40, 0, -0.10)),
            Gate::Regressed
        );
    }

    /// A retention that counted no pair, or left more of its slots uncounted
    /// than the run's bound (any, without one: the target rule), decides
    /// nothing; a loss on the pairs it did count still regresses.
    #[test]
    fn a_retention_without_counted_pairs_or_past_the_unpaired_bound_is_inconclusive() {
        assert_eq!(
            retention_within(paired(0, 0, 0, 0, 0.0), None),
            Gate::Inconclusive
        );
        assert_eq!(
            retention_within(paired(0, 0, 0, 40, 0.0), Some(0.02)),
            Gate::Inconclusive
        );
        assert_eq!(
            retention_within(paired(1, 1, 10, 30, 0.0), Some(0.02)),
            Gate::Inconclusive
        );
        assert_eq!(
            retention_within(paired(1, 1, 99, 1, 0.0), Some(0.02)),
            Gate::Retained
        );
        assert_eq!(
            retention_within(paired(1, 1, 99, 1, 0.0), None),
            Gate::Inconclusive
        );
        assert_eq!(
            retention_within(paired(0, 9, 10, 30, -0.9), Some(0.02)),
            Gate::Regressed
        );
    }

    /// An examiner's retention is retained only when its one-sided lower bound
    /// clears the margin; a gain past the margin is a regression, and a bound
    /// that does not clear it otherwise decides nothing.
    #[test]
    fn a_signed_retention_is_held_to_its_one_sided_bound() {
        let kept = json!({ "gain": 0.0, "one_sided_lower": -0.02, "alpha": 0.05 });
        assert_eq!(retention(true, kept), Gate::Retained);
        let lost = json!({ "gain": -0.20, "one_sided_lower": -0.30, "alpha": 0.05 });
        assert_eq!(retention(true, lost), Gate::Regressed);
        let unsure = json!({ "gain": 0.0, "one_sided_lower": -0.12, "alpha": 0.05 });
        assert_eq!(retention(true, unsure), Gate::Inconclusive);
    }

    /// A retention evaluation that measured nothing retains nothing.
    #[test]
    fn a_retention_without_its_measurement_is_inconclusive() {
        assert_eq!(retention(false, json!({})), Gate::Inconclusive);
        assert_eq!(retention(false, json!({ "wins": 1 })), Gate::Inconclusive);
        assert_eq!(retention(true, json!({ "gain": 0.1 })), Gate::Inconclusive);
    }
}
