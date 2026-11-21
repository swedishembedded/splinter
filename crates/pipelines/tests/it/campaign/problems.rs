// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for choosing the next training
// experiment from the evidence already paid for, for its clients. If your team
// needs expertise in planning model-improvement campaigns you can procure our
// services by sending an email to info@swedishembedded.com.

//! The campaign's problem state on its ledger: today's real state seeded
//! from the example and replayed idempotently, verdicts read from a
//! `compare-arms` report and from an arm campaign's gate, the
//! single-variable check on the gate's arm, the firewall, and the arm
//! `campaign next` writes.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{json, Value};
use splinter_core::rng::Seed;
use splinter_eval::campaign::problems::{Document, Effect, Evidence, Phase, Verdict};
use splinter_eval::promotion::{Aggregate, Pair, Promotion, Reason, Rule};
use splinter_orchestrator::Context;
use splinter_pipelines::arm_compare::{ArmComparison, Bounds, Contrast};
use splinter_pipelines::campaign::problems::{
    arm_text, campaign_next, check_arm, parse_document, problems_add, problems_show,
    problems_update, VerdictRequest, VerdictSource,
};
use splinter_pipelines::campaign::{campaign_dir, unit, CampaignSpec, CAMPAIGN_UNIT};
use splinter_store::stage_ledger::{KeyParts, OnCorrupt, StageLedger, StageWork};

use crate::common::{config, Scratch};

fn example(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/problem-state")
        .join(file)
}

fn context(scratch: &Scratch) -> Context {
    Context::new(config(scratch), false).unwrap()
}

fn spec(ctx: &Context, name: &str) -> CampaignSpec {
    CampaignSpec::parse(&format!(
        "name = {name:?}\ngenerations = 1\nbase = \"local:{}\"\nseed = \"7\"\n\
         [rollout]\nattempts = 2\nretire_at = 0.9\nretire_after = 4\n[train]\nrank = 4\n\
         [practice]\ndir = \"practice\"\nbench = \"bench\"\n",
        ctx.config().policy_base.display()
    ))
    .unwrap()
}

fn document(text: &str) -> Document {
    parse_document(text, std::path::Path::new("doc.toml")).unwrap()
}

#[derive(Deserialize)]
struct Recorded {
    verdicts: Vec<RecordedVerdict>,
}

#[derive(Deserialize)]
struct RecordedVerdict {
    candidate: String,
    experiment: String,
    verdict: Verdict,
    effect: Option<Effect>,
    reason: String,
    #[serde(default)]
    evidence: Vec<Evidence>,
}

/// The state of 2026-11-21: the example seeds it, the recorded verdicts
/// (each with an effect judged by the state's own rule to the verdict the
/// ledger reached) move the beliefs, and the next hypothesis is the concise
/// trace mode. Delivering every event again records nothing and leaves the
/// digest as it was.
#[test]
fn todays_state_replays_to_a_known_order_and_redelivery_records_nothing() {
    let scratch = Scratch::new("problems-today");
    let ctx = context(&scratch);
    let campaign = spec(&ctx, "e2c");
    let text = std::fs::read_to_string(example("problems.toml")).unwrap();
    let verdicts: Recorded =
        toml::from_str(&std::fs::read_to_string(example("verdicts.toml")).unwrap()).unwrap();

    let deliver = |ctx: &Context| {
        let mut last = problems_add(ctx, &campaign, document(&text)).unwrap();
        for v in &verdicts.verdicts {
            if v.effect.is_some() {
                assert_eq!(
                    last.state
                        .classify(&v.candidate, v.effect.as_ref())
                        .unwrap(),
                    v.verdict,
                    "{}: the rule and the ledger disagree",
                    v.experiment
                );
            }
            last = problems_update(
                ctx,
                &campaign,
                VerdictRequest {
                    candidate: v.candidate.clone(),
                    experiment: v.experiment.clone(),
                    source: VerdictSource::Stated(v.verdict),
                    evidence: v.evidence.clone(),
                    reason: Some(v.reason.clone()),
                },
            )
            .unwrap();
        }
        last
    };
    let first = deliver(&ctx);
    assert_eq!(first.events, 1 + verdicts.verdicts.len());

    let belief = |id: &str| first.state.candidate(id).unwrap().1.belief;
    for (id, expected) in [
        ("kb-unused.mount-instruction", 0.3),
        ("kb-unused.page-preload", 0.5),
        ("distill-transfer.v1-thinking", 0.3),
        ("4b-data.star-hinted", 0.1),
        ("4b-data.expert-iteration", 0.65),
        ("ratio-spikes.sampled-only", 0.65),
    ] {
        assert!((belief(id) - expected).abs() < 1e-9, "{id}: {}", belief(id));
    }
    let top: Vec<&str> = first
        .ranked
        .iter()
        .take(6)
        .map(|p| p.candidate.as_str())
        .collect();
    assert_eq!(
        top,
        [
            "overthink.concise-traces",
            "overthink.drop-unfinished",
            "overthink.exam-thinking-budget",
            "distill-transfer.v2-concise",
            "distill-transfer.v2-star",
            "4b-data.more-samples",
        ]
    );
    assert!(
        first.ranked.iter().all(|p| p.problem != "ratio-spikes"),
        "a resolved problem is never proposed"
    );

    let again = deliver(&ctx);
    assert!(!again.recorded);
    assert_eq!(again.events, first.events);
    assert_eq!(again.digest, first.digest);
    assert_eq!(problems_show(&ctx, &campaign).unwrap(), again);
}

const SMALL: &str = r#"
schema = "splinter.problem-state/1"

[[problems]]
id = "p"
statement = "the exam pass rate is low"
metric = "pass@1"

[[problems.candidates]]
id = "lr"
intervention = "rl-round"
statement = "a lower learning rate"
variable = "train.learning_rate"
value = 3e-6
belief = 0.5

[[problems.candidates]]
id = "jobs"
intervention = "engineering"
statement = "more trials at once"
variable = "practice.jobs"
value = 32
phase = "engineering"
belief = 0.5
"#;

fn comparison(level: f64, difference: f64, low: f64, high: f64, p: f64) -> ArmComparison {
    ArmComparison {
        format: "splinter.arm-comparison/1".into(),
        reference: "base".into(),
        // seed: a report field only; nothing is drawn from it here.
        seed: Seed::new(1),
        resamples: 10_000,
        level,
        sign_test: "exact".into(),
        arms: Vec::new(),
        contrasts: vec![Contrast {
            arm: "lr".into(),
            reference: "base".into(),
            paired_tasks: 14,
            arm_pass_at_1: None,
            reference_pass_at_1: None,
            difference: Some(difference),
            task_interval: Some(Bounds { low, high }),
            family_interval: None,
            families: None,
            better: 6,
            worse: 2,
            ties: 6,
            p_one_sided: p / 2.0,
            p_two_sided: p,
            p_one_sided_holm: p / 2.0,
            p_two_sided_holm: p,
        }],
    }
}

/// A `compare-arms` contrast becomes a verdict by the state's rule: an
/// interval holding 0 and the target is inconclusive and leaves the belief,
/// one below the target falsifies; a report at another level is refused.
#[test]
fn a_compare_arms_contrast_is_judged_against_the_target() {
    let scratch = Scratch::new("problems-compare");
    let ctx = context(&scratch);
    let campaign = spec(&ctx, "c");
    problems_add(&ctx, &campaign, document(SMALL)).unwrap();
    let update = |experiment: &str, report: ArmComparison| {
        problems_update(
            &ctx,
            &campaign,
            VerdictRequest {
                candidate: "lr".into(),
                experiment: experiment.into(),
                source: VerdictSource::Compare {
                    report,
                    arm: "lr".into(),
                },
                evidence: Vec::new(),
                reason: None,
            },
        )
    };
    let wide = update("e1", comparison(0.95, 0.03, -0.02, 0.08, 0.3)).unwrap();
    let (p, c) = wide.state.candidate("lr").unwrap();
    assert_eq!(c.history[0].verdict, Verdict::Inconclusive);
    assert_eq!((c.belief, p.attempts), (0.5, 1));
    assert!(update("e2", comparison(0.90, 0.0, -0.01, 0.01, 1.0)).is_err());
    let narrow = update("e3", comparison(0.95, 0.0, -0.01, 0.01, 1.0)).unwrap();
    let (_, c) = narrow.state.candidate("lr").unwrap();
    assert_eq!(c.history[1].verdict, Verdict::Falsified);
    assert!((c.belief - 0.3).abs() < 1e-12);
}

fn promoted() -> Promotion {
    let dev = Aggregate {
        schema: "bench.examination/2".into(),
        collection: "dev".into(),
        exams: Vec::new(),
        sealed: false,
        items: 96,
        alpha: 0.05,
        gain: Some(0.08),
        interval: Some((0.03, 0.13)),
        one_sided_lower: Some(0.04),
        families: Some(12),
        burns: Vec::new(),
    };
    Promotion {
        promoted: true,
        rule: Rule::Examined,
        exam_verified: true,
        reasons: Vec::<Reason>::new(),
        pair: Pair {
            candidate: Some("sha256:c".into()),
            predecessor: None,
        },
        dev: Some(dev),
        retention: None,
        sealed: None,
    }
}

/// Records generation 1's gate of the campaign `arm` names, as the campaign
/// controller does.
fn gate_recorded(ctx: &Context, arm: &CampaignSpec, promotion: &Promotion) {
    let ledger = StageLedger::open(&campaign_dir(ctx, &arm.name)).unwrap();
    let begun = KeyParts::new(CAMPAIGN_UNIT).config(&arm.identity()).key();
    ledger
        .run(CAMPAIGN_UNIT, &begun, OnCorrupt::Refuse, |_| {
            Ok::<_, String>(StageWork::new(
                json!({"identity": arm.identity(), "seed": 7}),
            ))
        })
        .unwrap();
    let gate = KeyParts::new("gate").key();
    let summary = serde_json::to_value(promotion).unwrap();
    ledger
        .run(&unit(1, "gate"), &gate, OnCorrupt::Refuse, |_| {
            Ok::<_, String>(StageWork::new(summary))
        })
        .unwrap();
}

/// The gate's verdict is read from the arm's ledger only when the arm
/// changes the candidate's variable and nothing else.
#[test]
fn a_gate_verdict_counts_only_for_a_single_variable_arm() {
    let scratch = Scratch::new("problems-gate");
    let ctx = context(&scratch);
    let campaign = spec(&ctx, "g");
    problems_add(&ctx, &campaign, document(SMALL)).unwrap();

    let mut two = spec(&ctx, "g-two");
    two.train.learning_rate = Some(3e-6);
    two.train.rank = 8;
    gate_recorded(&ctx, &two, &promoted());
    let request = |arm: &CampaignSpec, experiment: &str| VerdictRequest {
        candidate: "lr".into(),
        experiment: experiment.into(),
        source: VerdictSource::Gate {
            arm: Box::new(arm.clone()),
            generation: 1,
        },
        evidence: Vec::new(),
        reason: None,
    };
    let refused = problems_update(&ctx, &campaign, request(&two, "two")).unwrap_err();
    assert!(refused.to_string().contains("train.rank"), "{refused}");

    let mut one = spec(&ctx, "g-one");
    one.train.learning_rate = Some(3e-6);
    gate_recorded(&ctx, &one, &promoted());
    let report = problems_update(&ctx, &campaign, request(&one, "one")).unwrap();
    let (_, c) = report.state.candidate("lr").unwrap();
    assert_eq!(c.history[0].verdict, Verdict::Supported);
    assert_eq!(
        c.history[0].effect,
        Some(Effect {
            difference: 0.08,
            low: 0.03,
            high: 0.13,
            p: None
        })
    );
    assert_eq!(report.events, 2, "the refused verdict wrote nothing");
}

/// `campaign next` writes the arm with the one change; it reads back as
/// itself. An engineering arm passes the firewall; one that also moves the
/// learning rate does not; an engineering list that reaches the training
/// settings, and a variable naming a whole section, are refused when added.
#[test]
fn the_next_arm_changes_one_variable_and_the_firewall_holds() {
    let scratch = Scratch::new("problems-next");
    let ctx = context(&scratch);
    let campaign = spec(&ctx, "n");
    problems_add(&ctx, &campaign, document(SMALL)).unwrap();

    let next = campaign_next(&ctx, &campaign, Phase::Hypothesis).unwrap();
    assert_eq!(next.proposal.as_ref().unwrap().candidate, "lr");
    assert_eq!(next.changed, vec!["train.learning_rate".to_string()]);
    let arm = next.arm.unwrap();
    assert_eq!(arm.name, "n-lr");
    assert_eq!(arm.seed, campaign.seed);
    assert_eq!(CampaignSpec::parse(&arm_text(&arm).unwrap()).unwrap(), arm);
    let mut wide_seed = arm.clone();
    wide_seed.seed = Some(u64::MAX);
    assert_eq!(
        CampaignSpec::parse(&arm_text(&wide_seed).unwrap()).unwrap(),
        wide_seed,
        "a seed no TOML integer holds is written as text"
    );

    let engineering = campaign_next(&ctx, &campaign, Phase::Engineering).unwrap();
    assert_eq!(engineering.changed, vec!["practice.jobs".to_string()]);

    let allowed = vec!["practice.jobs".to_string()];
    let mut sneaky = arm.clone();
    sneaky.name = "n-sneaky".into();
    let refused = check_arm(&campaign, &sneaky, None, true, &allowed).unwrap_err();
    assert!(
        refused.to_string().contains("train.learning_rate"),
        "{refused}"
    );

    let widened = SMALL.replace(
        "schema = \"splinter.problem-state/1\"",
        "schema = \"splinter.problem-state/1\"\n[parameters]\nengineering = [\"practice.jobs\", \"train\"]",
    );
    assert!(problems_add(&ctx, &spec(&ctx, "w"), document(&widened)).is_err());
    let section = SMALL.replace("variable = \"train.learning_rate\"", "variable = \"train\"");
    assert!(problems_add(&ctx, &spec(&ctx, "s"), document(&section)).is_err());
    let _: Value = serde_json::to_value(&engineering.proposal).unwrap();
}
