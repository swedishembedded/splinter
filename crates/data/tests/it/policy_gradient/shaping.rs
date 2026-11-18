// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: how the view shapes what brain computes from the rewards: DAPO's
//! over-length penalty on the reward, the positive-only floor on the
//! advantages, and the baseline an attempt sets that cannot be rendered.

use super::*;

/// L1 of DAPO's soft over-length penalty, against the formula by hand: with a
/// budget L = 100 and a buffer B = 20, nothing up to L - B = 80 tokens, then
/// (80 - n) / 20 down to -1 at 100, and -1 past it.
#[test]
fn the_over_length_penalty_is_dapos_soft_ramp() {
    let overlong = Overlong::new(100, 20).unwrap();
    for (tokens, penalty) in [
        (0, 0.0),
        (80, 0.0),
        (81, -0.05),
        (90, -0.5),
        (99, -0.95),
        (100, -1.0),
        (101, -1.0),
        (u64::MAX, -1.0),
    ] {
        assert!(
            close(overlong.penalty(tokens), penalty),
            "{tokens}: {} not {penalty}",
            overlong.penalty(tokens)
        );
    }
    // A buffer of the whole budget ramps from the first token.
    assert!(close(Overlong::new(10, 10).unwrap().penalty(5), -0.5));
    for (max_tokens, buffer) in [(100, 0), (100, 101), (0, 0)] {
        assert!(
            Overlong::new(max_tokens, buffer).is_err(),
            "{max_tokens}/{buffer}"
        );
    }
}

/// The penalty is added to the reward each trial earns, from the completion
/// tokens its verdict records; a trial that recorded none is an error, not a
/// short trial; the projection records the thresholds for the manifest.
#[test]
fn a_trial_over_the_budget_earns_less_by_the_completion_tokens_it_recorded() {
    let shape = RewardShape::unshaped().with_overlong(Overlong::new(100, 20).unwrap());
    let evidence = |trial: &Trial, tokens: Option<u64>| {
        let mut evidence = json!({
            "trial": trial.name,
            "task": {"id": "cad/bracket", "version": trial.version},
            "model": trial.model,
            "metrics": {},
        });
        if let Some(tokens) = tokens {
            evidence["tokens"] = json!({"prompt": 7, "completion": tokens});
        }
        evidence
    };
    let mut corpus = Corpus::new();
    for (trial, tokens) in [
        (Trial::pass("short"), 50),
        (Trial::pass("long"), 90),
        (Trial::fail("cut", &[]), 120),
    ] {
        let evidence = evidence(&trial, Some(tokens));
        add(&mut corpus, &trial, Some(evidence));
    }
    let projection = view().rewards(shape.clone()).project(&corpus).unwrap();
    let mut by_trial: Vec<(String, f64)> = projection.groups[0]
        .trials
        .iter()
        .map(|t| (t.trial.clone(), t.reward))
        .collect();
    by_trial.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(by_trial.len(), 3);
    for ((name, reward), (expected_name, expected)) in
        by_trial
            .iter()
            .zip([("cut", -1.0), ("long", 0.5), ("short", 1.0)])
    {
        assert_eq!(name, expected_name);
        assert!(close(*reward, expected), "{name}: {reward}");
    }
    assert_eq!(projection.overlong, Some(Overlong::new(100, 20).unwrap()));
    assert!(rewards(&projection).iter().all(|r| r.is_finite()));

    let mut unmeasured = Corpus::new();
    for trial in [Trial::pass("t1"), Trial::fail("t2", &[])] {
        let evidence = evidence(&trial, None);
        add(&mut unmeasured, &trial, Some(evidence));
    }
    let error = view().rewards(shape).project(&unmeasured).unwrap_err();
    assert!(error.to_string().contains("completion token"), "{error}");
}

#[test]
fn positive_writes_every_attempt_and_asks_brain_to_floor_the_advantages_at_zero() {
    // Shaped rewards 1, 0.4 and 0: every attempt sets the group's baseline,
    // so all three are written; brain clamps the advantages below it at zero
    // (advantage_floor: zero), so nothing is pushed down.
    let shape = RewardShape::shaped(0.5, vec![("iou".into(), 1.0)]).unwrap();
    let group = corpus(&[
        Trial::pass("t1"),
        Trial::fail("t2", &[("iou", 0.8)]),
        Trial::fail("t3", &[("iou", 0.0)]),
    ]);
    let projection = view()
        .rewards(shape)
        .advantage(AdvantageKind::Positive)
        .project(&group)
        .unwrap();
    let written: Vec<String> = outcomes(&projection)
        .into_iter()
        .map(|o| o.rollout.unwrap())
        .collect();
    assert_eq!(written, ["t1", "t1", "t2", "t2", "t3", "t3"]);
    assert_eq!(projection.count(Exclusion::NoGroupSignal), 0);
    assert!(projection.groups[0].trials.iter().all(|t| t.written));
    assert_eq!(projection.advantage, Some(AdvantageKind::Positive));
    assert_eq!(AdvantageKind::Positive.floor(), AdvantageFloor::Zero);
    assert_eq!(AdvantageKind::Positive.norm(), AdvantageNorm::ZScore);
    for kind in [
        AdvantageKind::Standardised,
        AdvantageKind::MeanCentred,
        AdvantageKind::LeaveOneOut,
    ] {
        assert_eq!(kind.floor(), AdvantageFloor::None, "{kind:?}");
    }
}

/// An attempt the view cannot render (here a conversation shown images, for
/// a base not known to read them) still took part in its group: it is
/// written as a baseline-only record, its reward and group and nothing
/// supervised, so brain counts it in the group's baseline and trains none of
/// it (brain's contract v2, `baseline_only`).
#[test]
fn an_attempt_the_view_cannot_render_still_sets_its_groups_baseline() {
    let mut shown = Corpus::new();
    add(&mut shown, &Trial::pass("seen"), None);
    add_shown(&mut shown, &Trial::fail("blind", &[]), None, true);
    let projection = view().project(&shown).unwrap();
    assert_eq!(projection.count(Exclusion::ShownImages), 1);
    let baseline: Vec<&splinter_data::Record> = projection
        .records
        .iter()
        .filter(|r| matches!(&r.body, RecordBody::Chat { outcome: Some(o), .. } if o.baseline_only))
        .collect();
    assert_eq!(baseline.len(), 1);
    let RecordBody::Chat {
        messages,
        outcome: Some(outcome),
        ..
    } = &baseline[0].body
    else {
        panic!("chat");
    };
    assert!(
        messages.iter().all(|m| !m.train),
        "a baseline trains nothing"
    );
    assert_eq!(
        (outcome.reward, outcome.rollout.as_deref()),
        (0.0, Some("blind"))
    );
    let trained = outcomes(&projection);
    assert!(trained.iter().all(|o| o.group == outcome.group));
    let reports = &projection.groups[0].trials;
    assert!(reports
        .iter()
        .any(|t| t.trial == "blind" && t.baseline_only));
    // The line brain reads says so.
    let dir = std::env::temp_dir().join(format!("pg-baseline-{}", std::process::id()));
    write_dataset(&dir.join("d.jsonl"), &projection, &Unchecked).unwrap();
    let text = std::fs::read_to_string(dir.join("d.jsonl")).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let line: Value =
        serde_json::from_str(text.lines().find(|l| l.contains("baseline_only")).unwrap()).unwrap();
    assert_eq!(line["baseline_only"], true);
    assert_eq!(line["reward"], 0.0);
}
