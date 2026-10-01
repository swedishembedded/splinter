// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: training data is never the source of truth. Every objective is a
//! reproducible projection of the experience graph, compiled from a recipe
//! against a snapshot, and sibling decisions share their context instead of
//! copying it.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, coding_task, state, Scratch};
use splinter_expdb::analyze::TASK_COMPLETION;
use splinter_expdb::ingest::Collector;
use splinter_expdb::model::{
    Action, Content, Epistemic, Evaluation, EvaluatorRef, Outcome, PolicyRef, Target,
};
use splinter_expdb::train::{DataRef, Recipe, SampleBody};
use splinter_expdb::{Database, WriterIdentity};

fn collector(db: &Database) -> Collector {
    db.collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap()
}

fn verifier(
    target: splinter_expdb::RecordId,
    score: f64,
    confidence: f64,
    epistemic: Epistemic,
) -> Evaluation {
    Evaluation::new(
        Target::Record(target),
        EvaluatorRef::new("pytest", "8"),
        TASK_COMPLETION,
        score,
        confidence,
    )
    .epistemic(epistemic)
}

#[test]
fn sft_takes_the_steps_of_successful_attempts_with_the_context_each_was_taken_in() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    attempt(&mut c, 1, "good", 3, Outcome::Pass);
    attempt(&mut c, 1, "bad", 3, Outcome::Fail);
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot.compile(&Recipe::sft()).unwrap();
    assert_eq!(
        plan.samples.len(),
        3,
        "the failed attempt contributes nothing"
    );
    let SampleBody::Sft {
        context, target, ..
    } = &plan.samples[1].body
    else {
        panic!("not an sft sample")
    };
    let (context, target) = (
        snapshot.render(context).unwrap(),
        snapshot.render(target).unwrap(),
    );
    assert!(
        context.contains("make the failing test pass"),
        "the task is in the context: {context}"
    );
    assert!(
        context.contains("\"step\":0"),
        "the earlier step is in the context: {context}"
    );
    assert!(target.contains("\"step\":1") && target.contains("good"));
}

#[test]
fn sft_can_take_whole_episodes_instead_of_steps() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    attempt(&mut c, 1, "good", 3, Outcome::Pass);
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot.compile(&Recipe::sft().episodes()).unwrap();
    assert_eq!(plan.samples.len(), 1);
    let SampleBody::Sft { target, .. } = &plan.samples[0].body else {
        panic!()
    };
    let episode = snapshot.render(target).unwrap();
    for step in 0..3 {
        assert!(
            episode.contains(&format!("\"step\":{step}")),
            "step {step} is in the episode"
        );
    }
}

#[test]
fn sft_can_insist_on_verified_and_confident_outcomes() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (_, unverified) = attempt(&mut c, 1, "unverified", 1, Outcome::Pass);
    let (_, verified) = attempt(&mut c, 2, "verified", 1, Outcome::Pass);
    let (_, shaky) = attempt(&mut c, 3, "shaky", 1, Outcome::Pass);
    c.evaluate(verifier(verified, 1.0, 1.0, Epistemic::Fact))
        .unwrap();
    c.evaluate(verifier(shaky, 1.0, 0.4, Epistemic::Annotation))
        .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    assert_eq!(snapshot.compile(&Recipe::sft()).unwrap().samples.len(), 3);
    assert_eq!(
        snapshot
            .compile(&Recipe::sft().verified_only())
            .unwrap()
            .samples
            .len(),
        1
    );
    assert_eq!(
        snapshot
            .compile(&Recipe::sft().min_confidence(0.9))
            .unwrap()
            .samples
            .len(),
        2
    );
    let _ = unverified;
}

#[test]
fn dpo_pairs_the_decisions_taken_at_one_state_by_how_their_attempts_turned_out() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (good, _) = attempt(&mut c, 1, "good", 2, Outcome::Pass);
    let (bad, _) = attempt(&mut c, 1, "bad", 2, Outcome::Fail);
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot.compile(&Recipe::dpo()).unwrap();
    assert_eq!(
        plan.samples.len(),
        1,
        "only the first decisions share a state and a context"
    );
    let SampleBody::Preference {
        context,
        chosen,
        rejected,
        reward_gap,
    } = &plan.samples[0].body
    else {
        panic!("not a preference sample")
    };
    assert_eq!(*reward_gap, 1.0);
    assert_eq!(
        chosen,
        &DataRef::Action {
            decision: good[0].id
        }
    );
    assert_eq!(
        rejected,
        &DataRef::Action {
            decision: bad[0].id
        }
    );
    // The context is stored once and is the same for both.
    assert_eq!(
        snapshot.render(context).unwrap(),
        snapshot
            .render(&DataRef::Context {
                decision: bad[0].id
            })
            .unwrap()
    );
    assert!(snapshot
        .compile(&Recipe::dpo().min_gap(1.5))
        .unwrap()
        .samples
        .is_empty());
}

#[test]
fn dpo_uses_the_alternatives_of_a_fork_as_siblings() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (decisions, _) = attempt(&mut c, 1, "original", 3, Outcome::Fail);
    let mut branch = c
        .fork(&decisions[1], &PolicyRef::new("alternative", "1"))
        .unwrap();
    let alt = branch
        .decision()
        .commit(Action::new(
            "inspect",
            serde_json::json!({ "policy": "alternative" }),
        ))
        .unwrap();
    branch.finish(Outcome::Pass).unwrap();
    c.flush().unwrap();

    let plan = db
        .snapshot()
        .unwrap()
        .compile(&Recipe::dpo().min_gap(0.5))
        .unwrap();
    let pair = plan
        .samples
        .iter()
        .find_map(|s| match &s.body {
            SampleBody::Preference {
                chosen, rejected, ..
            } => Some((chosen.clone(), rejected.clone())),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        pair,
        (
            DataRef::Action { decision: alt.id },
            DataRef::Action {
                decision: decisions[1].id
            }
        )
    );
}

#[test]
fn dpo_ignores_unmeasured_outcomes_and_identical_choices() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    attempt(&mut c, 1, "same", 1, Outcome::Pass);
    attempt(&mut c, 1, "same", 1, Outcome::Fail); // the same choice, different luck
    attempt(&mut c, 2, "a", 1, Outcome::Pass);
    attempt(&mut c, 2, "b", 1, Outcome::Aborted); // never measured
    c.flush().unwrap();
    assert!(db
        .snapshot()
        .unwrap()
        .compile(&Recipe::dpo())
        .unwrap()
        .samples
        .is_empty());
}

#[test]
fn dpo_can_rank_by_the_credit_an_algorithm_assigned_instead_of_the_outcome() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (a, _) = attempt(&mut c, 1, "a", 1, Outcome::Fail);
    let (b, _) = attempt(&mut c, 1, "b", 1, Outcome::Fail);
    for (decision, value) in [(a[0].id, 0.9), (b[0].id, 0.1)] {
        c.assign_credit(splinter_expdb::model::CreditAssignment {
            target: decision,
            algorithm: "counterfactual-credit-v4".into(),
            value,
            confidence: 0.8,
            supporting_branches: vec![],
            derivation: None,
        })
        .unwrap();
    }
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    assert!(
        snapshot.compile(&Recipe::dpo()).unwrap().samples.is_empty(),
        "both attempts failed"
    );
    let plan = snapshot
        .compile(
            &Recipe::dpo()
                .credit("counterfactual-credit-v4")
                .min_gap(0.5),
        )
        .unwrap();
    let SampleBody::Preference { chosen, .. } = &plan.samples[0].body else {
        panic!()
    };
    assert_eq!(chosen, &DataRef::Action { decision: a[0].id });
}

#[test]
fn grpo_keeps_only_families_whose_rewards_differ() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    for (n, outcome) in [Outcome::Pass, Outcome::Fail, Outcome::Pass, Outcome::Fail]
        .into_iter()
        .enumerate()
    {
        attempt(&mut c, 1, &format!("p{n}"), 2, outcome); // learnable: half solve it
        attempt(&mut c, 2, &format!("p{n}"), 2, Outcome::Pass); // too easy
        attempt(&mut c, 3, &format!("p{n}"), 2, Outcome::Fail); // too hard
    }
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot
        .compile(&Recipe::grpo().group_size(4).min_reward_variance(0.15))
        .unwrap();
    assert_eq!(plan.samples.len(), 1);
    let SampleBody::Group { context, rollouts } = &plan.samples[0].body else {
        panic!("not a group")
    };
    assert_eq!(rollouts.len(), 4);
    assert!(snapshot
        .render(context)
        .unwrap()
        .contains("make the failing test pass"));
    let advantages: f64 = rollouts.iter().map(|r| r.advantage).sum();
    assert!(
        advantages.abs() < 1e-9,
        "advantages are relative to the group"
    );
    assert!(rollouts
        .iter()
        .filter(|r| r.reward == 1.0)
        .all(|r| r.advantage > 0.0));
    assert!(rollouts
        .iter()
        .filter(|r| r.reward == 0.0)
        .all(|r| r.advantage < 0.0));
    // Groups are exactly the requested size; smaller families are skipped.
    assert!(snapshot
        .compile(&Recipe::grpo().group_size(8))
        .unwrap()
        .samples
        .is_empty());
    // Uniform families are learnable only if asked for.
    assert_eq!(
        snapshot
            .compile(
                &Recipe::grpo()
                    .group_size(4)
                    .allow_uniform()
                    .min_reward_variance(0.0)
            )
            .unwrap()
            .samples
            .len(),
        3
    );
}

#[test]
fn ppo_transitions_carry_what_the_trainer_needs_including_the_terminal_reward() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (definition, instance, initial) = coding_task(1);
    let mut run = c
        .start_attempt(
            &definition,
            &instance,
            &initial,
            &PolicyRef::new("p", "1"),
            None,
        )
        .unwrap();
    let first = run
        .decision()
        .logprob(-0.3)
        .value_estimate(0.6)
        .commit(Action::new("a", serde_json::Value::Null))
        .unwrap();
    run.transition(&first, &state("s1"), Some(0.1), None)
        .unwrap();
    let second = run
        .decision()
        .logprob(-1.2)
        .commit(Action::new("b", serde_json::Value::Null))
        .unwrap();
    run.transition(&second, &state("s2"), None, None).unwrap();
    run.finish(Outcome::Pass).unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot.compile(&Recipe::ppo()).unwrap();
    let transitions: Vec<_> = plan
        .samples
        .iter()
        .map(|s| match &s.body {
            SampleBody::Transition {
                old_logprob,
                value_estimate,
                reward,
                done,
                state,
                next_state,
                ..
            } => (
                *old_logprob,
                *value_estimate,
                *reward,
                *done,
                *state,
                *next_state,
            ),
            _ => panic!("not a transition"),
        })
        .collect();
    assert_eq!(transitions.len(), 2);
    assert_eq!(
        transitions[0],
        (
            Some(-0.3),
            Some(0.6),
            Some(0.1),
            false,
            initial.id().unwrap(),
            state("s1").id().unwrap()
        )
    );
    assert_eq!(transitions[1].0, Some(-1.2));
    assert_eq!(
        transitions[1].1, None,
        "no estimate was recorded, so none is invented"
    );
    assert_eq!(
        (transitions[1].2, transitions[1].3),
        (Some(1.0), true),
        "the last step gets the outcome as its reward"
    );
    let plain = snapshot
        .compile(&Recipe::ppo().without_terminal_reward())
        .unwrap();
    let SampleBody::Transition { reward, .. } = &plain.samples[1].body else {
        panic!()
    };
    assert_eq!(*reward, None);
    let _ = Content::text("");
}

#[test]
fn prm_labels_steps_from_standing_evaluations_only() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (decisions, _) = attempt(&mut c, 1, "p", 3, Outcome::Pass);
    for (n, (who, score)) in [("critic", 0.9), ("critic", 0.2), ("flaky", 0.5)]
        .into_iter()
        .enumerate()
    {
        c.evaluate(Evaluation::new(
            Target::Record(decisions[n].id),
            EvaluatorRef::new(who, "1"),
            "reasoning_quality",
            score,
            0.8,
        ))
        .unwrap();
    }
    c.retract(EvaluatorRef::new("flaky", "1"), "noisy").unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot
        .compile(&Recipe::prm().criterion("reasoning_quality"))
        .unwrap();
    let scores: Vec<f64> = plan
        .samples
        .iter()
        .map(|s| match &s.body {
            SampleBody::StepLabel { score, .. } => *score,
            _ => panic!("not a step label"),
        })
        .collect();
    assert_eq!(scores, [0.9, 0.2]);
}

#[test]
fn a_plan_is_reproducible_from_its_recipe_and_snapshot_whatever_came_after() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    attempt(&mut c, 1, "a", 3, Outcome::Pass);
    attempt(&mut c, 2, "b", 3, Outcome::Pass);
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();
    snapshot.pin("training-run-1").unwrap();
    let recipe = Recipe::sft().seed(7).max_samples(4);
    let first = snapshot.compile(&recipe).unwrap();

    attempt(&mut c, 3, "c", 3, Outcome::Pass);
    c.flush().unwrap();
    let again = db
        .snapshot_at(snapshot.id())
        .unwrap()
        .compile(&recipe)
        .unwrap();

    assert_eq!(first.id().unwrap(), again.id().unwrap());
    assert_eq!(first.samples, again.samples);
    assert!(
        db.snapshot()
            .unwrap()
            .compile(&recipe)
            .unwrap()
            .id()
            .unwrap()
            != first.id().unwrap()
    );
}

#[test]
fn a_seed_chooses_which_samples_survive_a_cap_and_the_same_seed_always_chooses_the_same() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    for task in 0..6 {
        attempt(&mut c, task, "a", 3, Outcome::Pass);
    }
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();
    let pick = |seed| {
        snapshot
            .compile(&Recipe::sft().seed(seed).max_samples(5))
            .unwrap()
            .id()
            .unwrap()
    };
    assert_eq!(pick(1), pick(1));
    assert_ne!(pick(1), pick(2));
    assert_eq!(
        snapshot
            .compile(&Recipe::sft().max_samples(5))
            .unwrap()
            .samples
            .len(),
        5
    );
}

#[test]
fn no_family_can_dominate_a_dataset() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    attempt(&mut c, 1, "a", 10, Outcome::Pass);
    attempt(&mut c, 2, "a", 2, Outcome::Pass);
    c.flush().unwrap();
    let plan = db
        .snapshot()
        .unwrap()
        .compile(&Recipe::sft().max_per_family(3))
        .unwrap();
    assert_eq!(plan.samples.len(), 3 + 2);
}

#[test]
fn a_plan_is_exported_as_json_lines_and_the_export_is_only_a_projection() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    attempt(&mut c, 1, "good", 2, Outcome::Pass);
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();
    let plan = snapshot.compile(&Recipe::sft()).unwrap();

    let mut out = Vec::new();
    plan.export_jsonl(&snapshot, &mut out).unwrap();
    let lines: Vec<serde_json::Value> = String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["type"], "sft");
    assert!(lines[0]["context"]
        .as_str()
        .unwrap()
        .contains("make the failing test pass"));
    assert!(lines[1]["target"].as_str().unwrap().contains("good"));
}

#[test]
fn a_dataset_is_recorded_with_its_recipe_and_snapshot_and_traces_to_its_sources() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (decisions, _) = attempt(&mut c, 1, "good", 2, Outcome::Pass);
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();
    let plan = snapshot.compile(&Recipe::sft()).unwrap();

    let sources = plan.sources();
    assert_eq!(sources, decisions.iter().map(|d| d.id).collect::<Vec<_>>());
    let dataset = c
        .record_dataset(plan.dataset_node().unwrap(), &sources)
        .unwrap();
    c.flush().unwrap();
    let later = db.snapshot().unwrap();
    assert!(later
        .lineage_back(dataset)
        .unwrap()
        .contains(&decisions[0].id));
    let splinter_expdb::model::Body::Dataset(node) = later.get(dataset).unwrap().unwrap().body
    else {
        panic!()
    };
    assert_eq!(node.snapshot, snapshot.id());
    assert_eq!(node.samples, 2);
}
