// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that learn what a person
// advised from what they wrote, for its clients. If your team needs
// expertise in grounding a model's advice in a body of writing, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: `learn` over a person's letters with the `converse` kind teaches the
//! policy to talk as the person, from a dialogue the teacher holds with a
//! simulated other speaker.
//!
//! The teacher is shown the letter; the other speaker never is. The training
//! record is the whole conversation, the writer's every reply supervised and
//! the letter in none of it. A dialogue whose replies state a year or name
//! the letter does not hold teaches nothing: the grounding check fails it
//! and no record is made. The other speaker is sometimes told to ask, on its
//! last turn, for a detail the exchange has not given, so that the student
//! also sees what declining to invent looks like.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::gate::{gate_context, Brain, FakeTrainer, ANCHOR};
use common::Scripted;
use serde_json::json;
use splinter_agent::solve::{Model, MATERIAL_HEADING};
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::role::Role;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::dialogue::{probes_beyond_the_source, STUDENT_ROLE};
use splinter_pipelines::learn::{learn, ExamPlan, LearnRequest, Learned};
use splinter_pipelines::raft::PassageShare;
use splinter_pipelines::release::arm;
use splinter_pipelines::train::{TrainPlan, Trainer};

const LETTER: &str = "# To a young man

## Study

I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.
";

const PASSAGE: &str = "I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.";

const OPENING: &str = "My mornings vanish in idleness; how do you keep a habit of study?";

/// A second letter, of another family of sources: a judge is measured on
/// what one letter's reference is and another's is not.
const THRIFT: &str = "# To a young householder

## Thrift

I advise you to keep a strict account of every expense each week, and never spend a shilling you cannot name, for thrift is the first of the household virtues and a debt is a chain upon the neck.
";

const THRIFT_PASSAGE: &str = "I advise you to keep a strict account of every expense each week, and never spend a shilling you cannot name, for thrift is the first of the household virtues and a debt is a chain upon the neck.";

const THRIFT_OPENING: &str = "My money slips away unnoticed; how do you keep thrift in your house?";

/// A policy that writes one opening from the letter, plays the other speaker
/// with fixed follow-ups, and, as the teacher shown the letter, replies with
/// `third` as its last reply.
fn policy(third: &'static str) -> Scripted {
    Scripted::new(move |prompt| {
        if prompt.contains("You write training tasks") {
            let (opening, passage) = if prompt.contains("strict account") {
                (THRIFT_OPENING, THRIFT_PASSAGE)
            } else {
                (OPENING, PASSAGE)
            };
            json!({ "tasks": [{
                "instruction": opening,
                "reference": passage,
                "evidence": [{ "section": 0, "quote": passage }]
            }]})
            .to_string()
        } else if prompt.contains(STUDENT_ROLE) {
            let message = if prompt.contains("And then?") {
                "Why does the pen matter so much?"
            } else {
                "And then?"
            };
            json!({ "message": message }).to_string()
        } else if prompt.contains(MATERIAL_HEADING) && prompt.contains("strict account") {
            "Keep an account of every expense, and never spend what you cannot name.".to_string()
        } else if prompt.contains(MATERIAL_HEADING) {
            if prompt.contains("Why does the pen matter so much?") {
                third.to_string()
            } else if prompt.contains("And then?") {
                "Read some history or ethics each day, and write down what you read.".to_string()
            } else {
                "Fix a habit of study every morning, before anything else.".to_string()
            }
        } else {
            "I do not know.".to_string()
        }
    })
}

/// A judge that passes an answer sharing a key word with the reference: what
/// a reference about study and one about expense have not in common.
fn judge() -> Scripted {
    Scripted::new(|prompt| {
        let reference = prompt
            .split("REFERENCE:\\n")
            .nth(1)
            .and_then(|t| t.split("\\n\\nANSWER:").next())
            .unwrap_or_default();
        let answer = prompt.split("ANSWER:\\n").nth(1).unwrap_or_default();
        let shared = ["study", "expense"]
            .iter()
            .any(|key| reference.contains(key) && answer.contains(key));
        if shared {
            "PASS\nit gives the advice".into()
        } else {
            "FAIL\nit does not".into()
        }
    })
}

/// A trainer that records what it was asked to train on.
struct Student {
    plans: Mutex<Vec<TrainPlan>>,
}

impl Trainer for Student {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        self.plans.lock().unwrap().push(plan.clone());
        let trained = FakeTrainer::knowing(&[ANCHOR]).train(ctx, plan, cancel)?;
        ctx.add_model(
            arm(ctx.config(), Some(&trained.adapter)),
            Model::new(
                Arc::new(Scripted::new(|_| "I do not know.".into())),
                "scripted/learned",
            ),
        );
        Ok(trained)
    }

    fn train_preference(
        &self,
        _ctx: &Context,
        _plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        panic!("learn trains chat datasets only")
    }
}

/// The learn over two letters with `converse`, the judge role played by
/// `judge` (the policy, which is also the teacher, when `None`).
fn attempt(
    test: &str,
    third: &'static str,
    judge: Option<Scripted>,
    persona: Option<&str>,
    passages: Option<PassageShare>,
    voice: Option<f64>,
) -> (
    Result<Learned, OrchestratorError>,
    common::Scratch,
    Context,
    Scripted,
) {
    let (scratch, ctx) = gate_context(test, Brain::Missing);
    let policy = policy(third);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy.clone()), common::POLICY),
    );
    let judge_ref: ModelRef = "local:test/judge".parse().unwrap();
    let roles = judge.map_or_else(Default::default, |judge| {
        ctx.add_model(
            judge_ref.clone(),
            Model::new(Arc::new(judge), "scripted/judge"),
        );
        [(Role::Judge, judge_ref)].into()
    });
    let letters = scratch.0.join("letters");
    std::fs::create_dir_all(&letters).unwrap();
    std::fs::write(letters.join("study.md"), LETTER).unwrap();
    std::fs::write(letters.join("thrift.md"), THRIFT).unwrap();
    let student = Student {
        plans: Mutex::new(Vec::new()),
    };
    let learned = learn(
        &ctx,
        &LearnRequest {
            sources: vec![letters.display().to_string()],
            kinds: vec!["converse".into()],
            persona: persona.map(str::to_string),
            passages,
            voice,
            // A scripted policy answers no general task, so it has nothing
            // to rehearse.
            rehearsal: Some(0.0),
            roles,
            exam: ExamPlan {
                families: Some(0),
                ..ExamPlan::default()
            },
            no_release: true,
            distill: true,
            ..LearnRequest::default()
        },
        &student,
    );
    (learned, scratch, ctx, policy)
}

fn run(
    test: &str,
    third: &'static str,
) -> (splinter_pipelines::learn::LearnReport, common::Scratch) {
    let (learned, scratch, _ctx, _) = attempt(test, third, Some(judge()), None, None, None);
    let Learned::Ran(ran) = learned.unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    (ran.report, scratch)
}

#[test]
fn a_grounded_dialogue_becomes_one_record_of_the_whole_conversation() {
    let (report, _scratch) = run(
        "converse-learn-grounded",
        "The pen fixes what the memory lets slip, so write down what you have read.",
    );
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(
        (taught.verify.passed, taught.verify.failed),
        (2, 0),
        "{taught:#?}"
    );
    let dataset = report.dataset.as_ref().unwrap();
    assert_eq!(dataset.records, 2, "one dialogue per letter: {dataset:#?}");
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    let record: serde_json::Value = text
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .find(|record| record["messages"][1]["content"] == OPENING)
        .unwrap();
    let turns: Vec<(&str, bool)> = record["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m["role"].as_str().unwrap(), m["train"].as_bool().unwrap()))
        .collect();
    assert_eq!(
        turns,
        [
            ("system", false),
            ("user", false),
            ("assistant", true),
            ("user", false),
            ("assistant", true),
            ("user", false),
            ("assistant", true),
        ]
    );
    assert_eq!(record["messages"][1]["content"], OPENING);
    assert_eq!(record["messages"][3]["content"], "And then?");
    assert!(
        !text.contains(MATERIAL_HEADING),
        "the letter is teacher-only"
    );
    assert!(
        !text.contains(STUDENT_ROLE),
        "the other speaker's instructions are not training data"
    );
}

#[test]
fn a_dialogue_that_states_a_year_the_letter_does_not_hold_teaches_nothing() {
    let (report, _scratch) = run(
        "converse-learn-invents",
        "I took up this habit in 1762 and kept it all my life.",
    );
    let taught = report.teach.as_ref().unwrap();
    // The dialogue about study states a year its letter lacks: grounding
    // refutes it, whatever a judge says. The one about thrift stands.
    assert_eq!(
        (taught.verify.passed, taught.verify.failed),
        (1, 1),
        "{taught:#?}"
    );
    let dataset = report.dataset.as_ref().unwrap();
    assert_eq!(dataset.records, 1, "{dataset:#?}");
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    assert!(text.contains(THRIFT_OPENING) && !text.contains(OPENING) && !text.contains("1762"));
}

#[test]
fn one_dialogue_in_eight_ends_by_asking_beyond_the_letter_and_the_choice_is_stable() {
    let digests: Vec<_> = (0..400u32)
        .map(|n| splinter_core::digest::Digest::of(&n.to_le_bytes()))
        .collect();
    let probing = digests
        .iter()
        .filter(|d| probes_beyond_the_source(d))
        .count();
    assert!((25..=80).contains(&probing), "{probing} of 400");
    assert!(digests
        .iter()
        .all(|d| probes_beyond_the_source(d) == probes_beyond_the_source(d)));
}

#[test]
fn a_kind_only_a_judge_can_pass_is_refused_a_judge_that_is_also_the_teacher() {
    // With no judge named the policy plays it, and the policy is the
    // teacher whose dialogues it would grade.
    let (learned, _scratch, _ctx, _) = attempt(
        "converse-learn-self-judge",
        "Study each morning.",
        None,
        None,
        None,
        None,
    );
    let Err(refused) = learned else {
        panic!("a judge that grades itself is refused");
    };
    assert!(
        matches!(&refused, OrchestratorError::Refused(why) if why.contains("also the teacher")),
        "{refused:?}"
    );
}

#[test]
fn a_judge_that_cannot_tell_a_reference_from_another_is_refused_with_its_numbers() {
    let always_pass = Scripted::new(|_| "PASS\nit does".into());
    let (learned, _scratch, _ctx, _) = attempt(
        "converse-learn-weak-judge",
        "Study each morning.",
        Some(always_pass),
        None,
        None,
        None,
    );
    let Err(refused) = learned else {
        panic!("an imprecise judge is refused");
    };
    // Its passes admit answers to the training set, and half of them are
    // wrong: what it passes cannot be trusted, whatever it fails.
    assert!(
        matches!(&refused, OrchestratorError::Refused(why)
            if why.contains("not precise enough") && why.contains("on passes")),
        "{refused:?}"
    );
}

#[test]
fn a_persona_opens_every_training_conversation_and_the_manifest_records_it() {
    let (learned, _scratch, ctx, policy) = attempt(
        "converse-learn-persona",
        "The pen fixes what the memory lets slip, so write down what you have read.",
        Some(judge()),
        Some("Benjamin Franklin"),
        None,
        None,
    );
    let Learned::Ran(ran) = learned.unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    let dataset = ran.report.dataset.as_ref().unwrap();
    let prompt = splinter_core::prompt::persona_prompt("Benjamin Franklin");
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    assert!(!text.is_empty());
    for line in text.lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(record["messages"][0]["role"], "system");
        assert_eq!(record["messages"][0]["content"], prompt);
    }
    // The dataset names the prompt, and what is trained on it is asked under it.
    assert_eq!(
        ctx.system_prompt_of(std::slice::from_ref(&dataset.dataset))
            .unwrap()
            .as_deref(),
        Some(prompt.as_str())
    );
    // The other speaker's role never reaches the record.
    assert!(!text.contains(STUDENT_ROLE));
    // A run that learns to be a person trains on the writer's own text
    // beside the dialogues by default: records of the letters word for word
    // asked under the persona prompt or the line naming the writer, a share
    // of the tokens, in a dataset of their own that the training takes with
    // the dialogues.
    let voice = ran.report.voice.as_ref().expect("the writer's own text");
    assert!(voice.records >= 1, "{voice:?}");
    let voice_text = std::fs::read_to_string(&voice.path).unwrap();
    let mut answers = 0;
    for line in voice_text.lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        let system = record["messages"][0]["content"].as_str().unwrap();
        assert!(
            system == prompt || system == "You are Benjamin Franklin.",
            "{system}"
        );
        let answer = record["messages"][2]["content"].as_str().unwrap();
        assert!(
            LETTER.contains(answer) || THRIFT.contains(answer),
            "{answer}"
        );
        answers += 1;
    }
    assert_eq!(answers, voice.records);
    let candidate = ran.report.candidate.as_ref().unwrap();
    assert_eq!(
        candidate.datasets.len(),
        2,
        "the dialogues and the writer's text"
    );
    assert_eq!(candidate.datasets[1], voice.dataset);
    // The teacher answered under the prompt the student is trained under:
    // every one of its solves was sent the persona as its system turn, so the
    // records' answers were written as the person they open with.
    let prompts = policy.prompts.lock().unwrap();
    let systems = policy.systems.lock().unwrap();
    let teaching: Vec<&Vec<String>> = prompts
        .iter()
        .zip(systems.iter())
        .filter(|(prompt, _)| prompt.contains(MATERIAL_HEADING))
        .map(|(_, system)| system)
        .collect();
    assert!(!teaching.is_empty());
    for system in teaching {
        assert_eq!(
            system.as_slice(),
            std::slice::from_ref(&prompt),
            "{system:?}"
        );
    }
}

#[test]
fn a_dialogue_whose_reply_talks_about_the_material_teaches_nothing() {
    // Grounded, and a judge passes it, but the student is never shown the
    // material: a reply that speaks of it teaches the student to pretend to
    // a document.
    let (report, _scratch) = run(
        "converse-learn-document-talk",
        "According to the material, the pen fixes what the memory lets slip.",
    );
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(
        (taught.verify.passed, taught.verify.failed),
        (1, 1),
        "{taught:#?}"
    );
    let dataset = report.dataset.as_ref().unwrap();
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    assert!(!text.contains("According to the material"));
}

/// [`judge`], which also says whether a passage is a natural reply to a
/// message: the study message to the study passage and the thrift one to the
/// thrift passage.
fn judge_of_reference_and_fit() -> Scripted {
    let reference = judge();
    Scripted::new(move |prompt| {
        if !prompt.contains("MESSAGE:") {
            return reference_reply(&reference, prompt);
        }
        let message = prompt
            .split("MESSAGE:\\n")
            .nth(1)
            .and_then(|t| t.split("\\n\\nPASSAGE:").next())
            .unwrap_or_default();
        let passage = prompt.split("PASSAGE:\\n").nth(1).unwrap_or_default();
        let fits = (message.contains("study") && passage.contains("study"))
            || (message.contains("thrift") && passage.contains("expense"));
        if fits {
            "PASS\nit answers the message".into()
        } else {
            "FAIL\nit does not".into()
        }
    })
}

/// What `reference` says of `prompt`, asked as a model is asked.
fn reference_reply(reference: &Scripted, prompt: &str) -> String {
    let reference_text = prompt
        .split("REFERENCE:\\n")
        .nth(1)
        .and_then(|t| t.split("\\n\\nANSWER:").next())
        .unwrap_or_default();
    let answer = prompt.split("ANSWER:\\n").nth(1).unwrap_or_default();
    let _ = reference;
    let shared = ["study", "expense"]
        .iter()
        .any(|key| reference_text.contains(key) && answer.contains(key));
    if shared {
        "PASS\nit gives the advice".into()
    } else {
        "FAIL\nit does not".into()
    }
}

#[test]
fn the_writers_own_passage_joins_the_training_set_as_an_answer() {
    let (learned, _scratch, _ctx, _) = attempt(
        "converse-learn-authored",
        "The pen fixes what the memory lets slip, so write down what you have read.",
        Some(judge_of_reference_and_fit()),
        None,
        None,
        None,
    );
    let Learned::Ran(ran) = learned.unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    let authored = ran.report.authored.as_ref().unwrap();
    assert_eq!((authored.tasks, authored.kept), (2, 2), "{authored:#?}");
    let dataset = ran.report.dataset.as_ref().unwrap();
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    let records: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    // One record per message: where the writer's own passage fits it, the
    // passage is kept and the teacher's paraphrase of it is the duplicate.
    assert_eq!(records.len(), 2, "{dataset:#?}");
    let single_turn: Vec<&serde_json::Value> = records
        .iter()
        .filter(|r| r["messages"].as_array().unwrap().len() == 3)
        .collect();
    assert_eq!(single_turn.len(), 2, "no dialogue: {records:#?}");
    let answers: Vec<&str> = single_turn
        .iter()
        .map(|r| r["messages"][2]["content"].as_str().unwrap())
        .collect();
    assert!(answers.contains(&PASSAGE) && answers.contains(&THRIFT_PASSAGE));
}

#[test]
fn a_share_of_the_training_records_carries_passages_in_the_prompt() {
    let (learned, _scratch, _ctx, _) = attempt(
        "converse-learn-passages",
        "The pen fixes what the memory lets slip, so write down what you have read.",
        Some(judge()),
        None,
        Some(PassageShare {
            records: 1.0,
            with_evidence: 1.0,
        }),
        None,
    );
    let Learned::Ran(ran) = learned.unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    let dataset = ran.report.dataset.as_ref().unwrap();
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    assert!(!text.is_empty());
    for line in text.lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        let prompt = record["messages"][1]["content"].as_str().unwrap();
        // The passages of the letters are shown beside the question, in the
        // form an ask gives them, and the answers are the dialogue's own.
        assert!(prompt.contains(MATERIAL_HEADING), "{prompt}");
        assert!(prompt.contains("habit of study") || prompt.contains("strict account"));
        assert_eq!(record["messages"][2]["role"], "assistant");
    }
}
