// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! The Zork controller manual and the scripted models the acquisition specs
//! share: one model for every role, and a trainer double whose candidate
//! knows the topics of what it was trained on.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use serde_json::json;
use splinter_agent::CancelToken;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::acquire::AcquireRequest;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::train::{TrainPlan, Trainer};

use super::gate::{answer, put_base, question, FakeTrainer};
use super::{scratch_context, Scratch, Scripted};

pub const TOPICS: [&str; 3] = ["alpha", "beta", "gamma"];

/// The manual: a section a topic, each stating the codes of its register
/// (`alpha-0` to `alpha-7`, `alpha-10` to `alpha-17` and `alpha-20` to `alpha-27`), the third using the
/// first.
pub fn manual() -> String {
    let codes = |topic: &str| {
        (0..8)
            .chain(10..18)
            .chain(20..28)
            .map(|i| format!("{topic}-{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut text = String::from("# The Zork controller manual\n\n");
    for topic in TOPICS {
        text.push_str(&format!(
            "## `{topic}` register\n\nThe `{topic}` register holds the unit codes of its bank. It returns a code number for each unit of the bank, and the codes are {}. The controller must never reuse a code.\n\n",
            codes(topic)
        ));
    }
    // The third register is built on the first.
    text.replace(
        "The `gamma` register holds the unit codes of its bank.",
        "The `gamma` register holds the unit codes of its bank, counted over the `alpha` register.",
    )
}

pub fn write_manual(scratch: &Scratch) -> PathBuf {
    let dir = scratch.0.join("docs");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("manual.md");
    std::fs::write(&file, manual()).unwrap();
    file
}

/// The topic whose section a request shows: its heading is in the prompt.
pub fn topic_of(prompt: &str) -> Option<&'static str> {
    TOPICS
        .into_iter()
        .find(|t| prompt.contains(&format!("## `{t}` register")))
}

/// One scripted model for every role. As the generator it writes eight tasks
/// from the section it is shown, numbered 0 to 7 for an assessment and 10 to
/// 17 for training, so a training task never repeats a probe; as the teacher
/// it answers from the material; as the policy it knows nothing; as the
/// planner it says nothing a graph can be made from.
pub fn model() -> Scripted {
    authoring(10..18, &[])
}

/// [`model`] as it would be after training: it knows `knows` and, as the
/// generator, numbers its training tasks `numbers`.
pub fn authoring(numbers: std::ops::Range<usize>, knows: &[&str]) -> Scripted {
    writer(numbers, knows, |_| (0..8).collect())
}

/// [`model`] writing `per_call` probes a request, numbered on from the last
/// it wrote for the topic: a small generator that needs asking again.
pub fn writing_probes_in_batches(per_call: usize) -> Scripted {
    let written = std::sync::Mutex::new(std::collections::BTreeMap::<String, usize>::new());
    writer(10..18, &[], move |topic| {
        let mut written = written.lock().unwrap();
        let from = written.entry(topic.to_string()).or_default();
        let numbers = (*from..*from + per_call).collect();
        *from += per_call;
        numbers
    })
}

/// [`model`] writing `per_call` training tasks a request, numbered on from
/// the last it wrote for the topic: a small generator a round must ask again.
pub fn writing_tasks_in_batches(per_call: usize) -> Scripted {
    let written = std::sync::Mutex::new(std::collections::BTreeMap::<String, usize>::new());
    let probes = |_: &str| (0..8).collect();
    writer_with(&[], probes, move |topic| {
        let mut written = written.lock().unwrap();
        let from = written.entry(topic.to_string()).or_insert(10);
        let numbers = (*from..*from + per_call).collect();
        *from += per_call;
        numbers
    })
}

fn writer(
    numbers: std::ops::Range<usize>,
    knows: &[&str],
    probes: impl Fn(&str) -> Vec<usize> + Send + Sync + 'static,
) -> Scripted {
    writer_with(knows, probes, move |_| numbers.clone().collect())
}

fn writer_with(
    knows: &[&str],
    probes: impl Fn(&str) -> Vec<usize> + Send + Sync + 'static,
    training: impl Fn(&str) -> Vec<usize> + Send + Sync + 'static,
) -> Scripted {
    let knows: Vec<String> = knows.iter().map(|t| (*t).to_string()).collect();
    Scripted::new(move |prompt| {
        if prompt.contains("You write training tasks") {
            let Some(topic) = topic_of(prompt) else {
                return String::new();
            };
            let numbers: Vec<usize> = if prompt.contains("(assessment)") {
                probes(topic)
            } else {
                training(topic)
            };
            let tasks: Vec<_> = numbers
                .into_iter()
                .map(|i| {
                    json!({
                        "instruction": question(topic, i),
                        "subject": topic,
                        "reference": answer(topic, i),
                        "evidence": [{ "section": 0, "quote": answer(topic, i) }],
                    })
                })
                .collect();
            return json!({ "tasks": tasks }).to_string();
        }
        if prompt.contains("Reference material:") {
            return super::gate::reply(&TOPICS.map(String::from), prompt);
        }
        super::gate::reply(&knows, prompt)
    })
}

/// Training as a test double whose candidate knows the topics of everything
/// it was trained on.
pub struct Learner;

pub fn topics_in(plan: &TrainPlan) -> Vec<String> {
    let mut found = Vec::new();
    for dataset in &plan.datasets {
        let text = std::fs::read_to_string(&dataset.path).unwrap();
        for topic in TOPICS {
            if text.contains(&format!("What is the {topic} code number"))
                && !found.contains(&topic.to_string())
            {
                found.push(topic.to_string());
            }
        }
    }
    found
}

impl Trainer for Learner {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        let topics = topics_in(plan);
        let known: Vec<&str> = topics.iter().map(String::as_str).collect();
        FakeTrainer::knowing(&known).train(ctx, plan, cancel)
    }

    fn train_preference(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        FakeTrainer::knowing(&[]).train_preference(ctx, plan, cancel)
    }
}

pub fn request(manual: &Path) -> AcquireRequest {
    AcquireRequest {
        pass_at_k: PassAtK::new(2, None, None),
        probes_per_skill: 3,
        no_release: true,
        ..AcquireRequest::new(
            "Become exact about the Zork controller's registers",
            vec![manual.display().to_string()],
        )
    }
}

pub fn context(test: &str) -> (Scratch, Context) {
    let (scratch, ctx) = scratch_context(test, model(), false);
    put_base(&ctx);
    (scratch, ctx)
}
