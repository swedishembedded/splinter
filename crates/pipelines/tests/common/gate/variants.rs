// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate fixtures' tasks asked in other words: variants of a fact, stored
//! with the fact they are a variant of.

use splinter_core::experience::Task;
use splinter_orchestrator::Context;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use super::{fact, question, WORDINGS};

/// `fact(topic, i)` asked in other words ([`WORDINGS`]), stored.
pub fn variant_of_fact(ctx: &Context, topic: &str, i: usize, wording: usize) -> Task {
    let instruction = format!("{} {}", WORDINGS[wording], question(topic, i));
    variant_asking(ctx, topic, i, &instruction)
}

/// `fact(topic, i)` asking `instruction`, stored.
pub fn variant_asking(ctx: &Context, topic: &str, i: usize, instruction: &str) -> Task {
    let variant = fact(topic, i).with_instruction(instruction).unwrap();
    ctx.tasks().put(&variant).unwrap();
    variant
}

/// Stores a task set of variants: `variants` of the facts about `topic`
/// numbered `facts`, each in the first `wordings` ways, each recorded as a
/// variant of its fact.
pub fn variant_set(ctx: &Context, topic: &str, facts: &[usize], wordings: usize) -> TaskSetId {
    let mut members = Vec::new();
    for &i in facts {
        for wording in 0..wordings {
            members.push((i, variant_of_fact(ctx, topic, i, wording)));
        }
    }
    variant_set_of(ctx, topic, members)
}

/// Stores a task set of `variants`, each with the fact about `topic` it is
/// a variant of.
pub fn variant_set_of(ctx: &Context, topic: &str, variants: Vec<(usize, Task)>) -> TaskSetId {
    let members = variants
        .into_iter()
        .map(|(i, task)| TaskEntry {
            task: task.task.id,
            generator: Some("scripted/generator".into()),
            prompt: None,
            variant_of: Some(fact(topic, i).task.id),
            subject: None,
        })
        .collect();
    ctx.tasks()
        .put_set(&TaskSet {
            name: format!("variants of {topic} facts"),
            members,
        })
        .unwrap()
}
