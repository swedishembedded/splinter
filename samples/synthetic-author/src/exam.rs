// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The examination of an author: situations the corpus does not hold, put
//! through tools, graded by comparing the option a call names with the one the
//! beliefs determine.

use std::collections::BTreeSet;

use serde_json::Value;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::episode::situation_text;
use crate::facts::case;
use crate::persona::Persona;
use crate::scenario::{held_out_combo, scenario_pool, Archetype, Scenario};
use crate::tools::{Schema, HELD_OUT, TRAINING};
use crate::writer::HELD_OUT_TOPICS;

/// Keeps the examination apart from the corpus drawn from the same seed.
const EXAM_SALT: u64 = 0xE7A3_0000_0000_0002;
/// How many situations the examination draws from.
const POOL: usize = 60_000;

/// One situation to decide.
#[derive(Clone, Debug)]
pub struct Item {
    /// A stable name.
    pub id: String,
    /// The situation.
    pub scenario: Scenario,
    /// The case, named differently from every case of the corpus.
    pub case: String,
    /// The situation in words.
    pub briefing: String,
    /// The options, in the order they are listed.
    pub options: Vec<Archetype>,
    /// The tool to decide through.
    pub schema: Schema,
    /// Whether the topic is one the author wrote about.
    pub topic_seen: bool,
    /// Whether the facts combine as some article of the corpus has them.
    pub combo_seen: bool,
    /// Whether the tool is one the model was taught with.
    pub tool_seen: bool,
    /// What the beliefs determine.
    pub expected: Archetype,
}

/// `per_cell` items for each of the four kinds of difficulty (topic seen or
/// not, combination seen or not), half through a tool the model was taught
/// with and half through one it was not, with no case named as one in
/// `corpus_cases`.
#[must_use]
pub fn items(
    persona: &Persona,
    seed: u64,
    per_cell: usize,
    corpus_cases: &BTreeSet<String>,
) -> Vec<Item> {
    let mut taken: BTreeSet<String> = corpus_cases.clone();
    let mut filled = [[0_usize; 2]; 2];
    let mut out = Vec::with_capacity(per_cell * 4);
    for scenario in scenario_pool(seed ^ EXAM_SALT, POOL) {
        if filled.iter().flatten().all(|&n| n >= per_cell) {
            break;
        }
        let topic_seen = !HELD_OUT_TOPICS.contains(&scenario.topic);
        let combo_seen = !held_out_combo(seed, &scenario);
        let cell = &mut filled[usize::from(topic_seen)][usize::from(combo_seen)];
        let decision = persona.decide(&scenario);
        if *cell >= per_cell || !decision.determined {
            continue;
        }
        let mut rng = SplitMix64::new(seed ^ EXAM_SALT ^ (scenario.id as u64).wrapping_mul(0x9E37));
        let drawn = case(&mut rng, scenario.topic);
        if !taken.insert(drawn.name.clone()) {
            continue;
        }
        let k = *cell;
        *cell += 1;
        let tool_seen = k % 2 == 0;
        let schema = if tool_seen {
            TRAINING[(k / 2) % TRAINING.len()]
        } else {
            HELD_OUT[(k / 2) % HELD_OUT.len()]
        };
        out.push(Item {
            id: format!(
                "t{}c{}-{:03}",
                u8::from(topic_seen),
                u8::from(combo_seen),
                k
            ),
            briefing: situation_text(&scenario, &drawn),
            options: scenario.options.clone(),
            case: drawn.name,
            expected: decision.choice,
            scenario,
            schema,
            topic_seen,
            combo_seen,
            tool_seen,
        });
    }
    out
}

/// Whether the call a model made, if it made one, takes the option the beliefs
/// determine through the tool the item offered.
#[must_use]
pub fn grade(item: &Item, call: Option<(&str, &Value)>) -> bool {
    call.and_then(|(name, args)| item.schema.read(name, args, &item.options)) == Some(item.expected)
}
