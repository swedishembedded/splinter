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

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::episode::situation_text;
use crate::facts::case;
use crate::mine::Verified;
use crate::persona::Persona;
use crate::scenario::{held_out_combo, scenario_pool, Archetype, Features, Scenario};
use crate::tools::{Schema, HELD_OUT, TRAINING};
use crate::writer::HELD_OUT_TOPICS;

/// Keeps the examination apart from the corpus drawn from the same seed.
const EXAM_SALT: u64 = 0xE7A3_0000_0000_0002;
/// Keeps the order in which options are listed apart from every other draw of
/// the examination's seed.
const ORDER_SALT: u64 = 0x0D0E_0000_0000_0003;
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
    /// Whether the way the facts and options combine was not withheld from the
    /// corpus's planner. It says nothing of whether the corpus holds the
    /// combination: see `combo_trained`.
    pub combo_seen: bool,
    /// Whether some decision kept from the corpus has this item's facts and
    /// options, read off the corpus itself. Absent in the first version of the
    /// examination, which did not ask.
    pub combo_trained: Option<bool>,
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
        let tool_seen = k.is_multiple_of(2);
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
            combo_trained: None,
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

/// Which examination is drawn. The two share their situations; they differ in
/// the order options are listed in and in what is known about each item.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Version {
    /// The examination as first reported: every item lists its options in
    /// their canonical order, which puts the right answer first in about half
    /// the items, so a policy that always takes the first option beats a
    /// guess.
    V1,
    /// Options listed in an order drawn from the seed, the right answer at
    /// each position equally often within every cell to within one item, and
    /// every item says whether the corpus trained on its combination.
    #[default]
    V2,
}

/// What the kept decisions of a corpus cover: the combinations of facts and
/// open ways that some kept decision has.
#[derive(Clone, Debug, Default)]
pub struct Trained {
    combinations: BTreeSet<([bool; 6], Vec<Archetype>)>,
}

fn combination(features: &Features, options: &[Archetype]) -> ([bool; 6], Vec<Archetype>) {
    let facts = [
        features.exceeds_authority,
        features.urgent,
        features.binds_future,
        features.irreversible,
        features.members_harmed,
        features.outsider_offers_help,
    ];
    let mut ways = options.to_vec();
    ways.sort();
    ways.dedup();
    (facts, ways)
}

impl Trained {
    /// What `decisions` cover. A decision whose essay left a fact unstated
    /// covers no combination, as it makes no training record that names all
    /// the facts.
    #[must_use]
    pub fn from_decisions(decisions: &[Verified]) -> Self {
        Self {
            combinations: decisions
                .iter()
                .filter_map(|d| {
                    d.facts
                        .features(d.offer_made)
                        .map(|features| combination(&features, &d.open))
                })
                .collect(),
        }
    }

    /// Whether some decision has `features` and the ways `options`, however
    /// the latter are ordered.
    #[must_use]
    pub fn has(&self, features: &Features, options: &[Archetype]) -> bool {
        self.combinations.contains(&combination(features, options))
    }
}

/// What the examination is drawn from.
pub struct Spec<'a> {
    /// Which examination.
    pub version: Version,
    /// Items of each kind of difficulty.
    pub per_cell: usize,
    /// The names of the cases the corpus tells, which no item may use.
    pub corpus_cases: &'a BTreeSet<String>,
    /// What the corpus's kept decisions cover; read only by the second
    /// version.
    pub trained: &'a Trained,
}

/// The examination `spec` asks for.
#[must_use]
pub fn draw(persona: &Persona, seed: u64, spec: &Spec<'_>) -> Vec<Item> {
    let mut out = items(persona, seed, spec.per_cell, spec.corpus_cases);
    if spec.version == Version::V1 {
        return out;
    }
    let mut placed = [[0_usize; 2]; 2];
    for item in &mut out {
        let nth = &mut placed[usize::from(item.topic_seen)][usize::from(item.combo_seen)];
        let at = expected_position(seed, item, *nth);
        *nth += 1;
        item.options = listed(seed, item, at);
        item.combo_trained = Some(
            spec.trained
                .has(&item.scenario.features, &item.scenario.options),
        );
    }
    out
}

/// Where the right answer is listed for the `nth` item of a cell. Positions
/// come in blocks, each a seeded permutation of every position, so that
/// however many items a cell holds no position is used more than once more
/// than another.
fn expected_position(seed: u64, item: &Item, nth: usize) -> usize {
    let width = item.options.len();
    let (block, offset) = (nth / width, nth % width);
    let cell = u64::from(item.topic_seen) << 1 | u64::from(item.combo_seen);
    let mut rng = SplitMix64::new(seed ^ EXAM_SALT ^ ORDER_SALT ^ (cell << 40) ^ block as u64);
    let mut positions: Vec<usize> = (0..width).collect();
    rng.shuffle(&mut positions);
    positions[offset]
}

/// The options of `item` with the right one at `at` and the others in an order
/// the seed draws.
fn listed(seed: u64, item: &Item, at: usize) -> Vec<Archetype> {
    let mut rng = SplitMix64::new(
        seed ^ EXAM_SALT ^ ORDER_SALT ^ (item.scenario.id as u64).wrapping_mul(0x9E37),
    );
    let mut others: Vec<Archetype> = item
        .scenario
        .options
        .iter()
        .copied()
        .filter(|&o| o != item.expected)
        .collect();
    rng.shuffle(&mut others);
    others.insert(at, item.expected);
    others
}
