// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An episode: one situation with invented facts, the options it offers, and
//! no decision - the decision is the author's, and code computes it.

use serde::Serialize;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::facts::{case, Case};
use crate::persona::Persona;
use crate::scenario::{held_out_combo, scenario_pool, Archetype, Scenario, TOPICS};

/// Keeps the situations of a corpus apart from those of an exam drawn from
/// the same seed.
const CORPUS_SALT: u64 = 0x5EED_C0DE_0000_0001;
/// How many situations a corpus draws from.
const POOL: usize = 20_000;

/// One situation to be told or decided.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Episode {
    /// The situation's facts and options.
    pub scenario: Scenario,
    /// The invented facts of the case.
    pub case: Case,
    /// The situation in words, stating no decision.
    pub situation: String,
    /// The ways to act, in words, in the scenario's order.
    pub options: Vec<(Archetype, String)>,
}

/// How an option reads as a thing to do.
#[must_use]
pub fn option_phrase(option: Archetype, case: &Case) -> String {
    match option {
        Archetype::ActAlone => "settle the matter yourself, at once".into(),
        Archetype::ConsultThenAct => "bring it before the assembly and act on its word".into(),
        Archetype::DeferToOutsider => format!("accept the offer of {}", case.house),
        Archetype::HoldAndWait => "wait, and do nothing yet".into(),
    }
}

/// What the author did, in the words the writer is briefed with: the phrase
/// an article may not repeat.
#[must_use]
pub fn did_phrase(option: Archetype, case: &Case) -> String {
    match option {
        Archetype::ActAlone => "settled it yourself, at once, without waiting".into(),
        Archetype::ConsultThenAct => "put it before the assembly and acted on its word".into(),
        Archetype::DeferToOutsider => format!("accepted the offer of {}", case.house),
        Archetype::HoldAndWait => "waited, and did nothing yet".into(),
    }
}

/// The situation in words: the facts of the case and of the scenario.
#[must_use]
pub fn situation_text(scenario: &Scenario, case: &Case) -> String {
    let f = &scenario.features;
    let mut s = format!(
        "In the year {}, a matter arose at {} over {}: {}. About {} households depend on it. ",
        case.year, case.place, case.name, TOPICS[scenario.topic], case.households
    );
    s.push_str(if f.exceeds_authority {
        "The charter of the stewardship gives no power to do what is wanted. "
    } else {
        "The charter of the stewardship plainly allows what is wanted. "
    });
    s.push_str(if f.urgent {
        "Every day of delay costs dearly. "
    } else {
        "There is no pressing hurry; the matter could stand for a season. "
    });
    s.push_str(if f.binds_future {
        "Whatever is settled will bind the community for a generation. "
    } else {
        "Whatever is settled will hold only for this year. "
    });
    s.push_str(if f.irreversible {
        "Once done it cannot be undone. "
    } else {
        "It could be reversed next season. "
    });
    s.push_str(if f.members_harmed {
        "Several families are already suffering for it. "
    } else {
        "No member of the community has yet been hurt by it. "
    });
    if f.outsider_offers_help {
        s.push_str(&format!("{} has offered its help. ", case.house));
    }
    s.trim_end().to_string()
}

impl Episode {
    /// The facts of the episode as a case file: values to be told in the
    /// writer's own words, never a finished sentence to be copied.
    #[must_use]
    pub fn case_file(&self) -> serde_json::Value {
        let f = &self.scenario.features;
        serde_json::json!({
            "case": self.case.name,
            "place": self.case.place,
            "year": self.case.year,
            "households_depending": self.case.households,
            "matter": TOPICS[self.scenario.topic],
            "authority": if f.exceeds_authority { "beyond what the charter grants" } else { "within what the charter grants" },
            "time_pressure": if f.urgent { "severe" } else { "none" },
            "reach": if f.binds_future { "a generation" } else { "one year" },
            "undoable": !f.irreversible,
            "families_hurt_so_far": f.members_harmed,
            "offer_from": if f.outsider_offers_help { serde_json::json!(self.case.house) } else { serde_json::Value::Null },
        })
    }
}

/// `n` episodes of `topic` for `persona`, from the situations the beliefs
/// settle, skipping scenario ids in `used` (and adding the ones taken).
///
/// Cases are named differently from one another within the call.
#[must_use]
pub fn episodes_for(
    persona: &Persona,
    seed: u64,
    topic: usize,
    n: usize,
    used: &mut Vec<usize>,
) -> Vec<Episode> {
    let pool = scenario_pool(seed ^ CORPUS_SALT, POOL);
    let mut out: Vec<Episode> = Vec::with_capacity(n);
    for scenario in pool {
        if out.len() == n {
            break;
        }
        if scenario.topic != topic
            || used.contains(&scenario.id)
            || held_out_combo(seed, &scenario)
            || !persona.decide(&scenario).determined
        {
            continue;
        }
        let mut rng =
            SplitMix64::new(seed ^ CORPUS_SALT ^ (scenario.id as u64).wrapping_mul(0x9E37));
        let case = case(&mut rng, topic);
        if out.iter().any(|e| e.case.name == case.name) {
            continue;
        }
        used.push(scenario.id);
        out.push(Episode {
            situation: situation_text(&scenario, &case),
            options: scenario
                .options
                .iter()
                .map(|&o| (o, option_phrase(o, &case)))
                .collect(),
            case,
            scenario,
        });
    }
    out
}
