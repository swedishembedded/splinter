// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifier-rewarded training of language models
// for its clients. If your team needs expertise in turning a set of rules a
// model must keep into a reward that does not trust the model, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A task family for reinforcement learning: apply his method to a present-day
//! situation and keep every rule, rewarded by the same checks that accept a
//! helper's draft.
//!
//! The reward is read off the completion by code and has named parts, so a run
//! shows which rule it is learning to keep: the answer has the layout, it says
//! the right thing about whether his method applies, and every other rule holds
//! (the quotations are in the passages, the facts were given, a non-fit asks).
//! No model judges anything. The training pool and the pool the loop's own gate
//! draws from are disjoint, so the gate is never a score on what was trained.

use std::collections::BTreeMap;
use std::sync::Arc;

use splinter_sdk::model::rl::{Environment, Reward, Step, Task, Verifier};

use crate::curate::Document;
use crate::principles::Found;
use crate::respond::{self, Applicability};
use crate::scenario::{Case, Scenario};

/// Seeds at or above this draw from the gate's pool, below it from the training
/// pool: pin brain's two seeds either side of it and the pools never meet.
pub const GATE_SEED_BASE: u64 = 1 << 32;

/// What the reward gives for each rule kept, summing to one.
pub const FORMAT: f32 = 0.2;
pub const APPLICABILITY: f32 = 0.3;
pub const ALL_RULES: f32 = 0.5;

/// One situation, the passages he may be shown, and what a check needs.
#[derive(Clone, Debug)]
pub struct Item {
    pub scenario: Scenario,
    pub evidence: Vec<Found>,
    /// The documents the evidence is from, id and text.
    pub documents: Vec<(String, String)>,
}

/// How a prompt is turned into ids: the system message and the user turn.
pub type Encode = Arc<dyn Fn(&str, &str) -> anyhow::Result<Vec<u32>> + Send + Sync>;
/// How completion ids are turned back into text.
pub type Decode = Arc<dyn Fn(&[u32]) -> String + Send + Sync>;

/// The task family.
pub struct AdamsEnv {
    train: Vec<Item>,
    gate: Vec<Item>,
    encode: Encode,
}

impl AdamsEnv {
    /// A family over `items`, split into the pool the loop trains on and the
    /// pool its gate draws from: one in `gate_one_in` goes to the gate, by the
    /// scenario's id so the split does not depend on order.
    pub fn new(items: Vec<Item>, gate_one_in: usize, encode: Encode) -> AdamsEnv {
        let mut sorted = items;
        sorted.sort_by(|a, b| a.scenario.id.cmp(&b.scenario.id));
        let (gate, train): (Vec<Item>, Vec<Item>) = sorted.into_iter().partition(|i| {
            gate_one_in > 0
                && usize::from(blake3::hash(i.scenario.id.as_bytes()).as_bytes()[2]) % gate_one_in
                    == 0
        });
        AdamsEnv {
            train,
            gate,
            encode,
        }
    }

    /// How many items each pool holds: training, gate.
    pub fn pools(&self) -> (usize, usize) {
        (self.train.len(), self.gate.len())
    }

    /// Fail now, in words, if either pool is empty or a prompt cannot be
    /// encoded: brain panics on an environment that yields no task, and a
    /// tokenizer that cannot encode must not reach it.
    ///
    /// # Errors
    /// A pool is empty, or the first prompt of a pool cannot be encoded.
    pub fn preflight(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.train.is_empty(), "no scenario is left to train on");
        anyhow::ensure!(
            !self.gate.is_empty(),
            "no scenario is left for the gate; raise the gate share"
        );
        for (name, pool) in [("training", &self.train), ("gate", &self.gate)] {
            self.prompt(&pool[0]).map_err(|e| {
                anyhow::anyhow!("the {name} pool's first prompt cannot be encoded: {e}")
            })?;
        }
        Ok(())
    }

    /// The most tokens any prompt of either pool encodes to: what the training
    /// row must hold beside the completion.
    ///
    /// # Errors
    /// A prompt cannot be encoded.
    pub fn longest_prompt(&self) -> anyhow::Result<usize> {
        let mut longest = 0;
        for item in self.train.iter().chain(&self.gate) {
            longest = longest.max(self.prompt(item)?.len());
        }
        Ok(longest)
    }

    fn prompt(&self, item: &Item) -> anyhow::Result<Vec<u32>> {
        (self.encode)(
            respond::SYSTEM,
            &respond::student_prompt(&item.scenario, &item.evidence),
        )
    }
}

impl Environment for AdamsEnv {
    fn name(&self) -> &str {
        "adams-transfer"
    }

    fn tasks(&self, seed: u64) -> Vec<Task> {
        let (pool, index) = if seed >= GATE_SEED_BASE {
            (&self.gate, seed - GATE_SEED_BASE)
        } else {
            (&self.train, seed)
        };
        if pool.is_empty() {
            return Vec::new();
        }
        let round = index / pool.len() as u64;
        let item = &pool[(index % pool.len() as u64) as usize];
        let Ok(prompt) = self.prompt(item) else {
            return Vec::new();
        };
        let answer = serde_json::json!({
            "case": item.scenario.case,
            "situation": item.scenario.situation,
            "observations": item.scenario.observations,
            "request": item.scenario.request,
            "documents": item.documents.iter().map(|(id, body)| serde_json::json!({"id": id, "body": body})).collect::<Vec<_>>(),
        });
        vec![Task {
            id: format!("{}#{round}", item.scenario.id),
            prompt,
            answer,
        }]
    }
}

/// A document as the rules need it: an id and its text.
fn evidence_document(id: &str, body: &str) -> Document {
    use crate::document::{Authorship, Date, Period};
    Document {
        schema: crate::curate::SCHEMA,
        id: id.to_string(),
        source_id: String::new(),
        heading: String::new(),
        recipient: None,
        note: String::new(),
        date: Date {
            year: 1770,
            month: None,
            day: None,
        },
        period: Period::of(1770),
        authorship: Authorship::DraftInHand,
        authorship_confidence: 1.0,
        authorship_basis: String::new(),
        temporal_holdout: false,
        body: body.to_string(),
    }
}

/// The reward for `text` on a task whose `answer` is the JSON an item made:
/// the layout, the verdict, and every other rule, each a named part.
pub fn reward(text: &str, answer: &serde_json::Value) -> Reward {
    let Ok(case) = serde_json::from_value::<Case>(answer["case"].clone()) else {
        return Reward::default();
    };
    let strings = |key: &str| -> Vec<String> {
        answer[key]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let scenario = Scenario {
        id: String::new(),
        principle_id: String::new(),
        case,
        domain: String::new(),
        situation: answer["situation"].as_str().unwrap_or_default().to_string(),
        observations: strings("observations"),
        request: answer["request"].as_str().unwrap_or_default().to_string(),
        conditions_present: Vec::new(),
        conditions_missing: Vec::new(),
    };
    let documents: Vec<Document> = answer["documents"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|d| Some(evidence_document(d["id"].as_str()?, d["body"].as_str()?)))
                .collect()
        })
        .unwrap_or_default();
    let refs: Vec<&Document> = documents.iter().collect();

    let stated = Applicability::of(text);
    let has_block = crate::grounding::parse(text)
        .items
        .is_some_and(|items| !items.is_empty());
    let part = |earned: bool, worth: f32| if earned { worth } else { 0.0 };
    let parts: BTreeMap<String, f32> = BTreeMap::from([
        (
            "format".to_string(),
            part(stated.is_some() && has_block, FORMAT),
        ),
        (
            "applicability".to_string(),
            part(stated.is_some_and(|a| a.fits(case)), APPLICABILITY),
        ),
        (
            "all_rules".to_string(),
            part(respond::check(text, &scenario, &refs).is_ok(), ALL_RULES),
        ),
    ]);
    Reward {
        value: parts.values().sum(),
        parts,
    }
}

/// The verifier: decode the completion, apply the rules.
#[derive(Clone)]
pub struct AdamsVerifier {
    decode: Decode,
}

impl AdamsVerifier {
    pub fn new(decode: Decode) -> AdamsVerifier {
        AdamsVerifier { decode }
    }
}

impl Verifier for AdamsVerifier {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        reward(&(self.decode)(completion), &task.answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Period;
    use crate::testkit::doc;

    const WORDS_OF_HIS: &str = "Let the Committee write to every Town, that the Sense of the People may be known before the Assembly meets again.";

    fn item(n: usize, case: Case) -> Item {
        let id = format!("scenario-{n}");
        let found = Found {
            doc_id: "d1".into(),
            quote: "Let the Committee write to every Town".into(),
            period: Period::PreRevolution,
            recipient: None,
        };
        let d = doc("d1", 1770, "A B", WORDS_OF_HIS);
        Item {
            scenario: Scenario {
                id,
                principle_id: format!("P-{n}"),
                case,
                domain: "software".into(),
                situation: "Five regional teams want different migration dates for 800 engineers."
                    .into(),
                observations: vec![
                    "There are five regional teams.".into(),
                    "The platform group announced one date.".into(),
                ],
                request: "Advise the head of engineering.".into(),
                conditions_present: vec!["a grievance unanswered".into()],
                conditions_missing: if case.applies() {
                    Vec::new()
                } else {
                    vec!["whether the teams agree".into()]
                },
            },
            evidence: vec![found],
            documents: vec![(d.id.clone(), d.body)],
        }
    }

    fn items(n: usize) -> Vec<Item> {
        (0..n).map(|i| item(i, Case::Clear)).collect()
    }

    fn encode() -> Encode {
        Arc::new(|system: &str, user: &str| {
            Ok(format!("{system}|{user}").bytes().map(u32::from).collect())
        })
    }

    fn text_of(ids: &[u32]) -> String {
        ids.iter().map(|&b| b as u8 as char).collect()
    }

    #[test]
    fn the_pools_are_disjoint_whatever_the_order_and_a_share_goes_to_the_gate() {
        let env = AdamsEnv::new(items(100), 5, encode());
        let mut reversed = items(100);
        reversed.reverse();
        let other = AdamsEnv::new(reversed, 5, encode());
        let (train, gate) = env.pools();
        assert_eq!(train + gate, 100);
        assert!((10..=35).contains(&gate), "{gate} of 100 at one in five");
        assert_eq!(env.pools(), other.pools());
        let ids = |seeds: std::ops::Range<u64>, base: u64| -> Vec<String> {
            seeds
                .flat_map(|s| env.tasks(base + s))
                .map(|t| t.id)
                .collect()
        };
        let trained = ids(0..200, 0);
        let gated = ids(0..200, GATE_SEED_BASE);
        assert!(
            trained.iter().all(|t| !gated.contains(t)),
            "the gate never scores what was trained"
        );
    }

    #[test]
    fn a_task_is_deterministic_in_its_seed_and_carries_the_prompt_and_what_a_check_needs() {
        let env = AdamsEnv::new(items(20), 5, encode());
        let a = env.tasks(7);
        assert_eq!(a, env.tasks(7));
        assert_eq!(a.len(), 1);
        let text = text_of(&a[0].prompt);
        assert!(
            text.contains("You are Samuel Adams")
                && text.contains("Five regional teams")
                && text.contains("[d1]"),
            "{text}"
        );
        assert!(
            !text.to_lowercase().contains("clear"),
            "the case is never in the prompt"
        );
        let answer = &a[0].answer;
        assert_eq!(answer["case"], "clear");
        assert_eq!(answer["observations"].as_array().unwrap().len(), 2);
        assert_eq!(answer["documents"][0]["id"], "d1");
        assert_ne!(env.tasks(7)[0].id, env.tasks(8)[0].id);
    }

    #[test]
    fn the_longest_prompt_is_found_across_both_pools_and_an_unencodable_prompt_is_an_error() {
        let env = AdamsEnv::new(items(30), 5, encode());
        let longest = env.longest_prompt().unwrap();
        let some = env.tasks(0)[0].prompt.len();
        assert!(longest >= some && longest > 100, "{longest} vs {some}");
        let failing: Encode = Arc::new(|_, _| anyhow::bail!("no tokenizer"));
        assert!(AdamsEnv::new(items(30), 5, failing)
            .longest_prompt()
            .is_err());
    }

    #[test]
    fn a_seed_past_the_pool_wraps_instead_of_failing() {
        let env = AdamsEnv::new(items(10), 5, encode());
        assert_eq!(
            env.tasks(0)[0].id.split('#').next(),
            env.tasks(env.pools().0 as u64)[0].id.split('#').next()
        );
    }

    const FIT: &str = "Applicability: APPLIES\n\nI would have each team put its case in writing and gather them for the platform group.\n\nGrounding:\n- SOURCE_DIRECT: \"Let the Committee write to every Town\" [d1]\n- MODERN_OBSERVATION: There are five regional teams\n- PERSONA_TRANSFER: the committee method carried to the teams";

    fn answer_for(case: Case) -> serde_json::Value {
        let it = item(1, case);
        serde_json::json!({
            "case": case,
            "situation": it.scenario.situation,
            "observations": it.scenario.observations,
            "request": it.scenario.request,
            "documents": it.documents.iter().map(|(id, body)| serde_json::json!({"id": id, "body": body})).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn an_answer_that_keeps_every_rule_earns_the_whole_reward_in_named_parts() {
        let r = reward(FIT, &answer_for(Case::Clear));
        assert!((r.value - 1.0).abs() < 1e-6, "{r:?}");
        for part in ["format", "applicability", "all_rules"] {
            assert!(r.parts[part] > 0.0, "{part} in {:?}", r.parts);
        }
        assert!(
            (r.parts.values().sum::<f32>() - r.value).abs() < 1e-6,
            "the parts account for the value"
        );
    }

    #[test]
    fn an_answer_with_no_grounding_block_earns_nothing_for_layout_or_rules() {
        let r = reward(
            "Applicability: APPLIES\n\nJust advice.",
            &answer_for(Case::Clear),
        );
        assert_eq!(r.parts["format"], 0.0);
        assert_eq!(r.parts["all_rules"], 0.0);
        assert!(
            r.parts["applicability"] > 0.0,
            "it did say the right thing about applicability"
        );
    }

    #[test]
    fn the_wrong_verdict_earns_the_layout_but_neither_the_verdict_nor_the_rules() {
        let wrong = FIT.replace("APPLIES", "DOES_NOT_APPLY");
        let r = reward(&wrong, &answer_for(Case::Clear));
        assert!(r.parts["format"] > 0.0);
        assert_eq!((r.parts["applicability"], r.parts["all_rules"]), (0.0, 0.0));
        assert!((r.value - FORMAT).abs() < 1e-6);
    }

    #[test]
    fn a_fabricated_quotation_keeps_the_layout_and_the_verdict_but_loses_the_rules() {
        let bad = FIT.replace(
            "Let the Committee write to every Town",
            "Liberty is the first gift of nature to every citizen",
        );
        let r = reward(&bad, &answer_for(Case::Clear));
        assert!((r.value - (FORMAT + APPLICABILITY)).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn a_non_fit_that_says_it_needs_information_and_asks_keeps_every_rule() {
        let text = "Applicability: NEEDS_INFORMATION\n\nI cannot say yet. Do the five teams agree?\n\nGrounding:\n- SOURCE_INFERRED: my papers show I gathered opinion first\n- MODERN_OBSERVATION: There are five regional teams";
        assert!((reward(text, &answer_for(Case::MissingPrecondition)).value - 1.0).abs() < 1e-6);
        assert!(
            reward(text, &answer_for(Case::Clear)).value < 1.0,
            "the same answer to a fit is wrong"
        );
    }

    #[test]
    fn the_verifier_decodes_the_completion_and_applies_the_same_rules() {
        let v = AdamsVerifier::new(Arc::new(|ids: &[u32]| text_of(ids)));
        let ids: Vec<u32> = FIT.bytes().map(u32::from).collect();
        let task = Task {
            id: "t".into(),
            prompt: vec![1],
            answer: answer_for(Case::Clear),
        };
        assert!((v.verify(&task, &[], &ids).value - 1.0).abs() < 1e-6);
        assert_eq!(
            v.verify(&task, &[], &[]).value,
            0.0,
            "an empty completion earns nothing"
        );
    }
}
