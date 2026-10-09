// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what a learning system
// extracts from its users' conversations, for its clients. If your team needs
// expertise in checking extraction against known ground truth, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The subjects file: what each fact a recorded session was made to teach looks
//! like in a statement, as `claims gate --subjects` and `absorb --subjects`
//! read it (JSON Lines of `{"id", "terms": [...]}`; a statement carries the
//! fact when it holds every term). The ids are the ones the recorded
//! trajectories name in their root `extra`.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context as _;
use serde::Serialize;

use crate::facts::Manifest;
use crate::record::subject_id;
use crate::session::Recorded;

/// The file's name under the output directory.
pub const FILE: &str = "subjects.jsonl";

/// The index of recorded sessions, relative to the output directory.
pub const INDEX: &str = "sessions/index.jsonl";

#[derive(Serialize)]
struct Subject {
    id: String,
    terms: Vec<String>,
}

/// Writes the subjects of every recorded fact session under `out`: a fact's
/// keys, or for the canary the keys of its false statement. Returns how many.
///
/// # Errors
/// The manifest or the index cannot be read, a session names an unknown fact,
/// or the file cannot be written.
pub fn write(out: &Path) -> anyhow::Result<usize> {
    let manifest = Manifest::read(out)?;
    let index = std::fs::read_to_string(out.join(INDEX)).unwrap_or_default();
    let mut subjects: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (n, line) in index
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let recorded: Recorded =
            serde_json::from_str(line).with_context(|| format!("{INDEX} line {}", n + 1))?;
        if recorded.kind != "fact" {
            continue;
        }
        let id = recorded.fact.as_deref().unwrap_or_default();
        let fact = manifest
            .facts
            .iter()
            .find(|f| f.id == id)
            .with_context(|| format!("{INDEX} line {}: no fact {id}", n + 1))?;
        let keys = match (&fact.canary, recorded.canary) {
            (Some(canary), true) => &canary.keys,
            _ => &fact.keys,
        };
        subjects.insert(
            subject_id(id, recorded.canary),
            keys.iter().map(|k| k.text.clone()).collect(),
        );
    }
    let mut text = String::new();
    for (id, terms) in &subjects {
        text.push_str(&serde_json::to_string(&Subject {
            id: id.clone(),
            terms: terms.clone(),
        })?);
        text.push('\n');
    }
    std::fs::write(out.join(FILE), text)?;
    Ok(subjects.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{admit, Canary, Evidence, Origin, Proposal};
    use crate::keys::{Key, KeyKind};
    use crate::roles::Quotas;
    use crate::session::Style;

    fn key(text: &str) -> Key {
        Key {
            text: text.into(),
            kind: KeyKind::Number,
        }
    }

    #[test]
    fn the_subjects_file_describes_each_taught_fact_by_its_keys_and_the_canary_by_its_false_ones() {
        let dir = tempfile::tempdir().expect("dir");
        let statement = "The press was bought in 1785 for forty dollars.";
        let mut fact = admit(Proposal {
            seed: 1,
            quotas: &Quotas::PROTOCOL,
            origin: Origin::Letters,
            family: "f.txt",
            question: "When and for how much was the press bought?",
            statement,
            keys: vec![key("1785"), key("forty")],
            answer_type: "year",
            entities: vec![],
            evidence: Evidence {
                quote: statement.into(),
                source: "f.txt".into(),
            },
        })
        .expect("a fact");
        fact.canary = Some(Canary {
            statement: "The press was bought in 1792 for forty dollars.".into(),
            keys: vec![key("1792"), key("forty")],
        });
        let id = fact.id.clone();
        let manifest = Manifest {
            schema: crate::facts::SCHEMA.into(),
            seed: 1,
            quotas: Quotas::PROTOCOL,
            materials: "m".into(),
            generator: "g".into(),
            persona: "p".into(),
            facts: vec![fact],
            refusals: vec![],
            unknowns: vec![],
            judge: None,
            probes: None,
        };
        manifest.write(dir.path()).expect("manifest");
        let recorded = |kind: &str, canary: bool| Recorded {
            file: "x".into(),
            kind: kind.into(),
            fact: Some(id.clone()),
            topic: None,
            user_turns: 2,
            first_answer_holds_keys: None,
            correction_names_missing_keys: None,
            user_states_every_key: None,
            style: Style::Plain,
            canary,
            discarded_before: 0,
        };
        std::fs::create_dir_all(dir.path().join("sessions")).expect("dir");
        let lines: Vec<String> = [
            recorded("fact", false),
            recorded("fact", true),
            recorded("sham", false),
        ]
        .iter()
        .map(|r| serde_json::to_string(r).expect("json"))
        .collect();
        std::fs::write(dir.path().join(INDEX), lines.join("\n") + "\n").expect("index");
        assert_eq!(
            write(dir.path()).expect("written"),
            2,
            "a sham session teaches nothing"
        );
        let text = std::fs::read_to_string(dir.path().join(FILE)).expect("file");
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("json"))
            .collect();
        assert_eq!(rows[0]["id"], id.as_str());
        assert_eq!(rows[0]["terms"], serde_json::json!(["1785", "forty"]));
        assert_eq!(rows[1]["id"], format!("{id}#canary"));
        assert_eq!(rows[1]["terms"], serde_json::json!(["1792", "forty"]));
    }
}
