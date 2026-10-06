// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled synthetic cohorts with known
// hazards, to prove a risk-model pipeline end to end before it touches real
// records. If your team needs expertise in validating clinical prediction
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `synth`: a small deterministic synthetic cohort as a source file.
//!
//! The people come from brain's synthetic population, whose hazards are known
//! (cardiovascular-like death depends on age and the first risk factor `x1`,
//! cancer-like death on age, `x2` and a prior diagnosis; the groups differ in
//! a third, unrelated outcome). The sample keeps the two competing causes,
//! renamed to the ontology's `death:cvd` and `death:cancer`, and drops the
//! third, which the ontology has no definition for. Households of two share a
//! group, and two survey cycles stand for sources. The result is written as a
//! record file with a declaration: its digest, its declared usage terms and
//! the outcomes it supplies.

use std::path::Path;

use anyhow::{Context, Result};
use splinter_sdk::model::timeline::synthetic;

use crate::ontology::SYNTHETIC_DATASET;
use crate::source::{digest_of, Declaration, SubgroupDeclaration, FORMAT};

/// Participants when none is asked for.
pub const DEFAULT_PARTICIPANTS: usize = 12_000;
/// The seed when none is asked for.
pub const DEFAULT_SEED: u64 = 7;
/// The file `synth` writes, and its declaration.
pub const FILE: &str = "synthetic.jsonl";
/// The declaration `synth` writes.
pub const DECLARATION: &str = "synthetic.source.json";

/// The unit the first risk factor is stated in: a standard deviation of the
/// cohort's own distribution.
pub const X1_UNIT: &str = "sd";

/// The generator's outcome codes as the ontology names them.
const RENAMED: [(&str, &str); 2] = [("death:a", "death:cvd"), ("death:b", "death:cancer")];

/// The record lines of `n` synthetic participants, deterministic in `seed`.
pub fn lines(n: usize, seed: u64) -> Result<Vec<String>> {
    let (subjects, _) = synthetic::population(n, seed);
    subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let mut line = serde_json::to_value(subject)?;
            line["subject_id"] = format!("synthetic-{i}").into();
            line["group_id"] = format!("household-{}", i / 2).into();
            line["source"] = ["cycle-a", "cycle-b"][(i / 2) % 2].into();
            line["interventions"] = serde_json::json!([]);
            // The first risk factor is a standardised score: its unit is stated,
            // so the model records it and refuses a measurement in another.
            for o in line["observations"].as_array_mut().into_iter().flatten() {
                if o["var"] == "x1" {
                    o["unit"] = X1_UNIT.into();
                }
            }
            let events: Vec<serde_json::Value> = subject
                .events
                .iter()
                .filter_map(|e| {
                    let code = RENAMED
                        .iter()
                        .find(|(from, _)| *from == e.code)
                        .map(|(_, to)| *to);
                    let code = code.or((e.code == "dx").then_some("dx"))?;
                    Some(serde_json::json!({"t": e.t, "code": code}))
                })
                .collect();
            line["events"] = events.into();
            Ok(serde_json::to_string(&line)?)
        })
        .collect()
}

/// Writes the cohort and its declaration into `dir` and returns the
/// declaration. `usage` is the label of the terms the data is declared to
/// have come under.
pub fn write(dir: &Path, n: usize, seed: u64, usage: &str) -> Result<Declaration> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let file = dir.join(FILE);
    let text = lines(n, seed)?.join("\n") + "\n";
    std::fs::write(&file, text).with_context(|| format!("writing {}", file.display()))?;
    let declaration = Declaration {
        format: FORMAT.into(),
        dataset: SYNTHETIC_DATASET.into(),
        file: FILE.into(),
        blake3: digest_of(&file)?,
        records: n as u64,
        usage: usage.into(),
        supplies: vec!["death:cvd".into(), "death:cancer".into()],
        subgroups: vec![SubgroupDeclaration {
            name: "group_b".into(),
            var: "group".into(),
            level: "b".into(),
        }],
        generator: serde_json::json!({
            "name": "brain timeline synthetic::population",
            "participants": n,
            "seed": seed,
            "mapping": {"death:a": "death:cvd", "death:b": "death:cancer", "onset": "dropped"},
        }),
    };
    declaration.terms()?;
    let path = dir.join(DECLARATION);
    std::fs::write(&path, serde_json::to_vec_pretty(&declaration)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(declaration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cohort_is_deterministic_and_carries_only_ontology_outcomes() {
        let a = lines(300, 5).unwrap();
        assert_eq!(a, lines(300, 5).unwrap());
        assert_ne!(a, lines(300, 6).unwrap());
        let mut causes = std::collections::BTreeSet::new();
        for line in &a {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            for e in v["events"].as_array().unwrap() {
                causes.insert(e["code"].as_str().unwrap().to_owned());
            }
        }
        assert!(
            causes
                .iter()
                .all(|c| ["death:cvd", "death:cancer", "dx"].contains(&c.as_str())),
            "{causes:?}"
        );
        assert!(causes.contains("death:cvd") && causes.contains("death:cancer"));
    }

    #[test]
    fn the_first_risk_factor_states_its_unit_and_the_others_state_none() {
        for line in lines(50, 5).unwrap() {
            let v: serde_json::Value = serde_json::from_str(&line).unwrap();
            for o in v["observations"].as_array().unwrap() {
                let unit = o.get("unit").and_then(|u| u.as_str());
                assert_eq!(unit, (o["var"] == "x1").then_some(X1_UNIT), "{o}");
            }
        }
    }

    #[test]
    fn the_declaration_carries_the_digest_and_the_declared_terms() {
        let dir = tempfile::tempdir().unwrap();
        let a = write(dir.path(), 200, 3, "research_only").unwrap();
        let read = crate::source::read(&dir.path().join(DECLARATION)).unwrap();
        assert_eq!(read, a);
        assert!(read
            .terms()
            .unwrap()
            .permits_unrestricted_release()
            .is_err());
        let again = tempfile::tempdir().unwrap();
        assert_eq!(
            write(again.path(), 200, 3, "research_only").unwrap().blake3,
            a.blake3,
            "same seed, same digest"
        );
        assert!(write(again.path(), 10, 3, "public").is_err());
    }
}
