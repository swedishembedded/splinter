// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements declared, verifiable data sources for
// models trained on sensitive records: the file, its digest and the terms it
// came under travel together. If your team needs expertise in data-use
// governance for clinical models, you can procure our services by sending
// an email to info@swedishembedded.com.

//! A source file and its declaration.
//!
//! A record file never states its own terms. A [`Declaration`] beside it
//! names the dataset, the file, the file's digest, the terms the data came
//! under (one of the usage labels, e.g. `redistributable`, `research_only`,
//! `restricted_DUA`) and the outcomes it supplies. Reading one checks the
//! file against the digest, so what is imported is what was declared; the
//! terms travel with every record into the release, where they decide how
//! widely it may be handed on. Terms nobody declared are `unknown`, which
//! permits nothing.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::terms::{Terms, UsagePolicy};

/// The `format` of a declaration.
pub const FORMAT: &str = "splinter-health-source-v1";

/// A subgroup the release must not regress on, as a declaration states it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SubgroupDeclaration {
    /// The name its numbers travel under.
    pub name: String,
    /// The categorical variable.
    pub var: String,
    /// The level.
    pub level: String,
}

/// What is known of a source file besides its bytes.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Declaration {
    /// [`FORMAT`].
    pub format: String,
    /// The dataset's id.
    pub dataset: String,
    /// The record file, relative to the declaration.
    pub file: String,
    /// The digest of the file's bytes.
    pub blake3: String,
    /// Records in it.
    pub records: u64,
    /// The usage label of the terms the data came under.
    pub usage: String,
    /// The outcome codes the file supplies.
    pub supplies: Vec<String>,
    /// Subgroups the evaluation is also reported on.
    #[serde(default)]
    pub subgroups: Vec<SubgroupDeclaration>,
    /// How the file was made, in the maker's words.
    #[serde(default)]
    pub generator: serde_json::Value,
}

impl Declaration {
    /// The terms the usage label states, named for the dataset.
    pub fn terms(&self) -> Result<Terms> {
        let policy = UsagePolicy::parse(&self.usage).map_err(anyhow::Error::msg)?;
        Ok(policy.terms(&self.dataset))
    }

    /// The record file's path, beside the declaration at `declaration`.
    pub fn file_path(&self, declaration: &Path) -> PathBuf {
        declaration.with_file_name(&self.file)
    }
}

/// The digest of the file at `path`, as a declaration spells it.
pub fn digest_of(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Digest::of(&bytes).to_string())
}

/// Reads the declaration at `path` and checks the file it names against its
/// digest; refuses a file that is not the one declared and a usage label that
/// is not one.
pub fn read(path: &Path) -> Result<Declaration> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the declaration {}", path.display()))?;
    let declaration: Declaration = serde_json::from_str(&text)
        .with_context(|| format!("parsing the declaration {}", path.display()))?;
    anyhow::ensure!(
        declaration.format == FORMAT,
        "{}: format {:?} is not {FORMAT}",
        path.display(),
        declaration.format
    );
    declaration.terms()?;
    let file = declaration.file_path(path);
    let found = digest_of(&file)?;
    anyhow::ensure!(
        found == declaration.blake3,
        "{} has digest {found}, but {} declares {}: the file is not the one that was declared",
        file.display(),
        path.display(),
        declaration.blake3
    );
    Ok(declaration)
}

/// The terms of files under a directory of timelines, from a policy file.
///
/// The file maps a dataset (a file stem) or `*` to a usage label. A dataset
/// the file does not name, or no file at all, is `unknown`: nothing is
/// permitted until someone who holds the data's terms declares them.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Policy {
    /// Dataset (or `*`) to usage label.
    #[serde(default)]
    pub terms: std::collections::BTreeMap<String, String>,
    /// Subgroups the evaluation is also reported on.
    #[serde(default)]
    pub subgroups: Vec<SubgroupDeclaration>,
}

impl Policy {
    /// The policy in the file at `path`; the policy that declares nothing
    /// when there is no path.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        match path {
            None => Ok(Self::default()),
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("reading the policy file {}", path.display()))?;
                serde_json::from_str(&text)
                    .with_context(|| format!("parsing the policy file {}", path.display()))
            }
        }
    }

    /// The terms of `dataset`: its own entry, else `*`, else unknown.
    pub fn terms_of(&self, dataset: &str) -> Result<Terms> {
        let label = self
            .terms
            .get(dataset)
            .or_else(|| self.terms.get("*"))
            .map_or("unknown", String::as_str);
        let policy = UsagePolicy::parse(label)
            .map_err(|why| anyhow::anyhow!("policy for {dataset}: {why}"))?;
        Ok(policy.terms(dataset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(dir: &Path, usage: &str, body: &str) -> PathBuf {
        let file = dir.join("data.jsonl");
        std::fs::write(&file, body).unwrap();
        let declaration = Declaration {
            format: FORMAT.into(),
            dataset: "cohort".into(),
            file: "data.jsonl".into(),
            blake3: digest_of(&file).unwrap(),
            records: 1,
            usage: usage.into(),
            supplies: vec!["death:cvd".into()],
            subgroups: vec![],
            generator: serde_json::Value::Null,
        };
        let path = dir.join("data.source.json");
        std::fs::write(&path, serde_json::to_vec(&declaration).unwrap()).unwrap();
        path
    }

    #[test]
    fn a_file_that_is_not_the_one_declared_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = declared(dir.path(), "redistributable", "{}\n");
        assert!(read(&path).is_ok());
        std::fs::write(dir.path().join("data.jsonl"), "{\"changed\":1}\n").unwrap();
        let why = read(&path).unwrap_err().to_string();
        assert!(why.contains("not the one that was declared"), "{why}");
    }

    #[test]
    fn a_usage_label_states_the_terms_and_a_bad_one_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let open = read(&declared(dir.path(), "redistributable", "{}\n")).unwrap();
        assert!(open.terms().unwrap().permits_unrestricted_release().is_ok());
        let dua = read(&declared(dir.path(), "restricted_DUA", "{}\n")).unwrap();
        assert!(dua.terms().unwrap().permits_unrestricted_release().is_err());
        let bad = declared(dir.path(), "public", "{}\n");
        assert!(read(&bad)
            .unwrap_err()
            .to_string()
            .contains("not a usage policy"));
    }

    #[test]
    fn terms_nobody_declared_are_unknown_and_permit_nothing() {
        let policy = Policy::default();
        let terms = policy.terms_of("timelines").unwrap();
        assert!(terms
            .permits(splinter_sdk::vocabulary::terms::Use::Training)
            .is_err());
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("policy.json");
        std::fs::write(
            &file,
            r#"{"terms":{"timelines":"research_only","*":"unknown"}}"#,
        )
        .unwrap();
        let policy = Policy::load(Some(&file)).unwrap();
        assert!(policy
            .terms_of("timelines")
            .unwrap()
            .permits(splinter_sdk::vocabulary::terms::Use::Training)
            .is_ok());
        assert!(policy
            .terms_of("other")
            .unwrap()
            .permits(splinter_sdk::vocabulary::terms::Use::Training)
            .is_err());
    }
}
