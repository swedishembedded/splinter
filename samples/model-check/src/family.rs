// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A family: one TLA+ specification, its Lua reference implementation, and the
//! mutations of that implementation that are candidate bug-fix tasks.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use serde::Deserialize;

use crate::mutation::Mutation;

/// `spec.toml`: where the family's files are and what TLC must say about the
/// specification.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Spec {
    /// The family's name (the directory name).
    pub name: String,
    /// The Lua reference implementation, relative to the family directory.
    pub module: String,
    /// The TLA+ module, relative to the family directory.
    pub tla: String,
    /// TLC's configuration of the module, relative to the family directory.
    pub cfg: String,
    /// Distinct states TLC finds in the specification with that configuration.
    /// The reference implementation must reach exactly as many.
    pub distinct_states: u64,
    /// Invariant name -> the symptom, in words, a task statement may use.
    #[serde(default)]
    pub symptoms: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct MutationFile {
    #[serde(default)]
    mutation: Vec<Mutation>,
}

/// A loaded family directory.
#[derive(Debug, Clone)]
pub struct Family {
    /// The family directory.
    pub dir: PathBuf,
    /// Its `spec.toml`.
    pub spec: Spec,
    /// Its `mutations.toml`, in file order.
    pub mutations: Vec<Mutation>,
}

impl Family {
    /// Loads and validates the family in `dir`.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let spec_path = dir.join("spec.toml");
        let spec: Spec = toml::from_str(
            &std::fs::read_to_string(&spec_path)
                .with_context(|| format!("reading {}", spec_path.display()))?,
        )
        .with_context(|| format!("parsing {}", spec_path.display()))?;
        let mutations_path = dir.join("mutations.toml");
        let mutations = toml::from_str::<MutationFile>(
            &std::fs::read_to_string(&mutations_path)
                .with_context(|| format!("reading {}", mutations_path.display()))?,
        )
        .with_context(|| format!("parsing {}", mutations_path.display()))?
        .mutation;
        let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if spec.name != dir_name {
            bail!(
                "{}: name {:?} is not the directory name",
                spec_path.display(),
                spec.name
            );
        }
        let mut ids: Vec<&str> = mutations.iter().map(|m| m.id.as_str()).collect();
        ids.sort_unstable();
        if let Some(pair) = ids.windows(2).find(|w| w[0] == w[1]) {
            bail!(
                "{}: duplicate mutation id {:?}",
                mutations_path.display(),
                pair[0]
            );
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            spec,
            mutations,
        })
    }

    /// Every family directly below `specs`, sorted by name.
    pub fn load_all(specs: &Path) -> anyhow::Result<Vec<Self>> {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(specs)
            .with_context(|| format!("listing {}", specs.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.join("spec.toml").is_file())
            .collect();
        dirs.sort();
        dirs.iter().map(|d| Self::load(d)).collect()
    }

    /// The reference implementation's path.
    pub fn module_path(&self) -> PathBuf {
        self.dir.join(&self.spec.module)
    }

    /// The TLA+ module's name, which is also the stem of its file.
    pub fn tla_module(&self) -> anyhow::Result<&str> {
        Path::new(&self.spec.tla)
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| {
                format!(
                    "{}: tla {:?} has no file stem",
                    self.spec.name, self.spec.tla
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, spec_name: &str, mutations: &str) -> PathBuf {
        let family = dir.join(spec_name);
        std::fs::create_dir_all(&family).unwrap();
        std::fs::write(
            family.join("spec.toml"),
            format!(
                "name = \"{spec_name}\"\nmodule = \"m.lua\"\ntla = \"M.tla\"\ncfg = \"M.cfg\"\ndistinct_states = 7\n[symptoms]\nBounded = \"too many\"\n"
            ),
        )
        .unwrap();
        std::fs::write(family.join("mutations.toml"), mutations).unwrap();
        family
    }

    #[test]
    fn loads_spec_and_mutations_in_file_order() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write(
            tmp.path(),
            "q",
            "[[mutation]]\nid = \"b\"\nfind = \"x\"\nreplace = \"y\"\nviolates = \"Bounded\"\n[[mutation]]\nid = \"a\"\nfind = \"1\"\nreplace = \"2\"\nviolates = \"trace\"\n",
        );
        let family = Family::load(&dir).unwrap();
        assert_eq!(family.spec.distinct_states, 7);
        assert_eq!(family.tla_module().unwrap(), "M");
        assert_eq!(family.spec.symptoms["Bounded"], "too many");
        let ids: Vec<&str> = family.mutations.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["b", "a"]);
    }

    #[test]
    fn refuses_a_name_that_is_not_the_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write(tmp.path(), "q", "");
        std::fs::rename(&dir, tmp.path().join("other")).unwrap();
        assert!(Family::load(&tmp.path().join("other")).is_err());
    }

    #[test]
    fn refuses_duplicate_mutation_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let one = "[[mutation]]\nid = \"a\"\nfind = \"x\"\nreplace = \"y\"\nviolates = \"trace\"\n";
        let dir = write(tmp.path(), "q", &format!("{one}{one}"));
        assert!(Family::load(&dir).is_err());
    }
}
