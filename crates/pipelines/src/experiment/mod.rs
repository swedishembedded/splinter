// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continual-learning experiments whose exam is
// sealed before the first run, for its clients. If your team needs expertise
// in designing generation, retention and final-exam splits, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The experiment plan: which families each generation trains on, which
//! earlier families it is held to (retention), and which families are the
//! sealed final exam, as data a run reads and a check validates.
//!
//! A plan names families by their id in a catalog of task families, and the
//! catalog says what each is (language, domain, and the git tree id of its
//! files). A plan is valid when
//!
//! * every family it names is in the catalog and has a tree id;
//! * a family is trained in at most one generation, and every generation
//!   trains at least one;
//! * no sealed family is in any generation's training or retention list, and
//!   at least one family is sealed;
//! * a generation's retention families were trained by an earlier generation;
//! * no two families that play a part (trained or sealed) share a domain, so
//!   the final exam and the earlier generations ask about different things.
//!
//! At run time [`ExperimentPlan::admit`] decides whether a repository, known by
//! the tree id of its snapshot, may be read by a training stage of a given
//! generation: never a sealed family, never a family of another generation, and
//! never one the plan does not name. It is the tree id and not the name the
//! run was given that decides, so renaming a sealed family does not open it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_orchestrator::error::OrchestratorError;

/// One family a plan names, as the catalog describes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyRef {
    /// Its id in the catalog.
    pub id: String,
    /// The language it is written in.
    pub language: String,
    /// What it is about.
    pub domain: String,
    /// The git tree id of its files: what a snapshot of it records.
    pub tree: String,
    /// The digest of the workspace snapshot the catalog states for it, when
    /// it states one: a second identity a sealed family is matched by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
}

/// One generation of a plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Generation {
    /// The families this generation trains on.
    pub train: Vec<String>,
    /// The earlier families whose reserved exam files it is held to.
    pub retention: Vec<String>,
}

/// A validated plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentPlan {
    name: String,
    families: Vec<FamilyRef>,
    generations: Vec<Generation>,
    sealed: Vec<String>,
}

/// Why a plan is not valid.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Violation {
    /// A family the catalog does not have.
    #[error("family {family:?} is not in the catalog")]
    Unknown {
        /// The name.
        family: String,
    },
    /// A family the catalog gives no tree id, so a run could not be
    /// recognised as reading it.
    #[error("family {family:?} has no tree id in the catalog")]
    NoTree {
        /// The family.
        family: String,
    },
    /// A generation that trains nothing.
    #[error("generation {generation} trains no family")]
    EmptyGeneration {
        /// Its number, from 1.
        generation: usize,
    },
    /// A plan with no final exam.
    #[error("no family is sealed: there is no final exam")]
    NothingSealed,
    /// A family trained more than once.
    #[error("family {family:?} is trained more than once")]
    Repeated {
        /// The family.
        family: String,
    },
    /// A sealed family in a generation's training or retention list.
    #[error("sealed family {family:?} is scheduled in generation {generation}")]
    SealedScheduled {
        /// The family.
        family: String,
        /// The generation, from 1.
        generation: usize,
    },
    /// A retention family no earlier generation trained.
    #[error("generation {generation} is held to {family:?}, which no earlier generation trained")]
    RetentionNotYetTrained {
        /// The family.
        family: String,
        /// The generation, from 1.
        generation: usize,
    },
    /// Two families with a part to play in one domain.
    #[error("families {families:?} share the domain {domain:?}")]
    SharedDomain {
        /// The domain.
        domain: String,
        /// The families, in name order.
        families: Vec<String>,
    },
}

/// Why a plan could not be read.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// The plan or the catalog is not the TOML it should be.
    #[error("{what}: {reason}")]
    Parse {
        /// Which file.
        what: &'static str,
        /// What the parser said.
        reason: String,
    },
    /// The plan is not valid.
    #[error("the plan is not valid: {}", .0.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))]
    Invalid(Vec<Violation>),
}

/// Why a run was not admitted.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct NotAdmitted(String);

#[derive(Deserialize)]
struct PlanFile {
    plan: PlanHead,
    generation: Vec<Generation>,
    sealed: SealedFile,
}

#[derive(Deserialize)]
struct PlanHead {
    name: String,
}

#[derive(Deserialize)]
struct SealedFile {
    families: Vec<String>,
}

#[derive(Deserialize)]
struct CatalogFile {
    #[serde(default)]
    family: Vec<CatalogEntry>,
}

#[derive(Deserialize)]
struct CatalogEntry {
    id: String,
    language: String,
    domain: String,
    tree: Option<String>,
    #[serde(default)]
    environment: Option<String>,
}

fn parse_toml<T: serde::de::DeserializeOwned>(
    what: &'static str,
    text: &str,
) -> Result<T, PlanError> {
    toml::from_str(text).map_err(|e| PlanError::Parse {
        what,
        reason: e.to_string(),
    })
}

impl ExperimentPlan {
    /// The plan in `plan_text` checked against `catalog_text`, every
    /// violation reported together.
    pub fn parse(plan_text: &str, catalog_text: &str) -> Result<Self, PlanError> {
        let file: PlanFile = parse_toml("the plan", plan_text)?;
        let catalog: CatalogFile = parse_toml("the catalog", catalog_text)?;
        let entries: BTreeMap<&str, &CatalogEntry> =
            catalog.family.iter().map(|e| (e.id.as_str(), e)).collect();
        let mut found: Vec<Violation> = Vec::new();

        let named = file
            .generation
            .iter()
            .flat_map(|g| g.train.iter().chain(&g.retention))
            .chain(&file.sealed.families);
        let mut families: Vec<FamilyRef> = Vec::new();
        for id in named {
            if families.iter().any(|f| &f.id == id) {
                continue;
            }
            match entries.get(id.as_str()) {
                None => found.push(Violation::Unknown { family: id.clone() }),
                Some(entry) => match &entry.tree {
                    None => found.push(Violation::NoTree { family: id.clone() }),
                    Some(tree) => families.push(FamilyRef {
                        id: id.clone(),
                        language: entry.language.clone(),
                        domain: entry.domain.clone(),
                        tree: tree.clone(),
                        environment: entry.environment.clone(),
                    }),
                },
            }
        }

        if file.sealed.families.is_empty() {
            found.push(Violation::NothingSealed);
        }
        let sealed = |id: &String| file.sealed.families.contains(id);
        let mut trained: Vec<&String> = Vec::new();
        for (index, generation) in file.generation.iter().enumerate() {
            let number = index + 1;
            if generation.train.is_empty() {
                found.push(Violation::EmptyGeneration { generation: number });
            }
            for family in &generation.train {
                if trained.contains(&family) {
                    found.push(Violation::Repeated {
                        family: family.clone(),
                    });
                }
            }
            for family in &generation.retention {
                if !sealed(family) && !trained.contains(&family) {
                    found.push(Violation::RetentionNotYetTrained {
                        family: family.clone(),
                        generation: number,
                    });
                }
            }
            let mut scheduled: Vec<&String> = generation
                .train
                .iter()
                .chain(&generation.retention)
                .filter(|f| sealed(f))
                .collect();
            scheduled.sort();
            scheduled.dedup();
            found.extend(
                scheduled
                    .into_iter()
                    .map(|family| Violation::SealedScheduled {
                        family: family.clone(),
                        generation: number,
                    }),
            );
            trained.extend(&generation.train);
        }

        let mut by_domain: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for family in &families {
            let plays = sealed(&family.id) || trained.contains(&&family.id);
            if plays {
                by_domain
                    .entry(&family.domain)
                    .or_default()
                    .push(family.id.clone());
            }
        }
        for (domain, mut ids) in by_domain {
            ids.sort();
            if ids.len() > 1 {
                found.push(Violation::SharedDomain {
                    domain: domain.to_string(),
                    families: ids,
                });
            }
        }

        if !found.is_empty() {
            return Err(PlanError::Invalid(found));
        }
        Ok(Self {
            name: file.plan.name,
            families,
            generations: file.generation,
            sealed: file.sealed.families,
        })
    }

    /// The plan's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// How many generations it has.
    #[must_use]
    pub fn generations(&self) -> usize {
        self.generations.len()
    }

    /// The generations, in order.
    #[must_use]
    pub fn schedule(&self) -> &[Generation] {
        &self.generations
    }

    /// The sealed final-exam families.
    #[must_use]
    pub fn sealed(&self) -> &[String] {
        &self.sealed
    }

    /// What the catalog says of family `id`, when the plan names it.
    #[must_use]
    pub fn family(&self, id: &str) -> Option<&FamilyRef> {
        self.families.iter().find(|f| f.id == id)
    }

    /// The plan in the file `plan`, checked against the catalog `catalog`
    /// (the `catalog.toml` beside the plan when `None`); a plan that is not
    /// valid is refused with every violation.
    pub fn load(
        plan: &std::path::Path,
        catalog: Option<&std::path::Path>,
    ) -> Result<Self, OrchestratorError> {
        let beside = plan.with_file_name("catalog.toml");
        let catalog = catalog.unwrap_or(&beside);
        let read = |path: &std::path::Path| {
            std::fs::read_to_string(path)
                .map_err(|e| OrchestratorError::Refused(format!("{}: {e}", path.display())))
        };
        Self::parse(&read(plan)?, &read(catalog)?)
            .map_err(|e| OrchestratorError::Refused(e.to_string()))
    }

    /// The family whose files have tree id `tree`.
    fn family_with_tree(&self, tree: &str) -> Option<&FamilyRef> {
        self.families.iter().find(|f| f.tree == tree)
    }

    /// The id of the family a training stage of `generation` (from 1) may read
    /// when the repository's snapshot has tree id `tree`; refused when the
    /// plan does not allow it.
    pub fn admit(&self, generation: usize, tree: &str) -> Result<&str, NotAdmitted> {
        let refuse = |why: String| Err(NotAdmitted(why));
        if generation == 0 {
            return refuse("generations count from 1".into());
        }
        let Some(own) = self.generations.get(generation - 1) else {
            return refuse(format!(
                "the plan has {} generation(s), not {generation}",
                self.generations.len()
            ));
        };
        let Some(family) = self.family_with_tree(tree) else {
            return refuse(format!(
                "no family of plan {} has tree {tree}: its repository is not one the plan names",
                self.name
            ));
        };
        if self.sealed.contains(&family.id) {
            return refuse(format!(
                "{} is sealed: it is the final exam, and no stage of any generation may read it",
                family.id
            ));
        }
        if !own.train.contains(&family.id) {
            let home = self
                .generations
                .iter()
                .position(|g| g.train.contains(&family.id))
                .map_or_else(String::new, |i| format!("generation {}", i + 1));
            return refuse(format!(
                "{} is a family of {home}, and generation {generation} trains {:?}",
                family.id, own.train
            ));
        }
        Ok(&family.id)
    }
}

#[cfg(test)]
mod tests;
