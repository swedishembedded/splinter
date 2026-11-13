// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// self-generated verifiable task pools for policy-gradient training you can
// procure our services by sending an email to info@swedishembedded.com.

//! The task family of RSI2: the seeded generator's programs (source b) beside
//! the programs the current policy proposed from the Lua 5.1 manual (source
//! a), all put and graded under the neutral protocol.
//!
//! The proposals are kept in one append-only file, so a campaign that is
//! continued has the pool it had, and each proposal records which generation's
//! policy wrote it. The pool grows between arms; the dev and retention sets
//! are fixed when the campaign opens, so every arm is judged on the same
//! tasks.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_sdk::campaign::brain::{Graded, TaskFamily};
use splinter_sdk::campaign::CampaignError;
use splinter_sdk::vocabulary::digest::Digest;

use crate::{neutral, LuaPredict};

/// Which part of the campaign a proposal belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// Trained on.
    Train,
    /// Never trained on: the dev gate is decided on it.
    Dev,
}

/// One program the policy proposed, labelled by the interpreter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposal {
    /// The task id, from the digest of the program.
    pub id: String,
    /// The generation whose policy proposed it (1 is the base).
    pub generation: u32,
    /// Where it is used.
    pub part: Part,
    /// What the policy is asked: the question and the fenced program.
    pub request: String,
    /// The program.
    pub program: String,
    /// What the interpreter printed.
    pub expected: String,
}

/// The id of a proposed program.
#[must_use]
pub fn proposal_id(program: &str) -> String {
    let digest = Digest::of(program.trim().as_bytes()).to_string();
    let hex = digest.rsplit(':').next().unwrap_or(&digest);
    format!("prop-{}", &hex[..16])
}

/// A task of a campaign as handed to the examiner's overlap audit: the
/// program, its reference and its digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TaskRecord {
    /// The task id.
    pub id: String,
    /// `train`, `dev` or `retention`.
    pub role: &'static str,
    /// `generator` or the generation of the policy that proposed it.
    pub origin: String,
    /// The program.
    pub program: String,
    /// What the interpreter printed.
    pub reference: String,
    /// The digest of the program.
    pub digest: String,
}

/// The family: see the module documentation.
pub struct Rsi2Family {
    generator: LuaPredict,
    proposals: Mutex<BTreeMap<String, Proposal>>,
    file: PathBuf,
}

impl Rsi2Family {
    /// The family over `generator` and the proposals kept in `file` (created
    /// when absent).
    ///
    /// # Errors
    /// The file cannot be read or holds a line that is not a proposal.
    pub fn open(generator: LuaPredict, file: &Path) -> Result<Self, CampaignError> {
        let config = |what: String| CampaignError::Config(what);
        let mut proposals = BTreeMap::new();
        if file.exists() {
            let text = std::fs::read_to_string(file)
                .map_err(|e| config(format!("reading {}: {e}", file.display())))?;
            for (n, line) in text
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
            {
                let proposal: Proposal = serde_json::from_str(line)
                    .map_err(|e| config(format!("{} line {}: {e}", file.display(), n + 1)))?;
                proposals.insert(proposal.id.clone(), proposal);
            }
        }
        Ok(Self {
            generator,
            proposals: Mutex::new(proposals),
            file: file.to_path_buf(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, Proposal>> {
        self.proposals
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Whether any proposal was made by the policy of `generation`.
    #[must_use]
    pub fn has_generation(&self, generation: u32) -> bool {
        self.lock().values().any(|p| p.generation == generation)
    }

    /// Every proposal, by id.
    #[must_use]
    pub fn proposals(&self) -> Vec<Proposal> {
        self.lock().values().cloned().collect()
    }

    /// Whether `program` is already a task of the family, proposed or
    /// generated.
    #[must_use]
    pub fn holds(&self, program: &str) -> bool {
        self.lock().contains_key(&proposal_id(program))
            || self
                .generator
                .sources()
                .any(|source| source.trim() == program.trim())
    }

    /// Adds `proposals` to the pool (those not already held) and appends them
    /// to the file; returns how many were new.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn add(&self, proposals: Vec<Proposal>) -> Result<usize, CampaignError> {
        let mut held = self.lock();
        let mut fresh: Vec<Proposal> = Vec::new();
        for proposal in proposals {
            if !held.contains_key(&proposal.id) && !fresh.iter().any(|p| p.id == proposal.id) {
                fresh.push(proposal);
            }
        }
        if fresh.is_empty() {
            return Ok(0);
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file)
            .map_err(|e| CampaignError::Config(format!("opening {}: {e}", self.file.display())))?;
        for proposal in &fresh {
            let line = serde_json::to_string(proposal)
                .map_err(|e| CampaignError::Config(e.to_string()))?;
            writeln!(file, "{line}").map_err(|e| {
                CampaignError::Config(format!("writing {}: {e}", self.file.display()))
            })?;
        }
        file.sync_all()
            .map_err(|e| CampaignError::Config(format!("syncing {}: {e}", self.file.display())))?;
        let count = fresh.len();
        held.extend(fresh.into_iter().map(|p| (p.id.clone(), p)));
        Ok(count)
    }

    /// Every task of the family with its role, for the examiner's overlap
    /// audit.
    #[must_use]
    pub fn records(&self) -> Vec<TaskRecord> {
        let record = |id: &str, role, origin: String, program: &str, reference: &str| TaskRecord {
            id: id.to_string(),
            role,
            origin,
            program: program.to_string(),
            reference: reference.to_string(),
            digest: Digest::sha256_of(program.trim().as_bytes()).to_string(),
        };
        let mut records = Vec::new();
        for (ids, role) in [
            (self.generator.pool(), "train"),
            (self.generator.dev(), "dev"),
            (self.generator.retention(), "retention"),
        ] {
            for id in ids {
                if let Some(p) = self.generator.program(&id) {
                    records.push(record(
                        &id,
                        role,
                        "generator".into(),
                        &p.source,
                        &p.expected,
                    ));
                }
            }
        }
        for p in self.lock().values() {
            let role = match p.part {
                Part::Train => "train",
                Part::Dev => "dev",
            };
            records.push(record(
                &p.id,
                role,
                format!("policy-of-generation-{}", p.generation),
                &p.program,
                &p.expected,
            ));
        }
        records
    }

    fn ids(&self, part: Part, generated: Vec<String>) -> Vec<String> {
        let mut ids = generated;
        ids.extend(
            self.lock()
                .values()
                .filter(|p| p.part == part)
                .map(|p| p.id.clone()),
        );
        ids
    }
}

impl TaskFamily for Rsi2Family {
    fn describe(&self) -> Value {
        // The proposals are not part of it: the pool grows between arms and a
        // campaign that is continued is the same campaign.
        json!({"family": "rsi2", "protocol": "neutral", "generator": self.generator.describe()})
    }

    fn pool(&self) -> Vec<String> {
        self.ids(Part::Train, self.generator.pool())
    }

    fn dev(&self) -> Vec<String> {
        self.ids(Part::Dev, self.generator.dev())
    }

    fn retention(&self) -> Vec<String> {
        self.generator.retention()
    }

    fn prompt(&self, task: &str) -> Result<Vec<(String, String)>, CampaignError> {
        // The lock is released before a generated task is looked up: `known`
        // takes it again.
        let proposed = self.lock().get(task).map(|p| p.request.clone());
        let text = match proposed {
            Some(request) => request,
            None => neutral::request(&self.known(task)?.0),
        };
        Ok(vec![("user".into(), text)])
    }

    fn grade(&self, task: &str, completion: &str) -> Result<Graded, CampaignError> {
        let expected = self.known(task)?.1;
        let passed = neutral::answers(completion, &expected);
        Ok(Graded {
            reward: f64::from(u8::from(passed)),
            passed,
            feedback: (!passed)
                .then(|| "That is not what the program prints. Try again.".to_string()),
            privileged: Some(format!("The program prints exactly: {expected}")),
        })
    }
}

impl Rsi2Family {
    /// The program and what it prints, of a proposed or a generated task.
    fn known(&self, task: &str) -> Result<(String, String), CampaignError> {
        if let Some(p) = self.lock().get(task) {
            return Ok((p.program.clone(), p.expected.clone()));
        }
        self.generator
            .program(task)
            .map(|p| (p.source.clone(), p.expected.clone()))
            .ok_or_else(|| CampaignError::Config(format!("{task:?} is not a task of the family")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{default_interpreter, Sizes};

    fn family(dir: &Path) -> Rsi2Family {
        let sizes = Sizes {
            pool: 6,
            dev: 6,
            retention: 4,
        };
        // seed: the sets only have to be disjoint, whatever the draw.
        let generator = LuaPredict::generate(&default_interpreter(), dir, sizes, 7).unwrap();
        Rsi2Family::open(generator, &dir.join("proposals.jsonl")).unwrap()
    }

    fn proposal(program: &str, generation: u32, part: Part) -> Proposal {
        Proposal {
            id: proposal_id(program),
            generation,
            part,
            request: format!("What does this print?\n\n```lua\n{program}\n```"),
            program: program.into(),
            expected: "3".into(),
        }
    }

    #[test]
    fn proposals_join_the_pool_and_the_dev_set_by_their_part_and_are_graded_by_the_last_fence() {
        let dir = tempfile::tempdir().unwrap();
        let family = family(dir.path());
        let before = (family.pool().len(), family.dev().len());
        let n = family
            .add(vec![
                proposal("print(1+2)", 1, Part::Train),
                proposal("print(2+1)", 1, Part::Dev),
                proposal("print(1+2)", 2, Part::Train),
            ])
            .unwrap();
        assert_eq!(
            n, 2,
            "a program already held, or repeated in the batch, is added once"
        );
        assert_eq!(family.pool().len(), before.0 + 1);
        assert_eq!(family.dev().len(), before.1 + 1);
        let id = proposal_id("print(1+2)");
        assert!(family.prompt(&id).unwrap()[0].1.contains("print(1+2)"));
        assert!(family.grade(&id, "so\n```\n3\n```").unwrap().passed);
        assert!(!family.grade(&id, "3").unwrap().passed);
        assert!(family.holds("print(1+2)  "));
        let generated = &family.generator.pool()[0];
        let turns = family.prompt(generated).unwrap();
        assert!(
            turns[0].1.contains("```lua"),
            "a generated task is put to the policy too"
        );
    }

    #[test]
    fn a_continued_campaign_has_the_proposals_it_had_and_the_same_description() {
        let dir = tempfile::tempdir().unwrap();
        let first = family(dir.path());
        first
            .add(vec![proposal("print(1+2)", 1, Part::Train)])
            .unwrap();
        let described = first.describe();
        drop(first);
        let again = family(dir.path());
        assert!(again.has_generation(1) && !again.has_generation(2));
        assert_eq!(again.pool().len(), 7);
        assert_eq!(again.describe(), described);
    }

    #[test]
    fn the_overlap_audit_gets_every_task_with_its_role_program_and_digest() {
        let dir = tempfile::tempdir().unwrap();
        let family = family(dir.path());
        family
            .add(vec![proposal("print(1+2)", 1, Part::Dev)])
            .unwrap();
        let records = family.records();
        assert_eq!(records.len(), 6 + 6 + 4 + 1);
        let proposed = records
            .iter()
            .find(|r| r.origin.starts_with("policy"))
            .unwrap();
        assert_eq!((proposed.role, proposed.reference.as_str()), ("dev", "3"));
        assert!(proposed.digest.starts_with("sha256:"));
    }
}
