// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// self-generated verifiable task pools for policy-gradient training you can
// procure our services by sending an email to info@swedishembedded.com.

//! The task family of a lineage: the seeded generator's programs beside the
//! programs the current policy proposed from the Lua 5.1 manual, all put and
//! graded under the neutral protocol.
//!
//! The proposals are kept in one append-only file, so a lineage that is
//! continued has the pool it had, and each proposal records which round's
//! policy wrote it and from which manual section. The pool grows between
//! rounds; the dev and retention sets are fixed once they are drawn, so every
//! round is judged on the same tasks.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_sdk::campaign::brain::{Graded, TaskFamily};
use splinter_sdk::campaign::CampaignError;
use splinter_sdk::vocabulary::digest::Digest;

use crate::language::{Language, LUA51};
use crate::{neutral, LuaPredict};

/// Which part of the campaign a proposal belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// Trained on.
    Train,
    /// Never trained on: the dev gate is decided on it.
    Dev,
    /// Never trained on: what the policy already answered, kept as a
    /// retention probe beside the generator's.
    Keep,
}

/// One program the policy proposed, labelled by the interpreter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposal {
    /// The task id, from the digest of the program.
    pub id: String,
    /// The generation whose policy proposed it (1 is the base).
    pub generation: u32,
    /// The round of that generation (from 1; 0 for a proposal made before
    /// any round).
    #[serde(default)]
    pub round: u32,
    /// The manual section it was proposed from, as `part#position`.
    #[serde(default)]
    pub section: Option<String>,
    /// The task family it belongs to; absent for the original family, Lua
    /// 5.1, so lineages of one family journal what they always did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
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

/// The id of a program proposed for `language`: the original family's ids
/// are those of before families existed, any other family's are made from its
/// name and the program, so one text proposed for two languages is two tasks.
#[must_use]
pub fn proposal_id_in(language: &Language, program: &str) -> String {
    if language.is_original() {
        return proposal_id(program);
    }
    let digest =
        Digest::of(format!("{}\n{}", language.name, program.trim()).as_bytes()).to_string();
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
    /// The task family; absent for the original one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// The program.
    pub program: String,
    /// What the interpreter printed.
    pub reference: String,
    /// The digest of the program.
    pub digest: String,
}

/// The family: see the module documentation.
pub struct ProposalFamily {
    /// The families, each with its own generator of development programs;
    /// the first is the original one when there is only one.
    generators: Vec<(Language, LuaPredict)>,
    proposals: Mutex<BTreeMap<String, Proposal>>,
    file: PathBuf,
}

impl ProposalFamily {
    /// The family over `generators` (one per task family, with that family's
    /// own generator of development programs) and the proposals kept in
    /// `file` (created when absent).
    ///
    /// # Errors
    /// The file cannot be read or holds a line that is not a proposal.
    pub fn open(
        generators: Vec<(Language, LuaPredict)>,
        file: &Path,
    ) -> Result<Self, CampaignError> {
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
            generators,
            proposals: Mutex::new(proposals),
            file: file.to_path_buf(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, Proposal>> {
        self.proposals
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Whether any proposal of `family` was made by the policy of
    /// `generation` in `round`.
    #[must_use]
    pub fn has_round(&self, family: &str, generation: u32, round: u32) -> bool {
        self.lock().values().any(|p| {
            family_name(p.family.as_deref()) == family
                && p.generation == generation
                && p.round == round
        })
    }

    /// The families, in the order they were given.
    pub fn languages(&self) -> impl Iterator<Item = &Language> {
        self.generators.iter().map(|(language, _)| language)
    }

    /// The family task `id` belongs to, proposed or generated.
    #[must_use]
    pub fn family_of(&self, id: &str) -> Option<String> {
        if let Some(p) = self.lock().get(id) {
            return Some(family_name(p.family.as_deref()).to_string());
        }
        self.generators
            .iter()
            .find(|(_, g)| g.program(id).is_some())
            .map(|(language, _)| language.name.clone())
    }

    /// The proposal of task `id`, if it is one.
    #[must_use]
    pub fn proposal(&self, id: &str) -> Option<Proposal> {
        self.lock().get(id).cloned()
    }

    /// Every proposal, by id.
    #[must_use]
    pub fn proposals(&self) -> Vec<Proposal> {
        self.lock().values().cloned().collect()
    }

    /// Whether `program` is already a task of `language`, proposed or
    /// generated.
    #[must_use]
    pub fn holds(&self, language: &Language, program: &str) -> bool {
        self.lock().contains_key(&proposal_id_in(language, program))
            || self
                .generators
                .iter()
                .filter(|(l, _)| l.name == language.name)
                .any(|(_, g)| g.sources().any(|source| source.trim() == program.trim()))
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

    /// The program of task `id`, proposed or generated.
    #[must_use]
    pub fn program_of(&self, id: &str) -> Option<String> {
        self.known(id).ok().map(|(program, _)| program)
    }

    /// Every task of the family with its role, for the examiner's overlap
    /// audit: the dev and retention tasks, and of the proposals meant for
    /// training only those in `trained` (the ones a candidate was actually
    /// trained on).
    #[must_use]
    pub fn records(&self, trained: &std::collections::BTreeSet<String>) -> Vec<TaskRecord> {
        let record =
            |id: &str, role, origin: String, family: &str, program: &str, reference: &str| {
                TaskRecord {
                    id: id.to_string(),
                    role,
                    origin,
                    family: (family != LUA51).then(|| family.to_string()),
                    program: program.to_string(),
                    reference: reference.to_string(),
                    digest: Digest::sha256_of(program.trim().as_bytes()).to_string(),
                }
            };
        let mut records = Vec::new();
        for (language, generator) in &self.generators {
            for (ids, role) in [
                (generator.pool(), "train"),
                (generator.dev(), "dev"),
                (generator.retention(), "retention"),
            ] {
                for id in ids {
                    if let Some(p) = generator.program(&id) {
                        records.push(record(
                            &id,
                            role,
                            "generator".into(),
                            &language.name,
                            &p.source,
                            &p.expected,
                        ));
                    }
                }
            }
        }
        for p in self.lock().values() {
            if p.part == Part::Train && !trained.contains(&p.id) {
                continue;
            }
            let role = match p.part {
                Part::Train => "train",
                Part::Dev => "dev",
                Part::Keep => "retention",
            };
            records.push(record(
                &p.id,
                role,
                format!("policy-of-generation-{}", p.generation),
                family_name(p.family.as_deref()),
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

impl TaskFamily for ProposalFamily {
    fn describe(&self) -> Value {
        // The proposals are not part of it: the pool grows between arms and a
        // campaign that is continued is the same campaign.
        match self.generators.as_slice() {
            [(_, only)] => {
                json!({"family": "proposed-lua", "protocol": "neutral", "generator": only.describe()})
            }
            many => json!({
                "family": "proposed-programs",
                "protocol": "neutral",
                "generators": many.iter().map(|(l, g)| json!({"name": l.name, "generator": g.describe()})).collect::<Vec<_>>(),
            }),
        }
    }

    fn pool(&self) -> Vec<String> {
        self.ids(Part::Train, self.generated(LuaPredict::pool))
    }

    fn dev(&self) -> Vec<String> {
        self.ids(Part::Dev, self.generated(LuaPredict::dev))
    }

    fn retention(&self) -> Vec<String> {
        self.ids(Part::Keep, self.generated(LuaPredict::retention))
    }

    fn prompt(&self, task: &str) -> Result<Vec<(String, String)>, CampaignError> {
        // The lock is released before a generated task is looked up: `known`
        // takes it again.
        let proposed = self.lock().get(task).map(|p| p.request.clone());
        let text = match proposed {
            Some(request) => request,
            None => {
                let family = self.family_of(task).unwrap_or_else(|| LUA51.to_string());
                let program = self.known(task)?.0;
                match self.generators.iter().find(|(l, _)| l.name == family) {
                    Some((language, _)) => language.request(&program),
                    None => neutral::request(&program),
                }
            }
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

/// The family a proposal's field names: absent is the original.
fn family_name(field: Option<&str>) -> &str {
    field.unwrap_or(LUA51)
}

impl ProposalFamily {
    /// What each generator's `part` holds, family after family.
    fn generated(&self, part: fn(&LuaPredict) -> Vec<String>) -> Vec<String> {
        self.generators.iter().flat_map(|(_, g)| part(g)).collect()
    }

    /// The program and what it prints, of a proposed or a generated task.
    fn known(&self, task: &str) -> Result<(String, String), CampaignError> {
        if let Some(p) = self.lock().get(task) {
            return Ok((p.program.clone(), p.expected.clone()));
        }
        self.generators
            .iter()
            .find_map(|(_, g)| g.program(task))
            .map(|p| (p.source.clone(), p.expected.clone()))
            .ok_or_else(|| CampaignError::Config(format!("{task:?} is not a task of the family")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{default_interpreter, Sizes};

    fn family(dir: &Path) -> ProposalFamily {
        let sizes = Sizes {
            pool: 6,
            dev: 6,
            retention: 4,
        };
        // seed: the sets only have to be disjoint, whatever the draw.
        let generator = LuaPredict::generate(&default_interpreter(), dir, sizes, 7).unwrap();
        ProposalFamily::open(vec![(lua(), generator)], &dir.join("proposals.jsonl")).unwrap()
    }

    fn lua() -> Language {
        Language::lua51(&default_interpreter(), &[])
    }

    fn proposal(program: &str, generation: u32, part: Part) -> Proposal {
        Proposal {
            id: proposal_id(program),
            generation,
            round: 1,
            section: None,
            family: None,
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
        assert!(family.holds(&lua(), "print(1+2)  "));
        let generated = &family.pool()[0];
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
        assert!(
            again.has_round(LUA51, 1, 1)
                && !again.has_round(LUA51, 1, 2)
                && !again.has_round(LUA51, 2, 1)
        );
        assert_eq!(again.pool().len(), 7);
        assert_eq!(again.describe(), described);
    }

    #[test]
    fn retention_proposals_join_the_retention_set_and_an_untrained_proposal_is_not_audited_as_trained(
    ) {
        let dir = tempfile::tempdir().unwrap();
        let family = family(dir.path());
        let before = family.retention().len();
        family
            .add(vec![
                proposal("print(5)", 1, Part::Keep),
                proposal("print(6)", 2, Part::Train),
            ])
            .unwrap();
        assert_eq!(family.retention().len(), before + 1);
        let none = std::collections::BTreeSet::new();
        assert_eq!(
            family
                .records(&none)
                .iter()
                .filter(|r| r.origin.starts_with("policy"))
                .count(),
            1,
            "only the retention proposal: the training one was not trained on"
        );
        let trained = std::collections::BTreeSet::from([proposal_id("print(6)")]);
        assert_eq!(
            family
                .records(&trained)
                .iter()
                .filter(|r| r.origin.starts_with("policy"))
                .count(),
            2
        );
    }

    #[test]
    fn the_overlap_audit_gets_every_task_with_its_role_program_and_digest() {
        let dir = tempfile::tempdir().unwrap();
        let family = family(dir.path());
        family
            .add(vec![proposal("print(1+2)", 1, Part::Dev)])
            .unwrap();
        let records = family.records(&std::collections::BTreeSet::new());
        assert_eq!(records.len(), 6 + 6 + 4 + 1);
        let proposed = records
            .iter()
            .find(|r| r.origin.starts_with("policy"))
            .unwrap();
        assert_eq!((proposed.role, proposed.reference.as_str()), ("dev", "3"));
        assert!(proposed.digest.starts_with("sha256:"));
    }

    fn python() -> Language {
        let mut language = Language::python3(Path::new("python3"), "sha256:unused", &[]);
        language.interpreter.sha256 = None;
        language
    }

    #[test]
    fn several_families_keep_their_own_dev_items_ids_and_wording() {
        let dir = tempfile::tempdir().unwrap();
        let sizes = Sizes {
            pool: 0,
            dev: 6,
            retention: 2,
        };
        let (lua, python) = (lua(), python());
        let generators = vec![
            (
                lua.clone(),
                LuaPredict::generate_for(&lua, dir.path(), sizes, 7).unwrap(),
            ),
            (
                python.clone(),
                LuaPredict::generate_for(&python, dir.path(), sizes, 7).unwrap(),
            ),
        ];
        let family = ProposalFamily::open(generators, &dir.path().join("proposals.jsonl")).unwrap();
        let dev = family.dev();
        assert_eq!(dev.len(), 12, "every family's dev items are in the union");
        let ids: std::collections::BTreeSet<&String> = dev.iter().collect();
        assert_eq!(ids.len(), 12, "ids of two families never meet");
        let per_family = |name: &str| {
            dev.iter()
                .filter(|id| family.family_of(id).as_deref() == Some(name))
                .count()
        };
        assert_eq!((per_family(LUA51), per_family("python3")), (6, 6));
        let python_dev = dev.iter().find(|id| id.starts_with("python3-dev")).unwrap();
        assert!(family.prompt(python_dev).unwrap()[0]
            .1
            .starts_with("```python\n"));
        assert!(family.prompt(&dev[0]).unwrap()[0].1.starts_with("```lua\n"));
        // One text proposed in two languages is two tasks, and a repeat is a
        // repeat within a language only.
        assert_ne!(
            proposal_id_in(&lua, "print(1)"),
            proposal_id_in(&python, "print(1)")
        );
        assert_eq!(proposal_id_in(&lua, "print(1)"), proposal_id("print(1)"));
        assert!(!family.holds(&python, "print(1+2)"));
        let mut mine = proposal("print(1+2)", 1, Part::Train);
        mine.id = proposal_id_in(&python, "print(1+2)");
        mine.family = Some("python3".into());
        family.add(vec![mine]).unwrap();
        assert!(family.holds(&python, "print(1+2)"));
        assert!(!family.holds(&lua, "print(1+2)"));
        assert!(family.has_round("python3", 1, 1) && !family.has_round(LUA51, 1, 1));
    }

    /// A lineage of the original family alone journals the lines it always
    /// did, so one begun before families continues unchanged.
    #[test]
    fn a_proposal_of_the_original_family_journals_as_it_always_did() {
        let p = proposal("print(1)", 1, Part::Train);
        let line = serde_json::to_string(&p).unwrap();
        assert!(!line.contains("family"), "{line}");
        let again: Proposal = serde_json::from_str(&line).unwrap();
        assert_eq!(again, p);
        let task = TaskRecord {
            id: "t".into(),
            role: "train",
            origin: "generator".into(),
            family: None,
            program: "print(1)".into(),
            reference: "1".into(),
            digest: "d".into(),
        };
        assert!(!serde_json::to_string(&task).unwrap().contains("family"));
    }
}
