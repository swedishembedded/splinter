// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tool-use task generators with trusted,
// unreachable verifiers for training and examining small agents, for its
// clients. If your team needs expertise in verifiable agent tasks, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Task instances of the workspace skills ([`crate::skills`]), their splits,
//! their workspaces and their verifier.
//!
//! Every instance's answer is unknowable without the tools: the values sit
//! in workspace files, and helpers combine them by a rule drawn per
//! instance, so neither reading the request nor memorising earlier
//! instances gives it. The verifier reads the workspace: the answer file the
//! request names, or
//! the entries of an edited file; what it expects lives outside the
//! workspace and is never shown to the agent.
//!
//! These tasks are the sample's own and are used for demonstrations,
//! exploration and the probes, one of which draws only phrasings and
//! formats training never uses. The examination is not built from them: a
//! sealed exam is drawn by an independent implementation the sample never
//! reads.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::rng::SplitMix64;

pub use crate::skills::{Check, Plan, Skill, ANSWER_FILE, SKILLS};

/// Which use an instance is drawn for; each split has its own draw stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    /// Demonstrated by the expert.
    Train,
    /// Fresh instances the model explores on its own.
    Explore,
    /// Probed after every model stage; never trained on.
    Dev,
    /// Probed after every model stage, in phrasings and file formats no
    /// training instance uses.
    Heldout,
    /// Worked conversations in the pretraining corpus.
    World,
    /// Demonstrated by the expert for mid-training only.
    Midtrain,
}

impl Split {
    fn salt(self) -> u64 {
        // A world constant: separates the draw streams of the splits under
        // one run seed.
        match self {
            Self::Train => 0x7472_6169_6e00_0001,
            Self::Explore => 0x6578_706c_6f00_0002,
            Self::Dev => 0x6465_7600_0000_0003,
            Self::World => 0x776f_726c_6400_0004,
            Self::Heldout => 0x6865_6c64_6f00_0005,
            Self::Midtrain => 0x6d69_6474_7200_0006,
        }
    }

    /// Whether the split's instances become training material (the probes'
    /// do not).
    #[must_use]
    pub fn trained(self) -> bool {
        !matches!(self, Self::Dev | Self::Heldout)
    }

    /// The split's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Explore => "explore",
            Self::Dev => "dev",
            Self::Heldout => "heldout",
            Self::World => "world",
            Self::Midtrain => "midtrain",
        }
    }
}

/// One task instance: the request, the workspace, what to do and what is
/// checked.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instance {
    /// Stable id, `<split>-<index>-<skill>`.
    pub id: String,
    /// Its skill.
    pub skill: Skill,
    /// Its split.
    pub split: Split,
    /// The request the agent is given.
    pub instruction: String,
    /// The workspace's files, by relative path, with their content.
    pub files: BTreeMap<String, String>,
    /// Files that must be executable.
    pub executables: Vec<String>,
    /// What to do, without the answer: the expert's procedure follows it.
    pub plan: Plan,
    /// What the verifier checks.
    pub check: Check,
}

/// `count` instances of `split`, drawn from the run's `seed`; the skills
/// take turns.
#[must_use]
pub fn instances(seed: u64, split: Split, count: u32) -> Vec<Instance> {
    let mut rng = SplitMix64::new(seed ^ split.salt());
    (0..count as usize)
        .map(|i| {
            let skill = SKILLS[i % SKILLS.len()];
            let d = crate::skills::draw(&mut rng, skill, split == Split::Heldout);
            // What is trained on sometimes words its request in made-up
            // words, so the agent finds the files, keys and destination by
            // where they stand, not by the verbs around them; the probes
            // keep their wording.
            let instruction = if split.trained() && rng.below(3) == 0 {
                crate::phrasing::garble(&mut rng, &d.instruction)
            } else {
                d.instruction
            };
            Instance {
                id: format!("{}-{i:05}-{}", split.name(), skill.name()),
                skill,
                split,
                instruction,
                files: d.files,
                executables: d.executables,
                plan: d.plan,
                check: d.check,
            }
        })
        .collect()
}

/// Writes `instance`'s workspace into the empty directory `dir`.
///
/// # Errors
/// A file cannot be written.
pub fn materialise(instance: &Instance, dir: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    for (rel, content) in &instance.files {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::write(&path, content).with_context(|| format!("writing {}", path.display()))?;
        if instance.executables.contains(rel) {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .with_context(|| format!("making {} executable", path.display()))?;
        }
    }
    Ok(())
}

/// The verifier's judgement of one workspace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    /// The workspace holds what the task asks for.
    pub passed: bool,
    /// What was found, in a line.
    pub reason: String,
}

/// Judges the workspace `dir` an agent left behind for `instance`: the
/// trimmed content of the answer file equals the expected value, or the edited
/// file, read in its format, holds exactly the expected entries.
#[must_use]
pub fn verify(instance: &Instance, dir: &Path) -> Verdict {
    let verdict = |passed: bool, reason: String| Verdict { passed, reason };
    match &instance.check {
        Check::Answer { file, value } => match std::fs::read_to_string(dir.join(file)) {
            Ok(text) if text.trim() == value => verdict(true, "answer matches".into()),
            Ok(text) => verdict(
                false,
                format!(
                    "{file} holds {:?}",
                    text.trim().chars().take(40).collect::<String>()
                ),
            ),
            Err(_) => verdict(false, format!("no {file}")),
        },
        Check::File {
            file,
            format,
            entries,
        } => match std::fs::read_to_string(dir.join(file)) {
            Ok(text) => {
                let got: BTreeMap<String, String> = format.parse(&text).into_iter().collect();
                if &got == entries {
                    verdict(true, format!("{file} holds the expected entries"))
                } else {
                    let differ = entries
                        .iter()
                        .filter(|(k, v)| got.get(*k) != Some(v))
                        .count();
                    verdict(
                        false,
                        format!(
                            "{file}: {differ} entries differ, {} entries read",
                            got.len()
                        ),
                    )
                }
            }
            Err(_) => verdict(false, format!("{file} is gone")),
        },
    }
}

/// The request of `instance` as its task file holds it, final newline and
/// all: what a headless agent run of the exported task is given, so every
/// episode here is given the same.
#[must_use]
pub fn instruction_text(instance: &Instance) -> String {
    format!("{}\n", instance.instruction)
}

/// Writes `instance` as a Harbor task folder under `dir`: `instruction.md`,
/// `task.toml`, the workspace in `environment/`, and the check in `tests/`
/// (outside the workspace): `expected.txt` and a `test.sh` comparing
/// the answer file with it, or for an edit the expected file and a `test.sh`
/// comparing it byte for byte (the sample's own verifier reads entries, so
/// it also accepts another spelling of the same entries).
///
/// # Errors
/// A file cannot be written.
pub fn export_harbor(instance: &Instance, dir: &Path) -> anyhow::Result<()> {
    materialise(instance, &dir.join("environment"))?;
    let write = |rel: &str, text: &str| -> anyhow::Result<()> {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap_or(dir))?;
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
    };
    write("instruction.md", &instruction_text(instance))?;
    write(
        "task.toml",
        &format!(
            "[task]\nname = \"{}\"\n\n[metadata.minimind]\nskill = \"{}\"\nsplit = \"{}\"\n",
            instance.id,
            instance.skill.name(),
            instance.split.name()
        ),
    )?;
    match &instance.check {
        Check::Answer { file, value } => {
            write("tests/expected.txt", &format!("{value}\n"))?;
            write(
                "tests/test.sh",
                &format!(
                    "#!/bin/sh\n# Passes when the workspace's {file} holds the expected answer.\n\
                     # usage: tests/test.sh WORKSPACE\nhere=$(dirname \"$0\")\n\
                     want=$(tr -d ' \\n\\r\\t' < \"$here/expected.txt\")\n\
                     got=$(tr -d ' \\n\\r\\t' < \"$1/{file}\" 2>/dev/null) || exit 1\n\
                     [ \"$got\" = \"$want\" ]\n"
                ),
            )?;
        }
        Check::File {
            file,
            format,
            entries,
        } => {
            let mut ordered: Vec<(String, String)> = format.parse(&instance.files[file]);
            for (k, v) in &mut ordered {
                if let Some(new) = entries.get(k) {
                    v.clone_from(new);
                }
            }
            write("tests/expected", &format.render(&ordered))?;
            write(
                "tests/test.sh",
                &format!(
                    "#!/bin/sh\n# Passes when the edited file holds the expected entries, as written.\n\
                     # usage: tests/test.sh WORKSPACE\ncmp -s \"$(dirname \"$0\")/expected\" \"$1/{file}\"\n"
                ),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_file_of_a_workspace_is_the_directory_of_another() {
        // seed: drawn; a clash is rare, so many instances are checked.
        let seed = splinter_sdk::model::seed::draw().value();
        for i in instances(seed, Split::World, 20_000) {
            let mut names: Vec<&String> = i.files.keys().collect();
            if let Check::Answer { file, .. } = &i.check {
                names.push(file);
            }
            for n in &names {
                assert!(
                    !names.iter().any(|m| m.starts_with(&format!("{n}/"))),
                    "seed {seed}: {} {names:?}",
                    i.id
                );
            }
        }
    }

    #[test]
    fn the_splits_draw_different_instances_and_a_seed_repeats_them() {
        // seed: determinism is the property under test; any seed works.
        let train = instances(3, Split::Train, 10);
        assert_eq!(train, instances(3, Split::Train, 10));
        let dev = instances(3, Split::Dev, 10);
        assert!(train.iter().zip(&dev).all(|(t, d)| t.files != d.files));
    }

    #[test]
    fn the_expert_solves_every_skill_and_the_verifier_accepts_only_that() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        // seed: any seed; the expert and the verifier must hold for every
        // instance, held-out ones included.
        for split in [Split::Train, Split::Heldout] {
            for instance in instances(9, split, 20) {
                let ws = dir.path().join(&instance.id);
                materialise(&instance, &ws).unwrap_or_else(|e| panic!("{e}"));
                assert!(
                    !verify(&instance, &ws).passed,
                    "an untouched workspace never passes: {}",
                    instance.id
                );
                crate::expert::solve(&instance, &ws)
                    .unwrap_or_else(|e| panic!("{}: {e}", instance.id));
                let v = verify(&instance, &ws);
                assert!(
                    v.passed,
                    "{}: {} ({})",
                    instance.id, v.reason, instance.instruction
                );
            }
        }
    }
}
