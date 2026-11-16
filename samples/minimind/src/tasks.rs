// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tool-use task generators with trusted,
// unreachable verifiers for training and examining small agents, for its
// clients. If your team needs expertise in verifiable agent tasks, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The training families of workspace tool-use tasks, and their verifier.
//!
//! Every instance's answer is unknowable without the tools: the numbers sit
//! in workspace files, and `./calc` combines them by a rule drawn per
//! instance, so neither reading the request nor memorising earlier
//! instances gives it. The agent writes its answer to `answer.txt`; the
//! verifier compares that file with the expected value, which lives outside
//! the workspace and is never shown to the agent.
//!
//! These families are the sample's own and are used for demonstrations,
//! exploration and the probes. The examination is not built from them: a
//! sealed exam with held-out families and phrasings is drawn by an
//! independent implementation the sample never reads.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::world::word;

/// The file the agent writes its answer to.
pub const ANSWER_FILE: &str = "answer.txt";

/// A training family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    /// Two numbers in two files, combined by `./calc combine`.
    Combine,
    /// The code of one item in a registry file.
    Lookup,
    /// How many lines of a log name a word.
    Count,
}

/// Every training family, in draw order.
pub const FAMILIES: [Family; 3] = [Family::Combine, Family::Lookup, Family::Count];

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
}

impl Split {
    fn salt(self) -> u64 {
        // A world constant: separates the draw streams of the splits under
        // one run seed.
        match self {
            Self::Train => 0x7472_6169_6e00_0001,
            Self::Explore => 0x6578_706c_6f00_0002,
            Self::Dev => 0x6465_7600_0000_0003,
        }
    }

    /// The split's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Explore => "explore",
            Self::Dev => "dev",
        }
    }
}

/// One task instance: the request, the workspace and the expected answer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instance {
    /// Stable id, `<split>-<index>-<family>`.
    pub id: String,
    /// Its family.
    pub family: Family,
    /// Its split.
    pub split: Split,
    /// The request the agent is given.
    pub instruction: String,
    /// The workspace's files, by relative path, with their content.
    pub files: BTreeMap<String, String>,
    /// Files that must be executable.
    pub executables: Vec<String>,
    /// What `answer.txt` must hold.
    pub expected: String,
}

/// The `./calc` helper of an instance whose combining rule is
/// `(a * m + b * n) mod p`.
fn calc_script(m: u64, n: u64, p: u64) -> String {
    format!(
        "#!/bin/sh\n# calc combine A B: combines two whole numbers by this workspace's rule.\n\
         case \"$1\" in\n  combine) echo $(( ($2 * {m} + $3 * {n}) % {p} )) ;;\n  \
         *) echo \"usage: ./calc combine A B\" >&2; exit 2 ;;\nesac\n"
    )
}

fn draw(
    rng: &mut SplitMix64,
    family: Family,
) -> (String, BTreeMap<String, String>, Vec<String>, String) {
    let mut files = BTreeMap::new();
    match family {
        Family::Combine => {
            let (a, b) = (rng.between(10, 999), rng.between(10, 999));
            let (m, n, p) = (
                rng.between(3, 17),
                rng.between(3, 17),
                rng.between(101, 997),
            );
            let (fa, fb) = (
                format!("{}.txt", word(rng, 1).to_lowercase()),
                format!("{}.txt", word(rng, 2).to_lowercase()),
            );
            let fb = if fb == fa { format!("other-{fb}") } else { fb };
            files.insert(format!("inputs/{fa}"), format!("{a}\n"));
            files.insert(format!("inputs/{fb}"), format!("{b}\n"));
            files.insert("calc".into(), calc_script(m, n, p));
            let instruction = match rng.below(3) {
                0 => format!("Combine the number in inputs/{fa} with the number in inputs/{fb} using ./calc combine, in that order. Write the result to {ANSWER_FILE}."),
                1 => format!("Run ./calc combine on the values stored in inputs/{fa} and inputs/{fb} (first {fa}, then {fb}) and put the output in {ANSWER_FILE}."),
                _ => format!("What does ./calc combine give for the numbers in inputs/{fa} and inputs/{fb}, taken in that order? Save the answer in {ANSWER_FILE}."),
            };
            (
                instruction,
                files,
                vec!["calc".into()],
                ((a * m + b * n) % p).to_string(),
            )
        }
        Family::Lookup => {
            // Distinct names, so the item asked for has one code.
            let mut items: Vec<(String, u64)> = Vec::new();
            while items.len() < 6 {
                let name = word(rng, 2).to_lowercase();
                let code = rng.between(1000, 9999);
                if items.iter().all(|(n, _)| *n != name) {
                    items.push((name, code));
                }
            }
            let target = items[rng.below(items.len())].clone();
            let registry: String = items
                .iter()
                .map(|(name, code)| format!("{name}: {code}\n"))
                .collect();
            files.insert("registry.txt".into(), registry);
            let instruction = match rng.below(2) {
                0 => format!("registry.txt lists items with their codes. Write the code of {} to {ANSWER_FILE}.", target.0),
                _ => format!("Find the code registered for {} in registry.txt and write just that code to {ANSWER_FILE}.", target.0),
            };
            (instruction, files, vec![], target.1.to_string())
        }
        Family::Count => {
            let word_ = ["ERROR", "WARN", "TIMEOUT"][rng.below(3)];
            let lines = rng.between(6, 20) as usize;
            let mut log = String::new();
            let mut hits = 0;
            for i in 0..lines {
                if rng.below(3) == 0 {
                    hits += 1;
                    log.push_str(&format!("{i:03} {word_} {}\n", word(rng, 1).to_lowercase()));
                } else {
                    log.push_str(&format!("{i:03} ok {}\n", word(rng, 1).to_lowercase()));
                }
            }
            files.insert("log.txt".into(), log);
            let instruction = match rng.below(2) {
                0 => format!("How many lines of log.txt contain {word_}? Write the number to {ANSWER_FILE}."),
                _ => format!("Count the lines in log.txt that mention {word_} and write the count to {ANSWER_FILE}."),
            };
            (instruction, files, vec![], hits.to_string())
        }
    }
}

/// `count` instances of `split`, drawn from the run's `seed`; the families
/// take turns.
#[must_use]
pub fn instances(seed: u64, split: Split, count: u32) -> Vec<Instance> {
    let mut rng = SplitMix64::new(seed ^ split.salt());
    (0..count as usize)
        .map(|i| {
            let family = FAMILIES[i % FAMILIES.len()];
            let (instruction, files, executables, expected) = draw(&mut rng, family);
            Instance {
                id: format!("{}-{i:05}-{}", split.name(), family_name(family)),
                family,
                split,
                instruction,
                files,
                executables,
                expected,
            }
        })
        .collect()
}

fn family_name(family: Family) -> &'static str {
    match family {
        Family::Combine => "combine",
        Family::Lookup => "lookup",
        Family::Count => "count",
    }
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
    /// The workspace holds the expected answer.
    pub passed: bool,
    /// What was found, in a line.
    pub reason: String,
}

/// Judges the workspace `dir` an agent left behind for `instance`: the
/// trimmed content of `answer.txt` must equal the expected value.
#[must_use]
pub fn verify(instance: &Instance, dir: &Path) -> Verdict {
    match std::fs::read_to_string(dir.join(ANSWER_FILE)) {
        Ok(text) if text.trim() == instance.expected => Verdict {
            passed: true,
            reason: "answer matches".into(),
        },
        Ok(text) => Verdict {
            passed: false,
            reason: format!(
                "answer.txt holds {:?}",
                text.trim().chars().take(40).collect::<String>()
            ),
        },
        Err(_) => Verdict {
            passed: false,
            reason: "no answer.txt".into(),
        },
    }
}

/// Writes `instance` as a Harbor task folder under `dir`: `instruction.md`,
/// `task.toml`, the workspace in `environment/`, the verifier in `tests/`
/// (with the expected answer, outside the workspace) and a reference
/// solution in `solution/`.
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
    write("instruction.md", &format!("{}\n", instance.instruction))?;
    write(
        "task.toml",
        &format!(
            "[task]\nname = \"{}\"\n\n[metadata.minimind]\nfamily = \"{}\"\nsplit = \"{}\"\n",
            instance.id,
            family_name(instance.family),
            instance.split.name()
        ),
    )?;
    write("tests/expected.txt", &format!("{}\n", instance.expected))?;
    write(
        "tests/test.sh",
        "#!/bin/sh\n# Passes when the workspace's answer.txt holds the expected answer.\n\
         # usage: tests/test.sh WORKSPACE\nhere=$(dirname \"$0\")\n\
         want=$(tr -d ' \\n\\r\\t' < \"$here/expected.txt\")\n\
         got=$(tr -d ' \\n\\r\\t' < \"$1/answer.txt\" 2>/dev/null) || exit 1\n\
         [ \"$got\" = \"$want\" ]\n",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_splits_draw_different_instances_and_a_seed_repeats_them() {
        // seed: determinism is the property under test; any seed works.
        let train = instances(3, Split::Train, 6);
        assert_eq!(train, instances(3, Split::Train, 6));
        let dev = instances(3, Split::Dev, 6);
        assert!(train.iter().zip(&dev).all(|(t, d)| t.files != d.files));
    }

    #[test]
    fn the_verifier_accepts_the_expected_answer_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        // seed: any seed; the verifier must hold for every instance.
        for instance in instances(9, Split::Dev, 3) {
            let ws = dir.path().join(&instance.id);
            materialise(&instance, &ws).unwrap();
            assert!(
                !verify(&instance, &ws).passed,
                "an empty workspace never passes"
            );
            std::fs::write(ws.join(ANSWER_FILE), format!("{}\n", instance.expected)).unwrap();
            assert!(verify(&instance, &ws).passed);
            std::fs::write(ws.join(ANSWER_FILE), "12345678").unwrap();
            assert!(!verify(&instance, &ws).passed);
        }
    }

    #[test]
    fn the_calc_helper_computes_the_expected_answer_and_the_harbor_check_agrees() {
        let dir = tempfile::tempdir().unwrap();
        // seed: any seed; the helper and the expected value must agree.
        let instance = instances(21, Split::Train, 1).remove(0);
        assert_eq!(instance.family, Family::Combine);
        let task = dir.path().join("task");
        export_harbor(&instance, &task).unwrap();
        let ws = task.join("environment");
        let inputs: Vec<String> = instance
            .files
            .iter()
            .filter(|(k, _)| k.starts_with("inputs/"))
            .map(|(k, _)| k.clone())
            .collect();
        // The instruction names the files in their order.
        let mut named: Vec<&String> = inputs.iter().collect();
        named.sort_by_key(|k| instance.instruction.find(k.as_str()).unwrap());
        let value = |k: &str| instance.files[k].trim().to_string();
        let out = std::process::Command::new("./calc")
            .args(["combine", &value(named[0]), &value(named[1])])
            .current_dir(&ws)
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            instance.expected
        );
        let check = |ws: &Path| {
            std::process::Command::new("sh")
                .arg(task.join("tests/test.sh"))
                .arg(ws)
                .status()
                .unwrap()
                .success()
        };
        assert!(!check(&ws));
        std::fs::write(ws.join(ANSWER_FILE), &out.stdout).unwrap();
        assert!(check(&ws));
    }
}
