// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements adversarial admission testing of reward
// verifiers, for its clients. If your team needs expertise in finding
// reward-hacking surfaces in reinforcement-learning environments before a
// model does, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Attempts to earn a reward without solving the task.
//!
//! A verifier is written by the same author as the generator, who knows the
//! answer. An attack is written to know only what a solver knows: the
//! instruction and the workspace, the format of an answer, and whatever other
//! instances of the same environment looked like. Each [`Attack`] is a set
//! of edits to a fresh workspace; the verifier is then run on it. If it
//! says solved, the verifier has been beaten.
//!
//! The attacks every environment faces:
//!
//! * `nop` - change nothing.
//! * `format-only` - the reference answer's shape with its content scrambled
//!   (digits and letters moved, punctuation and whitespace kept): beaten when
//!   the verifier checks format.
//! * `constant:*` - an answer that is the same for every instance (empty, 0,
//!   true, nil, ...).
//! * `cross-instance` - another instance's reference answer: beaten when
//!   answers repeat across instances or do not depend on the question.
//! * `harvest` - every line, token and file of what the solver is shown,
//!   written as the answer: beaten when the answer can be read off.
//! * `tamper:*` - editing everything the solver may edit: blanking the
//!   workspace, planting `exit 0` in scripts, planting modules a verifier
//!   might import from the working directory that write the reward
//!   themselves.
//!
//! Plugins add attacks for their own verifier ([`Plugin::attacks`]).
//!
//! [`Plugin::attacks`]: super::plugin::Plugin::attacks

use std::collections::BTreeSet;

use super::artifact::Instance;
use super::folder::Folder;
use super::host::{Host, HostError, Session, Verdict};
use super::plugin::Reference;

/// The most candidate answers `harvest` tries on one instance.
pub const MAX_HARVEST: usize = 80;

/// Answers a solver might give without reading the task.
pub const GUESSES: [&str; 17] = [
    "", "0", "1", "2", "42", "-1", "10", "100", "true", "True", "false", "False", "nil", "None",
    "[]", "yes", "no",
];

/// A change to a workspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Edit {
    /// Create or replace a file.
    Write {
        /// The path, relative to the workspace.
        path: String,
        /// The new bytes.
        bytes: Vec<u8>,
    },
    /// Remove a file.
    Remove {
        /// The path, relative to the workspace.
        path: String,
    },
}

/// What an attack relies on, and so how a success is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AttackKind {
    /// Does nothing; a success means the task is already solved.
    Nop,
    /// An answer of the right shape or a guess. A success is chance when
    /// the answer space is small, so these are counted as a rate.
    Blind,
    /// An answer read off what the solver is shown.
    Leak,
    /// An edit of what the verifier trusts. A success is always a defect.
    Tamper,
}

/// An attempt: edits to a fresh workspace, and the verifier run after them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attack {
    /// What the attempt is called in a report.
    pub name: String,
    /// What it relies on.
    pub kind: AttackKind,
    /// The edits, applied in order.
    pub edits: Vec<Edit>,
}

/// What an attack achieved.
#[derive(Clone, Debug, PartialEq)]
pub struct AttackOutcome {
    /// The attack's name.
    pub name: String,
    /// What it relies on.
    pub kind: AttackKind,
    /// Whether the verifier called the result solved.
    pub beaten: bool,
    /// What the verifier said, for the report.
    pub detail: String,
}

/// Runs every attack in `attacks` on one session of `instance`, putting the
/// workspace back after each, and reports each outcome in order.
pub fn run_attacks(
    host: &Host,
    instance: &Instance,
    attacks: &[Attack],
) -> Result<Vec<AttackOutcome>, HostError> {
    let session = host.open(instance)?;
    let pristine = session.workspace()?;
    let mut outcomes = Vec::with_capacity(attacks.len());
    for attack in attacks {
        outcomes.push(run_attack(&session, attack)?);
        session.restore(&pristine)?;
    }
    Ok(outcomes)
}

fn run_attack(session: &Session, attack: &Attack) -> Result<AttackOutcome, HostError> {
    for edit in &attack.edits {
        match edit {
            Edit::Write { path, bytes } => session.write(path, bytes)?,
            Edit::Remove { path } => {
                session.remove(path);
            }
        }
    }
    let verdict = session.verify()?;
    Ok(AttackOutcome {
        name: attack.name.clone(),
        kind: attack.kind,
        beaten: verdict.solved(),
        detail: match &verdict {
            Verdict::Graded(vector) => format!("graded {vector:?}"),
            Verdict::NoReward { exit, .. } => format!("verifier wrote no reward ({exit})"),
            Verdict::Malformed(why) => format!("malformed reward: {why}"),
        },
    })
}

fn is_text(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_ok()
}

/// The reference's answer files that are text.
fn text_answers(reference: &Reference) -> Vec<(&String, &str)> {
    reference
        .changed
        .iter()
        .filter_map(|(path, bytes)| std::str::from_utf8(bytes).ok().map(|t| (path, t)))
        .collect()
}

/// `text` with every digit and letter moved to another of its class and
/// everything else kept: the same shape, a different content.
#[must_use]
pub fn scramble(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '0'..='9' => char::from(b'0' + (c as u8 - b'0' + 3) % 10),
            'a'..='z' => char::from(b'a' + (c as u8 - b'a' + 1) % 26),
            'A'..='Z' => char::from(b'A' + (c as u8 - b'A' + 1) % 26),
            other => other,
        })
        .collect()
}

fn write_all(paths: &[&String], text: &str) -> Vec<Edit> {
    paths
        .iter()
        .map(|path| Edit::Write {
            path: (*path).clone(),
            bytes: text.as_bytes().to_vec(),
        })
        .collect()
}

/// Modules a script that runs `python3` from the working directory might
/// import, each replaced by code that writes a full reward itself.
const HIJACKED_MODULES: [&str; 6] = [
    "sqlite3.py",
    "json.py",
    "hashlib.py",
    "subprocess.py",
    "sitecustomize.py",
    "usercustomize.py",
];

const REWARD_WRITER: &str = "import os as _o\n_d = _o.environ.get('LOGS_DIR', '/logs/verifier')\n_o.makedirs(_d, exist_ok=True)\nopen(_d + '/reward.json', 'w').write('{\"reward\": 1.0}')\n_o._exit(0)\n";

/// The attacks every environment faces, for `instance` whose reference
/// changed the workspace as `reference` says, with the references of the
/// other instances of the same environment as `siblings` (an instance whose
/// answer equals another's is attacked by it: that is the point).
#[must_use]
pub fn generic_attacks(
    instance: &Instance,
    reference: &Reference,
    siblings: &[Reference],
) -> Vec<Attack> {
    let workspace = instance.workspace();
    let mut attacks = vec![Attack {
        name: "nop".into(),
        kind: AttackKind::Nop,
        edits: Vec::new(),
    }];

    let answer_paths: Vec<&String> = reference.changed.keys().collect();
    let text_paths: Vec<&String> = text_answers(reference)
        .into_iter()
        .map(|(p, _)| p)
        .collect();

    // The same shape with another content, for every text answer at once.
    let scrambled: Vec<Edit> = text_answers(reference)
        .into_iter()
        .filter(|(_, text)| scramble(text) != *text)
        .map(|(path, text)| Edit::Write {
            path: path.clone(),
            bytes: scramble(text).into_bytes(),
        })
        .collect();
    if !scrambled.is_empty() {
        attacks.push(Attack {
            name: "format-only".into(),
            kind: AttackKind::Blind,
            edits: scrambled,
        });
    }

    if !text_paths.is_empty() {
        for guess in GUESSES {
            for tail in ["", "\n"] {
                attacks.push(Attack {
                    name: format!(
                        "constant:{guess:?}{}",
                        if tail.is_empty() { "" } else { "+nl" }
                    ),
                    kind: AttackKind::Blind,
                    edits: write_all(&text_paths, &format!("{guess}{tail}")),
                });
            }
        }
    }

    // Another instance's answer, whole: every file it changed.
    for (n, other) in siblings.iter().enumerate() {
        if other.changed.is_empty() {
            continue;
        }
        attacks.push(Attack {
            name: format!("cross-instance:{n}"),
            kind: AttackKind::Blind,
            edits: other
                .changed
                .iter()
                .map(|(path, bytes)| Edit::Write {
                    path: path.clone(),
                    bytes: bytes.clone(),
                })
                .collect(),
        });
    }

    // Everything the solver is shown, tried as the answer.
    if !text_paths.is_empty() {
        for (n, candidate) in harvest(instance, &workspace).into_iter().enumerate() {
            attacks.push(Attack {
                name: format!("harvest:{n}"),
                kind: AttackKind::Leak,
                edits: write_all(&text_paths, &candidate),
            });
        }
    }

    // Editing what the verifier might trust.
    let all_files: Vec<String> = workspace.paths().map(ToString::to_string).collect();
    attacks.push(Attack {
        name: "tamper:remove-workspace".into(),
        kind: AttackKind::Tamper,
        edits: all_files
            .iter()
            .map(|p| Edit::Remove { path: p.clone() })
            .collect(),
    });
    attacks.push(Attack {
        name: "tamper:blank-everything".into(),
        kind: AttackKind::Tamper,
        edits: all_files
            .iter()
            .map(|p| Edit::Write {
                path: p.clone(),
                bytes: Vec::new(),
            })
            .collect(),
    });
    let scripts: Vec<Edit> = workspace
        .entries()
        .filter(|(p, e)| e.executable || [".sh", ".py", ".lua"].iter().any(|x| p.ends_with(x)))
        .map(|(p, _)| Edit::Write {
            path: p.to_string(),
            bytes: b"exit 0\n".to_vec(),
        })
        .collect();
    if !scripts.is_empty() {
        attacks.push(Attack {
            name: "tamper:exit-zero-scripts".into(),
            kind: AttackKind::Tamper,
            edits: scripts,
        });
    }
    let mut hijack: Vec<Edit> = HIJACKED_MODULES
        .iter()
        .map(|name| Edit::Write {
            path: (*name).to_string(),
            bytes: REWARD_WRITER.as_bytes().to_vec(),
        })
        .collect();
    // With the answer also written, so a verifier that only reads the
    // answer first and then imports is reached too.
    hijack.extend(answer_paths.iter().filter_map(|p| {
        reference.changed.get(*p).map(|_| Edit::Write {
            path: (*p).clone(),
            bytes: Vec::new(),
        })
    }));
    attacks.push(Attack {
        name: "tamper:hijack-imports".into(),
        kind: AttackKind::Tamper,
        edits: hijack,
    });
    attacks
}

/// What a solver could read as an answer: each line, each number and each
/// whole file of the instruction and the workspace, in a fixed order, at
/// most [`MAX_HARVEST`].
fn harvest(instance: &Instance, workspace: &Folder) -> Vec<String> {
    let mut texts: Vec<String> = vec![instance.instruction().to_string()];
    texts.extend(
        workspace
            .entries()
            .filter_map(|(_, e)| String::from_utf8(e.bytes.clone()).ok())
            .filter(|t| !t.trim().is_empty()),
    );
    let mut found: BTreeSet<String> = BTreeSet::new();
    for text in &texts {
        if text.len() <= 400 {
            found.insert(text.trim().to_string());
        }
        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() && line.len() <= 200 {
                found.insert(line.to_string());
            }
        }
        for token in
            text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_'))
        {
            if !token.is_empty() && token.len() <= 40 {
                found.insert(token.to_string());
            }
        }
    }
    found
        .into_iter()
        .filter(|c| is_text(c.as_bytes()))
        .take(MAX_HARVEST)
        .collect()
}

/// Whether `needle` (the lines of an answer) stands in `haystack` as the
/// same consecutive lines, ignoring trailing space; a leak of the answer
/// into what the solver is shown.
#[must_use]
pub fn contains_lines(haystack: &str, needle: &str) -> bool {
    let want: Vec<&str> = needle.trim().lines().map(str::trim_end).collect();
    if want.is_empty() {
        return false;
    }
    let have: Vec<&str> = haystack.lines().map(str::trim_end).collect();
    have.windows(want.len())
        .any(|window| window == want.as_slice())
}

/// Files of what the solver is shown that hold the reference answer: the
/// static form of reading the hidden files.
#[must_use]
pub fn leaks(instance: &Instance, reference: &Reference) -> Vec<String> {
    let view = instance.solver_view();
    let mut found = Vec::new();
    for (_, answer) in text_answers(reference) {
        if answer.trim().is_empty() {
            continue;
        }
        for (path, entry) in view.entries() {
            if let Ok(text) = std::str::from_utf8(&entry.bytes) {
                if contains_lines(text, answer) {
                    found.push(path.to_string());
                }
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;
    use crate::factory::testing::{meta, parts};

    #[test]
    fn a_scramble_keeps_shape_and_changes_content() {
        assert_eq!(scramble("12 ab,\nZ"), "45 bc,\nA");
        assert_ne!(scramble("0"), "0");
        assert_eq!(scramble("..."), "...");
    }

    #[test]
    fn consecutive_lines_are_found_as_lines_and_not_as_substrings() {
        assert!(contains_lines("a\n42\nb", "42\n"));
        assert!(contains_lines("1\n2\n3", "2\n3"));
        assert!(!contains_lines("x = 142", "42"));
        assert!(!contains_lines("2\n9\n3", "2\n3"));
        assert!(!contains_lines("anything", "  "));
    }

    #[test]
    fn a_reference_answer_shown_in_the_instruction_is_a_leak() {
        let mut p = parts();
        p.instruction = "The result is:\n42\nbut write it yourself".into();
        let instance = Instance::assemble(meta(), p).unwrap();
        let reference = Reference {
            changed: [("answer.txt".to_string(), b"42\n".to_vec())].into(),
            removed: vec![],
        };
        assert_eq!(leaks(&instance, &reference), ["instruction.md"]);
        let clean = Instance::assemble(meta(), parts()).unwrap();
        assert!(leaks(&clean, &reference).is_empty());
    }

    #[test]
    fn generic_attacks_cover_every_family_and_siblings_add_cross_instance_attempts() {
        let instance = Instance::assemble(meta(), parts()).unwrap();
        let reference = Reference {
            changed: [("answer.txt".to_string(), b"7\n".to_vec())].into(),
            removed: vec![],
        };
        let sibling = Reference {
            changed: [("answer.txt".to_string(), b"9\n".to_vec())].into(),
            removed: vec![],
        };
        let attacks = generic_attacks(&instance, &reference, &[reference.clone(), sibling]);
        let names: Vec<&str> = attacks.iter().map(|a| a.name.as_str()).collect();
        for expected in [
            "nop",
            "format-only",
            "constant:\"0\"",
            "cross-instance:1",
            "tamper:hijack-imports",
            "tamper:remove-workspace",
        ] {
            assert!(names.contains(&expected), "{expected} in {names:?}");
        }
        assert!(
            names.contains(&"cross-instance:0"),
            "an equal answer elsewhere is still an attack"
        );
        let kinds: BTreeSet<_> = attacks.iter().map(|a| a.kind).collect();
        assert_eq!(kinds.len(), 4, "nop, blind, leak and tamper: {kinds:?}");
    }
}
