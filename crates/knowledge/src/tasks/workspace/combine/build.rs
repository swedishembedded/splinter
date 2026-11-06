// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! One combined task made of single tasks, or the reason it cannot be.
//!
//! The members' trees are laid out and each is put back to the clean
//! snapshot (its fault put right with its own reference change, its hidden
//! tests laid back): the members must agree on that snapshot, byte for byte.
//! The combined tree is that snapshot with every member's fault put in and
//! every member's hidden tests taken out, the way each member's tree had
//! them: a hidden file a member removed is removed, one it cut cases out of
//! is cut. Two members that cut the same file differently cannot both be
//! honoured and are refused.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use splinter_core::clock::Clock;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::family::composite;
use splinter_eval::verifiers::hidden_strength::HiddenStrength;
use splinter_eval::verifiers::test_report::TestExpectation;
use splinter_eval::verifiers::workspace::{Phase, WorkspaceCheck};
use splinter_sandbox::{Patch, SandboxError};

use super::single::Single;
use super::{request, CombineReason, Combined, Stop};
use crate::tasks::dedup::Seen;
use crate::tasks::workspace::files::{copy_file, files_under, io};
use crate::tasks::workspace::mutate::{admissible, hidden_text, Stop as FaultStop};
use crate::tasks::workspace::pack::{lay_over, pack_dir};
use crate::tasks::workspace::{
    FreshTree, Reason, WorkspaceError, WorkspaceGenerator, WorkspaceTask,
};

/// A member's tree, put back to the clean snapshot, and what it had.
struct Laid {
    /// The clean snapshot: the member's fault put right and its hidden
    /// tests laid back.
    tree: FreshTree,
    /// The content address of that snapshot.
    snapshot: Digest,
    /// The files its fault is in, as its own tree had them.
    faults: BTreeMap<String, Vec<u8>>,
    /// Each file its hidden tests are in, as its own tree had it: `None`
    /// for a file taken out, the file without the hidden cases when cut.
    hidden: BTreeMap<String, Option<Vec<u8>>>,
}

impl WorkspaceGenerator {
    /// The combined task of `members`, admitted only when it is shown to be
    /// one (see the module documentation of `combine`).
    pub(super) fn attempt(&self, members: &[&Single], clock: &dyn Clock) -> Result<Combined, Stop> {
        let mut members = members.to_vec();
        members.sort_by(|a, b| a.family.cmp(&b.family));
        let names: Vec<&str> = members.iter().map(|m| m.family.as_str()).collect();
        let family = composite(&names).ok_or(Stop::Refuse(CombineReason::NoFamily))?;
        let mut files = BTreeSet::new();
        if !members
            .iter()
            .flat_map(|m| &m.files)
            .all(|f| files.insert(f))
        {
            return Err(Stop::Refuse(CombineReason::SharedFiles));
        }

        let laid = members
            .iter()
            .map(|member| self.lay_member(member))
            .collect::<Result<Vec<_>, _>>()?;
        if laid.iter().any(|l| l.snapshot != laid[0].snapshot) {
            return Err(Stop::Refuse(CombineReason::DifferentSnapshots));
        }

        let hidden_files: BTreeSet<String> =
            laid.iter().flat_map(|l| l.hidden.keys().cloned()).collect();
        let originals = self.read_files(&laid[0], members.iter().flat_map(|m| &m.files))?;
        let overlay = self.temp_dir("splinter-combine-overlay-")?;
        for file in &hidden_files {
            copy_file(&laid[0].tree.path().join(file), &overlay.path().join(file))?;
        }
        let absent = self.put_faults_in(&laid, &hidden_files, overlay.path())?;
        // The first member's tree is now the combined tree.
        let tree = laid[0].tree.path();

        let request = request::combined(&members.iter().map(|m| &m.symptoms).collect::<Vec<_>>());
        let hidden = hidden_text(overlay.path(), tree, &hidden_files);
        let seen = Seen::new(self.plan.shingle_words, self.plan.max_overlap);
        for (member, laid) in members.iter().zip(&laid) {
            for file in &member.files {
                let mutated = String::from_utf8_lossy(&laid.faults[file]).into_owned();
                admissible(
                    &request,
                    &originals[file],
                    &mutated,
                    &hidden,
                    &hidden_files,
                    &seen,
                )
                .map_err(|stop| match stop {
                    FaultStop::Refuse(Reason::LeaksChange) => {
                        Stop::Refuse(CombineReason::LeaksChange)
                    }
                    FaultStop::Refuse(_) => Stop::Refuse(CombineReason::NotSelfContained),
                    FaultStop::Fail(error) => Stop::Fail(error),
                })?;
            }
        }

        let hidden_list: Vec<String> = hidden_files.iter().cloned().collect();
        let staged = self.stage(tree, &[], Some((overlay.path(), &hidden_list)))?;
        let env = self.environment(&staged.snapshot);
        let check = union_check(&members, staged.hidden(), &absent);
        let reference: String = members.iter().map(|m| m.reference.as_str()).collect();
        let task = self.build_task(&env, &check, &request.text, &family, Some(&reference), None)?;
        self.keep(&staged)?;
        self.prove(&env, &task, &check, &reference, &members, clock)?;

        let ids: Vec<Digest> = members.iter().map(|m| m.id.clone()).collect();
        Ok(Combined {
            task: WorkspaceTask {
                key: combined_key(&ids, &family)?,
                family,
                task,
                hidden: staged.hidden(),
                snapshot: staged.snapshot.clone(),
            },
            members: ids,
        })
    }

    /// `member`'s tree laid out and put back to the clean snapshot.
    fn lay_member(&self, member: &Single) -> Result<Laid, Stop> {
        let env = self.environment(&member.snapshot);
        let tree = FreshTree::lay(&env, &self.archive_of(&member.snapshot)?, &self.scratch)?;
        let mut faults = BTreeMap::new();
        for file in &member.files {
            let bytes = std::fs::read(tree.path().join(file))
                .map_err(|_| Stop::Refuse(CombineReason::NotRestorable))?;
            faults.insert(file.clone(), bytes);
        }
        let overlay = match &member.check.hidden {
            Some(digest) => Some(self.store.path(digest)?),
            None => None,
        };
        let mut hidden = BTreeMap::new();
        if let Some(archive) = &overlay {
            let listing = self.temp_dir("splinter-combine-listing-")?;
            lay_over(archive, listing.path())?;
            for file in files_under(listing.path())? {
                let kept = std::fs::read(tree.path().join(&file)).ok();
                hidden.insert(file, kept);
            }
        }
        let fix = Patch {
            diff: member.reference.clone().into_bytes(),
            files: Vec::new(),
        };
        match env.apply(tree.path(), &fix) {
            Ok(_) => {}
            Err(SandboxError::Workspace { .. }) => {
                return Err(Stop::Refuse(CombineReason::NotRestorable))
            }
            Err(other) => return Err(Stop::Fail(other.into())),
        }
        if let Some(archive) = &overlay {
            lay_over(archive, tree.path())?;
        }
        let snapshot = self.digest_of(tree.path())?;
        Ok(Laid {
            tree,
            snapshot,
            faults,
            hidden,
        })
    }

    /// The content address of the tree at `tree`, as it would be archived.
    fn digest_of(&self, tree: &Path) -> Result<Digest, Stop> {
        let work = self.temp_dir("splinter-combine-digest-")?;
        let archive = work.path().join("tree.tar");
        pack_dir(tree, &[], &archive)?;
        Ok(std::fs::File::open(&archive)
            .and_then(Digest::of_reader)
            .map_err(io(&archive))?)
    }

    /// `files` of the clean snapshot `laid` holds, as text.
    fn read_files<'a>(
        &self,
        laid: &Laid,
        files: impl Iterator<Item = &'a String>,
    ) -> Result<BTreeMap<String, String>, Stop> {
        let mut read = BTreeMap::new();
        for file in files {
            let path = laid.tree.path().join(file);
            let bytes = std::fs::read(&path).map_err(io(&path))?;
            read.insert(file.clone(), String::from_utf8_lossy(&bytes).into_owned());
        }
        Ok(read)
    }

    /// The clean snapshot of `laid` with every member's fault put in and
    /// every member's hidden files taken out as that member's tree had them
    /// (`overlay` holds the hidden files whole), in the first member's tree;
    /// and the hidden files that are absent from it.
    fn put_faults_in(
        &self,
        laid: &[Laid],
        hidden_files: &BTreeSet<String>,
        overlay: &Path,
    ) -> Result<BTreeSet<String>, Stop> {
        let tree = laid[0].tree.path();
        for l in laid {
            for (file, bytes) in &l.faults {
                write(&tree.join(file), bytes)?;
            }
        }
        let mut absent = BTreeSet::new();
        for file in hidden_files {
            let states: Vec<&Option<Vec<u8>>> =
                laid.iter().filter_map(|l| l.hidden.get(file)).collect();
            if states.iter().any(|state| state.is_none()) {
                let path = tree.join(file);
                std::fs::remove_file(&path).map_err(io(&path))?;
                absent.insert(file.clone());
                continue;
            }
            let whole = std::fs::read(overlay.join(file)).map_err(io(&overlay.join(file)))?;
            let cuts: BTreeSet<&Vec<u8>> = states
                .iter()
                .filter_map(|state| state.as_ref())
                .filter(|kept| **kept != whole)
                .collect();
            match cuts.len() {
                0 => {}
                1 => write(&tree.join(file), cuts.into_iter().next().unwrap_or(&whole))?,
                _ => return Err(Stop::Refuse(CombineReason::HiddenOverlap)),
            }
        }
        Ok(absent)
    }
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), Stop> {
    std::fs::write(path, bytes).map_err(io(path))?;
    Ok(())
}

/// The check of the combined task: every member's commands, each member's
/// expectations moved to where its commands now are, the hidden files, and
/// the files that stay in the tree and may not be touched.
fn union_check(
    members: &[&Single],
    hidden: Option<Digest>,
    absent: &BTreeSet<String>,
) -> WorkspaceCheck {
    let mut check = WorkspaceCheck {
        fail_to_pass: Vec::new(),
        pass_to_pass: Vec::new(),
        hidden,
        protected: Vec::new(),
        hidden_paths: Vec::new(),
        hidden_strength: None,
        expectations: Vec::new(),
        pinned: Vec::new(),
    };
    let mut hidden_paths: BTreeSet<String> = absent.clone();
    let mut protected = BTreeSet::new();
    let mut pinned = BTreeSet::new();
    for member in members {
        let (fail_at, pass_at) = (check.fail_to_pass.len(), check.pass_to_pass.len());
        check
            .fail_to_pass
            .extend(member.check.fail_to_pass.iter().cloned());
        check
            .pass_to_pass
            .extend(member.check.pass_to_pass.iter().cloned());
        check
            .expectations
            .extend(member.check.expectations.iter().map(|e| TestExpectation {
                index: e.index
                    + match e.phase {
                        Phase::FailToPass => fail_at,
                        Phase::PassToPass => pass_at,
                    },
                ..e.clone()
            }));
        hidden_paths.extend(member.check.hidden_paths.iter().cloned());
        protected.extend(member.check.protected.iter().cloned());
        pinned.extend(member.check.pinned.iter().cloned());
    }
    check.protected = protected.difference(&hidden_paths).cloned().collect();
    check.hidden_paths = hidden_paths.into_iter().collect();
    check.pinned = pinned.into_iter().collect();
    check.hidden_strength = members.iter().map(|m| m.check.hidden_strength).try_fold(
        HiddenStrength {
            variants: 0,
            caught: 0,
        },
        |sum, next| {
            next.map(|n| HiddenStrength {
                variants: sum.variants + n.variants,
                caught: sum.caught + n.caught,
            })
        },
    );
    check
}

/// What identifies a combined task across runs: the tasks it is made of and
/// its family.
fn combined_key(members: &[Digest], family: &str) -> Result<Digest, WorkspaceError> {
    #[derive(serde::Serialize)]
    struct Key<'a> {
        members: &'a [Digest],
        family: &'a str,
    }
    Ok(Digest::of(&canonical_json(&Key { members, family })?))
}
