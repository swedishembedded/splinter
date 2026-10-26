// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Tasks someone authored, imported as workspace tasks: a repository with
//! one seeded bug and an acceptance script kept outside it, or a task
//! family's workspace template with its checks.
//!
//! An authored task is held to the same bar as a seeded one before it is
//! admitted: its failing commands fail on the tree the solver is given on
//! every run, its guarding commands pass, its request neither quotes nor
//! points at the hidden files, and, when the author supplies the change that
//! solves it, the workspace verifier passes that change.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use splinter_core::clock::Clock;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::{Privileged, PrivilegedKind};
use splinter_core::selfcontained::check_self_contained;
use splinter_eval::verifiers::hidden_strength::HiddenStrength;
use splinter_eval::verifiers::test_report::TestExpectation;
use splinter_eval::verifiers::workspace::WorkspaceCheck;
use splinter_sandbox::CommandCall;

use super::held::Held;
use super::mutate::{task_key, Stop};
use super::pack::lay_over;
use super::profile::{list_files, UNITTEST_DISCOVER};
use super::{
    run_quiet, FreshTree, Reason, WorkspaceError, WorkspaceGeneration, WorkspaceGenerator,
    WorkspaceTask, REPRODUCTIONS,
};

/// The privileged kind of the hidden files' text, as the self-containment
/// rule reads it.
const HIDDEN_FILES: &str = "hidden-files";

/// The directory of a rendered repository that holds the repository.
const RENDERED_REPOSITORY: &str = "repo";

/// The file of a rendered repository that holds the acceptance script.
const RENDERED_ACCEPTANCE: &str = "accept.py";

/// The file of a rendered repository that holds the request.
const RENDERED_REQUEST: &str = "task.txt";

/// The file of a task family that holds its contract.
const FAMILY_CONTRACT: &str = "family.toml";

/// The directory of a task family that holds what only the check sees.
const FAMILY_WORLD: &str = "world";

/// An authored task, before it is checked.
#[derive(Clone, Debug)]
pub struct Fixture {
    /// The family the task belongs to, for reserving and grouping.
    pub family: String,
    /// The request, as the solver reads it.
    pub instruction: String,
    /// The directory the solver's tree is made from; an entry named `.git`
    /// is left out.
    pub tree: PathBuf,
    /// The files only the check sees, by path in the tree, with their
    /// content. They are laid over the solver's tree to check it and are
    /// not in `tree`.
    pub hidden: BTreeMap<String, Vec<u8>>,
    /// The commands that must fail on the tree and pass after the change.
    pub fail_to_pass: Vec<CommandCall>,
    /// The commands that must pass before and after.
    pub pass_to_pass: Vec<CommandCall>,
    /// The paths a change may not touch.
    pub protected: Vec<String>,
    /// The change that solves the task, as `git diff` writes it, when the
    /// author has one. Without it, that the task can be solved is not shown.
    pub reference: Option<String>,
    /// The tests each command must be seen to run and pass (see
    /// [`WorkspaceCheck::expectations`]); empty for a command graded by its
    /// exit status alone.
    pub expectations: Vec<TestExpectation>,
    /// Paths that must be byte-identical to the snapshot's after the change
    /// (see [`WorkspaceCheck::pinned`]).
    pub pinned: Vec<String>,
    /// How much a pass is worth, when the author or an earlier admission
    /// measured it (see [`WorkspaceCheck::hidden_strength`]).
    pub hidden_strength: Option<HiddenStrength>,
    /// The sven harness that solves the task, when the author adapted one
    /// to it. It travels with the task as a privileged item; the solver
    /// never sees it as part of the request.
    pub process: Option<String>,
}

impl Fixture {
    /// The fixture rendered into `dir`: a repository in `dir/repo` with one
    /// seeded bug, an acceptance script `dir/accept.py` kept outside it that
    /// the repository's tests do not catch, and the request in
    /// `dir/task.txt`. The acceptance script is the hidden check; the
    /// repository's own tests (`discover -s tests`) must keep passing and
    /// are protected.
    pub fn rendered(dir: &Path, family: &str) -> Result<Self, WorkspaceError> {
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read(&path).map_err(|source| WorkspaceError::Io { path, source })
        };
        let acceptance = read(RENDERED_ACCEPTANCE)?;
        let instruction = String::from_utf8_lossy(&read(RENDERED_REQUEST)?)
            .trim()
            .to_string();
        let argv = |parts: &[&str]| CommandCall {
            argv: parts.iter().map(|p| (*p).to_string()).collect(),
            stdin: None,
        };
        Ok(Self {
            family: family.to_string(),
            instruction,
            tree: dir.join(RENDERED_REPOSITORY),
            hidden: BTreeMap::from([(RENDERED_ACCEPTANCE.to_string(), acceptance)]),
            fail_to_pass: vec![argv(&["python3", RENDERED_ACCEPTANCE])],
            pass_to_pass: vec![argv(&UNITTEST_DISCOVER)],
            protected: vec!["tests".into()],
            reference: None,
            expectations: Vec::new(),
            pinned: Vec::new(),
            hidden_strength: None,
            process: None,
        })
    }

    /// The task family in `dir`: its contract `family.toml` names the
    /// family (`id`), the request and the workspace template with the paths
    /// the solver may write (`writable`, each a file or a `dir/**`); the
    /// template is the tree, every other path of it protected; what `world`
    /// holds is hidden, at the same path. A contract names no commands, so
    /// the family's checks are given.
    pub fn family_layout(
        dir: &Path,
        fail_to_pass: Vec<CommandCall>,
        pass_to_pass: Vec<CommandCall>,
    ) -> Result<Self, WorkspaceError> {
        let contract = dir.join(FAMILY_CONTRACT);
        let bad = |reason: &str| WorkspaceError::Fixture {
            path: contract.clone(),
            reason: reason.to_string(),
        };
        let text = std::fs::read_to_string(&contract).map_err(|source| WorkspaceError::Io {
            path: contract.clone(),
            source,
        })?;
        let doc: toml::Table = text
            .parse()
            .map_err(|e: toml::de::Error| bad(&e.to_string()))?;
        let text_of = |key: &str| {
            doc.get(key)
                .and_then(toml::Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
        };
        let family = text_of("id").ok_or_else(|| bad("it has no id"))?;
        let instruction = text_of("request").ok_or_else(|| bad("it has no request"))?;
        let workspace = doc
            .get("workspace")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| bad("it has no [workspace] table"))?;
        let template = workspace
            .get("template")
            .and_then(toml::Value::as_str)
            .ok_or_else(|| bad("[workspace] names no template"))?;
        let writable: Vec<&str> = workspace
            .get("writable")
            .and_then(toml::Value::as_array)
            .map(|a| a.iter().filter_map(toml::Value::as_str).collect())
            .unwrap_or_default();
        if let Some(pattern) = writable
            .iter()
            .find(|p| p.contains('*') && !p.ends_with("/**"))
        {
            return Err(bad(&format!(
                "writable pattern {pattern:?} is not a path or a `dir/**`"
            )));
        }
        let tree = dir.join(template);
        let protected = list_files(&tree)?
            .into_iter()
            .filter(|file| !writable.iter().any(|w| writable_matches(w, file)))
            .collect();
        let world = dir.join(FAMILY_WORLD);
        let mut hidden = BTreeMap::new();
        if world.is_dir() {
            for file in list_files(&world)? {
                let path = world.join(&file);
                let bytes =
                    std::fs::read(&path).map_err(|source| WorkspaceError::Io { path, source })?;
                hidden.insert(format!("{FAMILY_WORLD}/{file}"), bytes);
            }
        }
        Ok(Self {
            family: family.to_string(),
            instruction: instruction.to_string(),
            tree,
            hidden,
            fail_to_pass,
            pass_to_pass,
            protected,
            reference: None,
            expectations: Vec::new(),
            pinned: Vec::new(),
            hidden_strength: None,
            process: None,
        })
    }

    /// The same fixture with the change that solves it.
    #[must_use]
    pub fn with_reference(mut self, patch: impl Into<String>) -> Self {
        self.reference = Some(patch.into());
        self
    }

    fn validate(&self) -> Result<(), WorkspaceError> {
        let bad = |reason: &str| WorkspaceError::Fixture {
            path: self.tree.clone(),
            reason: reason.to_string(),
        };
        if self.family.trim().is_empty() {
            return Err(bad("it has no family"));
        }
        if self.instruction.trim().is_empty() {
            return Err(bad("it has no request"));
        }
        if self.fail_to_pass.is_empty() {
            return Err(bad(
                "it has no command that must go from failing to passing",
            ));
        }
        if let Some(path) = self.hidden.keys().find(|p| !is_inside(p)) {
            return Err(bad(&format!("hidden path {path:?} is not inside the tree")));
        }
        if let Some(path) = self.hidden.keys().find(|p| self.tree.join(p).exists()) {
            return Err(bad(&format!("hidden path {path:?} is in the tree")));
        }
        Ok(())
    }
}

/// Whether `file` is covered by the writable pattern `pattern`.
fn writable_matches(pattern: &str, file: &str) -> bool {
    match pattern.strip_suffix("/**") {
        Some(dir) => file.starts_with(&format!("{dir}/")),
        None => file == pattern,
    }
}

/// Whether `path` is relative and stays inside the tree it is joined to.
fn is_inside(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

impl WorkspaceGenerator {
    /// `fixture` as a workspace task, admitted when it passes the bar in
    /// the module documentation and its key is not in `held`; otherwise
    /// counted under the reason it was refused. `clock` stamps the
    /// throwaway experience the reference change is verified through.
    pub fn import_fixture(
        &self,
        fixture: &Fixture,
        held: &Held,
        clock: &dyn Clock,
    ) -> Result<WorkspaceGeneration, WorkspaceError> {
        fixture.validate()?;
        let mut generation = WorkspaceGeneration::default();
        match self.admit_fixture(fixture, held, clock) {
            Ok(task) => generation.admitted.push(task),
            Err(Stop::Refuse(reason)) => generation.refuse(reason),
            Err(Stop::Fail(error)) => return Err(error),
        }
        Ok(generation)
    }

    fn admit_fixture(
        &self,
        fixture: &Fixture,
        held: &Held,
        clock: &dyn Clock,
    ) -> Result<WorkspaceTask, Stop> {
        let overlay = self.temp_dir("splinter-fixture-")?;
        for (path, bytes) in &fixture.hidden {
            let file = overlay.path().join(path);
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent).map_err(|source| WorkspaceError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
            std::fs::write(&file, bytes)
                .map_err(|source| WorkspaceError::Io { path: file, source })?;
        }
        let paths: Vec<String> = fixture.hidden.keys().cloned().collect();
        let staged = self.stage(&fixture.tree, &[".git"], Some((overlay.path(), &paths)))?;
        let check = WorkspaceCheck {
            fail_to_pass: fixture.fail_to_pass.clone(),
            pass_to_pass: fixture.pass_to_pass.clone(),
            hidden: staged.hidden(),
            protected: fixture.protected.clone(),
            hidden_paths: paths,
            // An authored fixture's hidden tests are the author's to vouch for.
            hidden_strength: fixture.hidden_strength,
            expectations: fixture.expectations.clone(),
            pinned: fixture.pinned.clone(),
        };
        let change = Digest::of(&canonical_json(&check).map_err(WorkspaceError::from)?);
        let key = task_key(&staged.snapshot, &change, &fixture.family)?;
        if held.contains(&key) {
            return Err(Stop::Refuse(Reason::AlreadyHeld));
        }
        let seen = held.seen(
            &fixture.family,
            self.plan.shingle_words,
            self.plan.max_overlap,
        );
        if seen.repeats(&fixture.instruction).is_some() {
            return Err(Stop::Refuse(Reason::NearDuplicate));
        }
        let env = self.environment(&staged.snapshot);
        self.behaves_as_described(&env, &staged, &check)?;
        let hidden_text = Privileged {
            kind: PrivilegedKind::Other(HIDDEN_FILES.into()),
            content: fixture
                .hidden
                .values()
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                .collect::<Vec<_>>()
                .join("\n"),
            span: None,
        };
        if check_self_contained(&fixture.instruction, &[&hidden_text]).is_err() {
            return Err(Stop::Refuse(Reason::NotSelfContained));
        }
        let task = self.build_task(
            &env,
            &check,
            &fixture.instruction,
            &fixture.family,
            fixture.reference.as_deref(),
            fixture.process.as_deref(),
        )?;
        self.keep(&staged)?;
        if let Some(patch) = &fixture.reference {
            if !self.passes(&env, &task, patch, clock)? {
                return Err(Stop::Refuse(Reason::NotRestorable));
            }
        }
        Ok(WorkspaceTask {
            task,
            family: fixture.family.clone(),
            key,
            hidden: staged.hidden(),
            snapshot: staged.snapshot,
        })
    }

    /// Runs the check's commands on the solver's tree with the overlay laid
    /// over it, [`REPRODUCTIONS`] times: the failing ones must fail every
    /// time and the guarding ones pass every time.
    fn behaves_as_described(
        &self,
        env: &splinter_sandbox::WorkspaceEnvironment,
        staged: &super::staging::Staged,
        check: &WorkspaceCheck,
    ) -> Result<(), Stop> {
        // Each command with whether it is meant to pass, and what it did on
        // each run.
        let mut runs: Vec<(&CommandCall, bool, Vec<bool>)> = check
            .fail_to_pass
            .iter()
            .map(|call| (call, false, Vec::new()))
            .chain(
                check
                    .pass_to_pass
                    .iter()
                    .map(|call| (call, true, Vec::new())),
            )
            .collect();
        for _ in 0..REPRODUCTIONS {
            let tree = FreshTree::lay(env, &staged.archive, &self.scratch)?;
            if let Some(overlay) = &staged.overlay {
                lay_over(&overlay.path, tree.path())?;
            }
            for (call, _, outcomes) in &mut runs {
                let (result, took) = run_quiet(env, tree.path(), call, None)?;
                if result.timed_out || took > self.plan.min_time_budget {
                    return Err(Stop::Refuse(Reason::OverTimeBudget));
                }
                outcomes.push(result.exit_code == Some(0) && result.signal.is_none());
            }
        }
        for (_, wanted, outcomes) in runs {
            if outcomes.iter().any(|ok| *ok != outcomes[0]) {
                return Err(Stop::Refuse(Reason::Flaky));
            }
            if outcomes[0] != wanted {
                return Err(Stop::Refuse(Reason::NotFailingAsDescribed));
            }
        }
        Ok(())
    }
}
