// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `tasks drills`: comprehension drills made by code from a repository's
//! snapshot, stored as a task set.
//!
//! The drills are made by [`splinter_knowledge::tasks::workspace`] (see
//! `WorkspaceGenerator::drills` for what a drill is and what it is admitted
//! on). This stage captures the revision, optionally narrows the functions to
//! those a set of repair tasks touches, and stores the drills: a drill's
//! family is the file it exercises ([`splinter_core::family`]). A drill of a
//! family an experiment plan seals is refused ([`crate::sealed`]).
//!
//! The strategy hook for the automatic loop is [`DrillRequest::for_tasks`]:
//! given a set of workspace tasks (the ones a policy fails or solves only
//! sometimes), the drills are made for exactly the functions that contain the
//! lines of each task's reference change, so that a policy can be trained on
//! its verified answers to them before it attempts the repair tasks again.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Serialize;
use splinter_core::experience::PrivilegedKind;
use splinter_core::rng::Seed;
use splinter_knowledge::capture::{capture_tree, DirtyPolicy, TreeRequest};
use splinter_knowledge::tasks::workspace::{
    touched_lines, DrillPlan, DrillReason, Touched, WorkspacePlan, DRILL_GENERATOR,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

/// What `tasks drills` is asked.
#[derive(Clone, Debug)]
pub struct DrillRequest {
    /// The git repository (a work tree or a bare repository).
    pub repository: PathBuf,
    /// The revision whose snapshot is drilled.
    pub revision: String,
    /// Whether a revision checked out with uncommitted changes is allowed:
    /// the snapshot is the commit's tracked files either way.
    pub allow_dirty: bool,
    /// The most drills of any one function.
    pub per_function: usize,
    /// The seed the inputs are drawn by.
    pub seed: Seed,
    /// The source files to drill; every source file when empty.
    pub files: Vec<String>,
    /// Drill only the functions that contain the lines the reference changes
    /// of this set's workspace tasks change.
    pub for_tasks: Option<TaskSetId>,
}

/// What `tasks drills` did.
#[derive(Clone, Debug, Serialize)]
pub struct Drilled {
    /// The commit that was drilled.
    pub commit: String,
    /// The id of the git tree of its snapshot.
    pub tree_id: String,
    /// The seed the inputs were drawn by.
    pub seed: Seed,
    /// The most drills of one function.
    pub per_function: usize,
    /// Every drill admitted; `None` when none was.
    pub task_set: Option<TaskSetId>,
    /// Drills admitted.
    pub admitted: usize,
    /// Functions considered.
    pub functions: usize,
    /// Candidates, and functions with no input, not admitted, by reason.
    pub refused: BTreeMap<DrillReason, usize>,
    /// Drills admitted in each file (family).
    pub families: BTreeMap<String, usize>,
    /// Drills admitted for each `file::function`.
    pub drilled: BTreeMap<String, usize>,
    /// Files the parser could not read.
    pub unreadable: Vec<String>,
    /// With `for_tasks`: how the tasks were read.
    pub scope: Option<Scope>,
}

/// How the tasks of `for_tasks` were turned into functions.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Scope {
    /// Tasks read.
    pub tasks: usize,
    /// Tasks with no reference change to read lines from, left out.
    pub without_reference: usize,
    /// Functions the changes lie in (`file::function`).
    pub functions: Vec<String>,
    /// Files with a changed line that no function contains (a constant, an
    /// import), which have no drill to make for it.
    pub outside_functions: Vec<String>,
}

/// Makes the drills `request` asks for (see the module documentation).
///
/// # Errors
///
/// [`OrchestratorError`] when the repository or revision cannot be captured,
/// is sealed, or is dirty without `allow_dirty`; when `for_tasks` names a
/// set with a sealed family, a reference change that is not of this revision
/// or no function that contains a changed line; when a named file is not a
/// source file; or when the workspace backend or the stores fail.
pub fn drills(ctx: &Context, request: &DrillRequest) -> Result<Drilled, OrchestratorError> {
    let refuse =
        |what: &str, e: &dyn std::fmt::Display| OrchestratorError::Refused(format!("{what}: {e}"));
    let backend = ctx.environments().workspace()?;
    let captured = capture_tree(
        &TreeRequest {
            repository: &request.repository,
            revision: &request.revision,
            dirty: if request.allow_dirty {
                DirtyPolicy::Allow
            } else {
                DirtyPolicy::Refuse
            },
            terms: None,
        },
        &backend.host,
        &ctx.artifacts(),
        &backend.scratch,
        ctx.clock(),
    )
    .map_err(|e| refuse("capturing the repository", &e))?;
    crate::sealed::admit_tree(ctx, &captured.snapshot.tree_id)?;
    ctx.sources().put_source(&captured.source)?;
    let generator = crate::history::generator(ctx, WorkspacePlan::default())?;

    let mut plan = DrillPlan::new(request.per_function, request.seed);
    plan.files.clone_from(&request.files);
    let scope = match &request.for_tasks {
        Some(set) => {
            let (targets, scope) = scope_of(ctx, &generator, &captured.snapshot, set, request)?;
            plan.files = targets.keys().cloned().collect();
            plan.targets = Some(targets);
            Some(scope)
        }
        None => None,
    };
    let generation = generator
        .drills(&captured.snapshot, &plan)
        .map_err(|e| refuse("making drills", &e))?;

    let sealed = crate::sealed::families(ctx)?;
    let store = ctx.tasks();
    let mut all = Vec::new();
    let mut families: BTreeMap<String, usize> = BTreeMap::new();
    let mut drilled: BTreeMap<String, usize> = BTreeMap::new();
    for drill in &generation.admitted {
        if let Some(why) = crate::sealed::refuses_task(&sealed, &drill.task) {
            return Err(OrchestratorError::Refused(why));
        }
        let entry = TaskEntry {
            task: store.put(&drill.task)?,
            generator: Some(DRILL_GENERATOR.to_string()),
            prompt: Some(drill.key.clone()),
            variant_of: None,
            subject: None,
        };
        *families.entry(drill.family.clone()).or_default() += 1;
        *drilled
            .entry(format!("{}::{}", drill.family, drill.function))
            .or_default() += 1;
        all.push(entry);
    }
    let name = |what: &str| format!("{what} (of {})", captured.commit);
    let put =
        |what: &str, members: &Vec<TaskEntry>| -> Result<Option<TaskSetId>, OrchestratorError> {
            if members.is_empty() {
                return Ok(None);
            }
            Ok(Some(store.put_set(&TaskSet {
                name: name(what),
                members: members.clone(),
            })?))
        };
    Ok(Drilled {
        commit: captured.commit.clone(),
        tree_id: captured.snapshot.tree_id.clone(),
        seed: request.seed,
        per_function: request.per_function,
        task_set: put("comprehension drills", &all)?,
        admitted: all.len(),
        functions: generation.functions,
        refused: generation.refused,
        families,
        drilled,
        unreadable: generation.unreadable,
        scope,
    })
}

/// The functions to drill for the tasks of `set`, and how they were found.
fn scope_of(
    ctx: &Context,
    generator: &splinter_knowledge::tasks::workspace::WorkspaceGenerator,
    snapshot: &splinter_sandbox::TreeSnapshot,
    set: &TaskSetId,
    request: &DrillRequest,
) -> Result<(BTreeMap<String, BTreeSet<String>>, Scope), OrchestratorError> {
    crate::sealed::admit_set(ctx, set)?;
    let store = ctx.tasks();
    let mut touched: BTreeMap<String, Touched> = BTreeMap::new();
    let mut scope = Scope::default();
    for entry in store.get_set(set)?.members {
        let task = store.get(&entry.task)?;
        scope.tasks += 1;
        let Some(patch) = task.privileged.iter().find_map(|p| match p.kind {
            PrivilegedKind::Reference => Some(p.content.as_str()),
            _ => None,
        }) else {
            scope.without_reference += 1;
            continue;
        };
        for (file, change) in touched_lines(patch) {
            if !request.files.is_empty() && !request.files.contains(&file) {
                continue;
            }
            let held = touched.entry(file).or_default();
            held.lines.extend(change.lines);
            held.shown.extend(change.shown);
            held.shown.sort();
            held.shown.dedup();
        }
    }
    let targets = generator
        .enclosing_functions(snapshot, &touched)
        .map_err(|e| OrchestratorError::Refused(format!("reading the tasks' functions: {e}")))?;
    scope.outside_functions = touched
        .keys()
        .filter(|file| !targets.contains_key(*file))
        .cloned()
        .collect();
    scope.functions = targets
        .iter()
        .flat_map(|(file, names)| names.iter().map(move |name| format!("{file}::{name}")))
        .collect();
    if targets.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "no function of the revision contains a line the reference changes of the {} \
             task(s) of {set} change ({} without a reference change, {} file(s) changed \
             outside any function)",
            scope.tasks,
            scope.without_reference,
            scope.outside_functions.len()
        )));
    }
    Ok((targets, scope))
}
