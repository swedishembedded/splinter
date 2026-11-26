// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training tasks made from a codebase's own
// tests, for its clients. If your team needs expertise in teaching models
// your repositories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `task.mutate`: a revision of a repository made into Harbor tasks by its
//! own tests, what `practice` made before a recipe could.
//!
//! The revision (named by its commit, so the stage is the tree's) is
//! captured from the git repository the recipe names; how its tests run is
//! read by rule ([`RepositoryProfile::detect`]) unless the stage names the
//! command and the files under test, and the baseline is run twice: a red
//! or flaky one refuses the stage, since nothing made from it could be
//! trusted. Small faults are seeded by code into the files under test
//! ([`WorkspaceGenerator::from_mutations`]), each admitted only when the
//! project's own tests catch it reproducibly, the change reversed makes its
//! check pass, and the tests that catch it can be taken out of the
//! solver's tree.
//!
//! A family is the file a fault is in. `eval_percent` of the families,
//! drawn by the stage's seed, are held out: their tasks go to the `eval`
//! split and the rest to `train`, so nothing trained on shares a file with
//! what is examined. Each task is written as a Harbor folder
//! (`SPLIT/FAMILY-KEY`) that `tasks.pin` pins; `mutate.json` records what
//! was admitted, refused (by reason) and held out.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::json;
use splinter_core::clock::SystemClock;
use splinter_core::factory::identity::StageIdentity;
use splinter_knowledge::capture::{capture_tree, DirtyPolicy, TreeRequest};
use splinter_knowledge::tasks::workspace::{
    Held, RepositoryProfile, WorkspaceError, WorkspaceGenerator, WorkspacePlan,
};
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::workspace_host::WorkspaceHost;
use splinter_sandbox::{Limits, ProcessSandbox};
use splinter_store::artifacts::ArtifactStore;
use splinter_store::factory::products::Staging;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use super::holdout::held_out;
use super::{io, stage_error, write_json, Factory, RunError};
use crate::ops::MutateParams;
use crate::plan::PlannedStage;

fn params(stage: &PlannedStage) -> Result<MutateParams, RunError> {
    stage.params().map_err(|e| stage_error(&stage.key, e))
}

pub(crate) fn identity(
    _factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let params = params(stage)?;
    // The repository's path is where the tree is read from, not what it is:
    // the commit names the tree.
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        json!({
            "revision": params.revision, "test_command": params.test_command,
            "files": params.files, "max_tasks": params.max_tasks,
            "mutants_per_file": params.mutants_per_file, "eval_percent": params.eval_percent,
            "image": params.image,
        }),
        Vec::new(),
        "",
        Some(params.seed),
    ))
}

/// A family's name as a folder name: its letters and digits, the rest `_`.
fn slug(family: &str) -> String {
    family
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let fail = |why: String| stage_error(&stage.key, why);
    let refused = |what: &str, e: &dyn std::fmt::Display| fail(format!("{what}: {e}"));
    let params = params(stage)?;
    let repository = factory.resolved.repository.join(&params.repository);
    let scratch = factory.work(&stage.key).join("scratch");
    let _ = std::fs::remove_dir_all(&scratch);
    for dir in ["state", "host", "capture", "calls", "work", "profile"] {
        std::fs::create_dir_all(scratch.join(dir)).map_err(io(&scratch))?;
    }
    let state = StateRoot::new(scratch.join("state"));
    let store = ArtifactStore::new(&Workspace::at(&state), &state);
    let host = Arc::new(
        WorkspaceHost::resolve(DEFAULT_SEARCH_PATH, scratch.join("host"))
            .map_err(|e| refused("the workspace tools", &e))?,
    );
    let captured = capture_tree(
        &TreeRequest {
            repository: &repository,
            revision: &params.revision,
            // A commit is the tree it names, whatever the work tree holds.
            dirty: DirtyPolicy::Allow,
            terms: None,
        },
        &host,
        &store,
        &scratch.join("capture"),
        &SystemClock,
    )
    .map_err(|e| refused("capturing the revision", &e))?;
    let sandbox = ProcessSandbox::new(scratch.join("calls"), Vec::new(), &[], Limits::default());
    let defaults = WorkspacePlan::default();
    let generator = WorkspaceGenerator::new(Arc::new(sandbox), host, store, scratch.join("work"))
        .with_plan(WorkspacePlan {
            max_tasks: params.max_tasks.unwrap_or(defaults.max_tasks),
            mutants_per_file: params.mutants_per_file.unwrap_or(defaults.mutants_per_file),
            ..defaults
        })
        .map_err(|e| refused("the generator's bounds", &e))?;
    let tree = scratch.join("profile/tree");
    generator
        .environment(&captured.snapshot)
        .materialize(&captured.archive_path, &tree)
        .map_err(|e| refused("laying out the revision", &e))?;
    let mut profile = match (
        RepositoryProfile::detect(&tree),
        params.test_command.is_empty(),
    ) {
        (Ok(profile), _) => profile,
        (Err(WorkspaceError::NoProfile { .. }), false) => RepositoryProfile {
            test_command: Vec::new(),
            files_under_test: Vec::new(),
            baseline_secs: None,
            flaky: Vec::new(),
        },
        (Err(e), _) => {
            return Err(refused(
                "reading how the tests are run (name test_command and files)",
                &e,
            ))
        }
    };
    if !params.test_command.is_empty() {
        profile.test_command.clone_from(&params.test_command);
    }
    if !params.files.is_empty() {
        profile.files_under_test.clone_from(&params.files);
    }
    if profile.files_under_test.is_empty() {
        return Err(fail(
            "no file under test is known for this tree: name `files`".into(),
        ));
    }
    let qualified = generator
        .qualify(&profile, &captured.snapshot)
        .map_err(|e| refused("the baseline", &e))?;
    let generation = generator
        .from_mutations(
            &captured.snapshot,
            &qualified,
            &Held::default(),
            &SystemClock,
        )
        .map_err(|e| refused("seeding faults", &e))?;
    if generation.admitted.is_empty() {
        return Err(fail(format!(
            "no fault the tests catch was admitted; refused: {:?}",
            generation.refused
        )));
    }
    let families: BTreeSet<String> = generation
        .admitted
        .iter()
        .map(|t| t.family.clone())
        .collect();
    let examined = held_out(&families, params.eval_percent, params.seed);
    let mut splits: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for task in &generation.admitted {
        let split = if examined.contains(&task.family) {
            "eval"
        } else {
            "train"
        };
        let id = format!("{}-{}", slug(&task.family), &task.key.hex()[..12]);
        let folder = staging.path().join(split).join(&id);
        generator
            .export_harbor(&task.task, &folder, &params.image)
            .map_err(|e| refused(&format!("writing {split}/{id}"), &e))?;
        splits.entry(split).or_default().push(id);
    }
    let refused_by: BTreeMap<String, usize> = generation
        .refused
        .iter()
        .map(|(reason, count)| (format!("{reason:?}"), *count))
        .collect();
    write_json(
        &staging.path().join("mutate.json"),
        &json!({
            "schema": "splinter.task-mutate/1", "commit": params.revision,
            "test_command": qualified.profile.test_command, "tests": qualified.tests.len(),
            "admitted": generation.admitted.len(), "refused": refused_by,
            "families": families, "held_out": examined, "splits": splits,
        }),
    )
}
