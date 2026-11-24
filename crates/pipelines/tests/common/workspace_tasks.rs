// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A repository made into stored workspace tasks the way the repository
//! practice recipe makes them - snapshot, qualified baseline, mutation-seeded
//! faults - for the specs of the commands that read such a set (`tasks combine`, `tasks drills`,
//! `improve`).

#![allow(clippy::unwrap_used)]

use std::path::Path;

use splinter_knowledge::capture::{capture_tree, DirtyPolicy, TreeRequest};
use splinter_knowledge::tasks::workspace::{
    Held, RepositoryProfile, WorkspaceGenerator, WorkspacePlan, WORKSPACE_GENERATOR,
};
use splinter_orchestrator::Context;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

/// The tasks made from `repo` at `HEAD` (one fault per file, at most 32
/// tasks) as one stored set.
pub fn practised(ctx: &Context, repo: &Path) -> TaskSetId {
    let backend = ctx.environments().workspace().unwrap();
    let captured = capture_tree(
        &TreeRequest {
            repository: repo,
            revision: "HEAD",
            dirty: DirtyPolicy::Refuse,
            terms: None,
        },
        &backend.host,
        &ctx.artifacts(),
        &backend.scratch,
        ctx.clock(),
    )
    .unwrap();
    ctx.sources().put_source(&captured.source).unwrap();

    let generator = WorkspaceGenerator::new(
        backend.sandbox,
        backend.host,
        ctx.artifacts(),
        backend.scratch.join("generate"),
    )
    .with_plan(WorkspacePlan {
        max_tasks: 32,
        mutants_per_file: 1,
        ..WorkspacePlan::default()
    })
    .unwrap();
    let laid = tempfile::tempdir().unwrap();
    let tree = laid.path().join("tree");
    generator
        .environment(&captured.snapshot)
        .materialize(&generator.archive_of(&captured.snapshot).unwrap(), &tree)
        .unwrap();
    let profile = RepositoryProfile::detect(&tree).unwrap();
    let qualified = generator.qualify(&profile, &captured.snapshot).unwrap();
    let generation = generator
        .from_mutations(&captured.snapshot, &qualified, &Held::none(), ctx.clock())
        .unwrap();

    let members: Vec<TaskEntry> = generation
        .admitted
        .iter()
        .map(|made| TaskEntry {
            task: ctx.tasks().put(&made.task).unwrap(),
            generator: Some(WORKSPACE_GENERATOR.to_string()),
            prompt: Some(made.key.clone()),
            variant_of: None,
            subject: None,
        })
        .collect();
    let set = ctx
        .tasks()
        .put_set(&TaskSet {
            name: format!("workspace tasks of {}", captured.commit),
            members,
        })
        .unwrap();

    set
}
