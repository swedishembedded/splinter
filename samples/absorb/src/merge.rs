// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fact pools with provenance for measuring what
// a language model learned from conversation, for its clients. If your team
// needs expertise in building measurements a learning system cannot game, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `facts merge`: the pools of shards built side by side (each on a card of its
//! own) become one pool, with the tasks the judge is calibrated on in one set.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use splinter_sdk::sources::{self, SourceTarget};
use splinter_sdk::store::tasks::{TaskEntry, TaskSet};

use crate::build::SOURCE_DIR;
use crate::facts::{one_per_family, Manifest};
use crate::runtime;

/// Merges the manifests under `parts` into one under `out`.
///
/// # Errors
/// A part is not a manifest, the parts were not built with the same seed,
/// quotas, persona and generator, or a part's tasks are missing.
pub fn run(out: &Path, parts: &[PathBuf], models: Option<&PathBuf>) -> anyhow::Result<usize> {
    anyhow::ensure!(!parts.is_empty(), "name the shards to merge");
    let manifests: Vec<Manifest> = parts
        .iter()
        .map(|p| Manifest::read(p))
        .collect::<anyhow::Result<_>>()?;
    let first = &manifests[0];
    for (part, m) in parts.iter().zip(&manifests) {
        anyhow::ensure!(
            m.seed == first.seed
                && m.quotas == first.quotas
                && m.persona == first.persona
                && m.generator == first.generator,
            "{} was not built with the same seed, quotas, persona and generator as {}",
            part.display(),
            parts[0].display()
        );
    }

    std::fs::create_dir_all(out)?;
    let target = runtime::open(out, models)?.context();
    let mut members: Vec<TaskEntry> = Vec::new();
    for (part, m) in parts.iter().zip(&manifests) {
        let ctx = runtime::open(part, models)?.context();
        // The tasks' evidence is resolved through the sources they were read
        // from: captured again from where each shard captured them, they are
        // the same sources (a source is its content and its origin).
        let root = part.join(SOURCE_DIR);
        let mut groups: Vec<PathBuf> = std::fs::read_dir(&root)
            .with_context(|| format!("{} holds no sources", part.display()))?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        groups.retain(|g| g.is_dir());
        if groups.is_empty() {
            groups.push(root);
        }
        for path in groups {
            sources::add(&target, &SourceTarget::Path { path })?;
        }
        let set = ctx.tasks().get_set(&m.task_set)?;
        for entry in set.members {
            let task = ctx.tasks().get(&entry.task)?;
            target
                .tasks()
                .put(&task)
                .with_context(|| format!("copying a task of {}", part.display()))?;
            if members.iter().all(|e| e.task != entry.task) {
                members.push(entry);
            }
        }
    }
    let task_set = target.tasks().put_set(&TaskSet {
        name: "merged shards".into(),
        members,
    })?;

    let (facts, one_each) = one_per_family(
        first.seed,
        manifests.iter().flat_map(|m| m.facts.clone()).collect(),
    );
    let mut refusals: Vec<_> = manifests.iter().flat_map(|m| m.refusals.clone()).collect();
    refusals.extend(one_each);
    let count = facts.len();
    Manifest {
        facts,
        refusals,
        task_set,
        ..first.clone()
    }
    .write(out)?;
    Ok(count)
}
