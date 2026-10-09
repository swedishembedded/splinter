// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fact pools with provenance for measuring what
// a language model learned from conversation, for its clients. If your team
// needs expertise in building measurements a learning system cannot game, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `facts build`: the fact pool, read from a directory of family files by
//! Splinter's own task generator.
//!
//! The directory is what `splinter-jefferson materials` writes: one file per
//! letter family (the printings of one letter in different editions are one
//! file), so a file is the unit roles are decided by. A seeded hash picks
//! which families are read, the generator proposes `recall` tasks from their
//! sections, code admits them ([`splinter_sdk::knowledge`]'s gates), and each
//! admitted task becomes a fact when its statement holds checkable keys.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::Serialize;
use splinter_sdk::agent::CancelToken;
use splinter_sdk::sources::{self, SourceTarget};
use splinter_sdk::store::tasks::{TaskEntry, TaskSet};
use splinter_sdk::tasks::{self, Generation};
use splinter_sdk::vocabulary::experience::PrivilegedKind;

use crate::facts::{admit, Fact, Manifest, Refusal, SCHEMA};
use crate::roles::Quotas;
use crate::runtime;

/// The kind of task whose reference is one fact the sections state.
const KIND: &str = "recall";

/// Family files the generator is asked about in one call. It stops asking for
/// a kind whose proposals keep being refused, so a pool is generated a few
/// families at a time: one run of letters it cannot use does not end the rest.
const FAMILIES_PER_GENERATION: usize = 4;

/// The subdirectory of the materials directory that holds letters.
const LETTERS: &str = "letters";

/// Where the chosen family files are copied for capture, under the output.
pub(crate) const SOURCE_DIR: &str = "source";

/// What `facts build` needs.
pub struct Request {
    /// The directory of family files.
    pub materials: PathBuf,
    /// Where the run's state and manifest go.
    pub out: PathBuf,
    /// Seeds the choice of families and every later hash.
    pub seed: u64,
    /// The quotas the roles are later filled to; they set the roles' shares.
    pub quotas: Quotas,
    /// How many family files to read facts from.
    pub families: usize,
    /// The fewest words a family file has to run to be read.
    pub min_words: usize,
    /// The model that proposes the tasks.
    pub generator: String,
    /// The persona the policy answers under.
    pub persona: String,
    /// The model store, when not the configuration's.
    pub models: Option<PathBuf>,
    /// Build only shard `.0` of `.1` of the chosen families (counted from 0),
    /// so shards can be built on separate cards and merged.
    pub shard: Option<(usize, usize)>,
}

/// What the build did.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Family files read.
    pub families: usize,
    /// Facts in the pool.
    pub facts: usize,
    /// Families that hold at least one fact.
    pub families_with_facts: usize,
    /// Facts by the role their family is a candidate for.
    pub by_candidate_role: BTreeMap<String, usize>,
    /// Proposals the pool refused, by reason.
    pub refused: BTreeMap<String, usize>,
    /// The generator's own report: tasks admitted and rejected, by reason.
    pub generation: serde_json::Value,
    /// Seconds the build took.
    pub seconds: u64,
}

/// The `n` families to read: the files of at least `min_words` words, in the
/// order of a hash of the seed and the file's name, so that neither the
/// directory's order nor the files present beyond these change the choice.
#[must_use]
pub fn pick_families(
    files: &[(String, usize)],
    seed: u64,
    n: usize,
    min_words: usize,
) -> Vec<String> {
    let mut eligible: Vec<&String> = files
        .iter()
        .filter(|(_, words)| *words >= min_words)
        .map(|(name, _)| name)
        .collect();
    eligible.sort_by_key(|name| {
        let mut input = seed.to_le_bytes().to_vec();
        input.extend_from_slice(b"read:");
        input.extend_from_slice(name.as_bytes());
        (
            blake3::hash(&input).as_bytes()[..8].to_vec(),
            (*name).clone(),
        )
    });
    eligible.into_iter().take(n).cloned().collect()
}

/// The family files under `materials` (its `letters` directory when it has one).
fn family_files(materials: &Path) -> anyhow::Result<Vec<(String, usize)>> {
    let dir = if materials.join(LETTERS).is_dir() {
        materials.join(LETTERS)
    } else {
        materials.to_path_buf()
    };
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "txt") {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                files.push((name.to_string(), text.split_whitespace().count()));
            }
        }
    }
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "{} holds no .txt family file",
        dir.display()
    );
    Ok(files)
}

/// Builds the pool and writes the manifest.
///
/// # Errors
/// A file cannot be read or written, the generator cannot run, or it yields
/// no fact.
pub fn run(request: &Request) -> anyhow::Result<Report> {
    let started = std::time::Instant::now();
    let files = family_files(&request.materials)?;
    let chosen = pick_families(&files, request.seed, request.families, request.min_words);
    let chosen: Vec<String> = match request.shard {
        Some((index, of)) => chosen
            .into_iter()
            .enumerate()
            .filter(|(n, _)| n % of == index)
            .map(|(_, name)| name)
            .collect(),
        None => chosen,
    };
    anyhow::ensure!(
        !chosen.is_empty(),
        "no family file has {} words or more",
        request.min_words
    );
    let dir = if request.materials.join(LETTERS).is_dir() {
        request.materials.join(LETTERS)
    } else {
        request.materials.clone()
    };
    let splinter = runtime::open(&request.out, request.models.as_ref())?;
    let ctx = splinter.context();
    let generator = runtime::model_ref(&request.generator)?;
    let author = request.persona.as_str();

    let store = ctx.tasks();
    let mut facts: Vec<Fact> = Vec::new();
    let mut refusals: Vec<Refusal> = Vec::new();
    let mut members: Vec<TaskEntry> = Vec::new();
    let mut generation: Vec<serde_json::Value> = Vec::new();
    let mut rejected_by_generator: BTreeMap<String, usize> = BTreeMap::new();
    for (at, group) in chosen.chunks(FAMILIES_PER_GENERATION).enumerate() {
        let source_dir = request.out.join(SOURCE_DIR).join(format!("{at:03}"));
        std::fs::create_dir_all(&source_dir)?;
        for name in group {
            std::fs::copy(dir.join(name), source_dir.join(name))
                .with_context(|| format!("copying family file {name}"))?;
        }
        let added = sources::add(&ctx, &SourceTarget::Path { path: source_dir })?;
        let generated = tasks::generate(
            &ctx,
            &Generation {
                sources: std::slice::from_ref(&added.source.id),
                sections: &[],
                kinds: &[KIND.to_string()],
                generator: &generator,
                goal: None,
                author: Some(author),
                deadline: None,
                cancel: CancelToken::new(),
            },
        )?;
        eprintln!(
            "families {}..{}: {} tasks admitted, {:?} rejected",
            at * FAMILIES_PER_GENERATION,
            at * FAMILIES_PER_GENERATION + group.len(),
            generated.tasks,
            generated.rejected
        );
        for (reason, n) in &generated.rejected {
            *rejected_by_generator.entry(reason.clone()).or_default() += n;
        }
        for entry in store.get_set(&generated.task_set)?.members {
            let task = store.get(&entry.task)?;
            let statement = task
                .privileged
                .iter()
                .find(|p| p.kind == PrivilegedKind::Reference)
                .map(|p| p.content.as_str())
                .unwrap_or_default();
            let span = task.evidence.first();
            let family = span
                .and_then(|s| s.part.as_ref())
                .map(|p| p.name.clone())
                .unwrap_or_default();
            let quote = match span {
                Some(span) => String::from_utf8_lossy(&ctx.sources().read_span(span)?).into_owned(),
                None => String::new(),
            };
            match admit(
                request.seed,
                &request.quotas,
                &family,
                &task.instruction,
                statement,
                &quote,
            ) {
                Ok(fact) if facts.iter().all(|f| f.id != fact.id) => facts.push(fact),
                Ok(_) => {}
                Err(reason) => refusals.push(Refusal {
                    family,
                    question: task.instruction.clone(),
                    reason,
                }),
            }
            members.push(entry);
        }
        generation.push(serde_json::to_value(&generated)?);
    }
    anyhow::ensure!(
        !facts.is_empty(),
        "the generator yielded no fact: {} proposals refused by the pool; the generator's own \
         rejections: {rejected_by_generator:?}",
        refusals.len(),
    );
    let task_set = store.put_set(&TaskSet {
        name: "facts".into(),
        members,
    })?;

    let (facts, one_each) = crate::facts::one_per_family(request.seed, facts);
    refusals.extend(one_each);
    let mut corpus = String::new();
    for name in &chosen {
        corpus.push_str(&std::fs::read_to_string(dir.join(name))?.to_lowercase());
    }
    let unknowns = crate::unknowns::make(request.seed, request.quotas.hallucination, &corpus);
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    for refusal in &refusals {
        *refused.entry(refusal.reason.clone()).or_default() += 1;
    }
    let mut by_candidate_role: BTreeMap<String, usize> = BTreeMap::new();
    for fact in &facts {
        *by_candidate_role
            .entry(fact.candidate_role.name().to_string())
            .or_default() += 1;
    }
    let families_with_facts = facts
        .iter()
        .map(|f| f.family.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let report = Report {
        families: chosen.len(),
        facts: facts.len(),
        families_with_facts,
        by_candidate_role,
        refused,
        generation: serde_json::Value::Array(generation),
        seconds: started.elapsed().as_secs(),
    };
    Manifest {
        schema: SCHEMA.into(),
        seed: request.seed,
        quotas: request.quotas,
        materials: request.materials.display().to_string(),
        generator: request.generator.clone(),
        persona: request.persona.clone(),
        facts,
        refusals,
        unknowns,
        task_set,
        judge: None,
        probes: None,
    }
    .write(&request.out)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_families_read_are_a_seeded_choice_independent_of_directory_order() {
        let files: Vec<(String, usize)> = (0..50)
            .map(|n| (format!("letter-{n}.txt"), if n % 5 == 0 { 40 } else { 300 }))
            .collect();
        let mut shuffled = files.clone();
        shuffled.reverse();
        let a = pick_families(&files, 1, 10, 100);
        assert_eq!(a, pick_families(&shuffled, 1, 10, 100));
        assert_eq!(a.len(), 10);
        assert!(
            a.iter()
                .all(|f| !f.ends_with("0.txt") && !f.ends_with("5.txt")),
            "short files are left out: {a:?}"
        );
        assert_ne!(
            a,
            pick_families(&files, 2, 10, 100),
            "another seed, another choice"
        );
        // Reading more families extends the choice; it never replaces it.
        assert_eq!(pick_families(&files, 1, 15, 100)[..10], a[..]);
    }
}
