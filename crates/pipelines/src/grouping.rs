// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents whose held-out measurement
// cannot leak, for its clients. If your team needs expertise in evaluating a
// model on documents it has not seen when its sources overlap, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Naming, in a dataset, which records came from overlapping source text.
//!
//! Sources overlap: two editions print the same letter. A held-out split
//! that divides such records between training and scoring measures memory of
//! the print, not learning, so each record carries the group of its evidence
//! (`splinter_eval::overlap`), and the split keeps a group whole
//! (`splinter_data::holdout`).
//!
//! A group is named the same whatever dataset names it: the groups are
//! found over every text part of the sources the records come from, not
//! over the few parts one dataset happens to cite, and each is named by the
//! least content digest in it. So a record of a conversation written from
//! one print of a letter and a record of the writer's own text of the other
//! print - built as two datasets, trained as one - carry one name, and the
//! split holds both out or neither.
//!
//! A benchmark trial's records are grouped by its task, and every record is
//! drawn by the dataset's split seed in the stratum of its task's domain
//! ([`assign_draws`]), so held-out and monitoring sets hold every domain at
//! its share.

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Experience, Task};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_core::source::SourceId;
use splinter_data::{Corpus, Projection};
use splinter_eval::overlap::overlap_groups;

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// Sets each record of `projection` to the group of the source text it is
/// grounded in: the evidence of its experience, or, for a record of source
/// text alone, the content it names. A record grounded in several texts
/// takes the least of their groups; one grounded in none has no group.
pub(crate) fn assign_groups(
    ctx: &Context,
    corpus: &Corpus,
    projection: &mut Projection,
) -> Result<(), OrchestratorError> {
    let evidence = |e: &Experience| -> Vec<Digest> {
        e.evidence.iter().map(|span| span.source.clone()).collect()
    };
    let blobs: BTreeSet<Digest> = corpus
        .entries()
        .iter()
        .flat_map(|entry| evidence(&entry.experience))
        .chain(
            projection
                .records
                .iter()
                .flat_map(|record| record.metadata.sources.iter().cloned()),
        )
        .collect();
    let group_of = groups_of(ctx, corpus.sources(), blobs)?;
    let by_experience: BTreeMap<_, _> = corpus
        .entries()
        .iter()
        .map(|entry| (&entry.id, &entry.experience))
        .collect();
    for record in &mut projection.records {
        if let Some(family) = record
            .metadata
            .experiences
            .first()
            .and_then(|id| by_experience.get(id))
            .and_then(|e| workspace_family(e).or_else(|| bench_task(e)))
        {
            record.metadata.group = Some(family);
            continue;
        }
        let grounded_in: Vec<Digest> = match record.metadata.experiences.first() {
            Some(id) => by_experience
                .get(id)
                .map(|e| evidence(e))
                .unwrap_or_default(),
            None => record.metadata.sources.clone(),
        };
        record.metadata.group = grounded_in
            .iter()
            .filter_map(|blob| group_of.get(blob).cloned())
            .min();
    }
    Ok(())
}

/// The family of the task `experience` is an attempt at, when that task is
/// a change to a repository: the file its fault was seeded in, which the
/// task declares as its first concept.
fn workspace_family(experience: &Experience) -> Option<String> {
    (experience.task.kind == WORKSPACE_CHANGE)
        .then(|| experience.concepts.first().cloned())
        .flatten()
}

/// The group of a benchmark trial's records: its benchmark task, so that two
/// trials of one task are never divided between training and scoring.
fn bench_task(experience: &Experience) -> Option<String> {
    (experience.environment.kind == Environment::BENCH)
        .then(|| experience.environment.spec.pointer("/task/id"))
        .flatten()
        .and_then(serde_json::Value::as_str)
        .map(|id| format!("bench:{id}"))
}

/// The stratum the held-out split draws `experience`'s records in: the domain
/// its task exercises, as its provenance records it. A benchmark trial's is
/// the first part of the dimension its task declares (`support` of
/// `support.missing-info-defect-photo`); any other experience's is its task's
/// kind, so a dataset of several kinds of task is split across them.
fn stratum_of(experience: &Experience) -> Option<String> {
    if experience.environment.kind == Environment::BENCH {
        return experience
            .environment
            .spec
            .pointer("/task/dimension")
            .and_then(serde_json::Value::as_str)
            .and_then(|dimension| dimension.split('.').next())
            .filter(|domain| !domain.is_empty())
            .map(str::to_string);
    }
    Some(experience.task.kind.clone())
}

/// `projection` with every record drawn by `seed` in the stratum of the
/// experience it was projected from (`splinter_data::holdout`): the held-out
/// and monitoring sets then mirror the mix of strata, wherever the records
/// sit in the file. A record of no experience (source text alone) is drawn in
/// no stratum and follows its group. `None` derives the seed from the
/// experiences of the records, so the same experiences are split the same.
pub(crate) fn assign_draws(
    corpus: &Corpus,
    projection: Projection,
    seed: Option<u64>,
) -> Projection {
    let by_experience: BTreeMap<_, _> = corpus
        .entries()
        .iter()
        .map(|entry| (&entry.id, &entry.experience))
        .collect();
    let seed = seed.unwrap_or_else(|| content_seed(&projection));
    let stratum = |record: &splinter_data::Record| {
        record
            .metadata
            .experiences
            .first()
            .and_then(|id| by_experience.get(id))
            .and_then(|e| stratum_of(e))
    };
    projection.drawn_by(seed, &stratum)
}

/// A seed that is a function of the experiences `projection`'s records come
/// from, in no particular order: the first eight bytes of the digest of
/// their sorted ids.
fn content_seed(projection: &Projection) -> u64 {
    let ids: BTreeSet<&str> = projection
        .records
        .iter()
        .flat_map(|record| record.metadata.experiences.iter().map(|id| id.as_str()))
        .collect();
    let joined = ids.into_iter().collect::<Vec<_>>().join("\n");
    Digest::of(joined.as_bytes()).leading_u64()
}

/// The group of each text part of `sources` and of each of `blobs`: texts
/// that overlap share one ([`overlap_groups`]), named by the least content
/// digest in it.
pub(crate) fn groups_of(
    ctx: &Context,
    sources: &[SourceId],
    blobs: BTreeSet<Digest>,
) -> Result<BTreeMap<Digest, String>, OrchestratorError> {
    let store = ctx.sources();
    let mut digests = blobs;
    for id in sources {
        for part in ctx.sources().get_source(id)?.parts {
            if part.media_type.starts_with("text/") {
                digests.insert(part.content);
            }
        }
    }
    // In digest order, so the first text of a group is its least digest.
    let digests: Vec<Digest> = digests.into_iter().collect();
    let texts = digests
        .iter()
        .map(|digest| {
            store
                .read_blob(digest)
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    Ok(overlap_groups(&refs)
        .into_iter()
        .zip(&digests)
        .map(|(first, blob)| (blob.clone(), digests[first].as_str().to_string()))
        .collect())
}

/// How many independent source texts the text parts of `sources` make:
/// parts that overlap (two prints of one letter) are one. Sources with no
/// text part make none.
pub(crate) fn independent_text_groups(
    ctx: &Context,
    sources: &[SourceId],
) -> Result<usize, OrchestratorError> {
    let groups = groups_of(ctx, sources, BTreeSet::new())?;
    Ok(groups.values().collect::<BTreeSet<_>>().len())
}

/// The family of each text part of `sources`: prints of one text share a
/// family, named by the least content digest among them (the print the voice
/// view keeps).
pub(crate) fn part_families(
    ctx: &Context,
    sources: &[SourceId],
) -> Result<BTreeMap<Digest, String>, OrchestratorError> {
    groups_of(ctx, sources, BTreeSet::new())
}

/// The cluster of each of `tasks`: the group of the source text it is
/// grounded in, the least of them for a task grounded in several; `None`
/// for a task grounded in none. Tasks of one cluster draw on text that is the
/// same or overlaps, so they are not independent evidence.
pub fn task_clusters(
    ctx: &Context,
    tasks: &[Task],
) -> Result<Vec<Option<String>>, OrchestratorError> {
    let blobs: BTreeSet<Digest> = tasks
        .iter()
        .flat_map(|task| task.evidence.iter().map(|span| span.source.clone()))
        .collect();
    let sources: BTreeSet<SourceId> = tasks
        .iter()
        .flat_map(|task| task.evidence.iter())
        .filter_map(|span| span.part.as_ref().map(|part| part.source.clone()))
        .collect();
    let sources: Vec<SourceId> = sources.into_iter().collect();
    let group_of = groups_of(ctx, &sources, blobs)?;
    Ok(tasks
        .iter()
        .map(|task| {
            task.evidence
                .iter()
                .filter_map(|span| group_of.get(&span.source).cloned())
                .min()
        })
        .collect())
}
