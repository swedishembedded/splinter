// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out examinations of what a
// model learned from a person's writing, for its clients. If your team needs
// expertise in reserving an exam from overlapping sources before any training
// data is built, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Reserving the exam's families before anything is generated or built.
//!
//! A family is a group of text parts that print the same text, in whole or in
//! part ([`crate::grouping`]), named by the least content digest in it, as a
//! split names it. Which families the exam is made of is decided here, by a
//! stable hash of the name, before any task is generated and any dataset is
//! built, and the text of the reserved families - every edition of it - is
//! removed from the sources everything after this reads: a source becomes the
//! source without its reserved parts (which it records as skipped), and the
//! exam is written from a source of the reserved parts alone. Nothing
//! downstream has to remember to leave anything out, because there is nothing
//! to leave out.
//!
//! Only a repository source is divided. A family is examinable when its text
//! is long enough to write tasks from and short enough that reserving it does
//! not cost the run a large part of what it learns from, when every part of
//! it is in a repository source, and when no dataset the run continues from
//! was built from it.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::source::{
    CapturedSource, Origin, Part, PartContent, SkipReason, Skipped, Source, SourceId,
};

use crate::grouping::groups_of;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// The families a persona run reserves when it names no number.
pub const DEFAULT_EXAM_FAMILIES: usize = 50;

/// The families a persona run reserves for choosing its checkpoint when it
/// names no number: a dev suite no model is trained on and the final test
/// never sees.
pub const DEFAULT_DEV_FAMILIES: usize = 50;

/// The fewest words a family has to be examinable: a task needs a passage.
pub const MIN_FAMILY_WORDS: usize = 120;

/// What `reserve` is asked.
pub struct ReserveRequest<'a> {
    /// The sources of the run, as captured.
    pub sources: &'a [SourceId],
    /// How many families to reserve for the final test.
    pub families: usize,
    /// How many more to reserve for the dev suite a checkpoint is chosen on;
    /// they are as far from training as the final test's, and apart from it.
    pub dev_families: usize,
    /// Varies the choice; the same seed over the same sources reserves the
    /// same families whatever order the sources are given in.
    pub seed: u64,
    /// Datasets already trained on: a family any of them was built from is
    /// not examinable, for a model trained on them has seen it.
    pub touched_by: &'a [DatasetId],
}

/// One reserved family.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReservedFamily {
    /// Its name: the least content digest among its parts' texts.
    pub family: String,
    /// Parts of it, over every source and edition.
    pub parts: usize,
    /// Words in all of them.
    pub words: usize,
}

/// What was reserved, and the sources to use instead.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Reservation {
    /// The families reserved for the final test, by name.
    pub families: Vec<ReservedFamily>,
    /// The families reserved for the dev suite, by name.
    #[serde(default)]
    pub dev_families: Vec<ReservedFamily>,
    /// Families that could have been.
    pub examinable: usize,
    /// Families across the sources, examinable or not.
    pub total_families: usize,
    /// The sources everything after this reads, one for each of the run's
    /// sources, without the reserved parts.
    pub training: Vec<SourceId>,
    /// The sources the final test is written from: its reserved parts alone.
    pub exam: Vec<SourceId>,
    /// The sources the dev suite is written from.
    #[serde(default)]
    pub dev: Vec<SourceId>,
}

/// Refuses a request that cannot be honoured; see [`reserve`].
fn refuse(why: String) -> OrchestratorError {
    OrchestratorError::Refused(why)
}

/// The content digests of the text the records of `datasets` were built
/// from: the parts a voice record prints and the evidence of the
/// experiences a dialogue record was projected from.
pub(crate) fn touched_blobs(
    ctx: &Context,
    datasets: &[DatasetId],
) -> Result<BTreeSet<Digest>, OrchestratorError> {
    #[derive(Deserialize)]
    struct Line {
        metadata: Origin2,
    }
    #[derive(Deserialize)]
    struct Origin2 {
        #[serde(default)]
        sources: Vec<Digest>,
        #[serde(default)]
        experiences: Vec<splinter_core::experience::ExperienceId>,
    }
    let mut blobs = BTreeSet::new();
    let experiences = ctx.experiences();
    for id in datasets {
        let stored = ctx.datasets().get(id)?;
        let text = std::fs::read_to_string(&stored.path).map_err(io(&stored.path))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(record) = serde_json::from_str::<Line>(line) else {
                continue;
            };
            blobs.extend(record.metadata.sources);
            for experience in &record.metadata.experiences {
                if experiences.contains(experience)? {
                    blobs.extend(
                        experiences
                            .get(experience)?
                            .evidence
                            .iter()
                            .map(|span| span.source.clone()),
                    );
                }
            }
        }
    }
    Ok(blobs)
}

/// Reserves `request.families` families; see the module documentation.
///
/// # Errors
/// Refused when fewer examinable families exist than were asked for, or when
/// reserving them would leave fewer families to learn from than reserved.
pub fn reserve(
    ctx: &Context,
    request: &ReserveRequest<'_>,
) -> Result<Reservation, OrchestratorError> {
    let store = ctx.sources();
    let sources: Vec<Source> = request
        .sources
        .iter()
        .map(|id| store.get_source(id))
        .collect::<Result<_, _>>()?;
    let touched = touched_blobs(ctx, request.touched_by)?;
    let group_of = groups_of(ctx, request.sources, touched.clone())?;

    // Every text part, by family; a family is divisible only when all of its
    // parts are in repository sources.
    struct Family {
        parts: usize,
        words: usize,
        divisible: bool,
        touched: bool,
    }
    let mut table: BTreeMap<String, Family> = BTreeMap::new();
    let mut counted: BTreeSet<Digest> = BTreeSet::new();
    for source in &sources {
        let divisible = matches!(source.origin, Origin::Repository { .. });
        for part in source.parts.iter().filter(|p| is_text(p)) {
            let Some(name) = group_of.get(&part.content) else {
                continue;
            };
            let family = table.entry(name.clone()).or_insert(Family {
                parts: 0,
                words: 0,
                divisible: true,
                touched: false,
            });
            family.parts += 1;
            family.divisible &= divisible;
            if counted.insert(part.content.clone()) {
                let bytes = store.read_blob(&part.content)?;
                family.words += String::from_utf8_lossy(&bytes).split_whitespace().count();
            }
        }
    }
    for blob in &touched {
        if let Some(family) = group_of.get(blob).and_then(|name| table.get_mut(name)) {
            family.touched = true;
        }
    }
    let total_words: usize = table.values().map(|f| f.words).sum();
    let total_families = table.len();
    // A family larger than this would cost the run too much of what it
    // learns from; reserving a few of that size would be most of the corpus.
    let wanted = request.families + request.dev_families;
    let largest = total_words / (2 * wanted.max(1));
    let mut examinable: Vec<&String> = table
        .iter()
        .filter(|(_, f)| {
            f.divisible && !f.touched && (MIN_FAMILY_WORDS..=largest).contains(&f.words)
        })
        .map(|(name, _)| name)
        .collect();
    let pool = examinable.len();
    if total_families < 2 * wanted {
        return Err(refuse(format!(
            "reserving {wanted} of {total_families} families would leave less to learn from than \
             is examined: --exam-families and --dev-families ask for at most half of them together"
        )));
    }
    if pool < wanted {
        return Err(refuse(format!(
            "the exam needs {wanted} families reserved up front and the sources hold {pool} that can \
             be: of {total_families} families, the rest are shorter than {MIN_FAMILY_WORDS} \
             words, longer than {largest} words, in a source that is not a directory, or already \
             trained on. Name fewer with --exam-families and --dev-families, or add sources"
        )));
    }
    examinable.sort_by_cached_key(|name| {
        let mut input = request.seed.to_le_bytes().to_vec();
        input.extend_from_slice(name.as_bytes());
        Digest::of(&input).to_string()
    });
    examinable.truncate(wanted);
    let named = |names: &[&String]| -> Vec<ReservedFamily> {
        let mut families: Vec<ReservedFamily> = names
            .iter()
            .map(|name| ReservedFamily {
                family: (*name).clone(),
                parts: table[*name].parts,
                words: table[*name].words,
            })
            .collect();
        families.sort_by(|a, b| a.family.cmp(&b.family));
        families
    };
    let families = named(&examinable[..request.families]);
    let dev_families = named(&examinable[request.families..]);
    let final_set: BTreeSet<&str> = families.iter().map(|f| f.family.as_str()).collect();
    let dev_set: BTreeSet<&str> = dev_families.iter().map(|f| f.family.as_str()).collect();
    let reserved: BTreeSet<&str> = final_set.union(&dev_set).copied().collect();

    let mut training = Vec::with_capacity(sources.len());
    let mut exam = Vec::new();
    let mut dev = Vec::new();
    for source in &sources {
        let held = |part: &Part| {
            is_text(part)
                && group_of
                    .get(&part.content)
                    .is_some_and(|name| reserved.contains(name.as_str()))
        };
        if !source.parts.iter().any(held) {
            training.push(source.id.clone());
            continue;
        }
        let Origin::Repository {
            path,
            revision,
            skipped,
        } = &source.origin
        else {
            return Err(refuse(format!(
                "source {} is not a directory yet holds a reserved part",
                source.id
            )));
        };
        let content = |part: &Part| -> Result<PartContent, OrchestratorError> {
            Ok(PartContent {
                name: part.name.clone(),
                media_type: part.media_type.clone(),
                bytes: store.read_blob(&part.content)?,
            })
        };
        let (taken, kept): (Vec<&Part>, Vec<&Part>) = source.parts.iter().partition(|p| held(p));
        let in_dev = |part: &&Part| {
            group_of
                .get(&part.content)
                .is_some_and(|name| dev_set.contains(name.as_str()))
        };
        let (taken_dev, taken_final): (Vec<&Part>, Vec<&Part>) =
            taken.iter().copied().partition(|p| in_dev(p));
        let mut skipped_now = skipped.clone();
        skipped_now.extend(taken.iter().map(|part| Skipped {
            path: part.name.clone(),
            reason: SkipReason::Reserved,
        }));
        skipped_now.sort_by(|a, b| a.path.cmp(&b.path));
        let train_origin = Origin::Repository {
            path: path.clone(),
            revision: revision.clone(),
            skipped: skipped_now,
        };
        let exam_origin = Origin::Repository {
            path: path.clone(),
            revision: revision.clone(),
            skipped: Vec::new(),
        };
        let put = |origin: Origin, parts: Vec<&Part>| -> Result<SourceId, OrchestratorError> {
            let contents = parts
                .into_iter()
                .map(content)
                .collect::<Result<Vec<_>, _>>()?;
            let captured = CapturedSource::new(origin, contents, ctx.clock())
                .map_err(|e| refuse(format!("a source without its reserved parts: {e}")))?;
            Ok(store.put_source(&captured)?)
        };
        training.push(put(train_origin, kept)?);
        if !taken_final.is_empty() {
            exam.push(put(exam_origin.clone(), taken_final)?);
        }
        if !taken_dev.is_empty() {
            dev.push(put(exam_origin, taken_dev)?);
        }
    }
    Ok(Reservation {
        families,
        dev_families,
        examinable: pool,
        total_families,
        training,
        exam,
        dev,
    })
}

fn is_text(part: &Part) -> bool {
    part.media_type.starts_with("text/")
}
