// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that projects
// verified agent experience into supervised datasets, for its clients. If
// your team needs expertise in dataset curation for fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The dataset stage: experience sets projected through one view into a
//! dataset stored under the state root, named by its manifest.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::annotation::Strength;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::ExperienceId;
use splinter_core::source::SourceId;
use splinter_core::terms::{combine_stated, Terms};
use splinter_data::{
    manifest_path, Corpus, Cpt, Critic, DecisionView, DenoiseView, Exclusion, Format, Objective,
    OutcomeView, Preference, Projection, RecordBody, Rehearsal, Retrieval, SftFinal, SftStep,
    StoredDataset, VerifierView, View, Voice, DEFAULT_MAX_FAMILY_SHARE,
};
pub use splinter_data::{Control, Strip};
use splinter_knowledge::sections::sections;
use splinter_model::{BrainDatasetCheck, TokenCounter, TrainingCapabilities};
use splinter_store::error::StoreError;
use splinter_store::experiences::SetId;
use splinter_store::lineage::DatasetLineage;

use crate::describe::{describe, DescribeRequest, Described};
use crate::exam_seal;
use crate::grouping::{assign_groups, part_families};
use crate::raft::{with_passages, Abstainer, PassageShare};
use crate::retrieval::Retrieval as PassageSearch;
use crate::variants::refuse_variants;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};
use splinter_orchestrator::ids;
use splinter_orchestrator::runs::to_json;

pub use crate::dataset_views::{
    parse_strength, parse_strip, StripPolicy, ViewName, DEFAULT_MIN_STRENGTH,
};

/// One `dataset build`.
#[derive(Clone, Debug, Serialize)]
pub struct BuildRequest {
    /// The experience sets, by id.
    pub sets: Vec<SetId>,
    /// The view.
    pub view: ViewName,
    /// What the student sees; `None` keeps the view's default.
    pub strip: Option<Strip>,
    /// The weakest decision counted; `None` is [`DEFAULT_MIN_STRENGTH`].
    pub min_strength: Option<Strength>,
    /// The system prompt every conversation opens with, in place of the
    /// default: a person's, for a policy to be trained as them. `None` keeps
    /// the default.
    pub system_prompt: Option<String>,
    /// Write an objective brain cannot train in the export format.
    pub export_only: bool,
    /// The most records, kept evenly spread over the projection
    /// ([`Projection::thinned_to`]); `None` keeps every one.
    pub limit: Option<usize>,
    /// How the writer's own text is written, for the voice view.
    pub voice: VoiceBuild,
    /// The longest record, in tokens, the training that reads the dataset
    /// fits ([`Projection::within_block`]): a longer one is left out and
    /// counted as [`Exclusion::OverBlock`] instead of being refused in the
    /// middle of a run. `None` keeps every record.
    pub max_block: Option<usize>,
    /// Split a chat record longer than `max_block` into consecutive windows
    /// that each fit ([`Projection::split_over_block`]) instead of leaving it
    /// out: every window repeats the instruction and keeps each supervised
    /// turn exactly once. Meaningless without `max_block`, which leaves the
    /// records unbounded.
    pub split_over_block: bool,
    /// For the sft-step view: one record per trajectory with every admitted
    /// action supervised ([`SftStep::packed`]), not one record per action.
    pub pack: bool,
    /// For the sft-step view: a negative control to build instead of the
    /// verified dataset ([`Control`]).
    pub control: Control,
    /// For the sft-step view: read a trajectory exported as one agent step as
    /// one tool call per turn ([`SftStep::serialise_calls`]).
    pub serialise_calls: bool,
    /// For the sft-step view: keep only the actions whose tool calls are all
    /// named here, plus every action that calls no tool (the final answer);
    /// the rest become context and are counted as `ActionNotWanted`
    /// ([`SftStep::only_actions`]).
    pub only_actions: Option<Vec<String>>,
}

/// How the voice view writes the writer's text into records; the other
/// views ignore it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct VoiceBuild {
    /// Who the writer is: the requests are written as theirs and half the
    /// records are asked under their persona prompt, half under the line
    /// that says who the model is. Excludes `system_prompt`, which would
    /// replace both. `None` writes requests with no writer named, under
    /// `system_prompt` or the default.
    pub writer: Option<String>,
    /// The most tokens of the writer's text, kept evenly spread over it
    /// ([`Projection::thinned_to_tokens`]); `None` keeps all of it.
    pub token_budget: Option<usize>,
    /// The most of the tokens one family (the prints of one text, or one
    /// work) may supply; `None` is [`DEFAULT_MAX_FAMILY_SHARE`].
    pub max_family_share: Option<f64>,
    /// Ask each passage by what a model says it is about
    /// ([`crate::describe`]) instead of by its heading and opening; a
    /// passage that gets no admitted description keeps the latter. `None`
    /// asks every passage by its heading and opening.
    pub describe: Option<DescribeRequest>,
}

/// What `dataset build` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Built {
    /// The dataset (`train <dataset>`, `dataset export <dataset>`).
    pub dataset: DatasetId,
    /// The view it was projected through.
    pub view: String,
    /// The objective its records serve.
    pub objective: Objective,
    /// Its file format.
    pub format: Format,
    /// Records in it.
    pub records: usize,
    /// Candidates left out, by reason.
    pub excluded: BTreeMap<Exclusion, usize>,
    /// The dataset file.
    pub path: PathBuf,
    /// How the passages of a described voice dataset were described.
    pub described: Option<Described>,
    /// The experiences left out because they would have crossed the exam
    /// seal, and why ([`build_dropping_leaks`]); empty otherwise.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dropped: Vec<exam_seal::Leak>,
}

impl From<StoredDataset> for Built {
    fn from(stored: StoredDataset) -> Self {
        Self {
            dataset: stored.id,
            view: stored.manifest.view,
            objective: stored.manifest.objective,
            format: stored.manifest.format,
            records: stored.manifest.counts.records,
            excluded: stored.manifest.counts.excluded,
            path: stored.path,
            described: None,
            dropped: Vec::new(),
        }
    }
}

/// Passages to put in the prompt of a share of the records: what is retrieved
/// from, and how much of the dataset gets them ([`crate::raft`]).
pub struct Passages<'a> {
    /// The passages of the sources, searched.
    pub retrieval: &'a PassageSearch<'a>,
    /// Which records get them, and what each teaches.
    pub share: PassageShare,
    /// Writes the abstentions the share asks for, when it asks for any.
    pub abstainer: Option<&'a dyn Abstainer>,
}

/// Projects `request.sets` through `request.view` and stores the dataset.
pub fn build(ctx: &Context, request: &BuildRequest) -> Result<Built, OrchestratorError> {
    build_with(ctx, request, None)
}

/// [`build`], a share of the records given retrieved passages in their
/// prompt when `passages` says so.
pub fn build_with(
    ctx: &Context,
    request: &BuildRequest,
    passages: Option<&Passages<'_>>,
) -> Result<Built, OrchestratorError> {
    let (projection, described) = project_with(ctx, request, passages)?;
    let stored = store_dataset(ctx, &projection, request.export_only)?;
    Ok(Built {
        described,
        ..Built::from(stored)
    })
}

/// The projection [`build_with`] stores, and how a described voice dataset's
/// passages were described: for a caller that adds to the records before
/// they are stored.
pub fn project_with(
    ctx: &Context,
    request: &BuildRequest,
    passages: Option<&Passages<'_>>,
) -> Result<(Projection, Option<Described>), OrchestratorError> {
    let (projection, described, corpus) =
        project_without(ctx, request, passages, &BTreeSet::new())?;
    crate::sealed::admit_corpus(ctx, &corpus)?;
    exam_seal::seal(ctx, &corpus, &projection)?;
    Ok((projection, described))
}

/// [`build`], leaving out every experience that would cross the exam seal
/// ([`exam_seal::leaks`]) instead of refusing the build. The dataset is
/// projected again without them and sealed as usual, so it is refused if
/// anything still leaks. They are counted under [`Exclusion::ExamLeak`] and
/// listed, with why, in the result.
pub fn build_dropping_leaks(
    ctx: &Context,
    request: &BuildRequest,
) -> Result<Built, OrchestratorError> {
    let (first, _, corpus) = project_without(ctx, request, None, &BTreeSet::new())?;
    let dropped = exam_seal::leaks(ctx, &corpus, &first)?;
    let leaving: BTreeSet<ExperienceId> = dropped.iter().map(|l| l.experience.clone()).collect();
    let (mut projection, described, corpus) = project_without(ctx, request, None, &leaving)?;
    if !leaving.is_empty() {
        projection
            .excluded
            .insert(Exclusion::ExamLeak, leaving.len());
    }
    exam_seal::seal(ctx, &corpus, &projection)?;
    let stored = store_dataset(ctx, &projection, request.export_only)?;
    Ok(Built {
        described,
        dropped,
        ..Built::from(stored)
    })
}

/// [`project_with`] without the experiences in `left_out` and without the
/// exam seal: the projection and the corpus it was made from.
fn project_without(
    ctx: &Context,
    request: &BuildRequest,
    passages: Option<&Passages<'_>>,
    left_out: &BTreeSet<ExperienceId>,
) -> Result<(Projection, Option<Described>, Corpus), OrchestratorError> {
    let view = request.view;
    if request.strip.is_some() && !view.takes_strip() {
        return Err(OrchestratorError::Refused(format!(
            "view {} shows the student no task, so --strip does not apply",
            view.as_str()
        )));
    }
    if request.pack && view != ViewName::SftStep {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no packed form, so --pack does not apply",
            view.as_str()
        )));
    }
    if request.control != Control::Verified && view != ViewName::SftStep {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no negative controls, so --control does not apply",
            view.as_str()
        )));
    }
    if request.serialise_calls && view != ViewName::SftStep {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no run to take apart, so --serialise-calls does not apply",
            view.as_str()
        )));
    }
    if request.min_strength.is_some() && !view.takes_min_strength() {
        return Err(OrchestratorError::Refused(format!(
            "view {} reads no verdict, so --min-strength does not apply",
            view.as_str()
        )));
    }
    let store = ctx.experiences();
    let mut ids: Vec<ExperienceId> = Vec::new();
    for set in &request.sets {
        for member in store.get_set(set)?.members {
            if !ids.contains(&member) && !left_out.contains(&member) {
                ids.push(member);
            }
        }
    }
    let mut corpus = Corpus::load(&store, &ids)?;
    refuse_variants(ctx, corpus.entries().iter().map(|entry| &entry.experience))?;
    let mut sources = BTreeSet::new();
    for entry in corpus.entries() {
        for span in &entry.experience.evidence {
            if let Some(part) = &span.part {
                sources.insert(part.source.clone());
            }
        }
    }
    let terms = source_terms(ctx, sources.iter())?;
    for source in sources {
        corpus.add_source(source);
    }
    let strength = request.min_strength.unwrap_or(DEFAULT_MIN_STRENGTH);
    let strip = request.strip.clone().unwrap_or_default();
    let source_store = ctx.sources();
    let mut described = None;
    let mut projection = match view {
        ViewName::SftFinal => SftFinal::new(strength).with_strip(strip).project(&corpus),
        ViewName::SftStep => SftStep::new(strength)
            .with_strip(strip)
            .packed(request.pack)
            .control(request.control)
            .serialise_calls(request.serialise_calls)
            .only_actions(request.only_actions.clone().unwrap_or_default())
            .project(&corpus),
        ViewName::Critic => Critic::new(strength).with_strip(strip).project(&corpus),
        ViewName::Preference => Preference::new(strength).with_strip(strip).project(&corpus),
        ViewName::Verifier => VerifierView::new(strength)
            .with_strip(strip)
            .project(&corpus),
        ViewName::Decision => DecisionView::new(strength)
            .with_strip(strip)
            .project(&corpus),
        ViewName::Retrieval => Retrieval::new(strength, &source_store)
            .with_strip(strip)
            .project(&corpus),
        ViewName::Outcome => OutcomeView::new(strength)
            .with_strip(strip)
            .project(&corpus),
        ViewName::Denoise => DenoiseView::new().with_strip(strip).project(&corpus),
        ViewName::Rehearsal => Rehearsal::new().project(&corpus),
        ViewName::Cpt => Cpt::new(&source_store).project(&corpus),
        ViewName::Voice => {
            // The stretches are measured in the policy's own tokens, the
            // unit its training row is sized in; without its tokenizer, in
            // characters at a ratio that holds for text that tokenizes badly.
            let counter = TokenCounter::for_model(&ctx.config().policy_base).ok();
            let tokens = |text: &str| {
                counter
                    .as_ref()
                    .map_or_else(|| splinter_data::chars_as_tokens(text), |c| c.count(text))
            };
            let sectioner = |text: &str, media_type: &str| {
                sections(text, media_type)
                    .into_iter()
                    .map(|section| section.range)
                    .collect()
            };
            // Prints of one text are one family: one of them is projected.
            let families = part_families(ctx, corpus.sources())?;
            let family_of = |content: &Digest| families.get(content).cloned();
            let share = request
                .voice
                .max_family_share
                .unwrap_or(DEFAULT_MAX_FAMILY_SHARE);
            if !(share > 0.0 && share <= 1.0) {
                return Err(OrchestratorError::Refused(format!(
                    "the family share {share} is not a share in (0, 1]"
                )));
            }
            let voice = Voice::new(&source_store)
                .measured_by(&tokens)
                .sectioned_by(&sectioner)
                .keeping_one_print_of(&family_of)
                .bounding_families_to(share);
            let voice = match request.voice.writer.as_deref() {
                Some(_) if request.system_prompt.is_some() => {
                    return Err(OrchestratorError::Refused(
                        "the voice view writes its own system turns when it is told the \
                         writer, so no system prompt applies as well"
                            .into(),
                    ))
                }
                Some(writer) => voice.written_as(writer),
                None => voice,
            };
            let answer_tokens = |record: &splinter_data::Record| match &record.body {
                RecordBody::Chat { messages, .. } => messages
                    .iter()
                    .filter(|m| m.train)
                    .map(|m| tokens(&m.content))
                    .sum(),
                _ => 0,
            };
            let thinned = |projection: Projection| match request.voice.token_budget {
                Some(budget) => projection.thinned_to_tokens(budget, &answer_tokens),
                None => projection,
            };
            let projection = thinned(voice.project(&corpus)?);
            match &request.voice.describe {
                None => Ok(projection),
                // The passages the dataset keeps are the ones described;
                // the projection is made again with their descriptions.
                Some(describe_request) => {
                    let answers: Vec<&str> = projection
                        .records
                        .iter()
                        .filter_map(|record| match &record.body {
                            RecordBody::Chat { messages, .. } => messages
                                .iter()
                                .find(|m| m.train)
                                .map(|m| m.content.as_str()),
                            _ => None,
                        })
                        .collect();
                    let (written, report) = describe(ctx, describe_request, &answers)?;
                    described = Some(report);
                    let description_of =
                        |answer: &str| written.get(&Digest::of(answer.as_bytes())).cloned();
                    Ok(thinned(
                        voice.described_by(&description_of).project(&corpus)?,
                    ))
                }
            }
        }
    }?;
    projection.terms = terms;
    if let Some(limit) = request.limit {
        projection = projection.thinned_to(limit);
    }
    if let Some(prompt) = &request.system_prompt {
        projection = projection.with_system_prompt(prompt);
    }
    if view == ViewName::SftStep {
        // The prompt the runs were made under is the prompt the dataset trains
        // under, so a model trained on it is asked under it.
        projection = projection
            .settled_system_prompt()
            .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    }
    if let Some(max_block) = request.max_block {
        projection = within_policy_block(ctx, projection, max_block, request.split_over_block);
    }
    if let Some(passages) = passages {
        projection = with_passages(
            ctx,
            projection,
            &passages.share,
            passages.retrieval,
            passages.abstainer,
        )?;
    }
    assign_groups(ctx, &corpus, &mut projection)?;
    Ok((projection, described, corpus))
}

/// `projection` without the records longer than `max_block` of the policy's
/// own tokens, as the record's text and tool schemas are measured (the
/// tokenizer's, else a character ratio that holds for text that tokenizes
/// badly); with `split`, a longer chat record is split into windows that fit
/// ([`Projection::split_over_block`]) instead of left out. The chat
/// template's markup is not measured, so the bound is a floor on what the
/// trainer will refuse, not the trainer's own check.
fn within_policy_block(
    ctx: &Context,
    projection: Projection,
    max_block: usize,
    split: bool,
) -> Projection {
    let counter = TokenCounter::for_model(&ctx.config().policy_base).ok();
    let tokens = |text: &str| {
        counter
            .as_ref()
            .map_or_else(|| splinter_data::chars_as_tokens(text), |c| c.count(text))
    };
    let size = |record: &splinter_data::Record| {
        tokens(&serde_json::to_string(&record.body).unwrap_or_default())
    };
    if split {
        projection.split_over_block(max_block, &size)
    } else {
        projection.within_block(max_block, &size)
    }
}

/// The terms a dataset made from `sources` comes under: the most restrictive
/// of each axis over every source. A source that states none, or that the
/// store does not hold, counts as unknown, so one unread licence among stated
/// ones fails closed; `None` when no source states any.
pub fn source_terms<'a>(
    ctx: &Context,
    sources: impl IntoIterator<Item = &'a SourceId>,
) -> Result<Option<Terms>, OrchestratorError> {
    let store = ctx.sources();
    let mut stated = Vec::new();
    for source in sources {
        stated.push(match store.get_source(source) {
            Ok(found) => found.terms,
            Err(StoreError::UnknownSource(_)) => None,
            Err(other) => return Err(other.into()),
        });
    }
    Ok(combine_stated(stated.iter().map(Option::as_ref)))
}

/// Stores `projection` as a dataset and records where it came from: the
/// experience it was projected from, with the database pinned as it is, so
/// whatever is trained on it can be traced back and what it read stays
/// readable. Storing the same projection again changes nothing.
pub fn store_dataset(
    ctx: &Context,
    projection: &Projection,
    export_only: bool,
) -> Result<StoredDataset, OrchestratorError> {
    crate::bench::admit_projection(
        ctx,
        projection
            .records
            .iter()
            .flat_map(|record| record.metadata.experiences.iter().cloned()),
    )?;
    if !export_only {
        TrainingCapabilities::BRAIN
            .require(projection.objective)
            .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    }
    let stored = ctx.datasets().put(projection, &BrainDatasetCheck)?;
    record_dataset_lineage(ctx, &stored)?;
    Ok(stored)
}

/// Records where the stored dataset `stored` came from, if that was not
/// recorded yet. It is made official by one commit and its lineage by a second,
/// so a crash between them leaves a dataset without; this repairs it, and
/// `train` calls it before spending any time.
pub fn record_dataset_lineage(
    ctx: &Context,
    stored: &StoredDataset,
) -> Result<(), OrchestratorError> {
    ctx.workspace().record_dataset(
        &stored.id.0,
        &DatasetLineage {
            recipe: to_json("dataset manifest", &stored.manifest)?,
            records: stored.manifest.counts.records as u64,
        },
        &stored.manifest.experiences,
    )?;
    Ok(())
}

/// The stored dataset `id` (or a unique prefix of it) names, verified.
pub fn resolve_dataset(ctx: &Context, id: &str) -> Result<StoredDataset, OrchestratorError> {
    let stored = ctx.datasets().list()?.into_iter().map(|d| d.0);
    let id = DatasetId(ids::resolve("dataset", id, stored)?);
    Ok(ctx.datasets().get(&id)?)
}

/// What `dataset export` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Exported {
    /// The dataset.
    pub dataset: DatasetId,
    /// The copy of its records.
    pub path: PathBuf,
    /// The copy of its manifest.
    pub manifest: PathBuf,
}

/// Copies the dataset `id` names, with its manifest, into `out`, as
/// `<view>-<id prefix>.jsonl`. An existing file of that name is replaced
/// only when it already holds the same bytes.
pub fn export(ctx: &Context, id: &str, out: &Path) -> Result<Exported, OrchestratorError> {
    let stored = resolve_dataset(ctx, id)?;
    std::fs::create_dir_all(out).map_err(io(out))?;
    let name = format!(
        "{}-{}.jsonl",
        stored.manifest.view,
        &stored.id.0.hex()[..12]
    );
    let path = out.join(name);
    let manifest = manifest_path(&path);
    put_once(
        &path,
        &std::fs::read(&stored.path).map_err(io(&stored.path))?,
    )?;
    let manifest_bytes =
        canonical_json(&stored.manifest).map_err(|source| OrchestratorError::Json {
            what: "dataset manifest".into(),
            source,
        })?;
    put_once(&manifest, &manifest_bytes)?;
    Ok(Exported {
        dataset: stored.id,
        path,
        manifest,
    })
}

/// Writes `bytes` to `to`, refusing to replace a different file.
fn put_once(to: &Path, bytes: &[u8]) -> Result<(), OrchestratorError> {
    match std::fs::read(to) {
        Ok(existing) if existing == bytes => return Ok(()),
        Ok(_) => {
            return Err(OrchestratorError::Refused(format!(
                "{} exists and holds something else; export into another directory",
                to.display()
            )))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(io(to)(e)),
    }
    std::fs::write(to, bytes).map_err(io(to))
}

/// How many examples a training run reads from the dataset at `path`: for a
/// chat dataset, the supervised assistant turns of its records (a dialogue is
/// trained one answer at a time); for any other, its lines.
///
/// # Errors
/// The file cannot be read or holds a line that is not JSON.
pub fn examples_in(path: &Path) -> std::io::Result<usize> {
    let text = std::fs::read_to_string(path)?;
    let mut examples = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let record: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        examples += match record.get("messages").and_then(serde_json::Value::as_array) {
            Some(messages) => messages
                .iter()
                .filter(|m| {
                    m.get("role").and_then(serde_json::Value::as_str) == Some("assistant")
                        && m.get("train") == Some(&serde_json::Value::Bool(true))
                })
                .count(),
            None => 1,
        };
    }
    Ok(examples)
}

/// The tokens, as the policy's own tokenizer counts them, of the supervised
/// answers of the chat dataset at `path`; characters at the fallback ratio
/// when the policy's tokenizer is not at hand.
///
/// # Errors
/// The file cannot be read or holds a line that is not JSON.
pub fn supervised_tokens_in(ctx: &Context, path: &Path) -> std::io::Result<usize> {
    let counter = TokenCounter::for_model(&ctx.config().policy_base).ok();
    let tokens = |text: &str| {
        counter
            .as_ref()
            .map_or_else(|| splinter_data::chars_as_tokens(text), |c| c.count(text))
    };
    let text = std::fs::read_to_string(path)?;
    let mut total = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let record: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let answers = record
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter(|m| m.get("train") == Some(&serde_json::Value::Bool(true)))
            .filter_map(|m| m.get("content").and_then(serde_json::Value::as_str));
        total += answers.map(&tokens).sum::<usize>();
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dialogue is trained one answer at a time, so a run is sized by its
    /// answers, not its records.
    #[test]
    fn a_dialogue_is_as_many_examples_as_it_has_supervised_answers() {
        let path =
            std::env::temp_dir().join(format!("splinter-examples-{}.jsonl", std::process::id()));
        let turn = |role: &str, train: bool| serde_json::json!({"role": role, "content": "x", "train": train});
        let dialogue = serde_json::json!({"messages": [
            turn("system", false), turn("user", false), turn("assistant", true),
            turn("user", false), turn("assistant", true), turn("assistant", false)
        ]});
        let single =
            serde_json::json!({"messages": [turn("user", false), turn("assistant", true)]});
        let pair = serde_json::json!({"prompt": "q", "chosen": "a", "rejected": "b"});
        std::fs::write(&path, format!("{dialogue}\n\n{single}\n{pair}\n")).unwrap();
        assert_eq!(examples_in(&path).unwrap(), 4);
        std::fs::remove_file(&path).unwrap();
    }
}
