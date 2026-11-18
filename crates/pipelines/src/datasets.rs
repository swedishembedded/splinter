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
use splinter_core::dataset::DatasetId;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::{Environment, ExperienceId};
use splinter_core::model_ref::ModelRef;
use splinter_core::source::SourceId;
use splinter_core::terms::{combine_stated, Terms};
pub use splinter_data::curation::{Curation, HistoryCuration};
pub use splinter_data::holdout::SplitPolicy;
use splinter_data::{
    manifest_path, Corpus, Cpt, Critic, DecisionView, DenoiseView, Distill, Exclusion, Format,
    Objective, OutcomeView, PolicyGradient, Preference, Projection, RecordBody, Rehearsal,
    Retrieval, SftFinal, SftStep, StoredDataset, VerifierView, View, Voice,
    DEFAULT_MAX_FAMILY_SHARE,
};
pub use splinter_data::{AdvantageKind, BlockMeasure, GroupKind, GroupReport, RewardShape};
pub use splinter_data::{Control, Strip};
use splinter_knowledge::sections::sections;
use splinter_model::TrainingCapabilities;
use splinter_store::error::StoreError;
use splinter_store::lineage::DatasetLineage;

mod check;
mod request;

pub use check::{check, DatasetCheck};
pub use request::{BuildRequest, VoiceBuild};

use crate::describe::{describe, Described};
use crate::exam_seal;
use crate::grouping::{assign_draws, assign_groups, part_families};
use crate::raft::{with_passages, Abstainer, PassageShare};
use crate::retrieval::Retrieval as PassageSearch;
use crate::variants::refuse_variants;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};
use splinter_orchestrator::ids;
use splinter_orchestrator::runs::to_json;

pub use crate::dataset_views::{
    parse_advantage, parse_reward_metric, parse_strength, parse_strip, StripPolicy, ViewName,
    DEFAULT_MIN_STRENGTH,
};
pub use crate::rl_build::{RlBuild, DEFAULT_FAIL_CEILING};

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
    /// The groups of attempts a policy-gradient dataset compared: each
    /// attempt's reward and advantage.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupReport>,
    /// What `--max-block` measured records by, when it was given: the
    /// trainer's own count, or the floor that stands in where that is
    /// unavailable, with the reason. Not stored with the dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_measure: Option<BlockMeasure>,
    /// How the dataset's held-out and monitoring sets are drawn: the seed and
    /// the records of each stratum, as its manifest records them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split: Option<SplitPolicy>,
    /// The curation policy the records were built under, as the manifest
    /// records it; what it left out is counted in `excluded`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curation: Option<Curation>,
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
            groups: stored.manifest.groups,
            block_measure: None,
            split: stored.manifest.split,
            curation: stored.manifest.curation,
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
        block_measure: projection.block_measure.clone(),
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
        block_measure: projection.block_measure.clone(),
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
    if request.history != HistoryCuration::default() && view != ViewName::SftStep {
        return Err(OrchestratorError::Refused(format!(
            "view {} imitates no demonstration, so --repository-history does not apply",
            view.as_str()
        )));
    }
    if request.control != Control::Verified && view != ViewName::SftStep {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no negative controls, so --control does not apply",
            view.as_str()
        )));
    }
    if request.serialise_calls
        && !matches!(
            view,
            ViewName::SftStep | ViewName::PolicyGradient | ViewName::SelfDistill
        )
    {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no run to take apart, so --serialise-calls does not apply",
            view.as_str()
        )));
    }
    let trains_actions = matches!(
        view,
        ViewName::SftStep | ViewName::PolicyGradient | ViewName::SelfDistill
    );
    if request.images && !trains_actions {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no conversations that were shown images, so --images does not apply",
            view.as_str()
        )));
    }
    let rewards_attempts = matches!(view, ViewName::PolicyGradient | ViewName::SelfDistill);
    if !request.rl.is_default() && !rewards_attempts {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no rewards or advantages, so --reward-metric, --fail-ceiling, \
             --advantage and --cliff-distill do not apply",
            view.as_str()
        )));
    }
    if request.rl.cliff_distill && view != ViewName::PolicyGradient {
        return Err(OrchestratorError::Refused(
            "--cliff-distill distils the all-fail groups of the policy-gradient view; the \
             self-distill view distils every attempt already"
                .into(),
        ));
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
            .only_roles(request.roles.clone().unwrap_or_default())
            .images(request.images)
            .history(request.history)
            .project(&corpus),
        ViewName::PolicyGradient | ViewName::SelfDistill => PolicyGradient::new(strength)
            .distilling(match (view, request.rl.cliff_distill) {
                (ViewName::SelfDistill, _) => Distill::Every,
                (_, true) => Distill::Cliff,
                (_, false) => Distill::Off,
            })
            .with_strip(strip)
            .rewards(request.rl.shape()?)
            .advantage(request.rl.advantage)
            .serialise_calls(request.serialise_calls)
            .only_actions(request.only_actions.clone().unwrap_or_default())
            .only_roles(request.roles.clone().unwrap_or_default())
            .images(request.images)
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
            // The stretches are measured in characters at a ratio that holds
            // for text that tokenizes badly: no tokenizer runs here, and the
            // row a run trains is sized by the trainer's own count when the
            // dataset is checked before training.
            let tokens = splinter_data::chars_as_tokens;
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
    // A benchmark trial keeps the system prompt its agent was served, which
    // varies with the repository's commit; the prompt a model is asked under is
    // whatever its agent builds, so nothing is settled for it.
    let own_prompts = corpus
        .entries()
        .iter()
        .all(|entry| entry.experience.environment.kind == Environment::BENCH);
    if trains_actions && !own_prompts {
        // The prompt the runs were made under is the prompt the dataset trains
        // under, so a model trained on it is asked under it.
        projection = projection
            .settled_system_prompt()
            .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    }
    if let Some(max_block) = request.max_block {
        let base = crate::train::trained_base(
            ctx,
            request.base.as_ref().unwrap_or(&ModelRef::policy_default()),
        )?;
        projection = crate::block::within_policy_block(
            ctx,
            &base,
            projection,
            max_block,
            request.split_over_block,
        );
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
    let projection = assign_draws(&corpus, projection, request.split_seed);
    Ok((projection, described, corpus))
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
    let stored = ctx.datasets().put(projection, &splinter_data::Unchecked)?;
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

/// The tokens of the supervised answers of the chat dataset at `path`, as the
/// brain service's trainer counts them for the policy base (each answer with
/// the end of its turn); characters at the fallback ratio when no service can
/// be asked.
///
/// # Errors
/// The file cannot be read, or the service refused the dataset.
pub fn supervised_tokens_in(ctx: &Context, path: &Path) -> std::io::Result<usize> {
    let failed = |e: splinter_model::PolicyError| std::io::Error::other(e.to_string());
    if let Ok(brain) = ctx.brain() {
        let base = splinter_model::selection::local_model_name(&ctx.config().policy_base);
        let mut spec = splinter_model::served_train::CheckSpec::chat(&base, false, false);
        spec.per_record = false;
        let report =
            splinter_model::served_train::check_dataset(&brain, path, &spec).map_err(failed)?;
        report.refuse_invalid(path).map_err(failed)?;
        return Ok(usize::try_from(report.supervised_tokens.total).unwrap_or(usize::MAX));
    }
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
        total += answers.map(splinter_data::chars_as_tokens).sum::<usize>();
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
