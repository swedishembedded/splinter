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
use std::str::FromStr;

use serde::Serialize;
use splinter_core::annotation::Strength;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::canonical_json;
use splinter_core::experience::{ExperienceId, PrivilegedKind};
pub use splinter_data::Strip;
use splinter_data::{
    manifest_path, Corpus, Cpt, Critic, DecisionView, DenoiseView, Exclusion, Format, Fraction,
    Objective, OutcomeView, Preference, Projection, Retrieval, SftFinal, SftStep, StoredDataset,
    VerifierView, View,
};
use splinter_model::{BrainDatasetCheck, TrainingCapabilities};
use splinter_store::experiences::SetId;
use splinter_store::lineage::DatasetLineage;

use crate::grouping::assign_groups;
use crate::variants::refuse_variants;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};
use splinter_orchestrator::ids;
use splinter_orchestrator::runs::to_json;

/// The weakest decision a view counts when a command names none:
/// consistency admits critiques verified by their retry's outcome and
/// agreement among answers, and leaves judged verdicts to be asked for.
pub const DEFAULT_MIN_STRENGTH: Strength = Strength::Consistency;

/// The seed that picks which subjects keep their privileged context under
/// `--strip mix:F`, so the same dataset is built every time.
pub const MIX_SEED: u64 = 0;

/// The views a dataset can be built through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ViewName {
    /// A passed experience's instruction and final answer.
    SftFinal,
    /// Each action of a trajectory, in the context before it.
    SftStep,
    /// A task and candidate answer, and a verified critique of it.
    Critic,
    /// A chosen and a rejected answer to one task.
    Preference,
    /// A task and candidate answer, and `pass` or `fail`.
    Verifier,
    /// The passing action where a passing and a failed attempt part.
    Decision,
    /// An instruction and the source span it is grounded in.
    Retrieval,
    /// A whole trajectory and its derived reward.
    Outcome,
    /// A denoise task's corrupted passage and its original.
    Denoise,
    /// The raw text of source parts.
    Cpt,
}

impl ViewName {
    /// Every view, in the order `--help` lists them.
    pub const ALL: [Self; 10] = [
        Self::SftFinal,
        Self::SftStep,
        Self::Critic,
        Self::Preference,
        Self::Verifier,
        Self::Decision,
        Self::Retrieval,
        Self::Outcome,
        Self::Denoise,
        Self::Cpt,
    ];

    /// The name a command line uses.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SftFinal => "sft-final",
            Self::SftStep => "sft-step",
            Self::Critic => "critic",
            Self::Preference => "preference",
            Self::Verifier => "verifier",
            Self::Decision => "decision",
            Self::Retrieval => "retrieval",
            Self::Outcome => "outcome",
            Self::Denoise => "denoise",
            Self::Cpt => "cpt",
        }
    }

    fn takes_strip(self) -> bool {
        self != Self::Cpt
    }

    fn takes_min_strength(self) -> bool {
        !matches!(self, Self::Cpt | Self::Denoise)
    }
}

impl FromStr for ViewName {
    type Err = OrchestratorError;

    fn from_str(text: &str) -> Result<Self, OrchestratorError> {
        Self::ALL
            .into_iter()
            .find(|view| view.as_str() == text)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|v| v.as_str()).collect();
                OrchestratorError::Refused(format!(
                    "unknown view {text:?}; the views are {}",
                    names.join(", ")
                ))
            })
    }
}

/// A `--strip` policy: `all`, `keep:K,..` or `mix:F`.
pub fn parse_strip(text: &str) -> Result<Strip, OrchestratorError> {
    if text == "all" {
        return Ok(Strip::All);
    }
    if let Some(kinds) = text.strip_prefix("keep:") {
        let kinds = kinds
            .split(',')
            .map(privileged_kind)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Strip::Keep(kinds));
    }
    if let Some(fraction) = text.strip_prefix("mix:") {
        let value: f64 = fraction.parse().map_err(|_| {
            OrchestratorError::Refused(format!("mix:{fraction} needs a fraction in [0, 1]"))
        })?;
        let keep_fraction = Fraction::new(value)
            .map_err(|e| OrchestratorError::Refused(format!("mix:{fraction}: {e}")))?;
        return Ok(Strip::Mix {
            keep_fraction,
            seed: MIX_SEED,
        });
    }
    Err(OrchestratorError::Refused(format!(
        "{text:?} is not a strip policy: all, keep:<kind>,.. or mix:<fraction>"
    )))
}

/// A privileged kind as `keep:` names it: `passage`, `hint`, `critique`,
/// `reference`, `oracle`, or `other:<name>`.
fn privileged_kind(name: &str) -> Result<PrivilegedKind, OrchestratorError> {
    Ok(match name {
        "passage" => PrivilegedKind::Passage,
        "hint" => PrivilegedKind::Hint,
        "critique" => PrivilegedKind::Critique,
        "reference" => PrivilegedKind::Reference,
        "oracle" => PrivilegedKind::Oracle,
        other => match other.strip_prefix("other:").filter(|n| !n.is_empty()) {
            Some(name) => PrivilegedKind::Other(name.to_string()),
            None => {
                return Err(OrchestratorError::Refused(format!(
                    "unknown privileged kind {name:?}: passage, hint, critique, reference, \
                     oracle or other:<name>"
                )))
            }
        },
    })
}

/// A `--min-strength`: `executable`, `formal`, `consistency` or `judged`.
pub fn parse_strength(text: &str) -> Result<Strength, OrchestratorError> {
    Ok(match text {
        "executable" => Strength::Executable,
        "formal" => Strength::Formal,
        "consistency" => Strength::Consistency,
        "judged" => Strength::Judged,
        _ => {
            return Err(OrchestratorError::Refused(format!(
                "{text:?} is not a strength: executable, formal, consistency or judged"
            )))
        }
    })
}

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
        }
    }
}

/// Projects `request.sets` through `request.view` and stores the dataset.
pub fn build(ctx: &Context, request: &BuildRequest) -> Result<Built, OrchestratorError> {
    let view = request.view;
    if request.strip.is_some() && !view.takes_strip() {
        return Err(OrchestratorError::Refused(format!(
            "view {} shows the student no task, so --strip does not apply",
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
            if !ids.contains(&member) {
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
    for source in sources {
        corpus.add_source(source);
    }
    let strength = request.min_strength.unwrap_or(DEFAULT_MIN_STRENGTH);
    let strip = request.strip.clone().unwrap_or_default();
    let source_store = ctx.sources();
    let mut projection = match view {
        ViewName::SftFinal => SftFinal::new(strength).with_strip(strip).project(&corpus),
        ViewName::SftStep => SftStep::new(strength).with_strip(strip).project(&corpus),
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
        ViewName::Cpt => Cpt::new(&source_store).project(&corpus),
    }?;
    if let Some(prompt) = &request.system_prompt {
        projection = projection.with_system_prompt(prompt);
    }
    assign_groups(ctx, &corpus, &mut projection)?;
    let stored = store_dataset(ctx, &projection, request.export_only)?;
    Ok(Built::from(stored))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_policies_and_strengths_parse_or_are_refused() {
        assert_eq!(parse_strip("all").ok(), Some(Strip::All));
        assert_eq!(
            parse_strip("keep:hint,other:executable-check").ok(),
            Some(Strip::Keep(vec![
                PrivilegedKind::Hint,
                PrivilegedKind::Other("executable-check".into())
            ]))
        );
        assert!(matches!(parse_strip("mix:0.25"), Ok(Strip::Mix { .. })));
        for refused in ["none", "keep:hnt", "mix:2", "mix:x", "keep:other:"] {
            assert!(parse_strip(refused).is_err(), "{refused}");
        }
        assert_eq!(parse_strength("formal").ok(), Some(Strength::Formal));
        assert!(parse_strength("strong").is_err());
        for view in ViewName::ALL {
            assert_eq!(view.as_str().parse::<ViewName>().ok(), Some(view));
        }
        assert!("sft".parse::<ViewName>().is_err());
    }
}
