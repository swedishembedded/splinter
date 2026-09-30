// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The generator: a source part's sections shown to a generator model,
//! closed-book, and its proposals admitted by code.
//!
//! The model is asked through [`splinter_agent::solve::solve`] with a
//! closed-book task whose instruction is the prompt, so it runs exactly as
//! a solver or a judge does: no tools, bounded by a deadline and an
//! optional output-token budget. Its reply is parsed strictly (see the
//! shape in the prompt); a reply that does not parse is rejected whole,
//! with no second attempt.

use std::sync::Arc;

use splinter_agent::solve::{solve, SolveError, SolveOptions};
use splinter_lab::verifiers::mutation::MutationPolicy;
use splinter_sandbox::{ResolvedEnvironment, RuntimeEnvironment};
use splinter_store::digest::Digest;
use splinter_store::error::StoreError;
use splinter_store::experience::{Environment, ExperienceError, Task};
use splinter_store::source::{PartRef, SourceId};
use splinter_store::sources::SourceStore;
use sven_sdk::model::ModelProvider;

use super::admit::{Admission, Proposal};
use super::kind::{KindError, Material, SolverEnvironment, TaskKind};
use super::reply::SHAPE;
use super::{GenerationPolicy, GenerationReport, GENERATOR};
use crate::sections::{sections, Section};

/// The task kind of the closed-book request a generator model is sent.
pub const GENERATION_TASK_KIND: &str = "generate-tasks";

/// One text part of a stored source, split into the sections a generator
/// model is shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceText {
    pub(crate) part: PartRef,
    pub(crate) content: Digest,
    pub(crate) text: String,
    pub(crate) sections: Vec<Section>,
}

impl SourceText {
    /// Part `part` of source `source`, read verified from `store` and split
    /// by [`sections`] at its media type.
    pub fn load(store: &SourceStore, source: &SourceId, part: &str) -> Result<Self, GenerateError> {
        let stored = store.get_source(source)?;
        let found = stored.part(part).ok_or_else(|| {
            GenerateError::Store(StoreError::UnknownPart {
                source_id: source.clone(),
                part: part.to_string(),
            })
        })?;
        let bytes = store.read_blob(&found.content)?;
        let text = String::from_utf8(bytes).map_err(|_| GenerateError::NotText {
            source_id: source.clone(),
            part: part.to_string(),
        })?;
        let split = sections(&text, &found.media_type);
        Ok(Self {
            part: PartRef {
                source: source.clone(),
                name: part.to_string(),
            },
            content: found.content.clone(),
            text,
            sections: split,
        })
    }

    /// Only the sections at `positions` (in the order given), so a batch
    /// can cover part of a large source; refused for a position past the
    /// last section.
    pub fn select(mut self, positions: &[usize]) -> Result<Self, GenerateError> {
        let mut chosen = Vec::with_capacity(positions.len());
        for &position in positions {
            let section = self
                .sections
                .get(position)
                .ok_or(GenerateError::NoSection {
                    position,
                    sections: self.sections.len(),
                })?;
            chosen.push(section.clone());
        }
        self.sections = chosen;
        Ok(self)
    }

    /// The part the text is.
    #[must_use]
    pub fn part(&self) -> &PartRef {
        &self.part
    }

    /// The sections the model is shown, by position.
    #[must_use]
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// The text of the section at `position`.
    pub(crate) fn section_text(&self, position: usize) -> Option<&str> {
        self.sections.get(position)?.text(&self.text)
    }
}

/// Why a batch could not be generated. A model's bad proposal is a
/// [`super::Rejection`] in the report, never an error.
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    /// A generation parameter is out of range.
    #[error("invalid generation parameter {name}: {reason}")]
    Parameter {
        /// The parameter.
        name: &'static str,
        /// Why it is refused.
        reason: String,
    },
    /// A kind is not consistent.
    #[error(transparent)]
    Kind(#[from] KindError),
    /// A kind runs code in a runtime the generator was not offered.
    #[error("task kind {kind:?} needs runtime {runtime:?}, which the generator was not offered")]
    RuntimeNotOffered {
        /// The kind.
        kind: String,
        /// The runtime it names.
        runtime: String,
    },
    /// The source part is not UTF-8 text.
    #[error("part {part:?} of {source_id} is not UTF-8 text")]
    NotText {
        /// The source.
        source_id: SourceId,
        /// The part.
        part: String,
    },
    /// A section position past the last section.
    #[error("no section at position {position}; the text has {sections}")]
    NoSection {
        /// The position asked for.
        position: usize,
        /// Sections the text has.
        sections: usize,
    },
    /// The text has no sections to generate from.
    #[error("the text has no sections")]
    NoSections,
    /// The source store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// The request task could not be built.
    #[error(transparent)]
    Task(#[from] ExperienceError),
    /// The generator model could not be run.
    #[error("the generator model could not run: {0}")]
    Model(#[from] SolveError),
    /// A sandbox or verifier failed while admitting a task.
    #[error("admitting a task failed: {0}")]
    Admission(#[from] splinter_lab::verifiers::VerifyError),
    /// The blocking admission step did not complete.
    #[error("the admission step did not complete: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Generates tasks from sources with a generator model; see the module
/// documentation.
#[derive(Clone)]
pub struct ModelTaskGenerator {
    model: Arc<dyn ModelProvider>,
    identity: String,
    store: SourceStore,
    runtimes: Vec<RuntimeEnvironment>,
    policy: GenerationPolicy,
    mutation: MutationPolicy,
}

impl ModelTaskGenerator {
    /// A generator on `model`, whose identity is `identity` (the string an
    /// experience's provenance records for that model), resolving evidence
    /// through `store`, with the default policies and no runtime.
    #[must_use]
    pub fn new(
        model: Arc<dyn ModelProvider>,
        identity: impl Into<String>,
        store: SourceStore,
    ) -> Self {
        Self {
            model,
            identity: identity.into(),
            store,
            runtimes: Vec::new(),
            policy: GenerationPolicy::default(),
            mutation: MutationPolicy::default(),
        }
    }

    /// The runtimes code kinds run in: checks, generated tests and shown
    /// code are run only here.
    #[must_use]
    pub fn with_runtimes(mut self, runtimes: Vec<RuntimeEnvironment>) -> Self {
        self.runtimes = runtimes;
        self
    }

    /// The generation parameters.
    #[must_use]
    pub fn with_policy(mut self, policy: GenerationPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// How generated tests are validated by mutation.
    #[must_use]
    pub fn with_mutation(mut self, mutation: MutationPolicy) -> Self {
        self.mutation = mutation;
        self
    }

    /// The name admitted tasks record as their generator.
    #[must_use]
    pub fn generator_name(&self) -> String {
        format!("{GENERATOR}:{}", self.identity)
    }

    /// One batch: a request per entry of `kinds`, each for tasks of that
    /// kind from `source`'s sections, and every proposal admitted or
    /// rejected. The requests run in order; admission runs on a blocking
    /// thread, since it runs code.
    pub async fn generate(
        &self,
        source: &SourceText,
        kinds: &[&TaskKind],
    ) -> Result<GenerationReport, GenerateError> {
        self.policy.validate()?;
        if self.identity.trim().is_empty() {
            return Err(GenerateError::Parameter {
                name: "identity",
                reason: "the generator model needs an identity".into(),
            });
        }
        if source.sections.is_empty() {
            return Err(GenerateError::NoSections);
        }
        let mut proposals = Vec::with_capacity(kinds.len());
        for kind in kinds {
            kind.validate()?;
            let runtime = self.runtime_for(kind)?;
            let prompt = prompt(source, kind, self.policy.tasks_per_request);
            let reply = self.request(&prompt).await?;
            proposals.push(Proposal {
                kind: (*kind).clone(),
                runtime,
                prompt: Digest::of(prompt.as_bytes()),
                reply,
            });
        }
        let admission = Admission {
            store: self.store.clone(),
            policy: self.policy,
            mutation: self.mutation,
            generator: self.generator_name(),
        };
        let source = source.clone();
        tokio::task::spawn_blocking(move || admission.admit(&source, proposals)).await?
    }

    /// The offered runtime `kind` runs its code in, if it runs any.
    fn runtime_for(&self, kind: &TaskKind) -> Result<Option<RuntimeEnvironment>, GenerateError> {
        let runs_code = kind.answer.is_computed()
            || kind.environment == SolverEnvironment::Runtime
            || kind.requires.iter().any(|m| {
                matches!(
                    m,
                    Material::Checks | Material::Tests | Material::ChecksOrTests
                )
            });
        let Some(name) = kind.runtime.as_deref().filter(|_| runs_code) else {
            return Ok(None);
        };
        self.runtimes
            .iter()
            .find(|env| env.runtime().spec.name == name)
            .cloned()
            .map(Some)
            .ok_or_else(|| GenerateError::RuntimeNotOffered {
                kind: kind.name.clone(),
                runtime: name.to_string(),
            })
    }

    /// The model's reply to `prompt`, or `None` when its run ended without
    /// one.
    async fn request(&self, prompt: &str) -> Result<Option<String>, GenerateError> {
        let request = Task::new(
            GENERATION_TASK_KIND,
            vec![],
            Environment::closed_book(),
            prompt,
            vec![],
        )?;
        let mut options = SolveOptions::new(self.policy.deadline);
        options.max_output_tokens = self.policy.max_output_tokens;
        let solution = solve(
            &request,
            &ResolvedEnvironment::ClosedBook,
            self.model.clone(),
            options,
        )
        .await?;
        Ok(solution.final_output)
    }
}

/// The prompt for tasks of `kind` from `source`: the kind's brief, the
/// rules every task is held to, the reply's shape, and the sections by
/// position.
fn prompt(source: &SourceText, kind: &TaskKind, count: usize) -> String {
    let runtime = kind.runtime.as_deref().unwrap_or("-");
    let brief = kind
        .brief
        .replace("{count}", &count.to_string())
        .replace("{runtime}", runtime);
    let mut rules = vec![
        format!("Write at most {count} tasks."),
        "Each instruction is everything the student sees: it must stand on its own. Never \
         refer to the sections, a passage, a document or text the student is not shown, and do \
         not copy long runs of the sections into it."
            .to_string(),
        "Cite the evidence for each task: the position of each section used and, where the \
         answer rests on part of it, a quote copied exactly from that section."
            .to_string(),
    ];
    if kind.min_sections > 1 {
        rules.push(format!(
            "Each task must cite at least {} different sections.",
            kind.min_sections
        ));
    }
    if kind.shows_material {
        rules.push(
            "Put the material the task is about in `material`, and include it verbatim in the \
             instruction."
                .into(),
        );
    } else {
        rules.push("Leave `material` null.".into());
    }
    for material in &kind.requires {
        rules.push(match material {
            Material::Hints => "Give at least one hint in `hints`.".to_string(),
            Material::Checks => format!(
                "Give at least one check in `checks`: {runtime} code appended to the answer's \
                 code that exits 0 (or prints `stdout`) only when the answer is correct."
            ),
            Material::Tests => format!(
                "Give at least one test in `tests`: {runtime} code appended to the answer's \
                 code that exits 0 only when the answer is correct, and fails on small mistakes."
            ),
            Material::ChecksOrTests => format!(
                "Give at least one check in `checks` or test in `tests`: {runtime} code \
                 appended to the answer's code that exits 0 only when the answer is correct."
            ),
        });
    }
    let mut out = format!(
        "You write training tasks of the kind {:?}.\n\n{brief}\n\nRules:\n",
        kind.name
    );
    for rule in rules {
        out.push_str("- ");
        out.push_str(&rule);
        out.push('\n');
    }
    out.push_str(
        "\nReply with EXACTLY one JSON object and nothing else - no prose, no code fences:\n",
    );
    out.push_str(SHAPE);
    out.push_str("\n\nSECTIONS:\n");
    for position in 0..source.sections.len() {
        let text = source.section_text(position).unwrap_or_default();
        out.push_str(&format!("\n[{position}]\n{text}\n"));
    }
    out
}
