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
//! The model is asked through a typed sven call ([`Engine::call_with`])
//! with no tools, bounded by a deadline and an optional output-token
//! budget that cover every attempt: the kind's brief and rules are the
//! call's task, followed by [`Reply`]'s schema as the reply's shape
//! (shown in the prompt, since a local model cannot be constrained to
//! it), and its input the source's identity ([`SourceIdentity`]) and the
//! sections. sven reads the reply as that shape and sends one that is not
//! back for correction, up to `repairs` times; a reply still malformed
//! after that is rejected whole.
//!
//! The identity is shown because a question is only worth training on when
//! it names what it is about: the sections of a window often never say
//! which product or document they belong to, and "what baud rate does the
//! console run at?" has a different answer for every board.

use serde::Serialize;
use splinter_agent::solve::{Model, SolveOptions};
use splinter_core::digest::Digest;
use splinter_core::experience::ExperienceError;
use splinter_core::source::{Origin, PartRef, SourceId};
use splinter_lab::verifiers::mutation::MutationPolicy;
use splinter_record::error::StoreError;
use splinter_record::sources::SourceStore;
use splinter_sandbox::RuntimeEnvironment;
use sven_sdk::{CallError, CancelToken, Engine, Method, Toolset};

use super::admit::{Admission, Proposal, Refusal};
use super::kind::{KindError, Material, SolverEnvironment, TaskKind};
use super::reply::{Reply, REPLY_EXAMPLE};
use super::{GenerationPolicy, GenerationReport, Rejection, GENERATOR};
use crate::sections::{sections, title, Section};

/// The name of the typed call a generator model is sent.
pub const GENERATION_METHOD: &str = "generate_tasks";

/// The role the generator model is given.
const ROLE: &str = "You write training tasks from source material, exactly as briefed.";

/// What the generator model is shown: what the source is, and the
/// sections, by position.
#[derive(Serialize)]
struct Sections<'a> {
    source: &'a SourceIdentity,
    sections: Vec<ShownSection<'a>>,
}

/// What a source is, as far as it records it: the names a task's subject
/// may come from when the sections do not name one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SourceIdentity {
    /// `document`, `repository` or `command`.
    pub kind: &'static str,
    /// A document's file name, a repository's directory name, or the
    /// command line that was run.
    pub name: String,
    /// The part the sections are from: a file's path in the source, or an
    /// output stream.
    pub part: String,
    /// The part's title: its first Markdown heading, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The version the source records: a repository's commit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl SourceIdentity {
    /// `origin`'s identity for its part `part`, whose text is `text` of
    /// media type `media_type`.
    #[must_use]
    pub fn of(origin: &Origin, part: &str, text: &str, media_type: &str) -> Self {
        let file_name = |path: &str| {
            std::path::Path::new(path)
                .file_name()
                .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned())
        };
        let (name, version) = match origin {
            Origin::Document { path } => (file_name(path), None),
            Origin::Repository { path, revision, .. } => (
                file_name(path),
                revision.as_ref().and_then(|r| r.commit.clone()),
            ),
            Origin::Command { argv, .. } => (argv.join(" "), None),
        };
        Self {
            kind: origin.kind(),
            name,
            part: part.to_string(),
            title: title(text, media_type),
            version,
        }
    }

    /// The identity in one line: the name, the part when it is not the
    /// name, and the version (`manual.md`, `` `tool --help` stdout ``,
    /// `quark-q1 guide/manual.md at 3f2a9c1`).
    #[must_use]
    pub fn label(&self) -> String {
        let mut label = match self.kind {
            "command" => format!("`{}`", self.name),
            _ => self.name.clone(),
        };
        if self.part != self.name {
            label.push(' ');
            label.push_str(&self.part);
        }
        if let Some(version) = &self.version {
            label.push_str(" at ");
            label.push_str(version);
        }
        label
    }

    /// Every name the identity holds.
    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        [Some(self.name.as_str()), Some(self.part.as_str())]
            .into_iter()
            .chain([self.title.as_deref(), self.version.as_deref()])
            .flatten()
    }
}

#[derive(Serialize)]
struct ShownSection<'a> {
    position: usize,
    text: &'a str,
}

/// One text part of a stored source, split into the sections a generator
/// model is shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceText {
    pub(crate) part: PartRef,
    pub(crate) identity: SourceIdentity,
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
            identity: SourceIdentity::of(&stored.origin, part, &text, &found.media_type),
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

    /// What the source is, as the generator model is shown it.
    #[must_use]
    pub fn identity(&self) -> &SourceIdentity {
        &self.identity
    }

    /// The sections the model is shown, by position.
    #[must_use]
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// The text of the section at `position`.
    pub fn section_text(&self, position: usize) -> Option<&str> {
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
    /// A task cannot be varied; see [`super::can_vary`].
    #[error("task {task} cannot be varied: {reason}")]
    NotVaryable {
        /// The task.
        task: Digest,
        /// Why not.
        reason: &'static str,
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
    Model(#[from] CallError),
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
    pub(super) model: Model,
    pub(super) store: SourceStore,
    runtimes: Vec<RuntimeEnvironment>,
    pub(super) policy: GenerationPolicy,
    pub(super) mutation: MutationPolicy,
    cancel: Option<CancelToken>,
}

impl ModelTaskGenerator {
    /// A generator on `model`, known by its identity (the string an
    /// experience's provenance records for that model), resolving evidence
    /// through `store`, with the default policies and no runtime.
    #[must_use]
    pub fn new(model: Model, store: SourceStore) -> Self {
        Self {
            model,
            store,
            runtimes: Vec::new(),
            policy: GenerationPolicy::default(),
            mutation: MutationPolicy::default(),
            cancel: None,
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

    /// Stops the request in progress, and every later one, once `cancel`
    /// is cancelled: a stopped request has no reply, rejected as
    /// [`super::Rejection::NoReply`].
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// The name admitted tasks record as their generator.
    #[must_use]
    pub fn generator_name(&self) -> String {
        format!("{GENERATOR}:{}", self.model.identity)
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
        if self.model.identity.trim().is_empty() {
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
            let brief = brief(kind, self.policy.tasks_per_request);
            let shown = shown(source);
            let reply = self.request(GENERATION_METHOD, &brief, &shown).await?;
            proposals.push(Proposal {
                kind: (*kind).clone(),
                runtime,
                prompt: prompt_digest(&brief, &shown),
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

    /// The model's reply to `brief` over `shown`, asked as the typed call
    /// `method`: the tasks it proposed, or why there are none - no reply before a bound stopped the call,
    /// or a reply still malformed after correction.
    pub(super) async fn request<I: Serialize + ?Sized>(
        &self,
        method: &str,
        brief: &str,
        shown: &I,
    ) -> Result<Result<Reply, Refusal>, GenerateError> {
        let mut options = SolveOptions::new(self.policy.deadline);
        options.max_output_tokens = self.policy.max_output_tokens;
        options.cancel = self.cancel.clone();
        options.stream_idle = self.model.stream_idle;
        let method = Method::<Reply>::new(method)
            .task(brief)
            .role(ROLE)
            .max_repairs(self.policy.repairs);
        let engine = Engine::builder()
            .config(options.engine_config())
            .model_provider(options.provider(self.model.provider.clone()))
            .toolset(Toolset::none())
            .build()?;
        match engine
            .call_with(&method, shown, options.run_options())
            .await
        {
            Ok(reply) => Ok(Ok(reply)),
            Err(CallError::Invalid {
                attempts,
                detail,
                last,
                ..
            }) => Ok(Err((
                Rejection::Malformed,
                format!("{detail} (after {attempts} attempt(s)); last reply: {last}"),
            ))),
            Err(CallError::Stopped { conclusion }) => Ok(Err((
                Rejection::NoReply,
                format!("the call ended without a reply: {conclusion:?}"),
            ))),
            Err(e) => Err(e.into()),
        }
    }
}

/// The identity and sections of `source`, by position, as the model is
/// shown them.
fn shown(source: &SourceText) -> Sections<'_> {
    Sections {
        source: &source.identity,
        sections: (0..source.sections.len())
            .map(|position| ShownSection {
                position,
                text: source.section_text(position).unwrap_or_default(),
            })
            .collect(),
    }
}

/// The digest of what one request showed the model: the brief and its
/// input (the source's identity and the sections, or the task to vary).
pub(super) fn prompt_digest<I: Serialize + ?Sized>(brief: &str, shown: &I) -> Digest {
    let sections = serde_json::to_string(shown).unwrap_or_else(|e| {
        // Integers and strings always serialize.
        unreachable!("sections serialize: {e}")
    });
    Digest::of(format!("{brief}\n{sections}").as_bytes())
}

/// The brief for tasks of `kind`: the kind's own brief and the rules every
/// task is held to. The sections are the call's input; the reply's shape
/// is its return type.
fn brief(kind: &TaskKind, count: usize) -> String {
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
        "Cite the evidence for each task: the position of each section the answer comes from."
            .to_string(),
    ];
    if kind.names_subject() {
        rules.push(
            "Every instruction names its subject: the specific product, document, tool, \
             component or version it is about, as the sections or the `source` name it. Put \
             that name in `subject`, written exactly as the instruction writes it. Someone who \
             has never seen the source must get exactly one answer: a question whose answer \
             would differ for another product or version is wrong."
                .to_string(),
        );
    }
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
    out.push_str("\nA reply looks like this:\n");
    out.push_str(REPLY_EXAMPLE);
    out.push('\n');
    out
}
