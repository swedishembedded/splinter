// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn documents into tasks a
// model is examined and trained on, each grounded and checked by code, for
// its clients. If your team needs expertise in task synthesis you can trust,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Closed-book tasks from documents, and their answers, run in process.
//!
//! * `task.generate` asks a generator model for tasks of the named kinds
//!   from each document ([`splinter_knowledge::tasks::ModelTaskGenerator`],
//!   with the document captured into a source store of the computation's
//!   own); code admits a task only when its evidence quotes the document
//!   verbatim, it stands on its own, names its subject and is new
//!   ([`splinter_knowledge::tasks`]). Each admitted task is an item
//!   (`task.json`), holding the document its evidence is in as a source
//!   store of its own (`sources/`), so the item is graded against it
//!   wherever it goes; `rejected.jsonl` keeps every refusal with its reason.
//!   A kind that runs code is offered the runtime it names, found on the
//!   default search path, in the process sandbox.
//! * `answer.collect`, run once per task, has the stage's model answer it
//!   closed-book through the one solve every task runs through
//!   ([`splinter_agent::solve::solve`]) and grades the answer with the
//!   verifiers the task's kind names
//!   ([`splinter_knowledge::tasks::verifiers`]): the same verdicts, under
//!   the same producer names, as everywhere else. A quotation or grounding
//!   check reads the item's own `sources/`; a code check, and the answer
//!   of a kind solved in its runtime, runs in the process sandbox. A judge
//!   grades only where one is given, and none is here; a kind solved in a
//!   workspace, or graded by a check whose evidence or runtime is not
//!   there, is rejected by name. The item holds each attempt's
//!   `experience-N.json` and `verdicts-N.json`, and the task's `sources/`,
//!   so the answers carry the evidence they were graded against.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_agent::proposer::SvenProposer;
use splinter_agent::solve::{solve, SolveOptions};
use splinter_core::clock::{Clock, SystemClock};
use splinter_core::experience::{Provenance, Task};
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::items::{item_id, Item, ItemIndex, ITEMS_FILE};
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_eval::verifiers::Strongest;
use splinter_eval::verifiers::Verifier;
use splinter_knowledge::tasks::verifiers::{evidence_verifier, plain_verifier, runtime_verifier};
use splinter_knowledge::tasks::{
    Catalogue, GenerationPolicy, ModelTaskGenerator, SolverEnvironment, SourceText, TaskKind,
    VerifierKind,
};
use splinter_sandbox::environment::{ResolvedEnvironment, RUNTIME_KIND_PREFIX};
use splinter_sandbox::runtime::RuntimeRegistry;
use splinter_sandbox::{Limits, ProcessSandbox, RuntimeEnvironment, Sandbox};
use splinter_store::sources::SourceStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use crate::execute::{Call, Execute, Outcome};
use crate::registry::OperationSpec;

/// `task.generate`.
pub const GENERATE: &str = "task.generate";
/// `answer.collect`.
pub const ANSWER: &str = "answer.collect";

/// An admitted task in its item.
const TASK_FILE: &str = "task.json";

/// The source store, in a task's item, its evidence is in.
const SOURCES_DIR: &str = "sources";

/// The source store at `dir`.
fn source_store(dir: &Path) -> SourceStore {
    SourceStore::new(&Workspace::at(&StateRoot::new(dir.to_path_buf())))
}

/// Those of the runtimes `names` installed here, resolved in a process
/// sandbox under `scratch` that finds interpreters on the default search
/// path.
fn runtimes<'a>(
    scratch: &Path,
    names: impl IntoIterator<Item = &'a str>,
) -> Vec<RuntimeEnvironment> {
    let sandbox: Arc<dyn Sandbox> = Arc::new(ProcessSandbox::new(
        scratch.join("calls"),
        Vec::new(),
        &[],
        Limits::default(),
    ));
    let registry = RuntimeRegistry::builtin();
    let mut found: Vec<RuntimeEnvironment> = Vec::new();
    for name in names {
        if found.iter().all(|env| env.runtime().spec.name != name) {
            found.extend(RuntimeEnvironment::new(&registry, name, Arc::clone(&sandbox)).ok());
        }
    }
    found
}

fn port(name: &str, kind: ArtifactKind, many: bool) -> Port {
    Port {
        name: name.into(),
        kind,
        many,
        optional: false,
    }
}

fn write(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())
}

/// `task.generate`'s parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerateParams {
    /// The task kinds asked for, by name in the built-in catalogue.
    pub kinds: Vec<String>,
    /// The generator model's role.
    pub model: String,
    /// Output tokens one request may generate (the generator's default
    /// otherwise); a model that reasons before it answers needs more.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// How long one request may take, in seconds (the generator's default
    /// otherwise).
    #[serde(default)]
    pub deadline_seconds: Option<u64>,
    /// The most tasks one request asks for (the generator's default
    /// otherwise).
    #[serde(default)]
    pub tasks_per_request: Option<usize>,
}

/// The `task.generate` operation.
pub struct Generate;

fn generate_params(value: &Value) -> Result<GenerateParams, String> {
    serde_json::from_value(value.clone()).map_err(|e| e.to_string())
}

impl OperationSpec for Generate {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: GENERATE.into(),
            revision: 1,
            summary: "closed-book tasks a generator proposes from documents, admitted by code"
                .into(),
            inputs: vec![port("documents", ArtifactKind::Files, true)],
            outputs: vec![port("tasks", ArtifactKind::Files, false)],
            parameters: serde_json::to_value(schema_for!(GenerateParams)).unwrap_or(Value::Null),
            reproducibility: Reproducibility::Nondeterministic,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        let p = generate_params(value)?;
        if p.kinds.is_empty() {
            return Err("names no task kind".into());
        }
        let catalogue = Catalogue::builtin();
        match p.kinds.iter().find(|k| catalogue.get(k).is_none()) {
            Some(unknown) => Err(format!("no task kind {unknown:?} in the catalogue")),
            None => Ok(()),
        }
    }

    fn model_roles(&self, value: &Value) -> Vec<String> {
        generate_params(value)
            .map(|p| vec![p.model])
            .unwrap_or_default()
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

/// Every text document below `dir`, by its path relative to `root`.
fn texts(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if name.starts_with('.')
            || (dir == root && [ITEMS_FILE, "manifest.json", "stage.json"].contains(&name))
        {
            continue;
        }
        if path.is_dir() {
            texts(root, &path, out)?;
        } else if let Ok(text) = std::fs::read_to_string(&path) {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((relative.trim_start_matches("files/").to_owned(), text));
        }
    }
    Ok(())
}

impl Execute for Generate {
    fn implementation(&self) -> String {
        format!("{GENERATE} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let p = generate_params(call.params)?;
        let model = call
            .models
            .get(&p.model)
            .ok_or_else(|| format!("model role {} is not bound", p.model))?;
        let catalogue = Catalogue::builtin();
        let kinds: Vec<_> = p.kinds.iter().filter_map(|k| catalogue.get(k)).collect();
        let state = tempfile::tempdir().map_err(|e| e.to_string())?;
        let store = source_store(&state.path().join(SOURCES_DIR));
        let offered = runtimes(
            &state.path().join("sandbox"),
            kinds.iter().filter_map(|k| k.runtime.as_deref()),
        );
        let mut policy = GenerationPolicy::default();
        if let Some(tokens) = p.max_output_tokens {
            policy.max_output_tokens = Some(tokens);
        }
        if let Some(seconds) = p.deadline_seconds {
            policy.deadline = Duration::from_secs(seconds);
        }
        if let Some(count) = p.tasks_per_request {
            policy.tasks_per_request = count;
        }
        let generator =
            ModelTaskGenerator::new(Arc::new(SvenProposer::new(model.clone())), store.clone())
                .with_policy(policy)
                .with_runtimes(offered);
        let mut found = Vec::new();
        for root in call.inputs.get("documents").into_iter().flatten() {
            texts(root, root, &mut found)?;
        }
        let runtime = runtime()?;
        let (mut items, mut rejected) = (Vec::new(), Vec::new());
        for (document, text) in found {
            let media = if document.ends_with(".md") {
                "text/markdown"
            } else {
                "text/plain"
            };
            let part = document.rsplit('/').next().unwrap_or(&document).to_owned();
            let captured = CapturedSource::new(
                Origin::Document {
                    path: document.clone(),
                },
                vec![PartContent {
                    name: part.clone(),
                    media_type: media.into(),
                    bytes: text.into_bytes(),
                }],
                &SystemClock,
            )
            .map_err(|e| format!("{document}: {e}"))?;
            let id = store.put_source(&captured).map_err(|e| e.to_string())?;
            let source = SourceText::load(&store, &id, &part).map_err(|e| e.to_string())?;
            let report = runtime
                .block_on(generator.generate(&source, &kinds))
                .map_err(|e| format!("{document}: {e}"))?;
            for rejection in &report.rejections {
                rejected.push(json!({
                    "document": document, "kind": rejection.kind, "index": rejection.index,
                    "reason": format!("{:?}", rejection.reason), "detail": rejection.detail,
                }));
            }
            for generated in report.admitted {
                // Named by its content: the same task is the same item.
                let id = format!("{part}--{}", &generated.task.task.id.hex()[..16]);
                if !item_id(&id) {
                    return Err(format!("{document}: a task cannot be named {id:?}"));
                }
                let dir = call.out.join(&id);
                std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                write(&dir.join(TASK_FILE), &generated.task)?;
                source_store(&dir.join(SOURCES_DIR))
                    .put_source(&captured)
                    .map_err(|e| e.to_string())?;
                items.push(Item {
                    path: id.clone(),
                    id,
                });
            }
        }
        let lines: Vec<String> = rejected.iter().map(Value::to_string).collect();
        std::fs::write(call.out.join("rejected.jsonl"), lines.join("\n") + "\n")
            .map_err(|e| e.to_string())?;
        let index = ItemIndex::new(items, Vec::new()).map_err(|e| e.to_string())?;
        write(&call.out.join(ITEMS_FILE), &index)?;
        Ok(Outcome::Produced)
    }
}

fn one() -> u32 {
    1
}
fn default_deadline() -> u64 {
    300
}

/// `answer.collect`'s parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnswerParams {
    /// The answering model's role.
    pub model: String,
    /// Answers per task.
    #[serde(default = "one")]
    pub attempts: u32,
    /// How long one answer may take, in seconds.
    #[serde(default = "default_deadline")]
    pub deadline_seconds: u64,
}

/// The `answer.collect` operation.
pub struct Answer;

fn answer_params(value: &Value) -> Result<AnswerParams, String> {
    serde_json::from_value(value.clone()).map_err(|e| e.to_string())
}

impl OperationSpec for Answer {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: ANSWER.into(),
            revision: 1,
            summary: "a closed-book task answered in process and graded by its kind's verifiers"
                .into(),
            inputs: vec![port("task", ArtifactKind::Files, false)],
            outputs: vec![port("answers", ArtifactKind::Files, false)],
            parameters: serde_json::to_value(schema_for!(AnswerParams)).unwrap_or(Value::Null),
            reproducibility: Reproducibility::Nondeterministic,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        let p = answer_params(value)?;
        if p.attempts == 0 || p.deadline_seconds == 0 {
            return Err("needs at least one attempt and a deadline of 1 second".into());
        }
        Ok(())
    }

    fn model_roles(&self, value: &Value) -> Vec<String> {
        answer_params(value)
            .map(|p| vec![p.model])
            .unwrap_or_default()
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

/// Where `task` is answered, and the verifiers its kind grades it with:
/// the evidence read from `sources`, code run in `runtimes`. A reason
/// names what is missing for either.
struct Grading {
    environment: ResolvedEnvironment,
    verifiers: Strongest,
}

fn grading(task: &Task, sources: Option<&SourceStore>, scratch: &Path) -> Result<Grading, String> {
    let catalogue = Catalogue::builtin();
    let kind = catalogue
        .get(&task.task.kind)
        .ok_or_else(|| format!("no task kind {:?} in the catalogue", task.task.kind))?;
    let solved_in = task.environment.kind.strip_prefix(RUNTIME_KIND_PREFIX);
    let found = runtimes(
        scratch,
        kind.runtime.as_deref().into_iter().chain(solved_in),
    );
    let environment = match kind.environment {
        SolverEnvironment::ClosedBook => ResolvedEnvironment::ClosedBook,
        SolverEnvironment::Runtime => {
            let name = solved_in.ok_or("it names no runtime to be solved in")?;
            let env = found
                .iter()
                .find(|env| env.runtime().spec.name == name)
                .ok_or_else(|| format!("runtime {name} is not installed here"))?;
            ResolvedEnvironment::Runtime(env.clone())
        }
        SolverEnvironment::Workspace => {
            return Err(format!("its kind {} is solved in a workspace", kind.name))
        }
    };
    let mut out = Vec::new();
    for verifier in &kind.verifiers {
        // A judge grades only where one is given, as everywhere answers are
        // graded; none is given here.
        if *verifier != VerifierKind::Judged {
            out.push(verifier_of(kind, *verifier, sources, &found)?);
        }
    }
    Ok(Grading {
        environment,
        verifiers: Strongest::new(out),
    })
}

fn verifier_of(
    kind: &TaskKind,
    verifier: VerifierKind,
    sources: Option<&SourceStore>,
    runtimes: &[RuntimeEnvironment],
) -> Result<Box<dyn Verifier>, String> {
    let made = plain_verifier(verifier)
        .or_else(|| sources.and_then(|store| evidence_verifier(verifier, store)))
        .or_else(|| {
            (!runtimes.is_empty())
                .then(|| runtime_verifier(verifier, runtimes.to_vec()))
                .flatten()
        });
    made.ok_or_else(|| {
        let missing = match verifier {
            VerifierKind::Quotation | VerifierKind::Grounding => "the sources of its evidence",
            VerifierKind::Executable | VerifierKind::MutationValidated => "its runtime",
            _ => "what only a run has",
        };
        format!(
            "its kind {} is graded by {verifier:?}, which needs {missing}",
            kind.name
        )
    })
}

impl Execute for Answer {
    fn implementation(&self) -> String {
        format!("{ANSWER} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let p = answer_params(call.params)?;
        let model = call
            .models
            .get(&p.model)
            .ok_or_else(|| format!("model role {} is not bound", p.model))?;
        let dir = call
            .inputs
            .get("task")
            .and_then(|v| v.first())
            .ok_or("no task is bound")?;
        let task: Task = serde_json::from_str(
            &std::fs::read_to_string(dir.join(TASK_FILE)).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let stored = dir.join(SOURCES_DIR);
        let sources = stored.is_dir().then(|| source_store(&stored));
        if stored.is_dir() {
            // The answers keep the evidence they are graded against.
            crate::tree::link_tree(&stored, &call.out.join(SOURCES_DIR))
                .map_err(|e| e.to_string())?;
        }
        let scratch = tempfile::tempdir().map_err(|e| e.to_string())?;
        let grading = match grading(&task, sources.as_ref(), scratch.path()) {
            Ok(grading) => grading,
            Err(why) => return Ok(Outcome::Rejected(why)),
        };
        let runtime = runtime()?;
        for attempt in 0..p.attempts {
            let solution = runtime
                .block_on(solve(
                    &task,
                    &grading.environment,
                    model.provider.clone(),
                    SolveOptions::new(Duration::from_secs(p.deadline_seconds)),
                ))
                .map_err(|e| e.to_string())?;
            let provenance = Provenance {
                solver: model.identity.clone(),
                generator: None,
                policy: None,
                prompt_digests: Vec::new(),
                attempt: (p.attempts > 1).then_some(attempt),
                teacher: false,
                created_at: SystemClock.utc_now(),
            };
            let experience = solution
                .into_experience(task.clone(), provenance)
                .map_err(|e| e.to_string())?;
            let report = grading
                .verifiers
                .run(&task, &experience)
                .map_err(|e| e.to_string())?;
            write(
                &call.out.join(format!("experience-{attempt}.json")),
                &experience,
            )?;
            write(
                &call.out.join(format!("verdicts-{attempt}.json")),
                &report.annotations,
            )?;
        }
        Ok(Outcome::Produced)
    }
}
