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
//!   (`task.json`); `rejected.jsonl` keeps every refusal with its reason.
//! * `answer.collect`, run once per task, has the stage's model answer it
//!   closed-book through the one solve every task runs through
//!   ([`splinter_agent::solve::solve`]) and grades the answer with the
//!   verifiers the task's kind names
//!   ([`splinter_knowledge::tasks::verifiers::plain_verifier`]): the same
//!   verdicts, under the same producer names, as everywhere else (a judge
//!   grades only where one is given, and none is here). A kind graded by a
//!   verifier that needs a run's environment (a runtime, the source store,
//!   a workspace) is rejected by name. The item
//!   holds each attempt's `experience-N.json` and `verdicts-N.json`.

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
use splinter_knowledge::tasks::verifiers::plain_verifier;
use splinter_knowledge::tasks::{Catalogue, GenerationPolicy, ModelTaskGenerator, SourceText};
use splinter_sandbox::environment::ResolvedEnvironment;
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
        let store = SourceStore::new(&Workspace::at(&StateRoot::new(state.path().to_path_buf())));
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
                .with_policy(policy);
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

/// The verifiers `task`'s kind grades it with, or the one that needs a
/// run's environment.
fn verifiers_of(task: &Task) -> Result<Strongest, String> {
    let catalogue = Catalogue::builtin();
    let kind = catalogue
        .get(&task.task.kind)
        .ok_or_else(|| format!("no task kind {:?} in the catalogue", task.task.kind))?;
    let mut out = Vec::new();
    for verifier in &kind.verifiers {
        // A judge grades only where one is given, as everywhere answers are
        // graded; none is given here.
        if *verifier == splinter_knowledge::tasks::VerifierKind::Judged {
            continue;
        }
        out.push(plain_verifier(*verifier).ok_or_else(|| {
            format!(
                "its kind {} is graded by {verifier:?}, which needs a run's environment",
                task.task.kind
            )
        })?);
    }
    Ok(Strongest::new(out))
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
        let verifiers = match verifiers_of(&task) {
            Ok(v) => v,
            Err(why) => return Ok(Outcome::Rejected(why)),
        };
        let runtime = runtime()?;
        for attempt in 0..p.attempts {
            let solution = runtime
                .block_on(solve(
                    &task,
                    &ResolvedEnvironment::ClosedBook,
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
            let report = verifiers
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
