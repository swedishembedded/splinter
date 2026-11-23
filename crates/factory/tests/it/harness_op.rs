// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements programmable agent workflows that turn
// documents into structured data with every step on record, for its
// clients. If your team needs expertise in agent harnesses, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The `harness` operation: a sven harness run over each item's files, its
//! returned value the item's product; a harness that would run commands is
//! refused before anything runs.

use std::sync::Arc;

use serde_json::Value;
use splinter_agent::solve::Model;
use splinter_core::factory::execution::ModelBinding;
use splinter_factory::execute::Models;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, Services};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

use crate::fakes::{FakeBackend, Outcome};
use crate::repo::{connections, Repo};

/// A model that summarises by what it was given.
struct Summariser;

#[async_trait::async_trait]
impl ModelProvider for Summariser {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "summariser-1"
    }
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let text = serde_json::to_string(&request.messages).unwrap_or_default();
        let reply = if text.contains("alpha") {
            "Alpha comes first."
        } else {
            "Zeta comes last."
        };
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply.into())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

struct Summarisers;

impl Models for Summarisers {
    fn model(&self, _: &ModelBinding) -> Result<Model, String> {
        Ok(Model::new(Arc::new(Summariser), "scripted/summariser-1"))
    }
}

const HARNESS: &str = r#"api_version: "2"
name: summary
inputs:
  files: { type: json, description: "The notes." }
agent:
  profile: none
execution:
  limits: { model_turns: 4 }
workflow:
  - name: summarise
    task: "Summarise in one sentence: {{ inputs.files }}"
outputs:
  summary: { type: string, value: "{{ tasks.summarise.output }}" }
"#;

const RECIPE: &str = "schema: splinter.recipe/v2
name: summaries
inputs:
  notes: { kind: files, path: notes }
models:
  writer: \"hosted:org/writer\"
stages:
  - id: summaries
    uses: harness
    foreach: $inputs.notes
    inputs: { input: $item }
    with: { harness: harnesses/summary.yaml, model: writer }
";

#[test]
fn a_harness_stage_returns_each_items_value() {
    let repo = Repo::new();
    repo.write("notes/a.md", "alpha is the first letter");
    repo.write("notes/z.md", "zeta is a late letter");
    repo.write("harnesses/summary.yaml", HARNESS);
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(&Summarisers),
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let output: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("summaries/a.md/output.json")).unwrap(),
    )
    .unwrap();
    assert!(
        output.to_string().contains("Alpha comes first."),
        "{output}"
    );
}

#[test]
fn a_harness_that_would_run_commands_is_refused() {
    let repo = Repo::new();
    repo.write("notes/a.md", "alpha");
    repo.write(
        "harnesses/summary.yaml",
        &HARNESS.replace("profile: none", "profile: coding"),
    );
    let refused = resolve(&repo.root, RECIPE, connections())
        .unwrap_err()
        .0
        .join("\n");
    assert!(refused.contains("ExecuteShell"), "{refused}");
}
