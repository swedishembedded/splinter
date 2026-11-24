// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn documents into grounded,
// checkable knowledge for training and evaluation, for its clients. If your
// team needs expertise in knowledge extraction you can trust, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A document into grounded knowledge: `source.ingest` splits it into
//! sections that name where they came from, an `agent` stage proposes typed
//! knowledge per section with the quote it rests on, and `knowledge.admit`
//! keeps only what the sections say verbatim, every refusal with its reason.
//! `knowledge.query` gives each question the sections that bear on it.

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

/// An extractor that proposes, for each section, one object its text holds
/// (its value in its quote) and one it does not.
struct Extractor;

#[async_trait::async_trait]
impl ModelProvider for Extractor {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "extractor-1"
    }
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let text = serde_json::to_string(&request.messages).unwrap_or_default();
        let (quote, key, value) = if text.contains("Refunds") {
            (
                "Refunds are paid within 14 days",
                "paid within | Refunds",
                "14 days",
            )
        } else {
            ("Orders ship on weekdays", "ship on | Orders", "weekdays")
        };
        let reply = serde_json::json!({ "objects": [
            { "id": "a", "kind": "fact", "document": "?", "quote": quote,
              "data": { "key": key, "value": value } },
            { "id": "b", "kind": "fact", "document": "?", "quote": "Everything is free forever",
              "data": { "key": "price", "value": 0 } }
        ]})
        .to_string();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

struct Extractors;

impl Models for Extractors {
    fn model(&self, _: &ModelBinding) -> Result<Model, String> {
        Ok(Model::new(Arc::new(Extractor), "scripted/extractor-1"))
    }
}

const POLICY: &str = "# Shop policy\n\nOrders ship on weekdays from the main store.\n\n## Refunds\n\nRefunds are paid within 14 days of the return.\n";

const RECIPE: &str = "schema: splinter.recipe/v2
name: policy
inputs:
  docs: { kind: files, path: documents }
models:
  extractor: \"hosted:org/extractor\"
stages:
  - { id: sections, uses: source.ingest, inputs: { documents: $inputs.docs } }
  - id: proposed
    uses: agent
    foreach: $stages.sections.sections
    inputs: { input: $item }
    with: { instruction: prompts/extract.md, output_schema: schemas/objects.json, model: extractor }
  - id: knowledge
    uses: knowledge.admit
    inputs: { sections: $stages.sections.sections, proposals: $stages.proposed.output }
    with: { pack: shop-policy }
";

#[test]
fn a_document_yields_only_the_knowledge_its_sections_state() {
    let repo = Repo::new();
    repo.write("documents/policy.md", POLICY);
    repo.write(
        "prompts/extract.md",
        "Propose the facts the section states, each with its quote.",
    );
    repo.write(
        "schemas/objects.json",
        r#"{"type": "object", "required": ["objects"]}"#,
    );
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(&Extractors),
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();

    let source: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("sections/policy.md--002/source.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(source["document"], "policy.md");
    assert_eq!(source["title"], "Shop policy");
    assert_eq!(source["section"], 2);

    let pack: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("knowledge/pack.json")).unwrap(),
    )
    .unwrap();
    let admitted: Vec<&str> = pack["objects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["quote"].as_str().unwrap())
        .collect();
    assert_eq!(
        admitted,
        ["Orders ship on weekdays", "Refunds are paid within 14 days"]
    );
    let rejected = pack["rejected"].as_array().unwrap();
    assert_eq!(rejected.len(), 2, "{rejected:?}");
    assert!(rejected
        .iter()
        .all(|r| r["id"].as_str().unwrap().ends_with("#b")));
}

const ASKED: &str = "schema: splinter.recipe/v2
name: asked
inputs:
  docs: { kind: files, path: documents }
stages:
  - { id: sections, uses: source.ingest, inputs: { documents: $inputs.docs } }
  - id: found
    uses: knowledge.query
    inputs: { sections: $stages.sections.sections }
    with: { questions: questions.yaml, passages: 1 }
";

/// Retrieval: each question an item holding the sections that bear on it,
/// best first, each naming where it came from, so a stage over the items
/// (a teacher's brief, a retrieval dataset) reads what the documents say.
#[test]
fn each_question_is_given_the_sections_that_bear_on_it() {
    let repo = Repo::new();
    repo.write("documents/policy.md", POLICY);
    repo.write(
        "documents/opening.md",
        "# Opening hours\n\nThe store opens at nine and is closed on Sundays.\n",
    );
    repo.write(
        "questions.yaml",
        "- How many days until refunds are paid?\n- Is the store closed on Sundays?\n",
    );
    let resolved = resolve(&repo.root, ASKED, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let found = |item: &str| -> (String, Value) {
        let dir = dir.path().join("found").join(item);
        (
            std::fs::read_to_string(dir.join("question.md")).unwrap(),
            serde_json::from_str(&std::fs::read_to_string(dir.join("passages.json")).unwrap())
                .unwrap(),
        )
    };
    let (question, passages) = found("question-001");
    assert_eq!(question.trim(), "How many days until refunds are paid?");
    assert_eq!(passages.as_array().unwrap().len(), 1, "{passages:#}");
    assert_eq!(passages[0]["section"], "policy.md--002", "{passages:#}");
    assert_eq!(passages[0]["document"], "policy.md");
    assert!(passages[0]["text"].as_str().unwrap().contains("14 days"));
    let (_, passages) = found("question-002");
    assert_eq!(passages[0]["section"], "opening.md--001", "{passages:#}");
}
