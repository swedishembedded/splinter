// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continued pretraining on an organisation's
// own documents, for its clients. If your team needs expertise in adapting
// models to a domain's writing, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `dataset.text`: documents as a continued-pretraining dataset, by
//! splinter-data's CPT view: each document's text one record, a text shared
//! by two documents trained on once, whole documents held out for
//! validation, and the dataset trained like any chat dataset.

use serde_json::Value;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, Services};

use crate::fakes::{FakeBackend, Outcome};
use crate::repo::{connections, Repo};

// seed: the split of documents into train and validation is the seed's draw.
const RECIPE: &str = "schema: splinter.recipe/v2
name: read
inputs:
  docs: { kind: files, path: documents }
models:
  student: \"brain:Qwen/Qwen3-4B\"
stages:
  - id: text
    uses: dataset.text
    inputs: { documents: $inputs.docs }
    with: { target: student, seed: 3, validation: { percent: 50 } }
  - id: lora
    uses: model.train
    inputs: { dataset: $stages.text.dataset }
    with: { base: student, method: lora, params: { rank: 8 }, seed: 1 }
";

fn texts(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| {
            let line: Value = serde_json::from_str(l).unwrap();
            let turn = line["messages"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["train"] == true)
                .unwrap()
                .clone();
            turn["content"].as_str().unwrap().to_owned()
        })
        .collect()
}

#[test]
fn documents_become_a_continued_pretraining_dataset() {
    let repo = Repo::new();
    repo.write(
        "documents/a.md",
        "# Ports\n\nThe console runs at 115200 baud.\n",
    );
    repo.write("documents/b.md", "# Power\n\nThe board draws 2 A at 5 V.\n");
    repo.write(
        "documents/copy.md",
        "# Ports\n\nThe console runs at 115200 baud.\n",
    );
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = crate::fakes::FakeTrainer::default();
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    let mut factory = Factory::open(&resolved, dir.path(), services, "test").unwrap();
    factory.run().unwrap();
    let mut all = texts(&dir.path().join("text/train.jsonl"));
    let validation = texts(&dir.path().join("text/validation.jsonl"));
    assert_eq!(validation.len(), 1, "half the documents are held out");
    all.extend(validation);
    all.sort();
    assert_eq!(
        all,
        [
            "# Ports\n\nThe console runs at 115200 baud.\n",
            "# Power\n\nThe board draws 2 A at 5 V.\n"
        ]
    );
    let product: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("text/dataset.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(product["objective"], "cpt");
    assert_eq!(product["exclusions"]["duplicate"], 1, "{product:#}");
    assert!(
        dir.path().join("lora").is_dir(),
        "trained like a chat dataset"
    );
}
