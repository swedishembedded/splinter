// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! Every factory contract has a committed JSON Schema equal to the one its
//! Rust type generates, and a golden fixture that the schema accepts, that
//! round-trips through the type unchanged, and whose identity is stable.

#![allow(clippy::expect_used)]

use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use splinter_core::digest::canonical_json;
use splinter_core::factory::execution::{EnvValue, ExecutionSpec, SpecError};
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{DatasetProduct, EvalResult, ModelCandidate, TrialResult};
use splinter_core::factory::schemas;
use splinter_core::factory::task::{CollectionError, TaskCollection};

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/it/factory/fixtures")
        .join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect("the fixture exists"))
        .expect("the fixture is JSON")
}

fn schema(name: &str) -> Value {
    let (_, generated) = schemas::all()
        .into_iter()
        .find(|(n, _)| *n == name)
        .expect("a known schema");
    generated.to_value()
}

#[test]
fn the_committed_schemas_are_the_generated_ones() {
    for (name, generated) in schemas::all() {
        let path = repository()
            .join("schemas/factory")
            .join(schemas::file_name(name));
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        let expected = format!(
            "{}\n",
            serde_json::to_string_pretty(&generated).expect("json")
        );
        assert!(
            committed == expected,
            "{} differs from the schema {name} generates; regenerate it with `make schemas/generate`",
            path.display()
        );
    }
}

fn round_trips<T: Serialize + DeserializeOwned>(fixture_name: &str, schema_name: &str) -> T {
    let document = fixture(fixture_name);
    let validator = jsonschema::validator_for(&schema(schema_name)).expect("a valid schema");
    let errors: Vec<String> = validator
        .iter_errors(&document)
        .map(|e| e.to_string())
        .collect();
    assert!(
        errors.is_empty(),
        "{fixture_name} breaks {schema_name}: {errors:?}"
    );
    let value: T = serde_json::from_value(document.clone()).expect("the fixture reads as its type");
    assert_eq!(
        canonical_json(&value).expect("json"),
        canonical_json(&document).expect("json"),
        "{fixture_name} does not round-trip unchanged"
    );
    value
}

#[test]
fn every_golden_fixture_satisfies_its_schema_and_round_trips() {
    let stage: StageIdentity = round_trips("stage", "splinter.stage/2");
    let tasks: TaskCollection = round_trips("tasks", "splinter.tasks/1");
    let spec: ExecutionSpec = round_trips("execution", "splinter.execution/1");
    let trial: TrialResult = round_trips("trial", "splinter.trial/1");
    let _: DatasetProduct = round_trips("dataset", "splinter.dataset/1");
    let _: ModelCandidate = round_trips("candidate", "splinter.candidate/1");
    let _: EvalResult = round_trips("eval", "splinter.eval/1");
    let _: splinter_core::factory::attempt::AttemptReport =
        round_trips("attempt", "splinter.attempt/1");
    let operation: splinter_core::factory::operation::OperationDescriptor =
        round_trips("operation", "splinter.operation/1");
    let items: splinter_core::factory::items::ItemIndex = round_trips("items", "splinter.items/1");
    items.validate().expect("the golden index is valid");
    assert!(
        operation.problems().is_empty(),
        "{:?}",
        operation.problems()
    );
    spec.validate().expect("the golden spec is valid");
    assert!(trial.verified_success());
    assert_eq!(
        TaskCollection::new(tasks.name.clone(), tasks.tasks.clone()),
        Ok(tasks)
    );
    // Identities are content addresses: pinned so a serialisation change that
    // would orphan every stored product is caught here.
    assert_eq!(stage.digest(), StageIdentity { ..stage.clone() }.digest());
    let mut moved = stage.clone();
    moved.parameters["attempts"] = Value::from(5);
    assert_ne!(
        stage.digest(),
        moved.digest(),
        "a parameter is part of the identity"
    );
}

#[test]
fn a_stage_identity_ignores_the_order_its_inputs_were_listed_in() {
    let stage: StageIdentity = serde_json::from_value(fixture("stage")).expect("a stage");
    let a = splinter_core::digest::Digest::of(b"a");
    let b = splinter_core::digest::Digest::of(b"b");
    let build = |inputs| {
        StageIdentity::new(
            stage.operation.clone(),
            "x",
            Value::Null,
            inputs,
            "impl",
            None,
        )
        .digest()
    };
    assert_eq!(
        build(vec![("x".into(), a.clone()), ("y".into(), b.clone())]),
        build(vec![("y".into(), b), ("x".into(), a)])
    );
}

#[test]
fn a_specification_never_carries_a_credential_or_an_escaping_path() {
    let spec: ExecutionSpec = serde_json::from_value(fixture("execution")).expect("a spec");
    let mut secret = spec.clone();
    secret
        .agent
        .env
        .insert("OPENAI_API_KEY".into(), EnvValue::Literal("sk-123".into()));
    assert_eq!(
        secret.validate(),
        Err(SpecError::LiteralSecret("OPENAI_API_KEY".into()))
    );
    let mut escaping = spec.clone();
    escaping.task_dir = "../outside".into();
    assert!(matches!(escaping.validate(), Err(SpecError::BadPath(_))));
    let mut mount = spec;
    mount.sandbox.mounts[0].target = "relative".into();
    assert!(matches!(mount.validate(), Err(SpecError::BadTarget(_))));
}

#[test]
fn a_collection_refuses_a_repeated_or_escaping_task() {
    let tasks: TaskCollection = serde_json::from_value(fixture("tasks")).expect("tasks");
    let mut twice = tasks.tasks.clone();
    twice.push(tasks.tasks[0].clone());
    assert!(matches!(
        TaskCollection::new("c", twice),
        Err(CollectionError::Duplicate { .. })
    ));
    let mut escaping = tasks.tasks.clone();
    escaping[0].id = "../x".into();
    assert!(matches!(
        TaskCollection::new("c", escaping),
        Err(CollectionError::BadId(_))
    ));
    assert!(matches!(
        TaskCollection::new("c", vec![]),
        Err(CollectionError::Empty(_))
    ));
}

#[test]
fn an_operation_contract_refuses_loose_parameters_and_ambiguous_ports() {
    use splinter_core::factory::operation::{ArtifactKind, OperationDescriptor, Port};
    let mut op: OperationDescriptor =
        serde_json::from_value(fixture("operation")).expect("the fixture");
    op.parameters = serde_json::json!({ "type": "object" });
    op.inputs.push(Port {
        name: "tasks".into(),
        kind: ArtifactKind::Files,
        many: false,
        optional: false,
    });
    op.outputs.clear();
    op.name = "Attempt..collect".into();
    let text = op.problems().join("\n");
    for expected in [
        "additionalProperties: false",
        "input tasks is declared twice",
        "produces nothing",
        "not dotted plain names",
    ] {
        assert!(text.contains(expected), "{expected:?} missing from {text}");
    }
}

#[test]
fn an_item_index_refuses_an_escaping_or_repeated_item() {
    use splinter_core::factory::items::{Item, ItemIndex, ItemsError, Rejection};
    let item = |id: &str, path: &str| Item {
        id: id.into(),
        path: path.into(),
    };
    let index = ItemIndex::new(vec![item("b", "b"), item("a", "a")], vec![]).expect("valid");
    assert_eq!(index.items[0].id, "a", "kept in id order");
    assert!(matches!(
        ItemIndex::new(vec![item("..", "x")], vec![]),
        Err(ItemsError::BadId(_))
    ));
    assert!(matches!(
        ItemIndex::new(vec![item("a", "../x")], vec![]),
        Err(ItemsError::BadPath { .. })
    ));
    assert!(matches!(
        ItemIndex::new(
            vec![item("a", "a")],
            vec![Rejection {
                id: "a".into(),
                reason: "no".into()
            }]
        ),
        Err(ItemsError::Duplicate(_))
    ));
}
