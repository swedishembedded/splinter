// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The v2 recipe compiler against a registry of two test operations: a
//! branching graph compiles into stages each after what it consumes; every
//! problem of a recipe is reported at once; a subrecipe is expanded into the
//! graph under its stage's id; `foreach` binds `$item` to the collection's
//! kind.

use serde::Deserialize;
use serde_json::{json, Value};
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_factory::graph::{compile, OperationSpec, Registry, Source};

fn port(name: &str, kind: ArtifactKind, many: bool, optional: bool) -> Port {
    Port {
        name: name.into(),
        kind,
        many,
        optional,
    }
}

fn descriptor(name: &str, inputs: Vec<Port>, outputs: Vec<Port>) -> OperationDescriptor {
    OperationDescriptor {
        schema: OPERATION_SCHEMA.into(),
        name: name.into(),
        revision: 1,
        summary: name.into(),
        inputs,
        outputs,
        parameters: json!({ "type": "object", "additionalProperties": false }),
        reproducibility: Reproducibility::Deterministic,
        isolation: Isolation::Any,
    }
}

/// Makes files from nothing.
struct Make;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct MakeParams {
    count: u32,
}

impl OperationSpec for Make {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor(
            "source.make",
            vec![],
            vec![port("files", ArtifactKind::Files, false, false)],
        )
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        serde_json::from_value::<MakeParams>(params.clone())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

/// Makes a dataset of files, with a model.
struct Use;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UseParams {
    model: String,
}

impl OperationSpec for Use {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor(
            "files.use",
            vec![
                port("files", ArtifactKind::Files, false, false),
                port("extra", ArtifactKind::Files, true, true),
            ],
            vec![port("dataset", ArtifactKind::Dataset, false, false)],
        )
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        serde_json::from_value::<UseParams>(params.clone())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    fn model_roles(&self, params: &Value) -> Vec<String> {
        serde_json::from_value::<UseParams>(params.clone())
            .map(|p| vec![p.model])
            .unwrap_or_default()
    }
}

fn registry() -> Registry {
    Registry::default().with(Make).with(Use)
}

fn no_files(path: &str) -> Result<String, String> {
    Err(format!("{path}: no such file"))
}

#[test]
fn a_branching_graph_compiles_each_stage_after_what_it_consumes() {
    let recipe = r#"
schema: splinter.recipe/v2
name: branching
inputs: { docs: { kind: files, path: books } }
models: { student: "brain:Qwen/Qwen3-4B" }
stages:
  - { id: made, uses: source.make, with: { count: 3 } }
  - { id: left, uses: files.use, inputs: { files: $inputs.docs }, with: { model: student } }
  - id: right
    uses: files.use
    inputs: { files: $stages.made.files, extra: [$inputs.docs, $stages.made.files] }
    with: { model: student }
outputs: { left: $stages.left.dataset, right: $stages.right.dataset }
"#;
    let plan = compile(recipe, &registry(), &no_files).unwrap();
    let ids: Vec<&str> = plan.stages.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["made", "left", "right"]);
    let right = &plan.stages[2];
    assert_eq!(right.depends_on, ["made"]);
    assert_eq!(right.inputs["extra"].len(), 2);
    assert_eq!(right.params["model"], "student");
    assert_eq!(
        plan.outputs["right"],
        Source::Stage {
            stage: "right".into(),
            port: "dataset".into()
        }
    );
}

#[test]
fn every_problem_of_a_recipe_is_reported_at_once() {
    let recipe = r#"
schema: splinter.recipe/v2
name: broken
inputs: { docs: { kind: files, path: books } }
models: { student: "brain:Qwen/Qwen3-4B" }
stages:
  - { id: made, uses: source.make, with: { count: 3, colour: red } }
  - { id: made, uses: source.make, with: { count: 1 } }
  - { id: Bad, uses: source.make, with: { count: 1 } }
  - { id: nope, uses: no.such }
  - { id: early, uses: files.use, inputs: { files: $stages.late.dataset }, with: { model: student } }
  - { id: typed, uses: files.use, inputs: { files: $stages.early.dataset }, with: { model: teacher } }
  - { id: unbound, uses: files.use, inputs: { extras: $inputs.docs }, with: { model: student } }
  - { id: item, uses: files.use, inputs: { files: $item }, with: { model: student } }
  - { id: late, uses: files.use, inputs: { files: $inputs.docs }, with: { model: student } }
"#;
    let problems = compile(recipe, &registry(), &no_files)
        .unwrap_err()
        .0
        .join("\n");
    for expected in [
        "stage made: unknown field `colour`",
        "stage made is declared twice",
        "stage id \"Bad\" is not a plain name",
        "no operation \"no.such\"",
        "stage late is not above this one",
        "input files takes Files, and it is bound to Dataset",
        "model role teacher is not in `models:`",
        "files.use has no input extras",
        "stage unbound: input files is not bound",
        "$item is only for a stage with `foreach`",
    ] {
        assert!(
            problems.contains(expected),
            "{expected:?} missing from:\n{problems}"
        );
    }
}

#[test]
fn a_subrecipe_is_expanded_under_its_stage_and_its_outputs_are_the_stages() {
    let fragment = r#"
schema: splinter.recipe/v2
name: fragment
inputs: { files: { kind: files, path: unused } }
models: { model: "brain:unused" }
stages:
  - { id: used, uses: files.use, inputs: { files: $inputs.files }, with: { model: model } }
outputs: { dataset: $stages.used.dataset }
"#;
    let recipe = r#"
schema: splinter.recipe/v2
name: outer
models: { model: "brain:Qwen/Qwen3-4B" }
stages:
  - { id: made, uses: source.make, with: { count: 1 } }
  - { id: book, uses: recipe, inputs: { files: $stages.made.files }, with: { recipe: fragments/a.yaml } }
  - { id: after, uses: files.use, inputs: { files: $stages.made.files, extra: [] }, with: { model: model } }
outputs: { dataset: $stages.book.dataset }
"#;
    let load = |path: &str| match path {
        "fragments/a.yaml" => Ok(fragment.to_owned()),
        other => Err(format!("{other}: no such file")),
    };
    let plan = compile(recipe, &registry(), &load).unwrap();
    let ids: Vec<&str> = plan.stages.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["made", "book.used", "after"]);
    assert_eq!(
        plan.stages[1].inputs["files"],
        [Source::Stage {
            stage: "made".into(),
            port: "files".into()
        }]
    );
    assert_eq!(
        plan.outputs["dataset"],
        Source::Stage {
            stage: "book.used".into(),
            port: "dataset".into()
        }
    );

    // A subrecipe that includes itself is refused, not expanded forever.
    let looping = r#"
schema: splinter.recipe/v2
name: loop
stages:
  - { id: again, uses: recipe, with: { recipe: loop.yaml } }
"#;
    let load = |_: &str| Ok(looping.to_owned());
    let problems = compile(looping, &registry(), &load)
        .unwrap_err()
        .0
        .join("\n");
    assert!(problems.contains("nest deeper"), "{problems}");
}

#[test]
fn foreach_binds_item_to_the_kind_of_its_collection() {
    let recipe = r#"
schema: splinter.recipe/v2
name: each
models: { m: "brain:x" }
stages:
  - { id: made, uses: source.make, with: { count: 9 } }
  - { id: each, uses: files.use, foreach: $stages.made.files, inputs: { files: $item }, with: { model: m } }
"#;
    let plan = compile(recipe, &registry(), &no_files).unwrap();
    assert_eq!(
        plan.stages[1].foreach,
        Some(Source::Stage {
            stage: "made".into(),
            port: "files".into()
        })
    );
    assert_eq!(plan.stages[1].inputs["files"], [Source::Item]);
    assert_eq!(plan.stages[1].depends_on, ["made"]);
}

#[test]
fn a_file_of_another_schema_is_named_as_what_it_is() {
    let refused = compile(
        "schema: splinter.factory/v1\nname: x\n",
        &registry(),
        &no_files,
    )
    .unwrap_err()
    .0
    .join("\n");
    assert!(
        refused.contains("splinter.factory/v1 file, not a recipe"),
        "{refused}"
    );
}

#[test]
fn the_task_repositorys_recipes_compile_against_the_built_in_operations() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/it/fixtures/v2");
    let registry = splinter_factory::ops::registry();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let text = std::fs::read_to_string(&path).unwrap();
        let plan = compile(&text, &registry, &no_files)
            .unwrap_or_else(|e| panic!("{}: {}", path.display(), e.0.join("\n")));
        assert!(
            plan.stages.iter().all(|s| s.operation != "recipe"),
            "{}",
            path.display()
        );
    }
    // Every built-in contract is well formed.
    for name in registry.names() {
        let descriptor = registry.get(name).unwrap().descriptor();
        assert!(
            descriptor.problems().is_empty(),
            "{name}: {:?}",
            descriptor.problems()
        );
    }
}
