// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data and learning pipelines built from typed,
// reusable operations instead of one-off scripts, for its clients. If your
// team needs expertise in reproducible machine-learning data factories, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A `splinter.recipe/v2` recipe compiled into one typed graph of stages.
//!
//! ```yaml
//! schema: splinter.recipe/v2
//! name: vim-mastery
//! inputs:  { course: { kind: files, path: collections/vim.yaml } }
//! models:  { student: "brain:Qwen/Qwen3-4B" }
//! stages:
//!   - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: train } }
//!   - id: local
//!     uses: attempt.collect
//!     inputs: { tasks: $stages.tasks.tasks }
//!     with: { agent: sven, model: student, environment: environments/vim.yaml, attempts: 4, seed: 1 }
//!   - { id: book, uses: recipe, with: { recipe: recipes/fragments/sft.yaml }, inputs: { attempts: $stages.local.attempts } }
//! outputs: { dataset: $stages.book.dataset }
//! ```
//!
//! A stage names the operation it `uses` and binds each of the operation's
//! input ports to a reference: `$inputs.NAME`, `$stages.ID.PORT`, or `$item`
//! (one item of the stage's `foreach` collection), or to a list of them for
//! a port that takes several. `with:` holds the operation's parameters,
//! checked against its own typed parameters (an unknown one is an error);
//! the parameters that name models name roles of `models:`. A stage that
//! `uses: recipe` is a subrecipe: its graph is expanded into this one under
//! the stage's id (`book.sft`), its inputs bound to the stage's, its outputs
//! the stage's ports - there is no nested controller.
//!
//! A stage consumes only stages declared above it, so a recipe reads top to
//! bottom, cannot express a cycle, and its stages run in the order written. [`compile`] reports every problem at once - an unknown operation, port or
//! parameter, an unresolved or ill-typed reference, an unbound input, a
//! missing model role, a cycle, a repeated or malformed id - and otherwise
//! returns the [`CompiledPlan`]: stages in an order in which each comes after
//! what it consumes. That one plan is what execution, `splinter plan` and the
//! reproducibility checks read.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::factory::operation::{plain_name, ArtifactKind, Port};

use crate::recipe::{ModelRef, RecipeError};
use crate::registry::Registry;

/// The schema every v2 recipe names.
pub const RECIPE_V2_SCHEMA: &str = "splinter.recipe/v2";
/// The operation name of a subrecipe stage.
pub const SUBRECIPE: &str = "recipe";
/// How deep subrecipes may nest; deeper is refused as a likely cycle.
const MAX_NESTING: usize = 16;

/// An input of the recipe: a product of a kind, read from a path of the
/// task repository.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    /// What it holds.
    pub kind: ArtifactKind,
    /// Where it is, relative to the task repository.
    pub path: String,
}

/// One or several references bound to a port.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Binding {
    /// One reference.
    One(String),
    /// Several, for a port that takes several.
    Many(Vec<String>),
}

impl Binding {
    fn references(&self) -> Vec<&str> {
        match self {
            Self::One(r) => vec![r.as_str()],
            Self::Many(rs) => rs.iter().map(String::as_str).collect(),
        }
    }
}

/// One stage as a recipe writes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageSpec {
    /// Its id, unique in the recipe.
    pub id: String,
    /// The operation it runs, or [`SUBRECIPE`].
    pub uses: String,
    /// Its input ports, bound to references.
    #[serde(default)]
    pub inputs: BTreeMap<String, Binding>,
    /// The operation's parameters.
    #[serde(default = "empty_object", rename = "with")]
    pub params: Value,
    /// A collection the stage runs once per item of (`$item`).
    #[serde(default)]
    pub foreach: Option<String>,
    /// How many items of `foreach` run at once (one by default).
    #[serde(default)]
    pub concurrency: Option<u32>,
    /// How many items of `foreach` may fail, after their retries, and be
    /// recorded as absent rather than fail the stage (none by default).
    #[serde(default)]
    pub tolerate_failed_items: Option<crate::fan_out::FailedItems>,
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

/// A v2 recipe as written.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeV2 {
    /// Always [`RECIPE_V2_SCHEMA`].
    pub schema: String,
    /// Its name.
    pub name: String,
    /// Its inputs, by name.
    #[serde(default)]
    pub inputs: BTreeMap<String, InputSpec>,
    /// Model roles, by name.
    #[serde(default)]
    pub models: BTreeMap<String, ModelRef>,
    /// Its stages.
    pub stages: Vec<StageSpec>,
    /// What it produces, by name.
    #[serde(default)]
    pub outputs: BTreeMap<String, String>,
    /// Whether a resume may run on a build of other commits than the run
    /// started on: `pinned` (refused by name) unless the recipe says `mixed`.
    #[serde(default)]
    pub revisions: RevisionPolicy,
    /// Whether a brain server that reports no training-record contract may
    /// train: `checked` (it may, recorded as unknown; another contract is
    /// always refused) unless the recipe says `required`.
    #[serde(default)]
    pub contract: ContractPolicy,
}

/// How strictly a run holds its brain servers to the training-record
/// contract this build writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractPolicy {
    /// A server reporting another contract is refused; one reporting none
    /// (an older brain build) trains, with a warning, its contract recorded
    /// as unknown.
    #[default]
    Checked,
    /// A server reporting none is refused too.
    Required,
}

/// Whether a run may continue on another build than it started on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionPolicy {
    /// A resume on a build of other commits (or a dirty or unknown tree that
    /// is not the same executable) is refused, naming each difference.
    #[default]
    Pinned,
    /// A resume may run on another build; every stage's revision manifest
    /// still names the build that ran it.
    Mixed,
}

/// Where a bound product comes from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// A recipe input.
    Input(String),
    /// A port of an earlier stage.
    Stage {
        /// The stage's (expanded) id.
        stage: String,
        /// Its output port.
        port: String,
    },
    /// One item of the stage's `foreach` collection.
    Item,
}

/// One stage of the compiled graph.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompiledStage {
    /// Its id (a subrecipe's stages carry the subrecipe stage's id as a
    /// prefix: `book.sft`).
    pub id: String,
    /// The operation.
    pub operation: String,
    /// The operation's revision.
    pub revision: u32,
    /// Each input port's sources.
    pub inputs: BTreeMap<String, Vec<Source>>,
    /// The checked parameters.
    pub params: Value,
    /// The collection it runs once per item of.
    pub foreach: Option<Source>,
    /// How many items run at once.
    pub concurrency: u32,
    /// How many items may fail and be recorded as absent.
    pub tolerate_failed_items: Option<crate::fan_out::FailedItems>,
    /// The stages it consumes.
    pub depends_on: Vec<String>,
}

/// A compiled recipe.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompiledPlan {
    /// The recipe's name.
    pub name: String,
    /// Its inputs.
    pub inputs: BTreeMap<String, InputSpec>,
    /// Its model roles.
    pub models: BTreeMap<String, ModelRef>,
    /// Its stages, each after what it consumes.
    pub stages: Vec<CompiledStage>,
    /// Its outputs.
    pub outputs: BTreeMap<String, Source>,
    /// Whether a resume may run on another build.
    pub revisions: RevisionPolicy,
    /// How strictly servers are held to the record contract.
    pub contract: ContractPolicy,
}

/// `text` as a reference.
fn reference(text: &str) -> Result<Source, String> {
    if text == "$item" {
        return Ok(Source::Item);
    }
    if let Some(name) = text.strip_prefix("$inputs.") {
        if plain_name(name) {
            return Ok(Source::Input(name.to_owned()));
        }
    }
    if let Some(rest) = text.strip_prefix("$stages.") {
        if let Some((stage, port)) = rest.rsplit_once('.') {
            if stage.split('.').all(plain_name) && plain_name(port) {
                return Ok(Source::Stage {
                    stage: stage.to_owned(),
                    port: port.to_owned(),
                });
            }
        }
    }
    Err(format!(
        "{text:?} is not a reference ($inputs.NAME, $stages.ID.PORT or $item)"
    ))
}

/// What one expansion of a recipe resolves its references against.
struct Scope<'s> {
    /// The id prefix of this recipe's stages (`book.` in a subrecipe).
    prefix: String,
    /// The recipe's inputs, bound: to the outer recipe's sources in a
    /// subrecipe, to themselves at the top.
    inputs: BTreeMap<String, Vec<Source>>,
    /// The kinds of those inputs.
    input_kinds: BTreeMap<String, ArtifactKind>,
    /// Its model roles, mapped to the top recipe's roles.
    models: &'s BTreeMap<String, String>,
}

/// The compiler's working state.
struct Compiler<'r> {
    registry: &'r Registry,
    load: &'r dyn Fn(&str) -> Result<String, String>,
    stages: Vec<CompiledStage>,
    /// The output ports of every expanded stage, by id.
    ports: BTreeMap<String, BTreeMap<String, ArtifactKind>>,
    /// A subrecipe stage's outputs, as the sources they stand for.
    aliases: BTreeMap<(String, String), Source>,
    problems: Vec<String>,
}

impl Compiler<'_> {
    /// The kind of `source` as `scope` sees it, or why it is unknown.
    fn kind_of(&self, source: &Source, scope: &Scope<'_>) -> Result<ArtifactKind, String> {
        match source {
            Source::Input(name) => scope
                .input_kinds
                .get(name)
                .copied()
                .ok_or_else(|| format!("$inputs.{name} is not an input of the recipe")),
            Source::Stage { stage, port } => self
                .ports
                .get(stage)
                .ok_or_else(|| {
                    format!("stage {stage} is not above this one (a stage consumes only stages declared before it)")
                })?
                .get(port)
                .copied()
                .ok_or_else(|| format!("stage {stage} has no output {port}")),
            Source::Item => Err("$item is only for a stage with `foreach`".into()),
        }
    }

    /// `text` resolved in `scope`: an input to what it is bound to, a
    /// stage to its expanded id, a subrecipe's output to what it stands for.
    fn resolve(&self, text: &str, scope: &Scope<'_>) -> Result<Vec<Source>, String> {
        match reference(text)? {
            Source::Input(name) => scope
                .inputs
                .get(&name)
                .cloned()
                .ok_or_else(|| format!("$inputs.{name} is not an input of the recipe")),
            Source::Stage { stage, port } => {
                let id = format!("{}{stage}", scope.prefix);
                let source = self
                    .aliases
                    .get(&(id.clone(), port.clone()))
                    .cloned()
                    .unwrap_or(Source::Stage { stage: id, port });
                Ok(vec![source])
            }
            Source::Item => Ok(vec![Source::Item]),
        }
    }

    fn expand(&mut self, recipe: &RecipeV2, scope: &Scope<'_>, depth: usize) {
        for stage in &recipe.stages {
            let id = format!("{}{}", scope.prefix, stage.id);
            if !plain_name(&stage.id) {
                self.problems
                    .push(format!("stage id {:?} is not a plain name", stage.id));
                continue;
            }
            if self.ports.contains_key(&id) {
                self.problems.push(format!("stage {id} is declared twice"));
                continue;
            }
            if stage.uses == SUBRECIPE {
                self.subrecipe(&id, stage, scope, depth);
            } else {
                self.operation(&id, stage, scope);
            }
        }
    }

    fn operation(&mut self, id: &str, stage: &StageSpec, scope: &Scope<'_>) {
        // The registry outlives the compiler; held apart from `self`, which is
        // written below.
        let registry = self.registry;
        let Some(operation) = registry.get(&stage.uses) else {
            if let Some(why) = registry.withheld(&stage.uses) {
                self.problems.push(format!(
                    "stage {id}: operation {:?} cannot be used: {why}",
                    stage.uses
                ));
                return;
            }
            self.problems.push(format!(
                "stage {id}: no operation {:?}; there are: {}",
                stage.uses,
                registry.names().join(", ")
            ));
            return;
        };
        let descriptor = operation.descriptor();
        if let Err(why) = operation.check(&stage.params) {
            self.problems.push(format!("stage {id}: {why}"));
        }
        for role in operation.model_roles(&stage.params) {
            if !scope.models.contains_key(&role) {
                self.problems
                    .push(format!("stage {id}: model role {role} is not in `models:`"));
            }
        }
        let params = rename_roles(&stage.params, &operation.model_roles(&stage.params), scope);
        let foreach = stage.foreach.as_ref().and_then(|text| {
            match self.resolve(text, scope).map(|mut s| s.pop()) {
                Ok(Some(source)) if source != Source::Item => {
                    if let Err(why) = self.kind_of(&source, scope) {
                        self.problems.push(format!("stage {id}: foreach: {why}"));
                    }
                    Some(source)
                }
                Ok(_) => {
                    self.problems
                        .push(format!("stage {id}: foreach names a collection, not $item"));
                    None
                }
                Err(why) => {
                    self.problems.push(format!("stage {id}: foreach: {why}"));
                    None
                }
            }
        });
        let (concurrency, problems) =
            crate::fan_out::bounds(id, stage, operation, foreach.is_some());
        self.problems.extend(problems);
        if operation.executor().is_some() && descriptor.outputs.len() != 1 {
            self.problems.push(format!(
                "stage {id}: {} computes one product, and it declares {} outputs",
                descriptor.name,
                descriptor.outputs.len()
            ));
        }
        let mut inputs = BTreeMap::new();
        for name in stage.inputs.keys() {
            if descriptor.input(name).is_none() {
                self.problems.push(format!(
                    "stage {id}: {} has no input {name}; its inputs: {}",
                    descriptor.name,
                    port_names(&descriptor.inputs)
                ));
            }
        }
        for port in &descriptor.inputs {
            let Some(binding) = stage.inputs.get(&port.name) else {
                if !port.optional {
                    self.problems
                        .push(format!("stage {id}: input {} is not bound", port.name));
                }
                continue;
            };
            let references = binding.references();
            if references.len() != 1 && !port.many {
                self.problems
                    .push(format!("stage {id}: input {} takes one product", port.name));
            }
            let mut sources = Vec::new();
            for text in references {
                match self.resolve(text, scope) {
                    Ok(resolved) => {
                        for source in resolved {
                            self.check_kind(id, port, &source, foreach.as_ref(), scope);
                            sources.push(source);
                        }
                    }
                    Err(why) => self
                        .problems
                        .push(format!("stage {id}: {}: {why}", port.name)),
                }
            }
            inputs.insert(port.name.clone(), sources);
        }
        let mut depends_on: BTreeSet<String> = inputs
            .values()
            .flatten()
            .chain(foreach.iter())
            .filter_map(|s| match s {
                Source::Stage { stage, .. } => Some(stage.clone()),
                _ => None,
            })
            .collect();
        depends_on.remove(id);
        self.ports.insert(
            id.to_owned(),
            descriptor
                .outputs
                .iter()
                .map(|p| (p.name.clone(), p.kind))
                .collect(),
        );
        self.stages.push(CompiledStage {
            id: id.to_owned(),
            operation: descriptor.name,
            revision: descriptor.revision,
            inputs,
            params,
            foreach,
            concurrency,
            tolerate_failed_items: stage.tolerate_failed_items,
            depends_on: depends_on.into_iter().collect(),
        });
    }

    fn check_kind(
        &mut self,
        id: &str,
        port: &Port,
        source: &Source,
        foreach: Option<&Source>,
        scope: &Scope<'_>,
    ) {
        let found = match source {
            // An item has the kind of the collection it is an item of.
            Source::Item => match foreach {
                Some(collection) => self.kind_of(collection, scope),
                None => Err("$item is only for a stage with `foreach`".into()),
            },
            other => self.kind_of(other, scope),
        };
        match found {
            Ok(kind) if kind != port.kind => self.problems.push(format!(
                "stage {id}: input {} takes {:?}, and it is bound to {:?}",
                port.name, port.kind, kind
            )),
            Ok(_) => {}
            Err(why) => self
                .problems
                .push(format!("stage {id}: {}: {why}", port.name)),
        }
    }

    fn subrecipe(&mut self, id: &str, stage: &StageSpec, scope: &Scope<'_>, depth: usize) {
        if depth >= MAX_NESTING {
            self.problems.push(format!(
                "stage {id}: subrecipes nest deeper than {MAX_NESTING}; is one including itself?"
            ));
            return;
        }
        if let Some(extra) = stage
            .params
            .as_object()
            .and_then(|m| m.keys().find(|k| *k != "recipe"))
        {
            self.problems.push(format!(
                "stage {id}: a subrecipe stage takes only `recipe`, not {extra:?}"
            ));
        }
        if stage.foreach.is_some()
            || stage.concurrency.is_some()
            || stage.tolerate_failed_items.is_some()
        {
            self.problems.push(format!(
                "stage {id}: a subrecipe runs once; foreach, concurrency and tolerate_failed_items belong on its stages"
            ));
        }
        let Some(path) = stage.params.get("recipe").and_then(Value::as_str) else {
            self.problems.push(format!(
                "stage {id}: a subrecipe stage names `with: {{ recipe: PATH }}`"
            ));
            return;
        };
        let inner =
            match (self.load)(path).and_then(|text| parse(&text).map_err(|e| e.0.join("; "))) {
                Ok(inner) => inner,
                Err(why) => {
                    self.problems.push(format!("stage {id}: {path}: {why}"));
                    return;
                }
            };
        let (mut inputs, mut input_kinds) = (BTreeMap::new(), BTreeMap::new());
        for (name, spec) in &inner.inputs {
            let Some(binding) = stage.inputs.get(name) else {
                self.problems.push(format!(
                    "stage {id}: the subrecipe's input {name} is not bound"
                ));
                continue;
            };
            let mut sources = Vec::new();
            for text in binding.references() {
                match self.resolve(text, scope) {
                    Ok(resolved) => sources.extend(resolved),
                    Err(why) => self.problems.push(format!("stage {id}: {name}: {why}")),
                }
            }
            for source in &sources {
                match self.kind_of(source, scope) {
                    Ok(kind) if kind != spec.kind => self.problems.push(format!(
                        "stage {id}: the subrecipe's input {name} takes {:?}, and it is bound to {kind:?}",
                        spec.kind
                    )),
                    Ok(_) => {}
                    Err(why) => self.problems.push(format!("stage {id}: {name}: {why}")),
                }
            }
            inputs.insert(name.clone(), sources);
            input_kinds.insert(name.clone(), spec.kind);
        }
        for name in stage.inputs.keys() {
            if !inner.inputs.contains_key(name) {
                self.problems
                    .push(format!("stage {id}: the subrecipe has no input {name}"));
            }
        }
        // The subrecipe's roles are the outer recipe's of the same name.
        let mut models = BTreeMap::new();
        for role in inner.models.keys() {
            match scope.models.get(role) {
                Some(outer) => {
                    models.insert(role.clone(), outer.clone());
                }
                None => self.problems.push(format!(
                    "stage {id}: the subrecipe's model role {role} is not in the recipe's `models:`"
                )),
            }
        }
        let inner_scope = Scope {
            prefix: format!("{id}."),
            inputs,
            input_kinds,
            models: &models,
        };
        self.expand(&inner, &inner_scope, depth + 1);
        let mut ports = BTreeMap::new();
        for (name, text) in &inner.outputs {
            match self.resolve(text, &inner_scope).map(|mut s| s.pop()) {
                Ok(Some(source)) => {
                    match self.kind_of(&source, &inner_scope) {
                        Ok(kind) => {
                            ports.insert(name.clone(), kind);
                        }
                        Err(why) => self
                            .problems
                            .push(format!("stage {id}: output {name}: {why}")),
                    }
                    self.aliases.insert((id.to_owned(), name.clone()), source);
                }
                Ok(None) => {}
                Err(why) => self
                    .problems
                    .push(format!("stage {id}: output {name}: {why}")),
            }
        }
        self.ports.insert(id.to_owned(), ports);
    }
}

/// `params` with every role it names replaced by the top recipe's role of
/// the scope (a subrecipe's roles are the outer recipe's).
fn rename_roles(params: &Value, roles: &[String], scope: &Scope<'_>) -> Value {
    let mut out = params.clone();
    if let Value::Object(map) = &mut out {
        for value in map.values_mut() {
            if let Some(role) = value.as_str() {
                if roles.iter().any(|r| r == role) {
                    if let Some(top) = scope.models.get(role) {
                        *value = Value::from(top.clone());
                    }
                }
            }
        }
    }
    out
}

fn port_names(ports: &[Port]) -> String {
    if ports.is_empty() {
        return "none".into();
    }
    ports
        .iter()
        .map(|p| p.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// A v2 recipe's text, parsed (its schema read first).
///
/// # Errors
/// Not a v2 recipe, or not its shape.
pub fn parse(text: &str) -> Result<RecipeV2, RecipeError> {
    let schema = serde_yaml::from_str::<serde_yaml::Value>(text)
        .ok()
        .and_then(|v| v.get("schema").and_then(|s| s.as_str()).map(str::to_owned));
    match schema.as_deref() {
        Some(RECIPE_V2_SCHEMA) => {}
        Some(other) => {
            return Err(RecipeError(vec![format!(
                "this is a {other} file, not a recipe ({RECIPE_V2_SCHEMA})"
            )]))
        }
        None => {
            return Err(RecipeError(vec![format!(
                "no `schema:`; a recipe begins with `schema: {RECIPE_V2_SCHEMA}`"
            )]))
        }
    }
    serde_yaml::from_str(text).map_err(|e| RecipeError(vec![e.to_string()]))
}

/// Compile `text` against `registry`; `load` reads a subrecipe's text by its
/// path in the task repository.
///
/// # Errors
/// Every problem found (module docs).
pub fn compile(
    text: &str,
    registry: &Registry,
    load: &dyn Fn(&str) -> Result<String, String>,
) -> Result<CompiledPlan, RecipeError> {
    let (plan, problems) = compile_partly(text, registry, load)?;
    if problems.is_empty() {
        Ok(plan)
    } else {
        Err(RecipeError(problems))
    }
}

/// [`compile`]'s plan of every stage that compiled, with every problem
/// found: what a caller that checks further (the files the stages name, the
/// connections) needs to report all problems at once.
///
/// # Errors
/// The text is not a v2 recipe at all.
pub fn compile_partly(
    text: &str,
    registry: &Registry,
    load: &dyn Fn(&str) -> Result<String, String>,
) -> Result<(CompiledPlan, Vec<String>), RecipeError> {
    let recipe = parse(text)?;
    let mut problems = Vec::new();
    for name in recipe
        .inputs
        .keys()
        .chain(recipe.outputs.keys())
        .chain(recipe.models.keys())
    {
        if !plain_name(name) {
            problems.push(format!("{name:?} is not a plain name"));
        }
    }
    let roles: BTreeMap<String, String> = recipe
        .models
        .keys()
        .map(|role| (role.clone(), role.clone()))
        .collect();
    let scope = Scope {
        prefix: String::new(),
        inputs: recipe
            .inputs
            .keys()
            .map(|name| (name.clone(), vec![Source::Input(name.clone())]))
            .collect(),
        input_kinds: recipe
            .inputs
            .iter()
            .map(|(n, s)| (n.clone(), s.kind))
            .collect(),
        models: &roles,
    };
    let mut compiler = Compiler {
        registry,
        load,
        stages: Vec::new(),
        ports: BTreeMap::new(),
        aliases: BTreeMap::new(),
        problems,
    };
    compiler.expand(&recipe, &scope, 0);
    let mut outputs = BTreeMap::new();
    for (name, text) in &recipe.outputs {
        match compiler.resolve(text, &scope).map(|mut s| s.pop()) {
            Ok(Some(source)) => {
                if let Err(why) = compiler.kind_of(&source, &scope) {
                    compiler.problems.push(format!("output {name}: {why}"));
                }
                outputs.insert(name.clone(), source);
            }
            Ok(None) => {}
            Err(why) => compiler.problems.push(format!("output {name}: {why}")),
        }
    }
    // Declaration order is an order in which each stage comes after what it
    // consumes: a stage consumes only stages declared above it.
    let (problems, stages) = (compiler.problems, compiler.stages);
    let plan = CompiledPlan {
        name: recipe.name,
        inputs: recipe.inputs,
        models: recipe.models,
        stages,
        outputs,
        revisions: recipe.revisions,
        contract: recipe.contract,
    };
    Ok((plan, problems))
}
