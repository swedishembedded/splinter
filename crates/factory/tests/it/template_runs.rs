// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements starter repositories for learning
// campaigns that are proven to run, not only to parse, for its clients. If
// your team needs expertise in turning a learning recipe into a first
// working run, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Templates, executed: `init` writes each into an empty directory and every
//! recipe of it is then run end to end on the repository's fake services (a
//! scripted execution backend, a scripted trainer and scripted in-process
//! models), as its own comments tell a user to run it. What each stage of
//! the recipe made is read from the run directory, as a user reads it.

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use splinter_agent::solve::Model;
use splinter_core::factory::execution::ModelBinding;
use splinter_factory::execute::Models;
use splinter_factory::resolve::resolve;
use splinter_factory::run::dataset::dataset_of;
use splinter_factory::run::eval::result_of;
use splinter_factory::run::tasks::tasks_of;
use splinter_factory::run::training::candidate_of;
use splinter_factory::run::{fork, status, verify, Factory, Services};
use splinter_factory::templates::init;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::connections;

/// A model that always gives one reply.
struct Reply(String);

#[async_trait::async_trait]
impl ModelProvider for Reply {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "scripted-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(self.0.clone())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// The recipe's teacher (a hosted model) proposes `proposal`; its student
/// (a brain model) answers `answer`.
struct Classroom {
    proposal: Arc<Reply>,
    answer: Arc<Reply>,
}

impl Models for Classroom {
    fn model(&self, binding: &ModelBinding) -> Result<Model, String> {
        let provider = if binding.model.contains("deepseek") {
            &self.proposal
        } else {
            &self.answer
        };
        Ok(Model::new(provider.clone(), "scripted/model-1"))
    }
}

/// One task the teacher proposes from the template's guide, and the
/// student's answer to it.
const QUOTE: &str = "The example tool writes its log to example.log and keeps the last 7 logs";

fn classroom() -> Classroom {
    // Section 3 of the guide is its logging section; the licence comments
    // and the title come first.
    let proposal = serde_json::json!({ "tasks": [{
        "instruction": "How many logs does the example tool keep?",
        "subject": "example tool",
        "reference": "the last 7 logs",
        "evidence": [{ "section": 3, "quote": QUOTE }],
    }]});
    Classroom {
        proposal: Arc::new(Reply(proposal.to_string())),
        answer: Arc::new(Reply("It keeps the last 7 logs.".into())),
    }
}

/// The repository's fake services, with the models a recipe's typed calls use.
struct Fakes {
    backend: FakeBackend,
    trainer: FakeTrainer,
    models: Classroom,
}

impl Fakes {
    /// Candidates pass every task, the base fails them all, and the tasks
    /// named in `solved` are solved by whoever attempts them first (`times`
    /// attempts), so there is something to train on.
    fn new(solved: &[&str], times: usize) -> Self {
        let backend = FakeBackend {
            candidate: Some(Outcome::Pass),
            ..FakeBackend::with(Outcome::Fail)
        };
        for task in solved {
            backend.script(task, &vec![Outcome::Pass; times]);
        }
        Self {
            backend,
            trainer: FakeTrainer::default(),
            models: classroom(),
        }
    }
}

/// A template written into a scratch directory.
fn written(template: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    init(template, dir.path()).unwrap();
    dir
}

/// Runs the recipe `text` of the repository `repo` into `out`.
fn run_text(repo: &Path, text: &str, out: &Path, fakes: &Fakes) {
    let resolved =
        resolve(repo, text, connections()).unwrap_or_else(|e| panic!("the recipe resolves: {e:?}"));
    let services = Services {
        backend: &fakes.backend,
        trainer: Some(&fakes.trainer),
        models: Some(&fakes.models),
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    Factory::open(&resolved, out, services, "template-test")
        .unwrap()
        .run()
        .unwrap_or_else(|e| panic!("{}: {e}", out.display()));
}

/// Runs the recipe file `recipe` of the repository `repo` into `out`.
fn run_recipe(repo: &Path, recipe: &str, out: &Path, fakes: &Fakes) {
    let text = std::fs::read_to_string(repo.join(recipe)).unwrap();
    run_text(repo, &text, out, fakes);
}

/// The `(id, uses)` of every stage of the recipe text.
fn stages_of(text: &str) -> Vec<(String, String)> {
    let recipe: Value = serde_yaml::from_str(text).unwrap();
    recipe["stages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["id"].as_str().unwrap().to_owned(),
                s["uses"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

/// What a finished run of `recipe` must hold: every stage committed a
/// product, the run verifies on its own, and what each kind of stage makes
/// is there to be read.
fn assert_run_of(recipe: &str, out: &Path) {
    let report = status(out).unwrap();
    for (id, uses) in stages_of(recipe) {
        let committed = report
            .iter()
            .filter(|s| s.key == id || s.key.starts_with(&format!("{id}/")))
            .collect::<Vec<_>>();
        assert!(
            !committed.is_empty()
                && committed
                    .iter()
                    .all(|s| s.state == "completed" && s.product.is_some()),
            "stage {id} ({uses}) committed a product: {report:?}"
        );
        let view = out.join(&id);
        match uses.as_str() {
            "tasks.pin" => {
                assert!(
                    !tasks_of(&view).unwrap().tasks.is_empty(),
                    "{id} pins tasks"
                );
            }
            "dataset.project" => {
                let dataset = dataset_of(&view).unwrap();
                let train = std::fs::read_to_string(view.join("train.jsonl")).unwrap();
                assert!(
                    train.lines().count() >= 1,
                    "{id} wrote a record: {dataset:?}"
                );
            }
            "model.train" => {
                let candidate = candidate_of(&view).unwrap();
                assert!(
                    candidate
                        .artifacts
                        .iter()
                        .any(|a| a.path.ends_with(".safetensors")),
                    "{id} produced a candidate adapter"
                );
            }
            "evaluation" => {
                result_of(&view).unwrap_or_else(|e| panic!("{id} produced a result: {e}"));
            }
            "promotion" => {
                let release = out.join("release/release.json");
                assert!(release.is_file(), "{id} wrote {}", release.display());
            }
            "harness.search" => {
                for file in ["harness.yaml", "search.json"] {
                    assert!(view.join(file).is_file(), "{id} wrote {file}");
                }
            }
            _ => {}
        }
    }
    let problems = verify(out).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}

/// Runs the template's one recipe as its header says and checks the run.
fn runs_as_written(
    template: &str,
    recipe: &str,
    fakes: &Fakes,
) -> (tempfile::TempDir, std::path::PathBuf) {
    let repo = written(template);
    let out = repo.path().join("runs").join(template);
    run_recipe(repo.path(), recipe, &out, fakes);
    let text = std::fs::read_to_string(repo.path().join(recipe)).unwrap();
    assert_run_of(&text, &out);
    (repo, out)
}

#[test]
fn agent_sft_runs_as_written() {
    runs_as_written(
        "agent-sft",
        "recipes/agent-sft.yaml",
        &Fakes::new(&["train/001-hello"], 4),
    );
}

#[test]
fn knowledge_learning_runs_as_written() {
    runs_as_written(
        "knowledge-learning",
        "recipes/knowledge-learning.yaml",
        &Fakes::new(&[], 0),
    );
}

#[test]
fn harness_search_runs_as_written() {
    runs_as_written(
        "harness-search",
        "recipes/harness-search.yaml",
        &Fakes::new(&["train/001-hello"], 4),
    );
}

#[test]
fn repository_practice_runs_as_written() {
    // Whoever attempts a seeded fault solves it: the tasks are the generator's.
    let fakes = Fakes {
        backend: FakeBackend {
            default: Some(Outcome::Pass),
            candidate: Some(Outcome::Pass),
            ..FakeBackend::default()
        },
        ..Fakes::new(&[], 0)
    };
    runs_as_written(
        "repository-practice",
        "recipes/repository-practice.yaml",
        &fakes,
    );
}

#[test]
fn generations_run_as_written_and_the_next_is_a_fork_naming_the_champion() {
    let fakes = Fakes::new(&["train/001-hello"], 4);
    let (repo, first) = runs_as_written("generations", "recipes/generation.yaml", &fakes);
    let release = |run: &Path| -> Value {
        serde_json::from_slice(&std::fs::read(run.join("release/release.json")).unwrap()).unwrap()
    };
    let g1 = release(&first);
    assert!(
        g1["promoted"].is_string(),
        "the first generation promotes: {g1:#}"
    );

    // The recipe's own instructions for the next generation: name the
    // champion, make it the baseline, and fork the finished run.
    let text = std::fs::read_to_string(repo.path().join("recipes/generation.yaml")).unwrap();
    let next = text
        .replace(
            "models:\n",
            &format!("models:\n  champion: \"champion:{}\"\n", first.display()),
        )
        .replace("baseline: student", "baseline: champion");
    let second = repo.path().join("runs/g2");
    fork(&first, &second).unwrap();
    run_text(repo.path(), &next, &second, &fakes);
    assert_run_of(&next, &second);
    let g2 = release(&second);
    assert!(
        g2["champion"]["adapter"].is_string(),
        "the next generation holds a champion: {g2:#}"
    );
}
