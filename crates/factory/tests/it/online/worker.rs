// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning in which an
// agent practises with exactly the tools its task environment gives it, for
// its clients. If your team needs expertise in training agents on their own
// rollouts in real task environments, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: `train.online` with `agent:` runs each rollout on the execution
//! worker in its task's own environment, so the agent is offered exactly the
//! tools the task declares (here an MCP server its image installs, the only
//! tool its agent configuration allows) and its calls run there. Every
//! request samples at the run's temperature with a seed of the rollout's own
//! and asks for the policy trace; the task's own verifier grades each
//! rollout; the records the learner is sent are the rollout's tool calls,
//! exact, one per model call.
//!
//! Needs the real worker: `splinter-harbor-worker` and `sven` on `PATH`, and
//! the network to build the task's image (`make test/online/tools`).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use splinter_agent::policy_endpoint::Endpoint;
use splinter_factory::backend::HarborWorker;
use splinter_factory::connections::Connections;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, Services};
use splinter_factory::training::BrainTrainer;
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::ArtifactStore;

/// The fixture task set: one task whose agent's only tool is the `append`
/// tool of the `notes` MCP server its environment declares.
const FIXTURES: &str = "tests/it/fixtures/online-tools";
/// The tool as the agent runtime names it: `<server>-<tool>`.
const TOOL: &str = "notes-append";

// seed: a recipe states every seed; the rollouts draw theirs from it.
const RECIPE: &str = r#"
schema: splinter.recipe/v2
name: online-tools
inputs:
  course: { kind: files, path: collections/tools.yaml }
models:
  student: "brain:Qwen/Toy"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course } }
  - id: online
    uses: train.online
    inputs: { tasks: $stages.tasks.tasks }
    with:
      base: student
      groups: 3
      group_size: 4
      max_concurrent_attempts: 4
      max_staleness: 100
      rollout_tokens: 4
      step_tokens: 4
      queue_tokens: 64
      checkpoint_every: 1
      temperature: 0.9
      seed: 5
      agent: { agent: sven, environment: environments/tools.yaml }
"#;

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// The tools a request offers its model, by name.
fn offered(request: &Value) -> Vec<String> {
    request["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["function"]["name"].as_str().map(str::to_owned))
        .collect()
}

/// `message` as a brain server answers it: with the policy trace (at the
/// temperature the request asked for) and the version that decoded it.
fn traced(request: &Value, mut message: Value) -> Value {
    let turns = request["messages"].as_array().map_or(0, Vec::len) as u64;
    message["policy_trace"] = json!({
        "prompt_token_ids": (0..turns + 1).collect::<Vec<u64>>(),
        "generation_token_ids": [21, 22],
        "generation_log_probs": [-0.5, -0.25],
        "log_prob_temperature": request["temperature"].as_f64().unwrap_or(1.0),
    });
    message["brain"] = json!({
        "policy_version": 0,
        "logprob_kind": "policy",
        "tokenizer_digest": splinter_model::scripted_stream::SCRIPTED_TOKENIZER_DIGEST,
    });
    message
}

fn call(tool: &str, arguments: &Value) -> Value {
    json!({"role": "assistant", "content": null, "tool_calls": [{
        "id": "call-1", "type": "function",
        "function": {"name": tool, "arguments": arguments.to_string()}
    }]})
}

fn done() -> Value {
    json!({"role": "assistant", "content": "Done."})
}

/// Whether a tool's result is in the conversation.
fn answered(request: &Value) -> bool {
    request["messages"]
        .as_array()
        .is_some_and(|m| m.iter().any(|m| m["role"] == "tool"))
}

/// The fixture's policy: when its request's seed is even it calls the one
/// tool it is offered, once, with what the task asks for; otherwise it says
/// it is done at once (so a group's rewards differ).
fn notes_policy(request: &Value) -> anyhow::Result<Value> {
    let tools = offered(request);
    anyhow::ensure!(
        tools == [TOOL],
        "the agent is offered exactly the task's declared tool, not {tools:?}"
    );
    let solves = request["seed"].as_u64().is_some_and(|s| s % 2 == 0);
    Ok(traced(
        request,
        if answered(request) || !solves {
            done()
        } else {
            call(TOOL, &json!({"text": "hello from the tool"}))
        },
    ))
}

/// Arguments that satisfy `schema` (a tool's input schema): each required
/// property with the least value of its type.
fn arguments_for(schema: &Value) -> Value {
    match schema["type"].as_str() {
        Some("object") => {
            let mut object = serde_json::Map::new();
            for name in schema["required"].as_array().into_iter().flatten() {
                if let Some(name) = name.as_str() {
                    object.insert(name.into(), arguments_for(&schema["properties"][name]));
                }
            }
            Value::Object(object)
        }
        Some("array") => json!([arguments_for(&schema["items"])]),
        Some("integer" | "number") => json!(0),
        Some("boolean") => json!(false),
        _ => json!(""),
    }
}

/// A policy for any task set: it calls the first tool it is offered once,
/// with arguments its schema admits, then says it is done.
fn any_task_policy(request: &Value) -> anyhow::Result<Value> {
    let tools = request["tools"].as_array().cloned().unwrap_or_default();
    let first = tools
        .first()
        .ok_or_else(|| anyhow::anyhow!("no tool is offered"))?;
    let name = first["function"]["name"].as_str().unwrap_or_default();
    Ok(traced(
        request,
        if answered(request) {
            done()
        } else {
            call(name, &arguments_for(&first["function"]["parameters"]))
        },
    ))
}

/// One `train.online` run over the task folders under `tasks`, its rollouts
/// on the real worker against `policy`.
struct Ran {
    scratch: tempfile::TempDir,
    result: Result<(), splinter_factory::run::RunError>,
    exchanges: Vec<splinter_agent::policy_endpoint::Exchange>,
}

impl Ran {
    fn run_dir(&self) -> std::path::PathBuf {
        self.scratch.path().join("run")
    }
}

fn run_online(tasks: &Path, groups: u32, policy: fn(&Value) -> anyhow::Result<Value>) -> Ran {
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path().join("repo");
    copy_tree(tasks, &root.join("tasks/tools"));
    write(
        &root.join("collections/tools.yaml"),
        "schema: splinter.collection/v1\nname: tools\nroot: tasks/tools\n",
    );
    write(
        &root.join("environments/tools.yaml"),
        "schema: splinter.environment/v1\nsandbox: { resources: { timeout: 5m } }\nnetwork: { mode: model-gateway-only }\n",
    );
    let store = scratch.path().join("artifacts");
    let adapters = scratch.path().join("adapters");
    std::fs::create_dir_all(&store).unwrap();
    std::fs::create_dir_all(&adapters).unwrap();
    let trainer_service = Arc::new(
        ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")])
            .with_training(JobScript::over(store.clone()))
            .serving_adapters(adapters.clone()),
    );
    // The policy is listed, as brain lists a published adapter, under the id
    // the learner publishes, with the sampling controls it honours.
    let published = Arc::clone(&trainer_service);
    let served = Endpoint::start("served", Box::new(policy))
        .unwrap()
        .listing(Box::new(move || {
            published
                .jobs()
                .created()
                .iter()
                .filter_map(|doc| doc["publish_adapter"].as_str())
                .map(|id| {
                    json!({"id": id, "object": "model", "context_length": 8192,
                                 "supported_parameters": ["tools", "temperature", "top_p", "seed"]})
                })
                .collect()
        }));
    let endpoints = BrainEndpoints::new(vec![trainer_service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&store).unwrap())
        .with_adapter_directory(adapters)
        .with_refused_wait(Duration::from_millis(50));
    let trainer = BrainTrainer::new(endpoints, scratch.path().join("models"));
    // The agent reaches the policy over the base's connection: here the
    // scripted policy, under whatever id the learner publishes.
    let connections = Connections::parse(&format!(
        "connections:\n  brain: {{ protocol: openai, url: \"{}\" }}\n",
        served.url()
    ))
    .unwrap();
    let recipe = RECIPE.replace("groups: 3", &format!("groups: {groups}"));
    let resolved = resolve(&root, &recipe, connections).unwrap();
    let backend = HarborWorker::new("splinter-harbor-worker")
        .expect("splinter-harbor-worker is on PATH (make worker/install)");
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let result = Factory::open(&resolved, &scratch.path().join("run"), services, "test")
        .unwrap()
        .run()
        .map(drop);
    Ran {
        exchanges: served.exchanges(),
        scratch,
        result,
    }
}

/// Every request sampled at the run's temperature with a seed of its
/// rollout's own, and every tool result that came back.
fn tool_results(exchanges: &[splinter_agent::policy_endpoint::Exchange]) -> Vec<Value> {
    assert!(
        exchanges.iter().all(|e| e.reply.is_ok()),
        "{:?}",
        exchanges.iter().find(|e| e.reply.is_err())
    );
    for exchange in exchanges {
        let request = &exchange.request;
        assert!(
            (request["temperature"].as_f64().unwrap() - 0.9).abs() < 1e-6,
            "{request}"
        );
        assert!(
            request["seed"].is_u64(),
            "each rollout's own seed: {request}"
        );
    }
    exchanges
        .iter()
        .flat_map(|e| {
            e.request["messages"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter(|m| m["role"] == "tool")
        .collect()
}

#[test]
#[ignore = "needs splinter-harbor-worker and sven on PATH and the network to build the task's image: make test/online/tools"]
fn an_online_rollout_is_given_the_tools_its_task_declares_and_graded_by_its_verifier() {
    let ran = run_online(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES),
        3,
        notes_policy,
    );
    ran.result.as_ref().unwrap();
    let results = tool_results(&ran.exchanges);
    assert!(
        !results.is_empty()
            && results
                .iter()
                .all(|m| m["content"].to_string().contains("hello from the tool")),
        "every tool call ran in the task's environment and its result came back: {results:?}"
    );
    let view = ran.run_dir().join("online");
    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(view.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["rollouts"]["finished"], 12, "{report:#}");
    assert_eq!(report["rollouts"]["not_exact"], 0, "{report:#}");
    assert_eq!(report["rollouts"]["failures"], 0, "{report:#}");
    let mean = report["reward_mean"].as_f64().unwrap();
    assert!(
        mean > 0.0 && mean < 1.0,
        "the verifier passed the rollouts that used the tool and failed the others: {report:#}"
    );
    assert!(
        report["groups"]["committed"].as_u64().unwrap() >= 1,
        "groups whose rewards differ are trained: {report:#}"
    );
    let experiences = std::fs::read_to_string(view.join("experiences.jsonl")).unwrap();
    assert!(!experiences.is_empty());
}

/// Where `make test/online/tools/tasks` copies the task set it is given.
const TASK_SET: &str = "../../target/online-tools/tasks";

/// The MCP servers a task's `task.toml` declares, by name.
fn declared_servers(task: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(task.join("task.toml")).unwrap();
    let toml: toml::Table = text.parse().unwrap();
    toml.get("environment")
        .and_then(|e| e.get("mcp_servers"))
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|s| {
            s.get("name")
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

#[test]
#[ignore = "needs a task set copied by make test/online/tools/tasks TASKS=<dir>, the worker, sven and the network"]
fn any_task_sets_declared_tools_reach_the_agent_of_an_online_rollout() {
    let tasks = Path::new(env!("CARGO_MANIFEST_DIR")).join(TASK_SET);
    let folders: Vec<std::path::PathBuf> = std::fs::read_dir(&tasks)
        .unwrap_or_else(|e| {
            panic!(
                "{}: {e}: run make test/online/tools/tasks TASKS=<dir>",
                tasks.display()
            )
        })
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("task.toml").is_file())
        .collect();
    assert!(
        !folders.is_empty(),
        "{} holds no task folder",
        tasks.display()
    );
    let servers: Vec<String> = folders.iter().flat_map(|f| declared_servers(f)).collect();
    assert!(
        !servers.is_empty(),
        "the task set declares no tool: nothing to check"
    );
    let ran = run_online(&tasks, 1, any_task_policy);
    for exchange in &ran.exchanges {
        for tool in offered(&exchange.request) {
            assert!(
                servers.iter().any(|s| tool.starts_with(&format!("{s}-"))),
                "{tool} is a tool of a server the task declares ({servers:?})"
            );
        }
    }
    let results = tool_results(&ran.exchanges);
    assert!(!results.is_empty(), "the declared tools ran and answered");
    // A policy that does not solve the task earns rewards that do not differ,
    // so nothing may be trained; every rollout is still graded by its task's
    // verifier, never failed by the infrastructure.
    if let Err(e) = &ran.result {
        assert!(
            e.to_string().contains("committed no published version"),
            "{e}"
        );
    }
    let mut rewards = Vec::new();
    for entry in walkdir(&ran.run_dir().join("work")) {
        if entry.ends_with("verifier/reward.txt") {
            rewards.push(std::fs::read_to_string(&entry).unwrap().trim().to_string());
        }
    }
    assert_eq!(
        rewards.len(),
        4,
        "one graded trial per rollout: {rewards:?}"
    );
    assert!(rewards.iter().all(|r| r == "0" || r == "1"), "{rewards:?}");
}

fn walkdir(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walkdir(&path));
        } else {
            found.push(path);
        }
    }
    found
}

/// A worker program that never ends a batch: it starts a process of its own
/// (as a sandbox would) and both sleep; it answers `--version`.
fn stuck_worker(dir: &Path) -> std::path::PathBuf {
    let program = dir.join("worker");
    let pids = dir.join("pids");
    write(
        &program,
        &format!(
            "#!/bin/bash\n[ \"$1\" = --version ] && {{ echo stuck 1; exit 0; }}\nsleep 600 &\necho $! >> '{}'\necho $$ >> '{}'\nwait\n",
            pids.display(),
            pids.display()
        ),
    );
    std::fs::set_permissions(
        &program,
        std::os::unix::fs::PermissionsExt::from_mode(0o755),
    )
    .unwrap();
    program
}

#[test]
fn a_cancelled_batch_kills_the_worker_and_everything_it_started() {
    let dir = tempfile::tempdir().unwrap();
    let worker = HarborWorker::new(stuck_worker(dir.path())).unwrap();
    let out = dir.path().join("out");
    let asked = std::time::Instant::now();
    let cancelled = || asked.elapsed() > Duration::from_millis(500);
    let result = splinter_factory::backend::ExecutionBackend::execute_cancellable(
        &worker,
        splinter_factory::backend::Batch {
            specs: &[],
            tasks: dir.path(),
            files: dir.path(),
            out: &out,
            concurrency: 1,
        },
        &cancelled,
    );
    assert!(
        matches!(
            result,
            Err(splinter_factory::backend::BackendError::Cancelled)
        ),
        "{result:?}"
    );
    assert!(asked.elapsed() < Duration::from_secs(10), "stopped at once");
    let pids = std::fs::read_to_string(dir.path().join("pids")).unwrap();
    // A killed process is gone (or a zombie) once the kernel has delivered the signal: bounded.
    let alive = |pid: &str| {
        let state = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
        !(state.is_empty() || state.split_whitespace().nth(2) == Some("Z"))
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while pids.split_whitespace().any(alive) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    for pid in pids.split_whitespace() {
        assert!(
            !alive(pid),
            "process {pid} of the batch outlived its cancel"
        );
    }
}
