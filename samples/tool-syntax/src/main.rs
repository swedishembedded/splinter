// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does learning from its own verified experience teach a small model to find
//! out what it needs before acting?
//!
//! The task the model is given cannot be answered by reading. Three deployment
//! configurations sit in the workspace and nothing says which one the host is
//! running; the only way to find out is to ask the service. A model that
//! guesses has a one-in-two chance on any instance and cannot do better than
//! chance across a matched pair - which is what makes the score mean
//! something.
//!
//! Subcommands, in the order they are meant to be used:
//!
//! ```text
//! audit        the task catalog checks itself. No model, no GPU, no network.
//! baseline     what the base model scores, before anything is learned.
//! demonstrate  verified scripted episodes -> training data (train + held-out).
//! train        one monitored LoRA fine-tune on that data, on CUDA.
//! measure      the baseline's measurement for another arm: the trained adapter.
//! ```
//!
//! Swedish Embedded AB implements closed-loop learning systems - agents that
//! improve from their own verified experience rather than from hand-written
//! training data - for its clients. If your team needs expertise in agent
//! training loops, sample design, or promotion gating for small models,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::PathBuf;

use splinter_tool_syntax::{
    baseline_effective, capture_path, listed_models, records_from_performance, require_adapter,
    run_agent, run_verifier, run_witness, summary_path, to_jsonl, train, upstream_of, Action,
    ArmScore, Bounds, Endpoint, Episode, Family, Outcome, Recorder, ServedModel, TrainRequest,
};

const USAGE: &str = "\
usage: splinter-tool-syntax <command> [options]

  audit                    check the task catalog: no model, no GPU, no network
  baseline [options]       measure the base model on the frozen instances
  measure --arm NAME ...   the same measurement for a named arm, e.g. the trained
                           model served as an adapter
  demonstrate [options]    drive correct episodes with a scripted model and
                           emit the verified ones as training data
  train --data DIR --base DIR --out DIR --seed N [--steps N --rank N --alpha X
        --lr X --grad-accum N --eval-every N --patience N]
                           one monitored LoRA fine-tune on `demonstrate`'s output

options for baseline:
  --instances N            episodes per twin (default 4)
  --run-dir DIR            where episodes are materialised (default: a temp dir)
  --model ID               resident manifest id (default brain/qwen3)
  --base-url URL           the served endpoint (default: sven's own configuration)
  --api-key KEY            key for that endpoint
  --backend cuda --device gpuN
                           where train runs (every command takes them)
  --report FILE.json       write the machine-readable result here
  --seed N                 episode i samples with seed N+i (required to measure)
  --arm NAME               the arm's name in the report (measure)
  --adapter-digest D       measure only: refuse to run unless the server lists
                           the --model id as the adapter with digest D
  --weights DIR            demonstrate: also encode the dataset with this checkpoint
  --request TEXT           replaces the frozen request: a probe, never a measured arm
  --hint TEXT              appended to the request for THIS arm only
  --prompt-from FILE       requests.json from a baseline run (demonstrate only)

A hint is an exploration aid, not a change to the task. Collection may use one;
the measured arms must not, or the number says what the prompt did rather than
what the model learned.

The model is ordinary sven configuration, not a flag: point sven at a served
brain and the same binary measures whatever it is serving. The id must name the
RESIDENT (brain/qwen3), not the model-store spelling - only the resident ever
receives a trained adapter.
";

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    splinter_sdk::device::DeviceFlags::take(&mut args)
        .map_err(anyhow::Error::msg)?
        .apply()?;
    match args.first().map(String::as_str) {
        Some("audit") => audit(),
        Some("baseline") => measure(&args[1..], Some("baseline")),
        Some("measure") => measure(&args[1..], None),
        Some("train") => train_command(&args[1..]),
        Some("demonstrate") => demonstrate(&args[1..]),
        Some("--help") | Some("-h") | None => {
            print!("{USAGE}");
            Ok(())
        }
        Some(other) => {
            eprintln!("unknown command {other:?}\n");
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    }
}

fn family_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap_or_else(|| {
            unreachable!("a package two levels below the repository root has two ancestors")
        })
        .join("tasks/config-discovery")
}

/// The catalog checking itself. Delegates to the family's own audit, which is
/// the same one `make samples/tool-syntax/audit` runs - there is one audit, not
/// one per caller.
fn audit() -> anyhow::Result<()> {
    let root = family_root();
    let family = Family::load(&root).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "family: {} ({} predicates)",
        family.id(),
        family.predicates().names().len()
    );

    let status = std::process::Command::new("python3")
        .arg(root.join("world/audit.py"))
        .current_dir(&root)
        .status()?;
    if !status.success() {
        anyhow::bail!("the task catalog did not pass its own audit");
    }
    Ok(())
}

struct BaselineOptions {
    instances: usize,
    run_dir: PathBuf,
    model: ServedModel,
    /// Named rather than auto-detected on purpose. A sample that lets
    /// its endpoint be discovered is a sample that cannot say afterwards
    /// which server produced the number - and on a shared machine the answer
    /// changes between runs.
    base_url: Option<String>,
    api_key: Option<String>,
    report: Option<PathBuf>,
    /// Appended to the frozen request. Recorded in the report so an arm that
    /// used one can never be mistaken for one that did not.
    hint: Option<String>,
    /// A `requests.json` written by a previous `baseline` run, holding the
    /// prompt the agent really sends.
    prompt_from: Option<PathBuf>,
    /// The checkpoint a demonstration dataset is encoded against before it is
    /// written out (demonstrate only).
    weights: Option<PathBuf>,
    /// What an arm served from an adapter must prove: the digest the server
    /// lists for the model it is asked for (baseline and measure only).
    adapter_digest: Option<String>,
    /// Replaces the frozen request, for a probe of whether a behaviour
    /// follows the task or only its wording. Recorded in the report; an arm
    /// that has one is a probe, not a measured arm.
    request: Option<String>,
    /// The arm's name in the report.
    arm: Option<String>,
    /// Episode `i` samples with seed `seed + i`. No default: a seed nobody
    /// chose is a seed nobody can vary.
    seed: Option<u64>,
}

fn parse_baseline(args: &[String]) -> anyhow::Result<BaselineOptions> {
    let mut instances = 4usize;
    let mut run_dir = std::env::temp_dir().join(format!("tool-syntax-{}", std::process::id()));
    let mut model = ServedModel::qwen3();
    let mut base_url = None;
    let mut api_key = None;
    let mut report = None;
    let mut hint = None;
    let mut prompt_from = None;
    let mut weights = None;
    let mut adapter_digest = None;
    let mut request = None;
    let mut arm = None;
    let mut seed = None;

    let mut i = 0;
    while i < args.len() {
        let take = |i: &mut usize| -> anyhow::Result<String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--instances" => instances = take(&mut i)?.parse()?,
            "--run-dir" => run_dir = PathBuf::from(take(&mut i)?),
            "--model" => model = ServedModel::resident(take(&mut i)?),
            "--base-url" => base_url = Some(take(&mut i)?),
            "--api-key" => api_key = Some(take(&mut i)?),
            "--report" => report = Some(PathBuf::from(take(&mut i)?)),
            "--hint" => hint = Some(take(&mut i)?),
            "--prompt-from" => prompt_from = Some(PathBuf::from(take(&mut i)?)),
            "--weights" => weights = Some(PathBuf::from(take(&mut i)?)),
            "--adapter-digest" => adapter_digest = Some(take(&mut i)?),
            "--request" => request = Some(take(&mut i)?),
            "--arm" => arm = Some(take(&mut i)?),
            "--seed" => seed = Some(take(&mut i)?.parse()?),
            other => anyhow::bail!("unknown option {other:?}"),
        }
        i += 1;
    }
    Ok(BaselineOptions {
        instances,
        run_dir,
        model,
        base_url,
        api_key,
        report,
        hint,
        prompt_from,
        weights,
        adapter_digest,
        request,
        arm,
        seed,
    })
}

/// One measured arm. `baseline` is the arm of the untrained model; `measure`
/// names its arm (`--arm`), and with `--adapter-digest` refuses to run unless
/// the server proves it serves that adapter under the model id asked for.
fn measure(args: &[String], default_arm: Option<&str>) -> anyhow::Result<()> {
    let options = parse_baseline(args)?;
    let arm = options
        .arm
        .clone()
        .or_else(|| default_arm.map(String::from))
        .ok_or_else(|| anyhow::anyhow!("measure needs --arm NAME"))?;
    if default_arm.is_some() && options.adapter_digest.is_some() {
        anyhow::bail!("the baseline is the untrained model; use `measure --arm` for an adapter");
    }
    let seed = options
        .seed
        .ok_or_else(|| anyhow::anyhow!("--seed N is required: episode i samples with seed N+i"))?;
    let family = Family::load(&family_root()).map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("{arm}: {} on {}", family.id(), options.model);
    match &options.request {
        Some(request) => println!(
            "request OVERRIDDEN (a probe, not the frozen request): {}",
            request.replace('\n', " ")
        ),
        None => println!("request: {}", family.request().replace('\n', " ")),
    }
    println!();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    if let Some(digest) = &options.adapter_digest {
        let url = options
            .base_url
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--adapter-digest needs --base-url to ask"))?;
        let listed = runtime.block_on(listed_models(url, options.api_key.as_deref()))?;
        require_adapter(&listed, options.model.api_model(), digest)?;
        println!("served adapter verified: {} is {digest}", options.model);
    }
    // Control: the environment, solved by something that is not a model. If
    // the witness cannot solve an instance, everything below measures a broken
    // harness - and a score of zero looks exactly like a model that could not
    // do the task.
    {
        let dir = options.run_dir.join("control-witness");
        let episode = Episode::start(&family, &dir, &family.live_choices()[0])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        match run_witness(&family, &episode) {
            Ok(calls) => {
                println!("control: the witness solves this environment in {calls} call(s)")
            }
            Err(e) => anyhow::bail!(
                "the witness cannot solve this environment ({e}); every number below would \
                 measure the harness rather than the model"
            ),
        }
        println!();
    }

    // Everything the model is actually shown goes through here. An agent's
    // stored history has neither the system prompt nor the tool schemas, so
    // without this an episode cannot be turned into training data that
    // renders the way inference does.
    let recorder = match options.base_url.as_deref().and_then(upstream_of) {
        Some(upstream) => match runtime.block_on(Recorder::start(&upstream)) {
            Ok(recorder) => Some(recorder),
            Err(e) => {
                eprintln!("note: not recording model input ({e}); episodes will still run");
                None
            }
        },
        None => None,
    };
    if let Some(recorder) = &recorder {
        println!("recording model input via {}", recorder.base_url());
    }

    let mut score = ArmScore::new(&arm, options.model.api_model());
    let mut records = Vec::new();
    let mut answered_calls: Vec<usize> = Vec::new();

    for (index, hidden) in family
        .live_choices()
        .iter()
        .cycle()
        .take(options.instances * family.live_choices().len())
        .enumerate()
    {
        let dir = options.run_dir.join(format!("ep-{index:03}-{hidden}"));
        let through = recorder.as_ref().map(|r| r.base_url());
        let outcome = runtime.block_on(one_episode(
            &family,
            &dir,
            hidden,
            &options,
            through.as_deref(),
            seed + index as u64,
        ));
        let (outcome, detail, calls) = match outcome {
            Ok((verdict, detail, calls)) => (Outcome::from_verdict(&verdict), detail, Some(calls)),
            Err(e) => (Outcome::errored(e.to_string()), format!("{e}"), None),
        };
        let mark = match &outcome {
            Outcome::Answered { solved: true } => "PASS",
            Outcome::Answered { solved: false } => "fail",
            Outcome::Errored { .. } => "ERROR",
        };
        println!("  ep-{index:03} [{hidden:<10}] {mark}  {detail}");
        score.record(&outcome);
        if let (Outcome::Answered { .. }, Some(calls)) = (&outcome, calls) {
            answered_calls.push(calls);
        }
        records.push(serde_json::json!({
            "episode": index, "hidden_state": hidden, "outcome": outcome, "detail": detail,
            "tool_calls": calls,
        }));
    }

    println!();
    match score.rate() {
        Some(rate) => println!(
            "{arm}: {}/{} solved ({:.0}%) on {}",
            score.solved,
            score.denominator(),
            rate * 100.0,
            options.model
        ),
        None => println!("{arm}: nothing reached a verdict; there is no score to report"),
    }
    if let Some(caveat) = score.caveat() {
        eprintln!("{caveat}");
    }

    if let Some(recorder) = &recorder {
        let path = capture_path(&options.run_dir);
        if let Err(e) = runtime.block_on(recorder.dump(&path)) {
            eprintln!("note: could not write the captured model input: {e}");
        } else {
            println!("model input: {}", path.display());
        }
    }

    let mean_calls = (!answered_calls.is_empty())
        .then(|| answered_calls.iter().sum::<usize>() as f64 / answered_calls.len() as f64);
    if let Some(mean) = mean_calls {
        println!("tool calls per answered episode: {mean:.2}");
    }

    if let Some(path) = options.report {
        let body = serde_json::json!({
            "family": family.id(), "arm": arm, "seed": seed, "adapter_digest": options.adapter_digest, "model": options.model.api_model(),
            "hint": options.hint, "request_override": options.request,
            "score": {
                "solved": score.solved, "answered": score.answered, "errored": score.errored,
                "rate": score.rate(),
                "tool_calls_per_answered_episode": mean_calls,
            },
            "episodes": records,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&body)? + "\n")?;
        println!("report: {}", path.display());
    }
    Ok(())
}

/// One episode, start to verdict.
async fn one_episode(
    family: &Family,
    dir: &std::path::Path,
    hidden: &str,
    options: &BaselineOptions,
    through: Option<&str>,
    seed: u64,
) -> anyhow::Result<(splinter_tool_syntax::Verdict, String, usize)> {
    let mut episode = Episode::start(family, dir, hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let before = baseline_effective(family, &episode).map_err(|e| anyhow::anyhow!("{e}"))?;

    // The world is reachable only through the workspace's own CLI, and only
    // because these variables are set for the agent's tools.
    for (key, value) in episode.agent_env() {
        std::env::set_var(key, value);
    }
    // Tools run relative to the process's working directory, so the workspace
    // has to be it. Episodes are sequential for this reason.
    let previous = std::env::current_dir()?;
    std::env::set_current_dir(episode.workspace())?;

    // Through the recorder when there is one, so the captured request is the
    // one the server really answered.
    let frozen = options.request.as_deref().unwrap_or(family.request());
    let request = match &options.hint {
        Some(hint) => format!("{frozen}\n\n{hint}"),
        None => frozen.to_string(),
    };
    let result = run_agent(
        &request,
        &Endpoint {
            model: &options.model,
            base_url: through.or(options.base_url.as_deref()),
            api_key: options.api_key.as_deref(),
            seed,
        },
        Bounds {
            tool_calls: usize::try_from(family.tool_call_budget())?,
            deadline: std::time::Duration::from_secs(600),
        },
    )
    .await;

    std::env::set_current_dir(previous)?;
    // Stop the world before verifying: nothing the verifier reads should still
    // be able to change.
    episode.stop();

    let activity = result?;
    let predicates = run_verifier(family, &episode, &before).map_err(|e| anyhow::anyhow!("{e}"))?;
    let verdict = family
        .predicates()
        .evaluate(&predicates)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if let Ok(json) = serde_json::to_string_pretty(&activity.transcript) {
        let _ = std::fs::write(episode.dir().join("transcript.json"), json);
    }

    let mut detail = format!(
        "{} tool call(s), ended {}",
        activity.calls, activity.conclusion
    );
    if !verdict.solved() {
        detail.push_str(&format!(", unmet: {}", verdict.failed().join(", ")));
    }
    if !activity.failures.is_empty() {
        detail.push_str(&format!(
            "; {} failed: {}",
            activity.failures.len(),
            activity.failures.join(" | ")
        ));
    }
    Ok((verdict, detail, activity.calls))
}

// ---------------------------------------------------------------------------
// demonstrate - verified episodes from a scripted model
// ---------------------------------------------------------------------------

/// How many differently-worded demonstrations there are. Each is the same
/// plodding path (ask which deployment is live, look at it, change it, check
/// the change) in different words and with an optional extra look, so that a
/// model trained on some of them is scored on wording it has not seen.
const VARIANTS: usize = 4;

/// What the closing turn says once the work is done, per variant.
const CLOSINGS: [&str; VARIANTS] = [
    "The live deployment now has three retry attempts and the service validates it.",
    "Done: the deployment this host runs is set to three retry attempts, and validation passes.",
    "I changed the retry attempts of the live deployment to three; the service accepts the configuration.",
    "The active deployment's configuration now enables three retry attempts and validates cleanly.",
];

fn shell(command: &str, description: &str) -> Action {
    Action::new(
        "shell",
        serde_json::json!({ "shell_command": command, "description": description }),
    )
}

/// The demonstration, as actions only.
///
/// Deliberately the plodding version: ask which deployment is live, look at
/// it, change it, check the change. A demonstration that jumped straight to
/// the right file would teach the model to guess correctly rather than to find
/// out, and finding out is the capability under test.
fn demonstration(family: &Family, active: &str, variant: usize) -> anyhow::Result<Vec<Action>> {
    let config = format!("config/{active}.json");

    // The demonstration rewrites the whole file, so it must preserve what was
    // there: `production` overrides only the timeout, and dropping that would
    // change what the deployment does - which the verifier would rightly
    // reject.
    let template = family.root().join("workspace").join(&config);
    let mut contents: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&template)?)?;
    contents
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("{} is not a JSON object", template.display()))?
        .insert("retry_attempts".into(), serde_json::json!(3));
    let text = format!("{}\n", serde_json::to_string_pretty(&contents)?);

    let status = [
        "find out which deployment this host runs",
        "check which deployment is live",
        "which deployment does this host run",
        "see which deployment is active",
    ][variant];
    let limit = [50, 100, 50, 200][variant];
    let validate = [
        "check the service accepts the change",
        "validate the new configuration",
        "check the service accepts the change",
        "validate the service",
    ][variant];
    // Variant 2 inspects the effective configuration before the edit, 3 after.
    let (look_before, look_after) = match variant {
        2 => (Some("show the effective configuration"), None),
        3 => (None, Some("confirm the effective configuration")),
        _ => (None, None),
    };

    let mut actions = vec![
        shell("./svctl status", status),
        Action::new(
            "read_file",
            serde_json::json!({ "path": config, "offset": 1, "limit": limit }),
        ),
    ];
    if let Some(why) = look_before {
        actions.insert(1, shell("./svctl effective", why));
    }
    actions.push(Action::new(
        "write_file",
        serde_json::json!({ "path": config, "text": text, "append": false }),
    ));
    if let Some(why) = look_after {
        actions.push(shell("./svctl effective", why));
    }
    actions.push(shell("./svctl validate", validate));
    Ok(actions)
}

fn demonstrate(args: &[String]) -> anyhow::Result<()> {
    let options = parse_baseline(args)?;
    let family = Family::load(&family_root()).map_err(|e| anyhow::anyhow!("{e}"))?;

    // The prompt the agent really sends. Captured by `baseline`, because it is
    // the only place the whole thing exists - an agent's history holds neither
    // the system prompt nor the tool schemas.
    let capture = options
        .prompt_from
        .clone()
        .unwrap_or_else(|| options.run_dir.join("requests.json"));
    let captured: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&capture).map_err(|e| {
            anyhow::anyhow!(
                "{}: {e}\nRun `baseline --run-dir DIR` first; it writes the captured prompt there, \
                 or name one with --prompt-from.",
                capture.display()
            )
        })?)?;
    let prompt = captured
        .iter()
        .find(|r| {
            r.get("tools")
                .and_then(|t| t.as_array())
                .is_some_and(|t| !t.is_empty())
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{} holds no request carrying tool schemas; a record built from it would train the \
                 model on a prompt it never sees",
                capture.display()
            )
        })?;

    let sven = sven_binary()?;
    println!(
        "demonstrate: {} (tools via {})",
        family.id(),
        sven.display()
    );
    println!();

    // The last variant is held out: it is scored by the trainer as it learns
    // (never trained on) and is the one whose wording the model has not seen.
    let held_out_from = VARIANTS - 1;
    let mut train = Vec::new();
    let mut held_out = Vec::new();
    let mut solved = 0usize;
    let mut attempted = 0usize;

    for variant in 0..VARIANTS {
        for hidden in family.live_choices() {
            attempted += 1;
            let dir = options.run_dir.join(format!("demo-v{variant}-{hidden}"));
            match one_demonstration(&family, &dir, hidden, variant, prompt, &sven) {
                Ok((produced, detail)) => {
                    solved += 1;
                    println!("  demo-v{variant} [{hidden:<10}] SOLVED  {detail}");
                    if variant >= held_out_from {
                        held_out.extend(produced);
                    } else {
                        train.extend(produced);
                    }
                }
                Err(e) => println!("  demo-v{variant} [{hidden:<10}] failed  {e}"),
            }
        }
    }

    println!();
    println!(
        "demonstrate: {solved}/{attempted} verified; {} training and {} held-out record(s)",
        train.len(),
        held_out.len()
    );

    if train.is_empty() || held_out.is_empty() {
        anyhow::bail!(
            "no verified demonstration produced a record on one side of the split, so there is \
             nothing to train on or to score on"
        );
    }

    std::fs::create_dir_all(&options.run_dir)?;
    for (name, records) in [("train.jsonl", &train), ("heldout.jsonl", &held_out)] {
        let out = options.run_dir.join(name);
        std::fs::write(&out, to_jsonl(records)?)?;
        // Checked by the parser AND, given --weights, the encoder that will
        // train on it, before anything claims a GPU. Parsing alone is not
        // enough: a record can satisfy the wire schema completely and still
        // have no honest loss-mask boundary under the checkpoint's own chat
        // template.
        let checked = match &options.weights {
            Some(weights) => splinter_sdk::model::train::validate_dataset_for(&out, weights),
            None => {
                println!(
                    "note: no --weights, so {name} is only parsed, not encoded; a shape the \
                     template cannot mask would not be caught until training starts"
                );
                splinter_sdk::model::train::validate_dataset(&out)
            }
        };
        match checked {
            Ok(summary) => println!(
                "dataset: {} ({} record(s), {} message(s), {} supervised)",
                out.display(),
                summary.records,
                summary.messages,
                summary.trained_messages
            ),
            Err(e) => anyhow::bail!("the trainer would reject {name}: {e}"),
        }
    }
    Ok(())
}

/// Where the `sven` binary is: `SVEN_BIN`, else the first `sven` on `PATH`.
/// Named by the environment so a sample is not tied to one install or
/// one build profile.
fn sven_binary() -> anyhow::Result<PathBuf> {
    if let Ok(path) = std::env::var("SVEN_BIN") {
        return Ok(PathBuf::from(path));
    }
    std::env::var_os("PATH")
        .iter()
        .flat_map(std::env::split_paths)
        .map(|dir| dir.join("sven"))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            anyhow::anyhow!("cannot find the sven binary; put it on PATH or name it with SVEN_BIN")
        })
}

fn one_demonstration(
    family: &Family,
    dir: &std::path::Path,
    hidden: &str,
    variant: usize,
    prompt: &serde_json::Value,
    sven: &std::path::Path,
) -> anyhow::Result<(Vec<splinter_tool_syntax::Record>, String)> {
    let mut episode = Episode::start(family, dir, hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let before = baseline_effective(family, &episode).map_err(|e| anyhow::anyhow!("{e}"))?;

    let env = episode.agent_env();
    let actions = demonstration(family, hidden, variant)?;
    let performed = splinter_tool_syntax::perform_all(sven, episode.workspace(), &env, &actions)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    episode.stop();

    let predicates = run_verifier(family, &episode, &before).map_err(|e| anyhow::anyhow!("{e}"))?;
    let verdict = family
        .predicates()
        .evaluate(&predicates)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let steps: Vec<(String, String, String)> = performed
        .iter()
        .map(|p| {
            (
                p.action.tool.clone(),
                p.action.arguments.to_string(),
                p.observation.clone(),
            )
        })
        .collect();

    let records = records_from_performance(
        family.id(),
        hidden,
        prompt,
        family.request(),
        &steps,
        CLOSINGS[variant],
        &verdict,
    )
    .map_err(|why| anyhow::anyhow!("{why}"))?;

    let detail = format!(
        "{} action(s) performed, {} record(s)",
        performed.len(),
        records.len()
    );
    Ok((records, detail))
}

// ---------------------------------------------------------------------------
// train - one monitored LoRA fine-tune on the verified demonstrations
// ---------------------------------------------------------------------------

fn train_command(args: &[String]) -> anyhow::Result<()> {
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|at| args.get(at + 1))
    };
    let need = |name: &str| {
        value(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("{name} is required"))
    };
    let number = |name: &str, default: u32| -> anyhow::Result<u32> {
        value(name).map_or(Ok(default), |v| Ok(v.parse()?))
    };
    // No default seed: two runs that differ only in it show how much of a
    // result is the draw, and a hidden constant would make them the same run.
    let seed: u64 = need("--seed")?.parse()?;
    let data = PathBuf::from(need("--data")?);
    let request = TrainRequest {
        base: PathBuf::from(need("--base")?),
        train: data.join("train.jsonl"),
        held_out: data.join("heldout.jsonl"),
        attempt_dir: PathBuf::from(need("--out")?),
        steps: number("--steps", 24)?,
        rank: number("--rank", 16)?,
        alpha: value("--alpha").map_or(Ok(32.0), |v| v.parse())?,
        learning_rate: value("--lr").map(|v| v.parse()).transpose()?,
        grad_accum: number("--grad-accum", 4)?,
        seed,
        eval_every: number("--eval-every", 2)?,
        patience: number("--patience", 3)?,
    };
    let started = std::time::Instant::now();
    let narrate = |r: &splinter_sdk::model::train::StepReport| {
        println!(
            "step {}/{} loss {:.4} elapsed {:.1}s",
            r.step,
            r.steps,
            r.loss,
            started.elapsed().as_secs_f64()
        );
    };
    let summary = train(&request, &narrate)?;
    let out = serde_json::to_string_pretty(&summary.document)? + "\n";
    std::fs::write(summary_path(&request.attempt_dir), &out)?;
    println!("{out}");
    println!("adapter: {}", summary.adapter.display());
    println!("digest: {}", summary.adapter_digest);
    Ok(())
}
