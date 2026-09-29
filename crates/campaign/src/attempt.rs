// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A delegated task attempt as a command asks for it: the defaults, the
//! limits, and what a resume inherits from the run it continues.

use std::path::PathBuf;

use splinter_agent::{outcome::Outcome, AttemptOptions};
use splinter_store::runs::{read_manifest, Budget, RunManifest};

use crate::config::{allow_slow_local_prefill, Config};
use crate::models::ModelChoice;

/// Per-attempt wall-clock limit when a command names none.
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;
/// Generated tokens one attempt may spend. At the local model's measured
/// decode rate the wall-clock limit binds long before this does; it exists
/// for fast remote models, where it is the bound that stops a loop.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 100_000;
/// Billed USD one attempt on an OpenRouter model may spend unless the caller
/// sets another cap. Conservative on purpose: raising it is a visible flag.
pub const DEFAULT_REMOTE_MAX_COST_USD: f64 = 1.0;
/// Attempts per run, the first included: a first try and two resumes.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 3;

/// One attempt as a command line asks for it.
#[derive(Clone, Debug)]
pub struct AttemptRequest {
    /// The workspace the agent works in.
    pub workspace: Option<PathBuf>,
    /// The task; a resume falls back to the run's recorded one.
    pub task: Option<String>,
    /// Completion checks the attempt runs itself after the turn.
    pub checks: Vec<String>,
    pub model: ModelChoice,
    pub timeout_secs: u64,
    pub max_tool_rounds: Option<u32>,
    pub max_output_tokens: Option<u64>,
    pub max_cost_usd: Option<f64>,
    pub max_attempts: u32,
    /// Capture the exact model input at the wire.
    pub record_input: bool,
}

impl AttemptRequest {
    /// Refuses a request whose limits could never fire or whose model
    /// choice is inconsistent.
    pub fn validate(&self) -> anyhow::Result<()> {
        self.model.validate()?;
        anyhow::ensure!(
            self.max_cost_usd.is_none() || self.model.names_api_model(),
            "--max-cost-usd limits billed remote usage; local inference is not billed per token"
        );
        anyhow::ensure!(self.max_attempts >= 1, "--max-attempts must be at least 1");
        Ok(())
    }

    /// The attempt's usage limits, defaults applied. Only OpenRouter models
    /// get a default cost cap: they report a price per call, so the cap can
    /// be enforced. Other endpoints (a served brain, say) report none, and a
    /// default cap there would stop every run on its first usage report.
    #[must_use]
    pub fn budget(&self) -> Budget {
        let openrouter = self
            .model
            .remote
            .as_deref()
            .is_some_and(|m| m.starts_with("openrouter/"));
        Budget {
            max_output_tokens: Some(self.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)),
            max_cost_usd: self
                .max_cost_usd
                .or(openrouter.then_some(DEFAULT_REMOTE_MAX_COST_USD)),
        }
    }

    /// The runner's options for a fresh run, or - with `resume` naming a run -
    /// for a further attempt on it.
    pub fn options(&self, config: &Config, resume: Option<&str>) -> anyhow::Result<AttemptOptions> {
        let workspace = self
            .workspace
            .clone()
            .ok_or_else(|| anyhow::anyhow!("an attempt needs --workspace DIR"))?;
        let recorded = resume
            .map(|run_id| read_manifest(&config.state_root, run_id))
            .transpose()?;
        let task = match (&self.task, &recorded) {
            (Some(task), _) => task.clone(),
            (None, Some(manifest)) => manifest.task.clone(),
            (None, None) => anyhow::bail!("a run needs --task TEXT or --task-file FILE"),
        };
        let mut local = self.model.local_weights(config);
        local.adapter = adapter_for(&self.model, recorded.as_ref());
        Ok(AttemptOptions {
            workspace,
            task,
            checks: self.checks.clone(),
            model: self.model.selection(config, local),
            timeout_secs: self.timeout_secs,
            max_tool_rounds: self.max_tool_rounds,
            max_attempts: self.max_attempts,
            budget: self.budget(),
            allow_api_models: self.model.allow_api_models,
            record_input: self.record_input,
        })
    }
}

/// The adapter an attempt serves: an explicit one wins; a resume otherwise
/// keeps the adapter its run was recorded with. Dropping it silently would
/// turn a resumed attempt into a base-model run while the manifest still
/// says an adapter rode along.
fn adapter_for(model: &ModelChoice, recorded: Option<&RunManifest>) -> Option<PathBuf> {
    model
        .adapter
        .clone()
        .or_else(|| recorded.and_then(|m| m.local_adapter.clone()))
}

/// Runs one fresh attempt.
pub fn run(config: &Config, request: &AttemptRequest) -> anyhow::Result<Outcome> {
    request.validate()?;
    prepare(request);
    let options = request.options(config, None)?;
    Ok(splinter_agent::run(&config.state_root, options)?.0)
}

/// Resumes `run_id` from its checkpoint as a further attempt.
pub fn resume(config: &Config, run_id: &str, request: &AttemptRequest) -> anyhow::Result<Outcome> {
    request.validate()?;
    prepare(request);
    let options = request.options(config, Some(run_id))?;
    Ok(splinter_agent::resume(&config.state_root, run_id, options)?.0)
}

fn prepare(request: &AttemptRequest) {
    if request.model.remote.is_none() {
        allow_slow_local_prefill(request.timeout_secs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_store::{write_atomic, StateRoot};

    fn config(name: &str) -> Config {
        let root = std::env::temp_dir().join(format!("splinter-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        Config {
            state_root: StateRoot::new(root),
            default_local_weights: PathBuf::from("models/Qwen/Qwen3-0.6B"),
            openrouter_api_key: None,
            brain_api_key: None,
        }
    }

    fn request() -> AttemptRequest {
        AttemptRequest {
            workspace: Some(PathBuf::from("ws")),
            task: None,
            checks: vec![],
            model: ModelChoice::default(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_tool_rounds: None,
            max_output_tokens: None,
            max_cost_usd: None,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            record_input: false,
        }
    }

    fn record(config: &Config, run_id: &str, manifest: &RunManifest) {
        let path = config.state_root.run_dir(run_id).join("run.json");
        write_atomic(&path, &serde_json::to_string(manifest).unwrap()).unwrap();
    }

    /// A resume without a task continues the run's recorded one, and without
    /// an adapter serves the run's recorded adapter: the caller must not
    /// have to restate what the run carries, and a divergent restatement
    /// would silently change what the recovered attempt works on.
    #[test]
    fn a_resume_inherits_what_the_run_recorded() {
        let cfg = config("resume");
        record(
            &cfg,
            "loop-resume",
            &RunManifest {
                task: "confirm the sum".into(),
                local_adapter: Some(PathBuf::from("recorded.safetensors")),
                ..Default::default()
            },
        );
        let inherited = request().options(&cfg, Some("loop-resume")).unwrap();
        assert_eq!(inherited.task, "confirm the sum");
        assert_eq!(
            inherited.model.local().unwrap().adapter,
            Some(PathBuf::from("recorded.safetensors"))
        );

        let explicit = AttemptRequest {
            task: Some("explicit".into()),
            model: ModelChoice {
                adapter: Some(PathBuf::from("explicit.safetensors")),
                ..ModelChoice::default()
            },
            ..request()
        };
        let overridden = explicit.options(&cfg, Some("loop-resume")).unwrap();
        assert_eq!(overridden.task, "explicit", "an explicit task wins");
        assert_eq!(
            overridden.model.local().unwrap().adapter,
            Some(PathBuf::from("explicit.safetensors")),
            "an explicit adapter wins"
        );

        assert!(request().options(&cfg, Some("loop-missing")).is_err());
        assert!(
            request().options(&cfg, None).is_err(),
            "a fresh run needs a task"
        );
        let _ = std::fs::remove_dir_all(cfg.state_root.path());
    }

    /// Local inference is not billed per token, so a cost cap on it would be
    /// a limit that can never fire; it is refused instead of ignored. A
    /// remote run gets a visible default cap.
    #[test]
    fn cost_caps_apply_to_remote_models_only() {
        let capped_local = AttemptRequest {
            max_cost_usd: Some(0.5),
            ..request()
        };
        assert!(capped_local.validate().is_err());
        let remote = AttemptRequest {
            model: ModelChoice {
                remote: Some("openrouter/z-ai/glm-5.3-flash".into()),
                allow_api_models: true,
                ..ModelChoice::default()
            },
            ..request()
        };
        remote.validate().unwrap();
        assert_eq!(
            remote.budget().max_cost_usd,
            Some(DEFAULT_REMOTE_MAX_COST_USD)
        );
        assert_eq!(request().budget().max_cost_usd, None);
        assert_eq!(
            request().budget().max_output_tokens,
            Some(DEFAULT_MAX_OUTPUT_TOKENS)
        );
    }
}
