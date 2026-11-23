// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements programmable agent workflows that turn
// documents into structured data with every step on record, for its
// clients. If your team needs expertise in agent harnesses, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A sven harness run over files, in process: the factory's `harness`
//! operation.
//!
//! The harness's agent holds no tools (`profile: none`): the stage's files
//! reach it as text inputs, and its result is the value it returns. A run
//! that loops over files reads them from a directory the caller lays out.
//! It runs nothing: no shell, no gates, no network.
//! A harness that needs those works a task, in its sandbox, as an attempt
//! stage's `harness:` does. [`check`] refuses one before anything runs.

use std::path::Path;
use std::time::Duration;

use serde_json::Value;
use sven_sdk::config::Config;
use sven_sdk::{ApprovalPolicy, Engine, RunConclusion, Toolset};

use crate::harness_work::run_harness_in_tree;
use crate::solve::{Model, SolveOptions};
use crate::work::WITHHELD_TOOLS;

/// The capabilities a harness run over files may hold.
const FILE_CAPABILITIES: &[&str] = &["ReadFile", "WriteFile", "DeleteFile"];

/// Why `yaml` cannot run over files, or nothing.
///
/// # Errors
/// It does not compile, or declares a capability beyond the file tools.
pub fn check(yaml: &str) -> Result<(), String> {
    let plan = sven_sdk::compile_harness(yaml).map_err(|e| e.to_string())?;
    let beyond: Vec<&str> = plan
        .capabilities
        .iter()
        .map(String::as_str)
        .filter(|c| !FILE_CAPABILITIES.contains(c))
        .collect();
    if beyond.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "harness {} needs {}, and a harness over files has only {}: run it on a task, as an attempt stage's harness",
            plan.name,
            beyond.join(", "),
            FILE_CAPABILITIES.join(", ")
        ))
    }
}

/// A harness run over the files of a directory.
#[derive(Clone, Debug)]
pub struct HarnessCall {
    /// The harness document.
    pub yaml: String,
    /// Its inputs.
    pub inputs: Value,
    /// How long the run may take.
    pub deadline: Duration,
}

/// What a run came to.
#[derive(Clone, Debug, PartialEq)]
pub enum HarnessReply {
    /// The harness returned this value.
    Returned {
        /// The value.
        output: Value,
        /// The run's trajectory, when it kept one.
        trajectory: Option<Value>,
    },
    /// The harness failed by its own `fail`: the code and message.
    Failed(String),
}

impl HarnessCall {
    /// The run on `model` in `tree`, on a runtime of its own (the calling
    /// thread blocks).
    ///
    /// # Errors
    /// The harness is refused ([`check`]), the engine cannot start, or a
    /// bound stopped the run before a verdict: nothing was decided.
    pub fn run(self, model: &Model, tree: &Path) -> Result<HarnessReply, String> {
        check(&self.yaml)?;
        let options = SolveOptions::new(self.deadline);
        let mut config: Config = options.engine_config();
        config.tools.disabled = WITHHELD_TOOLS
            .iter()
            .copied()
            .chain(["shell"])
            .map(str::to_owned)
            .collect();
        let engine = Engine::builder()
            .config(config)
            .model_provider(options.provider(model.provider.clone()))
            .toolset(Toolset::coding())
            .project_root(tree)
            .approvals(ApprovalPolicy::Auto)
            .build()
            .map_err(|e| e.to_string())?;
        let harness = engine
            .load_harness_from_str(&self.yaml)
            .map_err(|e| e.to_string())?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let ran = runtime
            .block_on(run_harness_in_tree(
                &harness,
                tree,
                self.inputs,
                options.run_options(),
            ))
            .map_err(|e| e.to_string())?;
        match (ran.outcome.conclusion, ran.outcome.result) {
            (RunConclusion::Success, Some(Ok(output))) => Ok(HarnessReply::Returned {
                output,
                trajectory: ran
                    .run
                    .trajectory()
                    .and_then(|t| serde_json::to_value(t).ok()),
            }),
            (_, Some(Err(failure))) => Ok(HarnessReply::Failed(format!("{failure:?}"))),
            (conclusion, _) => Err(format!("the harness ended {conclusion:?} before a verdict")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::check;

    #[test]
    fn a_harness_over_files_holds_only_the_file_tools() {
        let files = r#"api_version: "2"
name: notes
inputs:
  topic: { type: string }
agent:
  profile: none
execution:
  limits: { model_turns: 10 }
workflow:
  - name: write
    task: "Write notes on {{ inputs.topic }}."
"#;
        assert_eq!(check(files), Ok(()));
        let gated = r#"api_version: "2"
name: tested
agent:
  profile: coding
execution:
  limits: { model_turns: 10 }
gates:
  test:
    run: sh run_tests.sh
workflow:
  - name: work
    task: "Make the tests pass."
    verify: [test]
"#;
        let refused = check(gated).unwrap_err();
        assert!(refused.contains("attempt stage"), "{refused}");
    }
}
