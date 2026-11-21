// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that ask their model
// backend what it can train across machines instead of assuming it, for
// its clients. If your team needs expertise in distributed training on a
// cluster, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Which brain actions run as ranks of a distributed job, as brain itself
//! says: an action that takes the `job_plan` parameter in its manifest
//! (`brain caps --json`) joins a job plan; no model family is assumed. The
//! served listing (`GET /v1/capabilities`) cannot say, since it leaves out
//! the parameters brain resolves on its own host, and a plan file is one.

use std::path::Path;
use std::process::Command;

use serde_json::Value;

use crate::error::PolicyError;

/// The parameter an action takes when it can run as one rank of a job plan.
const PLAN_PARAM: &str = "job_plan";

/// One action of one model, and whether it joins a job plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainAction {
    /// The model, as the manifest names it (`brain/qwen3`).
    pub model: String,
    /// The action (`lora_train`).
    pub action: String,
    /// Whether it takes `job_plan`.
    pub joins_plan: bool,
}

impl BrainAction {
    /// Whether the command words `arch action` name this action: the
    /// architecture as the manifest's model or its last segment.
    fn named_by(&self, arch: &str, action: &str) -> bool {
        self.action == action && (self.model == arch || self.model.rsplit('/').next() == Some(arch))
    }
}

/// Every action the manifests in `json` (the output of `brain caps --json`)
/// list.
///
/// # Errors
/// The text is not a list of manifests with actions.
pub fn parse_actions(json: &str) -> Result<Vec<BrainAction>, String> {
    let manifests: Value =
        serde_json::from_str(json).map_err(|e| format!("brain caps --json: {e}"))?;
    let manifests = manifests
        .as_array()
        .ok_or("brain caps --json: not a list of manifests")?;
    let mut actions = Vec::new();
    for manifest in manifests {
        let model = manifest["model"]
            .as_str()
            .ok_or("brain caps --json: a manifest names no model")?;
        for action in manifest["actions"].as_array().into_iter().flatten() {
            let Some(name) = action["name"].as_str() else {
                return Err(format!(
                    "brain caps --json: an action of {model} has no name"
                ));
            };
            let joins_plan = action["params"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|param| param["name"] == PLAN_PARAM);
            actions.push(BrainAction {
                model: model.to_string(),
                action: name.to_string(),
                joins_plan,
            });
        }
    }
    Ok(actions)
}

/// The actions the `brain` at `binary` lists (`brain caps --json`).
///
/// # Errors
/// The binary did not run or answered no manifests.
pub fn brain_actions(binary: &Path) -> Result<Vec<BrainAction>, PolicyError> {
    let refused = |reason: String| PolicyError::Endpoint {
        url: binary.display().to_string(),
        reason,
    };
    let output = Command::new(binary)
        .args(["caps", "--json"])
        .output()
        .map_err(|e| refused(format!("brain caps --json did not run: {e}")))?;
    if !output.status.success() {
        return Err(refused(format!(
            "brain caps --json failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    parse_actions(&String::from_utf8_lossy(&output.stdout)).map_err(refused)
}

/// Whether `command` (a brain invocation, `brain [flags] <arch> <action>
/// ...`) may run as the ranks of a job plan: refused, naming the actions
/// that can, when it names an action that takes no plan. A command that
/// names no action brain lists is not brain's to judge and passes.
///
/// # Errors
/// The command names a brain action that takes no `job_plan`.
pub fn check_plan_command(command: &[String], actions: &[BrainAction]) -> Result<(), String> {
    let named = command.windows(2).find_map(|pair| {
        actions
            .iter()
            .find(|action| action.named_by(&pair[0], &pair[1]))
    });
    match named {
        Some(action) if !action.joins_plan => {
            let able: Vec<String> = actions
                .iter()
                .filter(|a| a.joins_plan)
                .map(|a| format!("{} {}", a.model, a.action))
                .collect();
            Err(format!(
                "{} {} takes no job plan, so its ranks would not form one job; the actions \
                 that do: {}",
                action.model,
                action.action,
                if able.is_empty() {
                    "none".to_string()
                } else {
                    able.join(", ")
                }
            ))
        }
        _ => Ok(()),
    }
}

/// `command` (a `brain` invocation) with brain's node-local model staging
/// (`--stage-models DIR --stage-from MODELS`, global flags, right after the
/// program): every argument naming a path inside a model-store entry under
/// `models` is read from a copy under `dir`, made once per node. A command
/// that already stages is left as it is.
#[must_use]
pub fn with_model_staging(command: &[String], dir: &Path, models: &Path) -> Vec<String> {
    let mut staged = command.to_vec();
    if command.len() < 2 || command.iter().any(|word| word == STAGE_FLAG) {
        return staged;
    }
    staged.splice(1..1, staging_args(dir, models));
    staged
}

/// brain's global flags that stage `models`' entries under the node-local `dir`.
#[must_use]
pub fn staging_args(dir: &Path, models: &Path) -> Vec<String> {
    vec![
        STAGE_FLAG.into(),
        dir.display().to_string(),
        "--stage-from".into(),
        models.display().to_string(),
    ]
}

/// The flag brain stages model-store entries on node-local disk with.
const STAGE_FLAG: &str = "--stage-models";

#[cfg(test)]
mod tests {
    use super::*;

    const CAPS: &str = r#"[
        {"model":"brain/qwen3","summary":"","actions":[
            {"name":"generate","params":[{"name":"prompt"}]},
            {"name":"lora_train","params":[{"name":"rank"},{"name":"job_plan"},{"name":"job_rank"}]}]},
        {"model":"brain/gpt2","summary":"","actions":[
            {"name":"train","params":[{"name":"job_plan"}]}]}
    ]"#;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn staging_goes_right_after_the_program_and_only_once() {
        let staged = with_model_staging(
            &words("/b/brain --device gpu0 qwen3 lora_train"),
            Path::new("/scratch/local/models"),
            Path::new("/shared/models"),
        );
        assert_eq!(
            staged,
            words(
                "/b/brain --stage-models /scratch/local/models --stage-from /shared/models \
                 --device gpu0 qwen3 lora_train"
            )
        );
        let again = with_model_staging(&staged, Path::new("/other"), Path::new("/shared/models"));
        assert_eq!(again, staged, "a command that stages is left alone");
    }

    #[test]
    fn the_actions_that_join_a_plan_are_read_from_the_manifests() {
        let actions = parse_actions(CAPS).unwrap();
        let joining: Vec<_> = actions
            .iter()
            .filter(|a| a.joins_plan)
            .map(|a| format!("{} {}", a.model, a.action))
            .collect();
        assert_eq!(joining, ["brain/qwen3 lora_train", "brain/gpt2 train"]);
        assert!(parse_actions("{}").is_err());
        assert!(parse_actions("not json").is_err());
    }

    #[test]
    fn a_command_naming_an_action_without_a_plan_is_refused_with_those_that_have_one() {
        let actions = parse_actions(CAPS).unwrap();
        let able =
            words("/b/brain --device gpu{local_rank} --backend cuda qwen3 lora_train --steps 8");
        assert!(check_plan_command(&able, &actions).is_ok());
        let unable = words("/b/brain --backend cuda qwen3 generate --prompt hi");
        let error = check_plan_command(&unable, &actions).unwrap_err();
        assert!(error.contains("brain/qwen3 generate"), "{error}");
        assert!(error.contains("brain/gpt2 train"), "{error}");
        assert!(check_plan_command(&words("python3 train.py"), &actions).is_ok());
    }
}
