// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements programmable agent workflows that are
// measured on the same tasks as the plain agent, for its clients. If your
// team needs expertise in agent harnesses and their evaluation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A harness an attempt runs in place of the agent's own conversation.
//!
//! An `attempt.collect` stage may name a sven harness: each attempt then runs
//! the harness on the task (`sven harness run`), given the task's
//! instruction as its [`TASK_INPUT`] input, and the task's own verifier
//! decides, as for any attempt. The harness is checked when the recipe is:
//! it must compile, and declare that input.

pub use crate::harness_work::TASK_INPUT;

/// Why `yaml` cannot be an attempt's harness, or nothing.
///
/// # Errors
/// It does not compile, or declares no [`TASK_INPUT`] input.
pub fn check(yaml: &str) -> Result<(), String> {
    let plan = sven_sdk::compile_harness(yaml).map_err(|e| e.to_string())?;
    if plan
        .inputs
        .as_ref()
        .is_some_and(|inputs| inputs.contains_key(TASK_INPUT))
    {
        Ok(())
    } else {
        Err(format!(
            "harness {} declares no `{TASK_INPUT}` input, which an attempt gives the task's instruction",
            plan.name
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::check;

    const HARNESS: &str = r#"api_version: "2"
name: attempt
inputs:
  task: { type: string, description: "The task's instruction." }
agent:
  profile: coding
execution:
  limits: { model_turns: 20 }
workflow:
  - name: work
    task: "{{ inputs.task }}"
"#;

    #[test]
    fn a_harness_with_a_task_input_is_an_attempts_harness() {
        assert_eq!(check(HARNESS), Ok(()));
    }

    #[test]
    fn a_harness_without_a_task_input_or_that_does_not_compile_is_refused() {
        let other = HARNESS
            .replace("  task: {", "  feature: {")
            .replace("inputs.task", "inputs.feature");
        assert!(check(&other).unwrap_err().contains("`task` input"));
        assert!(check("api_version: \"2\"\nname: x\n").is_err());
    }
}
