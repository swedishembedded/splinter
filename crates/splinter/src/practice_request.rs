// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The `practice` request the command line asks for.

use splinter_sdk::practice::PracticeRequest;
use splinter_sdk::train::Tuning;

use crate::cli::PracticeArgs;

/// The `practice` the command line asked for.
pub(crate) fn practice_request(args: PracticeArgs) -> PracticeRequest {
    PracticeRequest {
        pass_at_k: args.pass_at_k.pass_at_k(),
        repository: args.repository,
        revision: args.rev,
        test_command: args.test_command,
        files: args.files,
        allow_dirty: args.allow_dirty,
        max_tasks: args.max_tasks,
        mutants_per_file: args.mutants_per_file,
        exam_families: args.exam_families,
        exam_seed: args.exam_seed,
        follow_ups: args.follow_ups,
        max_tool_calls: args.max_tool_calls,
        max_repeated_results: args.max_repeated_results,
        budget: args.budget,
        no_release: args.no_release,
        gate: args.gate.plan(),
        steps: args.steps,
        rank: args.rank,
        tuning: args.optimiser.applied_to(Tuning {
            bf16_base: args.bf16_base,
            records_per_step: args.records_per_step,
            eval_every: args.monitoring.eval_every,
            patience: args.monitoring.patience,
            monitor_share: args.monitoring.monitor_share,
            keep_evaluations: args.monitoring.keep_evaluations,
            seed: args.seed,
            max_block: args.max_block,
            ..Tuning::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::cli::{Cli, Command};

    fn request(words: &[&str]) -> PracticeRequest {
        let argv = std::iter::once("splinter")
            .chain(std::iter::once("practice"))
            .chain(words.iter().copied());
        match Cli::try_parse_from(argv) {
            Ok(Cli {
                command: Some(Command::Practice(args)),
                ..
            }) => practice_request(*args),
            other => panic!("{words:?} is not a practice: {other:?}"),
        }
    }

    #[test]
    fn a_repository_and_a_revision_are_what_is_asked_and_the_test_command_comes_last() {
        let asked = request(&[
            "./shop",
            "--rev",
            "v1.2",
            "--exam-families",
            "2",
            "--k",
            "3",
            "--max-block",
            "8192",
            "--test-command",
            "python3",
            "-m",
            "pytest",
            "-x",
        ]);
        assert_eq!(asked.repository, std::path::PathBuf::from("./shop"));
        assert_eq!(asked.revision, "v1.2");
        assert_eq!(asked.exam_families, Some(2));
        assert_eq!(asked.pass_at_k.k, 3);
        assert_eq!(asked.tuning.max_block, Some(8192));
        assert_eq!(asked.test_command, ["python3", "-m", "pytest", "-x"]);
    }

    #[test]
    fn the_revision_is_head_unless_named() {
        let asked = request(&["./shop"]);
        assert_eq!(asked.revision, "HEAD");
        assert!(asked.test_command.is_empty() && !asked.no_release);
    }

    #[test]
    fn the_gate_settings_reach_the_request() {
        let asked = request(&["./shop", "--alpha-spending", "--target-effect", "0.25"]);
        assert!(asked.gate.alpha_spending);
        assert_eq!(asked.gate.target_effect, Some(0.25));
        assert_eq!(request(&["./shop"]).gate, Default::default());
    }
}
