// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The `practice` request the command line asks for.

use splinter_sdk::data::ReplayPolicy;
use splinter_sdk::experiment::ExperimentPlan;
use splinter_sdk::practice::{
    CorrectionPlan, HistorySource, PlanSlot, PracticeObjective, PracticeRequest,
};
use splinter_sdk::train::Tuning;
use splinter_sdk::Error;

use crate::cli::{PracticeArgs, ReplayArg};

/// The `practice` the command line asked for; refused when the plan it names
/// cannot be read or is not valid.
pub(crate) fn practice_request(args: PracticeArgs) -> Result<PracticeRequest, Error> {
    let experiment = match (&args.plan, args.generation) {
        (Some(file), Some(generation)) => Some(PlanSlot {
            plan: ExperimentPlan::load(file, None)?,
            generation: generation as usize,
        }),
        _ => None,
    };
    Ok(PracticeRequest {
        experiment,
        pass_at_k: args.pass_at_k.pass_at_k(),
        repository: args.repository,
        revision: args.rev,
        test_command: args.test_command,
        files: args.files,
        allow_dirty: args.allow_dirty,
        max_tasks: args.max_tasks,
        mutants_per_file: args.mutants_per_file,
        history: args.history.map(|range| HistorySource {
            max_commits: args.history_commits.map_or(
                splinter_sdk::knowledge::tasks::workspace::DEFAULT_MAX_COMMITS,
                |n| n as usize,
            ),
            range,
        }),
        instruction: args.instruction_version,
        exam_families: args.exam_families,
        exam_seed: args.exam_seed,
        canaries: args.canaries,
        follow_ups: args.follow_ups,
        max_tool_calls: args.max_tool_calls,
        max_repeated_results: args.max_repeated_results,
        budget: args.budget,
        no_release: args.no_release,
        teacher: args.teacher,
        corrections: args.correction.map(|teacher| CorrectionPlan {
            teacher,
            max: args.corrections,
        }),
        objective: args.objective,
        replay_policy: replay_policy(args.objective, args.replay)?,
        rebuild: args.rebuild,
        curriculum: args.curriculum.strategy(),
        progress: args.curriculum.params(),
        curriculum_seed: args.curriculum.curriculum_seed,
        gate: args.gate.plan(),
        steps: args.steps,
        rank: args.rank,
        tuning: args.optimiser.applied_to(Tuning {
            records_per_step: args.records_per_step,
            eval_every: args.monitoring.eval_every,
            patience: args.monitoring.patience,
            monitor_share: args.monitoring.monitor_share,
            keep_evaluations: args.monitoring.keep_evaluations,
            seed: args.seed,
            max_block: args.max_block,
            base_precision: args.base_precision,
            // The importance ratios are taken at the temperature the pass@k
            // attempts are sampled at (`--temperature`), which the request
            // carries.
            policy_gradient: args.policy_gradient.options(None),
            ..Tuning::default()
        }),
    })
}

/// The replay the command line asks for. A rollout objective replays
/// nothing, so a policy named beside it is refused instead of dropped.
fn replay_policy(
    objective: PracticeObjective,
    replay: Option<ReplayArg>,
) -> Result<ReplayPolicy, Error> {
    match (objective.learns_from_groups(), replay) {
        (true, Some(_)) => Err(Error::Refused(format!(
            "--objective {objective} trains the champion's adapter on its own rollouts and \
             replays no earlier record: drop --replay"
        ))),
        (true, None) => Ok(ReplayPolicy::None),
        (false, replay) => Ok(replay
            .map(ReplayArg::policy)
            .transpose()?
            .unwrap_or_default()),
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
            }) => practice_request(*args).unwrap(),
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
    fn a_correction_teacher_reaches_the_request_with_its_cap() {
        let plain = request(&["./shop"]);
        assert!(
            plain.corrections.is_none(),
            "no teacher, no correction stage"
        );

        let asked = request(&[
            "./shop",
            "--correction",
            "local:teacher-7b",
            "--corrections",
            "3",
        ]);
        let plan = asked.corrections.expect("a correction plan");
        assert_eq!(plan.teacher.to_string(), "local:teacher-7b");
        assert_eq!(plan.max, 3);
    }

    #[test]
    fn a_curriculum_strategy_and_its_signal_reach_the_request_and_default_to_the_frontier() {
        let plain = request(&["./shop"]);
        assert_eq!(
            plain.curriculum,
            splinter_sdk::curriculum::schedule::Strategy::Frontier
        );
        assert_eq!(plain.curriculum_seed, None);

        let asked = request(&[
            "./shop",
            "--curriculum",
            "learning-progress",
            "--curriculum-seed",
            "9",
            "--lp-window",
            "3",
            "--lp-explore",
            "0.25",
        ]);
        assert_eq!(
            asked.curriculum,
            splinter_sdk::curriculum::schedule::Strategy::LearningProgress
        );
        assert_eq!(asked.curriculum_seed.map(|s| s.value()), Some(9));
        assert_eq!((asked.progress.window, asked.progress.epsilon), (3, 0.25));
    }

    #[test]
    fn a_plan_and_its_generation_reach_the_request_and_a_bad_plan_is_refused() {
        let scratch = std::env::temp_dir().join(format!("splinter-plan-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let catalog = "[[family]]\nid = \"a\"\nlanguage = \"python\"\ndomain = \"one\"\ntree = \"t1\"\n\
                       [[family]]\nid = \"b\"\nlanguage = \"lua\"\ndomain = \"two\"\ntree = \"t2\"\n";
        std::fs::write(scratch.join("catalog.toml"), catalog).unwrap();
        let plan = scratch.join("plan.toml");
        std::fs::write(
            &plan,
            "[plan]\nname = \"p\"\n[[generation]]\ntrain = [\"a\"]\nretention = []\n\
             [sealed]\nfamilies = [\"b\"]\n",
        )
        .unwrap();
        let plan_arg = plan.display().to_string();
        let asked = request(&["./shop", "--plan", &plan_arg, "--generation", "1"]);
        assert_eq!(asked.experiment.as_ref().map(|s| s.generation), Some(1));

        // A plan that seals what it trains is refused before any run starts.
        std::fs::write(
            &plan,
            "[plan]\nname = \"p\"\n[[generation]]\ntrain = [\"a\"]\nretention = []\n\
             [sealed]\nfamilies = [\"a\"]\n",
        )
        .unwrap();
        let argv = [
            "splinter",
            "practice",
            "./shop",
            "--plan",
            &plan_arg,
            "--generation",
            "1",
        ];
        let Some(Command::Practice(args)) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("not a practice");
        };
        assert!(practice_request(*args).unwrap_err().is_refusal());
        let _ = std::fs::remove_dir_all(&scratch);
        // The two flags come together.
        assert!(
            Cli::try_parse_from(["splinter", "practice", "./shop", "--plan", "p.toml"]).is_err()
        );
    }

    #[test]
    fn the_revision_is_head_unless_named() {
        let asked = request(&["./shop"]);
        assert_eq!(asked.revision, "HEAD");
        assert!(asked.test_command.is_empty() && !asked.no_release);
    }

    #[test]
    fn the_gate_settings_reach_the_request() {
        let asked = request(&["./shop", "--target-effect", "0.25"]);
        assert_eq!(asked.gate.target_effect, Some(0.25));
        assert_eq!(request(&["./shop"]).gate, Default::default());
    }

    #[test]
    fn the_objective_defaults_to_imitation_and_a_rollout_objective_reaches_the_request() {
        let plain = request(&["./shop"]);
        assert_eq!(plain.objective, PracticeObjective::SftStep);
        assert!(plain.tuning.policy_gradient.named().is_empty());
        assert_eq!(plain.tuning.base_precision, None);

        let asked = request(&[
            "./shop",
            "--objective",
            "self-distill",
            "--clip-eps",
            "0.15",
            "--kl-beta",
            "0.05",
            "--teacher-policy",
            "start",
            "--advantage-bound",
            "2",
            "--base-precision",
            "fp8",
        ]);
        assert_eq!(asked.objective, PracticeObjective::SelfDistill);
        // A rollout objective replays nothing.
        assert_eq!(asked.replay_policy, ReplayPolicy::None);
        let options = asked.tuning.policy_gradient;
        assert_eq!(
            (options.clip_eps, options.kl_beta),
            (Some(0.15), Some(0.05))
        );
        assert_eq!(options.advantage_bound, Some(2.0));
        assert_eq!(
            asked.tuning.base_precision,
            Some(splinter_sdk::model::train::BasePrecision::Fp8)
        );
        assert!(
            Cli::try_parse_from(["splinter", "practice", "./shop", "--objective", "sft"]).is_err()
        );
    }

    #[test]
    fn a_replay_named_beside_a_rollout_objective_is_refused_not_dropped() {
        let argv = [
            "splinter",
            "practice",
            "./shop",
            "--objective",
            "policy-gradient",
        ];
        let with = |extra: &[&str]| {
            let Some(Command::Practice(args)) = Cli::try_parse_from(argv.iter().chain(extra))
                .unwrap()
                .command
            else {
                panic!("not a practice");
            };
            practice_request(*args)
        };
        assert!(with(&[]).is_ok());
        assert!(with(&["--replay", "uniform"]).unwrap_err().is_refusal());
    }
}
