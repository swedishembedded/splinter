// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements command lines that turn a sentence about
// what to learn into a recorded learning run, for its clients. If your team
// needs expertise in agent tooling, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The `learn` request the command line asks for.

use splinter_sdk::learn::{ExamPlan, LearnRequest};
use splinter_sdk::raft::PassageShare;
use splinter_sdk::train::Tuning;
use splinter_sdk::vocabulary::role::Role;

use crate::cli::LearnArgs;

/// Whether the run is to end in a release: a named adapter is published
/// instead, and `--no-release` stops before the gate.
pub(crate) fn release_asked(args: &LearnArgs) -> bool {
    !args.no_release && args.adapter.is_none()
}

/// The `learn` the command line asked for. With no kinds named, a planner
/// chooses them: naming the kinds is deciding them.
pub(crate) fn learn_request(args: LearnArgs) -> LearnRequest {
    LearnRequest {
        pass_at_k: args.pass_at_k.pass_at_k(),
        plan: args.kinds.is_empty(),
        sources: args.sources,
        goal: args.goal,
        persona: args.persona,
        base: args.base,
        adapter: args.adapter,
        voice: args.voice,
        describe_voice: args.describe_voice,
        select_on_dev: args.select_on_dev,
        rehearsal: args.rehearsal,
        exam: ExamPlan {
            families: args.exam_families,
            tasks_per_family: args.exam_tasks_per_family,
            dev_families: args.dev_families,
            dev_tasks_per_family: args.dev_tasks_per_family,
            resamples: args.exam_resamples.map(|n| n as usize),
        },
        passages: args.with_passages.map(|records| match args.abstain {
            Some(abstentions) => PassageShare::with_abstentions(records, abstentions),
            None => PassageShare::evidence(records),
        }),
        kinds: args.kinds,
        budget: args.budget,
        dry_run: args.dry_run,
        no_release: args.no_release,
        no_frontier: args.no_frontier,
        distill: args.distill,
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
            ..Tuning::default()
        }),
        roles: [
            (Role::Planner, args.planner),
            (Role::Teacher, args.teacher),
            (Role::Generator, args.generator),
            (Role::Judge, args.judge),
        ]
        .into_iter()
        .filter_map(|(role, model)| model.map(|m| (role, m)))
        .collect(),
        ..LearnRequest::default()
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::cli::{Cli, Command};

    fn request(words: &[&str]) -> LearnRequest {
        let mut argv = vec!["splinter"];
        argv.extend_from_slice(words);
        match Cli::try_parse_from(argv).unwrap().command {
            Some(Command::Learn(args)) => learn_request(*args),
            other => panic!("not a learn: {other:?}"),
        }
    }

    #[test]
    fn learn_plans_unless_the_kinds_are_named() {
        let planned = request(&["learn", "docs"]);
        assert!(planned.plan && planned.kinds.is_empty());
        let named = request(&["learn", "docs", "--kinds", "recall,advise"]);
        assert!(!named.plan, "naming the kinds is deciding them");
        assert_eq!(named.kinds, ["recall", "advise"]);
    }

    #[test]
    fn a_run_is_asked_for_a_release_unless_it_is_stopped_before_one_or_publishes_an_adapter() {
        let asked = |words: &[&str]| {
            let mut argv = vec!["splinter"];
            argv.extend_from_slice(words);
            match Cli::try_parse_from(argv).unwrap().command {
                Some(Command::Learn(args)) => release_asked(&args),
                other => panic!("not a learn: {other:?}"),
            }
        };
        assert!(asked(&["learn", "docs"]));
        assert!(!asked(&["learn", "docs", "--no-release"]));
        assert!(
            !asked(&["learn", "docs", "--adapter", "maren"]),
            "a named adapter is published, so there is no release to be missing"
        );
    }

    #[test]
    fn a_base_and_an_adapter_name_reach_the_request() {
        let r = request(&[
            "learn",
            "docs",
            "--base",
            "Qwen/Qwen3-8B",
            "--adapter",
            "jefferson",
        ]);
        assert_eq!(r.base.as_deref(), Some("Qwen/Qwen3-8B"));
        assert_eq!(r.adapter.as_deref(), Some("jefferson"));
        let plain = request(&["learn", "docs"]);
        assert_eq!((plain.base, plain.adapter), (None, None));
    }

    #[test]
    fn the_distill_and_training_flags_reach_the_request() {
        let r = request(&[
            "learn",
            "docs",
            "--distill",
            "--steps",
            "300",
            "--rank",
            "16",
            "--lr",
            "0.0002",
            "--alpha",
            "32",
            "--weight-decay",
            "0.01",
            "--bf16-base",
        ]);
        assert!(r.distill);
        assert_eq!((r.steps, r.rank), (Some(300), Some(16)));
        assert!(r.tuning.bf16_base);
        assert_eq!(r.tuning.learning_rate, Some(0.0002));
        assert_eq!(
            (r.tuning.alpha, r.tuning.weight_decay),
            (Some(32.0), Some(0.01))
        );
        assert_eq!(r.voice, None, "the run decides from its persona");
        assert_eq!(
            request(&["learn", "docs", "--voice", "0.25"]).voice,
            Some(0.25)
        );
        assert_eq!(request(&["learn", "docs", "--voice", "0"]).voice, Some(0.0));
        assert_eq!(r.rehearsal, None, "the run decides from its persona");
        assert_eq!(
            request(&["learn", "docs", "--rehearsal", "0.4"]).rehearsal,
            Some(0.4)
        );
        assert_eq!(
            request(&["learn", "docs", "--rehearsal", "0"]).rehearsal,
            Some(0.0)
        );
        assert_eq!(
            request(&["learn", "docs", "--seed", "7"]).tuning.seed,
            Some(7)
        );
    }
}
