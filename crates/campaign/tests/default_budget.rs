// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that finish within the time
// they are given, for its clients. If your team needs expertise in turning an
// open-ended learning job into a bounded one, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a learning run with no budget on a large corpus has no end, and a
//! sentence cannot carry one. The configuration may name a default budget
//! (`SPLINTER_BUDGET`); a run that names none runs under it, one that names its
//! own runs under that, and a value that is not a duration is refused by name.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::{config, Scratch, Scripted};
use splinter_agent::solve::Model;
use splinter_campaign::learn::{learn, parse_budget, LearnRequest, Learned};
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::train::{TrainPlan, Trainer};
use splinter_campaign::{CampaignError, Context};
use splinter_policy::train::{Trained, TrainedPreference};
use sven_sdk::CancelToken;

struct NoTraining;

impl Trainer for NoTraining {
    fn train(&self, _: &Context, _: &TrainPlan, _: &CancelToken) -> Result<Trained, CampaignError> {
        panic!("a dry run trains nothing")
    }

    fn train_preference(
        &self,
        _: &Context,
        _: &TrainPlan,
        _: &CancelToken,
    ) -> Result<TrainedPreference, CampaignError> {
        panic!("a dry run trains nothing")
    }
}

fn planned(
    test: &str,
    default_budget: Option<&str>,
    asked: Option<&str>,
) -> Result<Option<u64>, CampaignError> {
    let scratch = Scratch::new(test);
    let mut settings = config(&scratch);
    settings.default_budget = default_budget.map(str::to_string);
    let ctx = Context::new(settings, false).unwrap().with_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(Scripted::new(|_| String::new())), common::POLICY),
    );
    let manual = scratch.0.join("manual.md");
    std::fs::write(&manual, "# Manual\n\nThe pump runs at 60 rpm.\n").unwrap();
    let learned = learn(
        &ctx,
        &LearnRequest {
            sources: vec![manual.display().to_string()],
            kinds: vec!["recall".into()],
            budget: asked.map(|b| parse_budget(b).unwrap()),
            dry_run: true,
            ..LearnRequest::default()
        },
        &NoTraining,
    )?;
    let Learned::Planned(plan) = learned else {
        panic!("a dry run only plans");
    };
    Ok(plan.budget_secs)
}

#[test]
fn a_run_that_names_no_budget_runs_under_the_configured_default() {
    assert_eq!(
        planned("budget-default", Some("2h"), None).unwrap(),
        Some(7200)
    );
    assert_eq!(planned("budget-none", None, None).unwrap(), None);
}

#[test]
fn a_budget_the_run_names_wins_over_the_default() {
    assert_eq!(
        planned("budget-named", Some("2h"), Some("30m")).unwrap(),
        Some(1800)
    );
}

#[test]
fn a_default_that_is_not_a_duration_is_refused_by_name() {
    let error = planned("budget-bad", Some("soon"), None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("SPLINTER_BUDGET") && error.contains("soon"),
        "{error}"
    );
}
