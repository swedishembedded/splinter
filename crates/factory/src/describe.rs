// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment planners that show the work an
// experiment will do before it spends anything, for its clients. If your
// team needs expertise in machine-learning experiment orchestration, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Each planned stage's work, in a person's words, as `splinter plan` shows
//! it.

use std::collections::BTreeMap;

use crate::collection::LoadedCollection;
use crate::graph::{self, CompiledPlan};
use crate::ops::{
    names, AttemptParams, EvaluationParams, ImportParams, ProjectParams, PromotionParams,
    TrainParams,
};
use crate::plan::PlannedStage;

/// What `stage` will do, for a person.
pub(crate) fn work(
    stage: &PlannedStage,
    compiled: &CompiledPlan,
    collections: &BTreeMap<String, LoadedCollection>,
) -> String {
    let model = |role: &str| {
        compiled
            .models
            .get(role)
            .map_or_else(|| role.to_owned(), ToString::to_string)
    };
    match stage.operation.as_str() {
        names::PIN => {
            let count = collections
                .get(&stage.key)
                .map_or(0, |c| c.collection.tasks.len());
            let file = stage
                .input("collection")
                .and_then(|i| compiled.inputs.get(i))
                .map_or("?", |i| i.path.as_str());
            format!("pin {count} tasks of {file}")
        }
        names::ATTEMPTS => {
            let Ok(p) = stage.params::<AttemptParams>() else {
                return String::new();
            };
            let unsolved = stage.sources("unsolved_by");
            let selected = if !p.tasks.is_empty() {
                format!("{} named tasks", p.tasks.len())
            } else if unsolved.is_empty() {
                "its tasks".to_owned()
            } else {
                format!("the tasks unsolved by {}", unsolved.join(", "))
            };
            let on = p
                .model
                .as_deref()
                .map_or_else(|| "its own model".to_owned(), model);
            if p.interactive {
                format!(
                    "the sessions people play of {selected} with {} on {on} (`splinter play`)",
                    p.agent
                )
            } else {
                format!(
                    "{selected} x {} attempts of {} on {on}, {} at a time",
                    p.attempts, p.agent, p.concurrency
                )
            }
        }
        names::IMPORT => {
            let Ok(p) = stage.params::<ImportParams>() else {
                return String::new();
            };
            let selected = if p.tasks.is_empty() {
                "its tasks".to_owned()
            } else {
                format!("{} named tasks", p.tasks.len())
            };
            let jobs: Vec<&str> = stage
                .inputs_of("jobs")
                .into_iter()
                .filter_map(|i| compiled.inputs.get(i))
                .map(|i| i.path.as_str())
                .collect();
            format!(
                "import the trials of {selected} from Harbor jobs {}, running nothing",
                jobs.join(", ")
            )
        }
        names::PROJECT => stage.params::<ProjectParams>().map_or_else(
            |_| String::new(),
            |p| {
                format!(
                    "{} {} records of {} attempts for {}",
                    p.projection,
                    p.objective,
                    p.admission,
                    model(&p.target)
                )
            },
        ),
        names::TRAIN => stage.params::<TrainParams>().map_or_else(
            |_| String::new(),
            |p| {
                format!(
                    "{} {} of {} with {}",
                    p.engine,
                    p.method,
                    model(&p.base),
                    p.params
                )
            },
        ),
        names::EVALUATION => stage.params::<EvaluationParams>().map_or_else(
            |_| String::new(),
            |p| {
                format!(
                    "{} against {} under {}",
                    stage.source("candidate").unwrap_or("?"),
                    model(&p.baseline),
                    p.plan
                )
            },
        ),
        names::PROMOTION => {
            if stage.params::<PromotionParams>().is_ok_and(|p| p.promote) {
                "promote the best improved candidate".into()
            } else {
                "record the candidates without promoting any".into()
            }
        }
        operation => match &stage.foreach {
            Some(collection) => format!(
                "{operation} once per item of {}, {} at a time",
                source_name(collection),
                stage.concurrency
            ),
            None => format!("{operation} once"),
        },
    }
}

/// A source as the recipe names it.
fn source_name(source: &graph::Source) -> String {
    match source {
        graph::Source::Input(name) => format!("$inputs.{name}"),
        graph::Source::Stage { stage, port } => format!("$stages.{stage}.{port}"),
        graph::Source::Item => "$item".into(),
    }
}
