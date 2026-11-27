// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that prove a training run
// used the cards it was given, for its clients. If your team needs expertise
// in verifying multi-card and multi-node training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! How a served run was spread, as brain reports it (the `parallel` object of
//! a training report), checked against what [`BrainEndpoints::train_span`]
//! asked. A run that trained on fewer cards than requested is flagged on
//! stderr and in the run's record; a report without the object (an older
//! brain) is recorded as not reported, never as one rank.

use serde_json::Value;
use splinter_core::training::{ParallelRecord, ParallelRequest, TrainingParallelism};

use crate::endpoint::{BrainEndpoints, TrainSpan};

/// What `span` asks of the servers: the ranks in all, the servers they are on
/// and the ranks of the first, which holds rank 0 and reports.
fn requested(span: &TrainSpan) -> ParallelRequest {
    match span {
        TrainSpan::OneCard => ParallelRequest {
            world_size: span.ranks(),
            servers: 1,
            first_server_ranks: 1,
        },
        TrainSpan::Cards(cards) => ParallelRequest {
            world_size: span.ranks(),
            servers: 1,
            first_server_ranks: *cards,
        },
        TrainSpan::Servers(job) => ParallelRequest {
            world_size: span.ranks(),
            servers: u32::try_from(job.cards.len()).unwrap_or(u32::MAX),
            first_server_ranks: job.cards.first().copied().unwrap_or(0),
        },
    }
}

/// The run's `parallel` object checked against the request, with every
/// warning printed. An object that is present but unreadable is an error: it
/// cannot be told from a run that was not spread.
///
/// # Errors
/// The `parallel` object does not have the shape brain documents.
pub(super) fn observe(brain: &BrainEndpoints, report: &Value) -> Result<ParallelRecord, String> {
    let reported = match &report["parallel"] {
        Value::Null => None,
        object => Some(
            serde_json::from_value::<TrainingParallelism>(object.clone())
                .map_err(|e| format!("the service's parallel report: {e}"))?,
        ),
    };
    let record = ParallelRecord::check(requested(brain.train_span()), reported);
    for warning in record.warnings() {
        eprintln!("brain train: WARNING {warning}");
    }
    Ok(record)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::endpoint::ServersJob;

    fn endpoints(span: TrainSpan) -> BrainEndpoints {
        BrainEndpoints::new(vec!["http://127.0.0.1:1/v1".into()], "k")
            .unwrap()
            .with_train_span(span)
    }

    fn two_servers() -> TrainSpan {
        TrainSpan::Servers(ServersJob {
            cards: vec![2, 2],
            coordinator: "127.0.0.1:29500".into(),
            secret_sha256: "ab".repeat(32),
        })
    }

    fn parallel(world: u32) -> Value {
        json!({"parallel": {
            "world_size": world, "ranks": [0, 2],
            "devices": [{"rank": 0, "device": "gpu0", "name": "GH200"},
                        {"rank": 1, "device": "gpu1", "name": "GH200"}],
            "collective": "nccl", "transport": "tcp", "network": null, "sharding": "replicated"
        }})
    }

    #[test]
    fn a_matching_report_is_recorded_without_warnings() {
        let record = observe(&endpoints(two_servers()), &parallel(4)).unwrap();
        assert!(record.warnings().is_empty(), "{record:?}");
        assert_eq!(record.reported.unwrap().world_size, 4);
    }

    #[test]
    fn a_report_of_fewer_ranks_is_flagged() {
        let record = observe(&endpoints(two_servers()), &parallel(2)).unwrap();
        assert!(
            record.mismatches.iter().any(|m| m.contains("asked for 4")),
            "{record:?}"
        );
    }

    #[test]
    fn a_report_without_the_object_is_not_reported() {
        let record = observe(&endpoints(two_servers()), &json!({})).unwrap();
        assert!(record.reported.is_none());
        assert!(record.mismatches.is_empty());
        assert!(record.warnings()[0].contains("not reported"));
        let one = observe(&endpoints(TrainSpan::OneCard), &json!({})).unwrap();
        assert!(one.reported.is_none() && one.warnings().is_empty());
    }

    #[test]
    fn an_unreadable_object_is_an_error() {
        let bad = json!({"parallel": {"world_size": "four"}});
        assert!(observe(&endpoints(TrainSpan::OneCard), &bad).is_err());
    }
}
