// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that prove a training run
// used the cards it was given, for its clients. If your team needs expertise
// in verifying multi-card and multi-node training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! How a training run was spread over cards and servers, as brain reports it,
//! against what was asked of it (re-exported by [`crate::training`]).

use serde::{Deserialize, Serialize};

/// One rank's device in brain's `parallel` report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelDevice {
    /// The rank.
    pub rank: u32,
    /// `gpu<N>` or `cpu`.
    pub device: String,
    /// The card's name as its driver reports it; `None` when not observed.
    #[serde(default)]
    pub name: Option<String>,
}

/// The `parallel` object of a brain training report: how the run was spread,
/// as it ran. A field brain did not observe is `None`, never a default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingParallelism {
    /// Ranks in the whole group, across every server.
    pub world_size: u32,
    /// `[first, end)`: the ranks the reporting server ran.
    pub ranks: [u32; 2],
    /// One entry per rank the reporting server ran.
    #[serde(default)]
    pub devices: Vec<ParallelDevice>,
    /// What summed the gradients: `nccl` or `host`.
    #[serde(default)]
    pub collective: Option<String>,
    /// How the ranks reached each other: `in_process` or `tcp`.
    #[serde(default)]
    pub transport: Option<String>,
    /// The network NCCL picked between nodes.
    #[serde(default)]
    pub network: Option<serde_json::Value>,
    /// What the group sharded: `replicated`, `optimizer`, `gradients` or `parameters`.
    #[serde(default)]
    pub sharding: Option<String>,
}

/// How a run was asked to spread: what splinter sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelRequest {
    /// Ranks asked for in total.
    pub world_size: u32,
    /// Servers the job was spread over.
    pub servers: u32,
    /// Ranks asked of the first server, which holds rank 0 and reports.
    pub first_server_ranks: u32,
}

/// What a run was asked to spread as, what brain reported, and where they
/// differ. A run that trained on fewer cards than it asked for, or on the
/// host transport instead of the cards' collective, is a failed expectation
/// whatever its loss says.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelRecord {
    /// What was asked.
    pub requested: ParallelRequest,
    /// What brain reported; `None` when its report has no `parallel` object
    /// (an older brain): not reported, which is not one rank.
    pub reported: Option<TrainingParallelism>,
    /// Each way the report differs from the request; empty when it matches
    /// or when nothing was reported.
    pub mismatches: Vec<String>,
}

impl ParallelRecord {
    /// `reported` checked against `requested`.
    #[must_use]
    pub fn check(requested: ParallelRequest, reported: Option<TrainingParallelism>) -> Self {
        let mut mismatches = Vec::new();
        if let Some(got) = &reported {
            let [first, end] = got.ranks;
            if got.world_size != requested.world_size {
                mismatches.push(format!(
                    "asked for {} rank(s) and the run trained on {}",
                    requested.world_size, got.world_size
                ));
            }
            if first != 0 || end.saturating_sub(first) != requested.first_server_ranks {
                mismatches.push(format!(
                    "asked the first server for ranks [0, {}) and it ran [{first}, {end})",
                    requested.first_server_ranks
                ));
            }
            if usize::try_from(end.saturating_sub(first)).ok() != Some(got.devices.len()) {
                mismatches.push(format!(
                    "the report names {} device(s) for the {} rank(s) it ran",
                    got.devices.len(),
                    end.saturating_sub(first)
                ));
            }
            if requested.world_size > 1 && got.collective.as_deref() != Some("nccl") {
                mismatches.push(format!(
                    "asked for {} ranks and their gradients were summed by {} instead of nccl",
                    requested.world_size,
                    got.collective.as_deref().unwrap_or("no collective")
                ));
            }
            let transport = if requested.servers > 1 {
                "tcp"
            } else {
                "in_process"
            };
            if requested.world_size > 1 && got.transport.as_deref() != Some(transport) {
                mismatches.push(format!(
                    "asked for {} server(s), which reach each other over {transport}, and the run used {}",
                    requested.servers,
                    got.transport.as_deref().unwrap_or("no transport")
                ));
            }
        }
        Self {
            requested,
            reported,
            mismatches,
        }
    }

    /// What the record warns of: every mismatch, and that a spread run could
    /// not be confirmed because brain reported nothing. Empty otherwise.
    #[must_use]
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings: Vec<String> = self
            .mismatches
            .iter()
            .map(|m| format!("parallelism not as requested: {m}"))
            .collect();
        if self.reported.is_none() && self.requested.world_size > 1 {
            warnings.push(format!(
                "parallelism not reported: {} ranks were asked for and the service's report \
                 does not say how many trained",
                self.requested.world_size
            ));
        }
        warnings
    }

    /// One line for a run summary.
    #[must_use]
    pub fn line(&self) -> String {
        let Some(got) = &self.reported else {
            return format!(
                "not reported ({} rank(s) asked for)",
                self.requested.world_size
            );
        };
        let devices: Vec<String> = got
            .devices
            .iter()
            .map(|d| match &d.name {
                Some(name) => format!("{} {name}", d.device),
                None => d.device.clone(),
            })
            .collect();
        let verdict = if self.mismatches.is_empty() {
            "as requested".to_string()
        } else {
            format!("NOT AS REQUESTED ({})", self.mismatches.join("; "))
        };
        format!(
            "{} rank(s) ({} asked for), this server ran [{}, {}) on {}; collective {}, transport {}, sharding {}: {verdict}",
            got.world_size,
            self.requested.world_size,
            got.ranks[0],
            got.ranks[1],
            if devices.is_empty() { "no device named".to_string() } else { devices.join(", ") },
            got.collective.as_deref().unwrap_or("not observed"),
            got.transport.as_deref().unwrap_or("not observed"),
            got.sharding.as_deref().unwrap_or("not observed"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parallelism(
        world: u32,
        ranks: [u32; 2],
        collective: &str,
        transport: &str,
    ) -> TrainingParallelism {
        TrainingParallelism {
            world_size: world,
            ranks,
            devices: (ranks[0]..ranks[1])
                .map(|rank| ParallelDevice {
                    rank,
                    device: format!("gpu{rank}"),
                    name: Some("card".into()),
                })
                .collect(),
            collective: Some(collective.into()),
            transport: Some(transport.into()),
            network: None,
            sharding: Some("replicated".into()),
        }
    }

    const TWO_SERVERS: ParallelRequest = ParallelRequest {
        world_size: 4,
        servers: 2,
        first_server_ranks: 2,
    };

    #[test]
    fn a_report_matching_the_request_has_no_mismatch() {
        let record =
            ParallelRecord::check(TWO_SERVERS, Some(parallelism(4, [0, 2], "nccl", "tcp")));
        assert!(record.mismatches.is_empty(), "{record:?}");
        assert!(record.warnings().is_empty());
        assert!(record.line().contains("as requested"), "{}", record.line());
    }

    /// A run that silently trained on fewer cards than asked is a failed
    /// expectation, not a success.
    #[test]
    fn a_report_below_the_request_is_flagged() {
        let one_card = ParallelRecord::check(
            TWO_SERVERS,
            Some(parallelism(1, [0, 1], "host", "in_process")),
        );
        let text = one_card.warnings().join("\n");
        assert!(
            text.contains("asked for 4 rank(s) and the run trained on 1"),
            "{text}"
        );
        assert!(text.contains("host instead of nccl"), "{text}");
        assert!(text.contains("over tcp"), "{text}");
        assert!(one_card.line().contains("NOT AS REQUESTED"));
        let host = ParallelRecord::check(
            ParallelRequest {
                world_size: 4,
                servers: 1,
                first_server_ranks: 4,
            },
            Some(parallelism(4, [0, 4], "host", "in_process")),
        );
        assert_eq!(host.mismatches.len(), 1, "{host:?}");
    }

    /// An older brain's report has no object: not reported, never one rank.
    #[test]
    fn a_missing_object_is_not_reported_rather_than_one_rank() {
        let spread = ParallelRecord::check(TWO_SERVERS, None);
        assert!(spread.mismatches.is_empty());
        assert!(spread.line().starts_with("not reported"));
        assert!(spread.warnings()[0].contains("not reported"));
        let single = ParallelRecord::check(
            ParallelRequest {
                world_size: 1,
                servers: 1,
                first_server_ranks: 1,
            },
            None,
        );
        assert!(
            single.warnings().is_empty(),
            "nothing was asked beyond one rank"
        );
    }
}
