// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tamper-evident hand-over of agent benchmark
// results into training pipelines, for its clients. If your team needs
// expertise in turning benchmark runs into verified training data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A harness trial's request log (`model-requests.jsonl`), divided among its
//! conversations: which model role each request was sent to, whether it
//! carried an image, and each request with the response it got.

use splinter_core::harness::{ModelCall, RoleSource};
use splinter_data::request_log::{made_a_step, read_request_log, LoggedRequest};

use super::REQUESTS;
/// The requests of one conversation as the request log attributes them.
#[derive(Clone, Debug)]
pub(super) struct Attribution {
    /// The one model role all its requests name, if they do.
    pub(super) role: Option<String>,
    pub(super) images: bool,
    /// Which requests of the log they are, in order.
    requests: Vec<usize>,
}

impl Attribution {
    pub(super) fn source(&self) -> RoleSource {
        if self.role.is_some() {
            RoleSource::RequestLog
        } else {
            RoleSource::Unknown
        }
    }
}

/// What `model-requests.jsonl` says about each request: the model role it
/// was sent to, whether it carries an image, and the request itself with the
/// response it got.
pub(super) struct RequestLog {
    requests: Vec<LoggedRequest>,
    pub(super) any_image: bool,
}

impl RequestLog {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self, String> {
        let requests = read_request_log(bytes).map_err(|e| format!("{REQUESTS} {e}"))?;
        let any_image = requests.iter().any(|r| r.media);
        Ok(Self {
            requests,
            any_image,
        })
    }

    /// Divides the requests among `conversations`, in order, one request per
    /// agent step. Only an exact division counts: the request count must
    /// equal the agent steps of all the conversations, or else the count of
    /// the requests that made a step - those answered with text or a call; an
    /// answer that said nothing, which the agent records no step for and asks
    /// again after, and a request the trial ended before answering, made
    /// none - or the requests cannot be told apart and nothing is attributed.
    pub(super) fn attribute(
        &self,
        conversations: &[&atif::Trajectory],
    ) -> Option<Vec<Attribution>> {
        let steps: Vec<usize> = conversations
            .iter()
            .map(|t| {
                t.steps
                    .iter()
                    .filter(|s| s.source == atif::StepOrigin::Agent)
                    .count()
            })
            .collect();
        let total = steps.iter().sum::<usize>();
        let stepping: Vec<usize> = if self.requests.len() == total {
            (0..total).collect()
        } else {
            (0..self.requests.len())
                .filter(|&n| self.requests[n].call.as_ref().is_some_and(made_a_step))
                .collect()
        };
        if stepping.len() != total {
            return None;
        }
        let mut start = 0;
        Some(
            steps
                .into_iter()
                .map(|n| {
                    let own = &stepping[start..start + n];
                    let first = own.first().and_then(|&r| self.requests[r].role.clone());
                    let one_role = own.iter().all(|&r| self.requests[r].role == first);
                    start += n;
                    Attribution {
                        role: first.filter(|_| one_role),
                        images: own.iter().any(|&r| self.requests[r].media),
                        requests: own.to_vec(),
                    }
                })
                .collect(),
        )
    }

    /// The calls of the requests attributed to a conversation; `None` unless
    /// the log holds a readable response to every one of them.
    pub(super) fn calls_of(&self, attribution: &Attribution) -> Option<Vec<ModelCall>> {
        attribution
            .requests
            .iter()
            .map(|&r| self.requests[r].call.clone())
            .collect()
    }
}
