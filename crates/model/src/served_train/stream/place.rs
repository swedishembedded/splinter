// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online training services whose runs survive
// the processes that drive them, for its clients. If your team needs expertise
// in resumable training jobs, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Where a run's job is, and cancelling one a stopped controller left.

use serde::Serialize;

use super::{http_client, message_of, StreamError, StreamJob, CALL};
use crate::endpoint::BrainEndpoints;

/// Where a run's job is: the endpoint that placed it and the job's id. A caller keeps it beside the run's state, so
/// that a controller that stopped can cancel the job it left running ([`StreamJob::cancel_left`]): brain starts no
/// other job of a run while one runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobPlace {
    /// The endpoint (its `/v1` base).
    pub url: String,
    /// The job's id there.
    pub id: String,
}

impl StreamJob {
    /// Where this job is, for the caller to keep ([`JobPlace`]).
    #[must_use]
    pub fn job_place(&self) -> JobPlace {
        JobPlace {
            url: self.url.clone(),
            id: self.id.clone(),
        }
    }

    /// Cancels the job at `place` when its server still runs it: brain writes the job's training state on a cancel,
    /// so what it applied is kept, and lets go of the run, whose next job may then start. A job the server does not
    /// know (it ended, or the server restarted since) and a server that cannot be reached leave nothing to cancel.
    ///
    /// # Errors
    /// The server refused the cancel for another reason.
    pub fn cancel_left(brain: &BrainEndpoints, place: &JobPlace) -> Result<(), StreamError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| StreamError::Failed(format!("starting a runtime: {e}")))?;
        let client = runtime.block_on(async { http_client() })?;
        let sent = runtime.block_on(async {
            client
                .delete(format!("{}/jobs/{}", place.url, place.id))
                .bearer_auth(brain.key())
                .timeout(CALL)
                .send()
                .await
        });
        let Ok(response) = sent else { return Ok(()) };
        let status = response.status();
        if status.is_success() || matches!(status.as_u16(), 404 | 410) {
            return Ok(());
        }
        let text = runtime.block_on(response.text()).unwrap_or_default();
        Err(StreamError::Failed(format!(
            "cancelling the job {} left at {} (HTTP {}): {}",
            place.id,
            place.url,
            status.as_u16(),
            message_of(&text)
        )))
    }
}

/// `error` and every error under it, which reqwest's own message leaves out (a closed connection, a refusal).
pub(super) fn with_sources(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(&format!(": {cause}"));
        source = cause.source();
    }
    text
}
