// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online training services whose clients and
// servers agree on when a run is ready and when it has ended, for its clients.
// If your team needs expertise in resumable training jobs, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What the stream client waits for the server to have: the run's policy
//! served before rollouts are drawn from it, a finish answered, a finished
//! run's result written.

use std::time::Duration;

use serde_json::Value;

use super::super::job::message_of;
use super::place::with_sources;
use super::{StreamError, StreamJob, CALL, DEFAULT_RETRY, START_WAIT};

/// How long a finished run's result is asked for while its job still ends, and how often.
const RESULT_WAIT: Duration = Duration::from_secs(600);
const RESULT_POLL: Duration = Duration::from_millis(200);

/// How often the server's model list is read while a run's policy is awaited.
const SERVED_POLL: Duration = Duration::from_millis(250);

impl StreamJob {
    /// Waits until the server lists the run's policy (`publish_adapter`)
    /// among its models: brain lists it once the base is loaded and the
    /// first version is published (from the start for a run that continues
    /// one it serves), and a rollout drawn before then is refused as a model
    /// it does not know. What the run reports meanwhile is kept for
    /// [`Self::events`].
    ///
    /// # Errors
    /// The run ended or was forgotten first, the policy is not listed within
    /// `within`, or the server's model list cannot be read.
    pub fn await_served(&mut self, within: Duration) -> Result<(), StreamError> {
        let started = std::time::Instant::now();
        loop {
            if self.lists_policy()? {
                return Ok(());
            }
            while let Ok(frame) = self.frames.try_recv() {
                let ended = matches!(frame["type"].as_str(), Some("done" | "forgotten"));
                let why = message_of(&frame.to_string());
                self.held.push_back(frame);
                if ended {
                    return Err(StreamError::Failed(format!(
                        "the run ended before its server served its policy {}: {why}",
                        self.publish_adapter
                    )));
                }
            }
            if started.elapsed() >= within {
                return Err(StreamError::Failed(format!(
                    "the server did not serve the run's policy {} within {}s",
                    self.publish_adapter,
                    within.as_secs()
                )));
            }
            std::thread::sleep(SERVED_POLL);
        }
    }

    /// Whether the server's model list (`GET /models`) names the run's policy.
    fn lists_policy(&mut self) -> Result<bool, StreamError> {
        let url = format!("{}/models", self.url);
        let sent = self.runtime.block_on(async {
            self.client
                .get(url)
                .bearer_auth(&self.key)
                .timeout(CALL)
                .send()
                .await
        });
        let response = sent.map_err(|e| StreamError::Transport(format!("{}: {e}", self.url)))?;
        let status = response.status().as_u16();
        let text = self.runtime.block_on(response.text()).unwrap_or_default();
        if status != 200 {
            return Err(self.refusal(status, &text));
        }
        let listed: Value = serde_json::from_str(&text).map_err(|e| {
            StreamError::Transport(format!("{}: a model list that is not JSON: {e}", self.url))
        })?;
        Ok(listed["data"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|m| m["id"].as_str() == Some(self.publish_adapter.as_str())))
    }

    /// The ended run's outputs (`GET /jobs/{id}/result`): its final adapter
    /// in the artifact store (`adapter_digest`, a BLAKE3 hex name) among
    /// them. The versions it published are named by their adapter files'
    /// SHA-256, which the artifact store does not name.
    ///
    /// A run that was told to finish ends after its last receipts (it still writes its outputs, and brain answers
    /// 409 until it has): the result is asked for again until it exists, for at most ten minutes. A read the
    /// server does not answer is sent again, as a finish is.
    ///
    /// # Errors
    /// The run has not ended within the wait, restarted, or the endpoint failed.
    pub fn result(&mut self) -> Result<Value, StreamError> {
        let url = format!("{}/jobs/{}/result", self.url, self.id);
        let started = std::time::Instant::now();
        let (status, text) = loop {
            let (response, _) =
                self.answered("result", || self.client.get(&url).bearer_auth(&self.key))?;
            let status = response.status().as_u16();
            let text = self.runtime.block_on(response.text()).unwrap_or_default();
            if status != 409 || started.elapsed() >= RESULT_WAIT {
                break (status, text);
            }
            std::thread::sleep(RESULT_POLL);
        };
        if status != 200 {
            return Err(self.refusal(status, &text));
        }
        let body: Value = serde_json::from_str(&text).map_err(|e| {
            StreamError::Transport(format!("{}: a result that is not JSON: {e}", self.url))
        })?;
        Ok(body["outputs"].clone())
    }

    /// No more groups: the run trains what it holds, commits and ends.
    ///
    /// A finish the server does not answer (a connection it closed, a server
    /// restarting) is sent again for as long as a new job is asked for: a
    /// finish is idempotent while the job finishes, and a job that has ended
    /// since an unanswered one (gone, 410) took it.
    ///
    /// # Errors
    /// The run restarted, or the endpoint did not answer within the wait.
    pub fn finish(&mut self) -> Result<(), StreamError> {
        let url = format!("{}/jobs/{}/finish?epoch={}", self.url, self.id, self.epoch);
        let (response, unanswered) =
            self.answered("finish", || self.client.post(&url).bearer_auth(&self.key))?;
        let status = response.status().as_u16();
        let text = self.runtime.block_on(response.text()).unwrap_or_default();
        if (200..300).contains(&status) || (unanswered && status == 410) {
            return Ok(());
        }
        Err(self.refusal(status, &text))
    }

    /// Keeps the version `version` the run retains served under its own id
    /// past the retention while the caller refers to it (`held`), or lets it
    /// go; the version's id. Holding is idempotent: a hold the server does
    /// not answer is sent again, as a finish is.
    ///
    /// # Errors
    /// The run restarted, the version is no longer served, or the endpoint
    /// refused.
    pub fn hold(&mut self, version: u64, held: bool) -> Result<String, StreamError> {
        let url = format!("{}/jobs/{}/hold?epoch={}", self.url, self.id, self.epoch);
        let body = serde_json::json!({"version": version, "held": held});
        let (response, _) = self.answered("hold", || {
            self.client
                .post(&url)
                .bearer_auth(&self.key)
                .header("content-type", "application/json")
                .body(body.to_string())
        })?;
        let status = response.status().as_u16();
        let text = self.runtime.block_on(response.text()).unwrap_or_default();
        if status != 200 {
            return Err(self.refusal(status, &text));
        }
        let answer: Value = serde_json::from_str(&text).map_err(|e| {
            StreamError::Transport(format!("{}: a hold answer that is not JSON: {e}", self.url))
        })?;
        Ok(answer["version_id"]
            .as_str()
            .unwrap_or_default()
            .to_string())
    }

    /// Sends the idempotent call `what` built by `request` and returns the
    /// server's answer, and whether an earlier send went unanswered. A send
    /// the server does not answer (a connection it closed, an idle keep-alive
    /// it dropped, a server restarting) is sent again for as long as a new
    /// job is asked for.
    ///
    /// # Errors
    /// The endpoint did not answer within the wait.
    fn answered(
        &self,
        what: &str,
        request: impl Fn() -> reqwest::RequestBuilder,
    ) -> Result<(reqwest::Response, bool), StreamError> {
        let started = std::time::Instant::now();
        let mut unanswered = false;
        loop {
            match self
                .runtime
                .block_on(async { request().timeout(CALL).send().await })
            {
                Ok(response) => return Ok((response, unanswered)),
                Err(e) if started.elapsed() < START_WAIT => {
                    eprintln!(
                        "splinter: the streaming run's {what} got no answer ({}); sending it again",
                        with_sources(&e)
                    );
                    unanswered = true;
                    std::thread::sleep(DEFAULT_RETRY);
                }
                Err(e) => {
                    return Err(StreamError::Transport(format!(
                        "{}: no answer to {what} for {}s: {}",
                        self.url,
                        START_WAIT.as_secs(),
                        with_sources(&e)
                    )))
                }
            }
        }
    }
}
