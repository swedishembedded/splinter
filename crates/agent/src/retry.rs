// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose model calls survive
// the dropped connections of a shared inference service, for its clients. If
// your team needs expertise in making agent traffic to a serving cluster
// robust, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Bounded retry of a model request that failed in a way that may pass.
//!
//! A shared service closes idle connections and restarts; a request sent on
//! such a connection fails with "connection closed before message completed"
//! before the model has produced anything, or cuts a streaming reply off
//! mid-body ("error reading a body from connection: unexpected EOF"). Asking
//! again from the start is safe (a reply is only used whole) and usually
//! succeeds, so [`Retrying`] asks again, a bounded number of times with a
//! growing pause, for those failures only: a refusal, a malformed request or
//! a deadline is returned at once. Every model call of the typed calls and
//! the solves goes through it, so a campaign does not lose a whole
//! generation to one dropped connection.

use std::sync::Arc;
use std::time::Duration;

use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseStream};

/// How often a request is asked again after a failure that may pass.
pub const RETRIES: u32 = 4;

/// The pause before the first repeat; each later one doubles it.
pub const FIRST_PAUSE: Duration = Duration::from_millis(500);

/// Whether `error` is a connection-level or server-availability failure that
/// asking again may get past: the request never reached a model that
/// answered, or the connection ended before the reply's body did. A body
/// that arrived whole and does not parse is not one. A text sven has already
/// classified as transient counts.
#[must_use]
pub fn is_transient(error: &anyhow::Error) -> bool {
    is_transient_text(&format!("{error:#}"))
}

/// [`is_transient`] for an error already written as text.
#[must_use]
pub fn is_transient_text(text: &str) -> bool {
    let text = text.to_lowercase();
    [
        "transient",
        "connection closed before message completed",
        "connection reset",
        "error reading a body from connection",
        "unexpected eof",
        "broken pipe",
        "error sending request",
        "http 502",
        "http 503",
        "http 504",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

/// A provider that asks again when a request fails transiently.
pub struct Retrying {
    inner: Arc<dyn ModelProvider>,
    retries: u32,
    first_pause: Duration,
}

impl Retrying {
    /// `inner`, retried [`RETRIES`] times from a pause of [`FIRST_PAUSE`].
    #[must_use]
    pub fn wrap(inner: Arc<dyn ModelProvider>) -> Arc<dyn ModelProvider> {
        Self::with(inner, RETRIES, FIRST_PAUSE)
    }

    /// `inner`, retried `retries` times from a pause of `first_pause`.
    #[must_use]
    pub fn with(
        inner: Arc<dyn ModelProvider>,
        retries: u32,
        first_pause: Duration,
    ) -> Arc<dyn ModelProvider> {
        Arc::new(Self {
            inner,
            retries,
            first_pause,
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Retrying {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let mut pause = self.first_pause;
        for attempt in 0..=self.retries {
            match self.inner.complete(request.clone()).await {
                Err(e) if attempt < self.retries && is_transient(&e) => {
                    eprintln!(
                        "splinter: the model request failed transiently ({e:#}); asking again in {pause:?} ({}/{})",
                        attempt + 1,
                        self.retries
                    );
                    tokio::time::sleep(pause).await;
                    pause = pause.saturating_mul(2);
                }
                other => return other,
            }
        }
        unreachable!("the last attempt returns")
    }

    fn catalog_max_output_tokens(&self) -> Option<u32> {
        self.inner.catalog_max_output_tokens()
    }

    fn catalog_context_window(&self) -> Option<u32> {
        self.inner.catalog_context_window()
    }

    fn config_context_window(&self) -> Option<u32> {
        self.inner.config_context_window()
    }

    fn config_max_output_tokens(&self) -> Option<u32> {
        self.inner.config_max_output_tokens()
    }

    async fn probe_context_window(&self) -> Option<u32> {
        self.inner.probe_context_window().await
    }

    fn supports_images(&self) -> bool {
        self.inner.supports_images()
    }

    fn supports_audio(&self) -> bool {
        self.inner.supports_audio()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    /// Fails its first `failures` requests with `message`, then answers.
    struct Dropping {
        failures: u32,
        message: &'static str,
        asked: AtomicU32,
    }

    #[async_trait::async_trait]
    impl ModelProvider for Dropping {
        fn name(&self) -> &str {
            "scripted"
        }

        fn model_name(&self) -> &str {
            "scripted"
        }

        async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
            if self.asked.fetch_add(1, Ordering::SeqCst) < self.failures {
                anyhow::bail!("{}", self.message);
            }
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    const DROPPED: &str = "openai request failed: error sending request for url (http://x/v1/chat/completions): connection closed before message completed";

    fn dropping(failures: u32, message: &'static str) -> Arc<Dropping> {
        Arc::new(Dropping {
            failures,
            message,
            asked: AtomicU32::new(0),
        })
    }

    #[tokio::test]
    async fn a_dropped_connection_is_asked_again_and_the_call_succeeds() {
        let inner = dropping(1, DROPPED);
        let provider = Retrying::with(inner.clone(), 3, Duration::from_millis(1));
        assert!(provider
            .complete(CompletionRequest::default())
            .await
            .is_ok());
        assert_eq!(inner.asked.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn the_retries_are_bounded_and_the_last_error_is_returned() {
        let inner = dropping(u32::MAX, DROPPED);
        let provider = Retrying::with(inner.clone(), 2, Duration::from_millis(1));
        let error = provider
            .complete(CompletionRequest::default())
            .await
            .err()
            .expect("it keeps failing");
        assert!(format!("{error:#}").contains("connection closed"));
        assert_eq!(
            inner.asked.load(Ordering::SeqCst),
            3,
            "one ask and two repeats"
        );
    }

    #[test]
    fn a_reply_body_cut_off_mid_stream_is_transient() {
        // What a router restart left an agent stage with: the reply's chunked
        // body ended before its next chunk.
        assert!(is_transient_text(
            "error decoding response body: error reading a body from connection: \
             unexpected EOF during chunk size line"
        ));
        assert!(!is_transient_text(
            "error decoding response body: expected value at line 1 column 1"
        ));
    }

    #[tokio::test]
    async fn a_failure_that_is_not_transient_is_returned_at_once() {
        let inner = dropping(
            u32::MAX,
            "openai request failed: HTTP 400 context length exceeded",
        );
        let provider = Retrying::with(inner.clone(), 3, Duration::from_millis(1));
        assert!(provider
            .complete(CompletionRequest::default())
            .await
            .is_err());
        assert_eq!(inner.asked.load(Ordering::SeqCst), 1);
    }
}
