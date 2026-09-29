// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One prompt in, one reply's text out.

use futures::StreamExt;
use sven_sdk::model::{CompletionRequest, Message, ModelProvider, ResponseEvent, Role};

/// Sends `prompt` as one user message and collects the streamed reply text.
///
/// The request streams (the OpenAI-compatible drivers parse every response
/// as SSE), carries its own output budget and asks for a JSON object on
/// drivers that support the constraint - it is how every one-shot question
/// Splinter asks a model is sent.
pub async fn complete_text(provider: &dyn ModelProvider, prompt: &str) -> anyhow::Result<String> {
    let req = CompletionRequest {
        messages: vec![Message {
            role: Role::User,
            content: sven_sdk::model::MessageContent::Text(prompt.to_string()),
        }],
        // The OpenAI-compat driver parses every response as SSE, so a
        // non-streaming request would return a plain JSON object the
        // parser extracts nothing from - the stream would end empty and
        // every strict parse would fail on it.
        stream: true,
        // A reasoning model spends its output budget on thinking before
        // any answer text arrives; the drivers' 4096-token default ends
        // such a turn at `MaxTokens` with zero visible text. One section
        // asking for EVERY fact needs room for the reasoning AND the
        // object, so the completion carries its own cap. The local
        // provider ignores the override (its 512-token budget and
        // context check are its own), so this stays remote-only.
        max_output_tokens_override: Some(32_768),
        // Observed drift: a full-sheet extraction prompt once drew a
        // markdown answer despite the JSON-only instruction. Drivers that
        // support it accept a JSON-object constraint on the wire; drivers
        // that don't ignore the field, and the strict parse stays the gate.
        response_format: Some(sven_sdk::model::ResponseFormat::JsonObject),
        ..Default::default()
    };
    let mut stream = provider.complete(req).await?;
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::TextDelta(delta) => text.push_str(&delta),
            // The budget ran out with no visible answer - likely a reasoning
            // model that spent everything on thinking. An empty Ok here would
            // surface downstream as a parse failure on a reply that never
            // existed; name the real cause instead.
            ResponseEvent::MaxTokens if text.is_empty() => {
                anyhow::bail!(
                    "completion hit the output-token limit before any answer text; \
                    raise max_output_tokens_override or simplify the prompt"
                );
            }
            ResponseEvent::Error(what) => {
                anyhow::bail!("stream failed: {what}");
            }
            ResponseEvent::Done => break,
            _ => {}
        }
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captures the request a completion carries, so tests can assert on the
    /// wire contract without a server.
    struct CapturingProvider {
        reply: &'static str,
        seen: std::sync::Mutex<Vec<CompletionRequest>>,
    }

    /// Plays a fixed event script, for testing stream failure paths.
    struct ScriptedProvider(Vec<anyhow::Result<ResponseEvent>>);

    #[async_trait::async_trait]
    impl ModelProvider for ScriptedProvider {
        fn name(&self) -> &str {
            "scripted"
        }
        fn model_name(&self) -> &str {
            "script-1"
        }
        async fn complete(
            &self,
            _req: CompletionRequest,
        ) -> anyhow::Result<sven_sdk::model::ResponseStream> {
            let script = self
                .0
                .iter()
                .map(|e| match e {
                    Ok(ev) => Ok(ev.clone()),
                    Err(e) => Err(anyhow::anyhow!("{e}")),
                })
                .collect::<Vec<_>>();
            Ok(Box::pin(futures::stream::iter(script)))
        }
    }

    #[async_trait::async_trait]
    impl ModelProvider for CapturingProvider {
        fn name(&self) -> &str {
            "capturing"
        }
        fn model_name(&self) -> &str {
            "capture-1"
        }
        async fn complete(
            &self,
            req: CompletionRequest,
        ) -> anyhow::Result<sven_sdk::model::ResponseStream> {
            self.seen.lock().unwrap().push(req);
            let reply = self.reply;
            Ok(Box::pin(futures::stream::iter(vec![
                Ok(ResponseEvent::TextDelta(reply.to_string())),
                Ok(ResponseEvent::Done),
            ])))
        }
    }

    /// A reasoning model that burns its whole output budget on thinking ends
    /// at MaxTokens with ZERO visible text. That must surface as an error
    /// naming the budget, not Ok("") - an empty Ok sends the caller chasing
    /// a parse failure on a reply that never existed.
    #[tokio::test]
    async fn max_tokens_with_no_text_is_an_error_not_an_empty_ok() {
        let provider = ScriptedProvider(vec![
            Ok(ResponseEvent::ThinkingDelta("pondering...".into())),
            Ok(ResponseEvent::MaxTokens),
            Ok(ResponseEvent::Done),
        ]);
        let err = complete_text(&provider, "extract facts").await.unwrap_err();
        assert!(
            err.to_string().contains("output-token limit"),
            "error should name the output-token limit, got: {err:#}"
        );
    }

    /// Text before the limit is not lost to the same error: it flows on to
    /// the strict parser, which reports the raw reply as evidence.
    #[tokio::test]
    async fn max_tokens_with_text_keeps_the_text() {
        let provider = ScriptedProvider(vec![
            Ok(ResponseEvent::TextDelta(r#"{"answer": "168 MHz"}"#.into())),
            Ok(ResponseEvent::MaxTokens),
            Ok(ResponseEvent::Done),
        ]);
        let text = complete_text(&provider, "q").await.unwrap();
        assert_eq!(text, r#"{"answer": "168 MHz"}"#);
    }

    /// A fatal mid-stream error is a hard failure of the completion, per the
    /// ResponseEvent::Error contract - never a silently truncated Ok.
    #[tokio::test]
    async fn stream_error_events_fail_the_completion() {
        let provider = ScriptedProvider(vec![
            Ok(ResponseEvent::TextDelta("partial".into())),
            Ok(ResponseEvent::Error("connection reset".into())),
        ]);
        let err = complete_text(&provider, "q").await.unwrap_err();
        assert!(
            err.to_string().contains("connection reset"),
            "error should carry the stream failure, got: {err:#}"
        );
    }

    /// Facts extraction runs against models that drift out of the requested
    /// shape (a full-sheet prompt produced a markdown answer in one observed
    /// run). Drivers that support it accept a JSON-object constraint, so the
    /// completion must ask for one; the strict parser stays as the gate.
    #[tokio::test]
    async fn completions_constrain_the_reply_to_a_json_object() {
        let provider = CapturingProvider {
            reply: r#"{"facts": []}"#,
            seen: std::sync::Mutex::new(Vec::new()),
        };
        let text = complete_text(&provider, "extract facts").await.unwrap();
        assert_eq!(text, r#"{"facts": []}"#);
        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].response_format,
            Some(sven_sdk::model::ResponseFormat::JsonObject)
        );
        assert!(seen[0].stream, "driver parses every reply as SSE");
    }
}
