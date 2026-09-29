// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that run their model
// in-process - no separately versioned serving process to drift from the
// build. If your team needs expertise in local inference integration or
// provider abstractions, you can procure our services by sending an email
// to info@swedishembedded.com.

//! One generation on the locked model: render, prefill in chunks, decode,
//! scan tool calls, and stream the visible text as it is produced.

use capability::{CancelToken, Invocation, Outcome, Progress};
use data::qwen_tokenizer::QwenBpe;
use data::rng::Rng;
use qwen3::chat::{self, SeqState};
use qwen3::model::Qwen;
use qwen3::sample::generate_kv_stream_cancellable;
use sven_sdk::model::ResponseEvent;

use super::PREFILL_CHUNK_TOKENS;

/// Everything one generation reads beyond the locked model and the parsed
/// request: the sampler's head, tokenizer and stop tokens, the inline
/// context budget, the stream visible text goes out on, and the token an
/// abandoned turn arms.
pub(super) struct Generation<'a> {
    pub head: &'a [f32],
    pub tok: &'a QwenBpe,
    pub eos: &'a [u32],
    pub context_tokens: u32,
    pub tx: &'a tokio::sync::mpsc::Sender<anyhow::Result<ResponseEvent>>,
    pub cancel: &'a CancelToken,
}

/// One generation, start to finish: render, decode, scan, finish. Runs on a
/// dedicated OS thread with the model lock held, streaming visible text
/// through the stream as the scanner produces it.
///
/// Cancellation is cooperative: the token is polled between prefill chunks
/// (sized by [`PREFILL_CHUNK_TOKENS`]) and between decode steps, never
/// inside one (brain's documented contract). A dropped receiver - the turn
/// above this stream was abandoned - arms it through the failed send; the
/// runner's `stop_generation` arms it directly. Either way an interrupted
/// turn stops within one chunk of where it is, not after the whole prompt
/// or the whole generation cap.
pub(super) fn generate_once(
    model: &Qwen,
    inv: &Invocation,
    gen: &Generation<'_>,
) -> anyhow::Result<Outcome> {
    let mut req = chat::parse_request(gen.tok, inv).map_err(anyhow::Error::msg)?;
    let context = usize::try_from(gen.context_tokens).unwrap_or(usize::MAX);
    anyhow::ensure!(
        req.ids.len() < context,
        "prompt ({} tokens) fills the engine's context ({context}); nothing left to generate",
        req.ids.len()
    );
    // A prompt leaves exactly so much room. A caller that asked for more
    // than fits (explore's 32k object budget against a 16k cache) gets the
    // remaining room, not an error: the budget was an upper bound, and the
    // rendered prompt's size is only known here, after the chat-template
    // render. An oversized PROMPT is the hard error above.
    req.max_new = req.max_new.min(context - req.ids.len());
    let mut rng = Rng::new(req.seed);
    // Two numbers an operator needs to tell a slow device from a wedged
    // generation: how much prompt there is, and how long the first token
    // took to arrive after it.
    let started = std::time::Instant::now();
    eprintln!("serve: prompt {} tokens", req.ids.len());
    let mut first_token: Option<std::time::Duration> = None;
    // A clone of the token outlives the sequence: the emit path arms it when
    // the consumer above this stream is gone.
    let abandon = gen.cancel.clone();
    let mut seq = SeqState::new(&req, gen.cancel.clone());
    let mut ids_out: Vec<u32> = Vec::with_capacity(req.max_new);
    // A failed send means the consumer is gone - the turn was abandoned
    // above this stream - so arm the token; the next `advance` observes it
    // and ends the sequence.
    let emit = &mut |p: Progress| {
        if let Some(text) = p.delta {
            if first_token.is_none() {
                let elapsed = started.elapsed();
                first_token = Some(elapsed);
                eprintln!(
                    "serve: prompt {} tokens, first token after {:.1}s",
                    req.ids.len(),
                    elapsed.as_secs_f32()
                );
            }
            if gen
                .tx
                .blocking_send(Ok(ResponseEvent::TextDelta(text)))
                .is_err()
            {
                abandon.cancel();
            }
        }
    };
    let generated = generate_kv_stream_cancellable(
        model,
        &req.ids,
        req.max_new,
        req.temp,
        req.top_k,
        req.top_p,
        gen.eos,
        &mut rng,
        gen.head,
        gen.cancel,
        PREFILL_CHUNK_TOKENS,
        &mut |_i, t| {
            ids_out.push(t);
            // `advance` answers "should we stop?"; the callback answers
            // "keep going?" - the same inversion the serving path applies.
            !seq.advance(gen.tok, &ids_out, emit)
        },
    );
    // `finish` flushes the scanner's held-back tail through the same emit,
    // so the streamed deltas and the outcome's text stay identical.
    Ok(seq.finish(gen.tok, &generated, emit))
}

/// The finished outcome's non-text events. Visible text was streamed while
/// it was scanned; the tail carries the tool calls the scanner extracted -
/// complete, post-finish - then usage, then done.
pub(super) fn events_from(outcome: Outcome) -> Vec<anyhow::Result<ResponseEvent>> {
    let mut events = Vec::new();
    if let Some(calls) = outcome
        .outputs
        .get("tool_calls")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| v.as_array().cloned())
    {
        for (index, call) in calls.iter().enumerate() {
            let arguments = match call.get("arguments") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            events.push(Ok(ResponseEvent::ToolCall {
                index: index as u32,
                id: call
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                name: call
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                arguments,
            }));
        }
    }
    events.push(Ok(ResponseEvent::Usage {
        input_tokens: outcome
            .outputs
            .get("prompt_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as u32,
        output_tokens: outcome
            .outputs
            .get("completion_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as u32,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        cost_usd: None,
    }));
    events.push(Ok(ResponseEvent::Done));
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_finished_outcomes_tail_is_tool_calls_usage_and_done() {
        let outcome = Outcome::new()
            .set("text", serde_json::json!("did it"))
            .set("prompt_tokens", serde_json::json!(120))
            .set("completion_tokens", serde_json::json!(34))
            .set(
                "tool_calls",
                serde_json::json!(r#"[{"id":"c1","name":"write","arguments":"{}"}]"#),
            );
        let events = events_from(outcome);
        // The visible text was streamed while it was scanned; the tail must
        // not repeat it.
        assert_eq!(events.len(), 3, "tool call, usage, done");
        assert!(matches!(
            &events[0],
            Ok(ResponseEvent::ToolCall { name, arguments, .. })
                if name == "write" && arguments == "{}"
        ));
        assert!(matches!(
            &events[1],
            Ok(ResponseEvent::Usage { input_tokens, output_tokens, cost_usd: None, .. })
                if *input_tokens == 120 && *output_tokens == 34
        ));
        assert!(matches!(events[2], Ok(ResponseEvent::Done)));
    }
}
