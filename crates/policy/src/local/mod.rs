// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that run their model
// in-process - no separately versioned serving process to drift from the
// build. If your team needs expertise in local inference integration or
// provider abstractions, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The local model provider: brain's chat pipeline, linked in-process.
//!
//! sven's [`ModelProvider`] seam is what remote providers (OpenAI,
//! Anthropic, OpenRouter) hang off; this module hangs a LOCAL model off the
//! same seam, so the loop agent runs the same engine the wire providers do
//! (same tool loop, same event stream, same usage accounting) with no HTTP
//! hop and no dependency on a separately running, separately versioned
//! `brain serve`: Splinter is built against the brain SDK it links, and
//! loads weights and adapters directly from disk at startup.
//!
//! The generation is [`brain::ChatPipeline`]'s: the chat template, tool
//! schemas, KV-cached decode, chunked prefill, the reasoning and tool-call
//! scanner and cooperative cancellation are brain's, the same functions its
//! served chat endpoint runs. A LoRA adapter is folded into the base at load
//! by brain, the fold its held-out scorer uses, so a served adapter is
//! numerically the model it was trained to be. This module maps sven's
//! request onto a [`brain::ChatRequest`] and the reply back onto sven's
//! event stream.

mod events;
mod request;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use brain::{CancelToken, ChatPipeline, TextGenerationPipeline};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent};

use crate::error::PolicyError;
use events::events_from;
use request::{chat_request, Sampling};

/// Sampling defaults for agent work, applied per request. Low temperature:
/// an agent is executing a procedure, not writing prose; the small models
/// this provider serves drift into repetition well before they drift into
/// creativity at higher temperatures. The generation cap is bounded to what
/// one agentic step needs - a completion that has not concluded within a few
/// hundred tokens is looping, and a larger cap would let one never-ending
/// generation spend an entire attempt budget.
const AGENT_SAMPLING: Sampling = Sampling {
    max_new_tokens: 512,
    temperature: 0.2,
    top_k: 20,
};

/// A loaded chat model (optionally with a folded LoRA adapter), ready to
/// complete. One sequence decodes at a time - the model carries its KV
/// cache across a generation - so requests serialize behind a lock; that is
/// the shape of a single-user local agent, not a serving fleet.
pub struct LocalQwen {
    pipeline: Arc<Mutex<ChatPipeline>>,
    model_name: String,
    /// The in-flight generation's cancel token, if one is running.
    /// [`Self::stop_generation`] arms it when the work using the model ends
    /// before the turn does, so a decode stops within one prefill chunk or
    /// one token instead of running to its cap.
    in_flight: Arc<Mutex<Option<CancelToken>>>,
    /// Generations currently running. [`Self::stop_generation`] waits on
    /// this reaching zero so the process never exits under a live device
    /// call - exit racing a Vulkan submit is a segfault, not a controlled
    /// outcome.
    live: Arc<AtomicUsize>,
}

/// Where the weights come from and which adapter rides on top.
#[derive(Clone, Debug)]
pub struct LocalWeights {
    /// Checkpoint directory (or file) - the same layout brain's model store
    /// uses. A directory resolves to the checkpoint inside it.
    pub base: PathBuf,
    /// Optional LoRA adapter file, folded in at load.
    pub adapter: Option<PathBuf>,
    /// Inline context budget (tokens): the KV cache is built for exactly
    /// this many, so a prompt plus its generation must fit inside it.
    pub context_tokens: u32,
}

impl LocalQwen {
    /// Loads weights, tokenizer and (optionally) an adapter, and builds the
    /// decode engine. This is the expensive step - do it once, at startup,
    /// not per request.
    pub fn load(weights: &LocalWeights, model_name: &str) -> Result<Self, PolicyError> {
        let base = resolve_base(&weights.base)?;
        let mut builder =
            TextGenerationPipeline::builder(utf8(&base)?).capacity(weights.context_tokens.max(1));
        // A brain-format checkpoint carries no tokenizer; the one beside it
        // is the one it was trained with. A GGUF embeds its own.
        if let Some(tokenizer) = base
            .parent()
            .map(|dir| dir.join("tokenizer.json"))
            .filter(|path| path.is_file())
        {
            builder = builder.tokenizer(utf8(&tokenizer)?);
        }
        if let Some(adapter) = &weights.adapter {
            builder = builder.adapter(utf8(adapter)?);
        }
        let pipeline = builder.load().map_err(|e| PolicyError::Load {
            path: base.clone(),
            reason: e.to_string(),
        })?;
        Ok(Self {
            pipeline: Arc::new(Mutex::new(ChatPipeline::from(pipeline))),
            model_name: model_name.to_string(),
            in_flight: Arc::new(Mutex::new(None)),
            live: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Cancels the in-flight generation, if any, and waits up to `grace` for
    /// it to actually stop, so the process exits AFTER the device is quiet:
    /// exiting under a live prefill or decode crashes the process instead of
    /// ending it. [`crate::LoadedModel`] calls it when it is dropped.
    /// Returns whether every generation has stopped; on `false` the exit is
    /// best-effort (the generation is cancelling and will stop, just not
    /// inside the grace window).
    #[must_use]
    pub fn stop_generation(&self, grace: std::time::Duration) -> bool {
        if let Some(cancel) = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            cancel.cancel();
        }
        let deadline = std::time::Instant::now() + grace;
        while self.live.load(Ordering::SeqCst) > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        self.live.load(Ordering::SeqCst) == 0
    }
}

/// `path` as the UTF-8 string brain's loaders take, or an error naming it.
fn utf8(path: &Path) -> Result<&str, PolicyError> {
    path.to_str().ok_or_else(|| PolicyError::NotUtf8 {
        path: path.to_path_buf(),
    })
}

/// A directory pointing at a checkpoint resolves to the checkpoint inside
/// it; a file passes through. Mirrors brain's own `resolve_base`.
pub fn resolve_base(specified: &Path) -> Result<PathBuf, PolicyError> {
    if specified.is_file() {
        return Ok(specified.to_path_buf());
    }
    for name in ["model.safetensors", "model.brain.safetensors"] {
        let candidate = specified.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    // Anything else: refuse here, where the caller can name the directory,
    // rather than inside the checkpoint open.
    Err(PolicyError::NoCheckpoint {
        path: specified.to_path_buf(),
    })
}

#[async_trait::async_trait]
impl ModelProvider for LocalQwen {
    fn name(&self) -> &str {
        "brain"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn complete(
        &self,
        req: CompletionRequest,
    ) -> anyhow::Result<sven_sdk::model::ResponseStream> {
        let request = chat_request(&req, &AGENT_SAMPLING);
        let pipeline = Arc::clone(&self.pipeline);
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let cancel = CancelToken::armed();
        // Register this generation so a raced attempt end can stop it
        // through `stop_generation`, and account it in `live` for that
        // method's bounded wait. A generation still registered is
        // superseded: this request would queue behind it on the lock.
        {
            let mut slot = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(previous) = slot.as_ref() {
                previous.cancel();
            }
            *slot = Some(cancel.clone());
        }
        self.live.fetch_add(1, Ordering::SeqCst);
        let live = Arc::clone(&self.live);
        // Generation runs on an OS thread the async runtime does not own and
        // never joins. A `spawn_blocking` task would make the runtime's
        // shutdown wait for a decode to run to its cap - an abandoned turn
        // (timeout, interrupt) would hang the process inside
        // `Runtime::drop`. Here an abandoned stream is what arms the cancel:
        // its receiver is gone, the next send fails, and the decode stops at
        // the next token. `LoadedModel` additionally calls `stop_generation`
        // when it is dropped, so the device is quiet - not merely abandoned -
        // when the process ends.
        let spawned = std::thread::Builder::new()
            .name("loop-generate".to_string())
            .spawn(move || {
                // A device error surfaces as a panic (brain's backend reports
                // wgpu errors that way); catch it here so it reaches the
                // consumer as a failed stream instead of a dead thread.
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let pipeline = pipeline
                        .lock()
                        .map_err(|e| anyhow::anyhow!("chat pipeline lock poisoned: {e}"))?;
                    pipeline
                        .generate_stream(&request, &cancel, |delta| {
                            let Some(event) = events::delta_event(delta) else {
                                return;
                            };
                            // A failed send means the consumer is gone - the
                            // turn was abandoned above this stream - so stop.
                            if tx.blocking_send(Ok(event)).is_err() {
                                cancel.cancel();
                            }
                        })
                        .map_err(|e| anyhow::anyhow!("{e}"))
                }));
                let tail = match run {
                    Ok(Ok(response)) => events_from(response),
                    Ok(Err(e)) => vec![Err(anyhow::anyhow!("generation failed: {e:#}"))],
                    Err(panic) => vec![Err(anyhow::anyhow!(
                        "device panic during generation: {}",
                        panic_message(&panic)
                    ))],
                };
                for event in tail {
                    if tx.blocking_send(event).is_err() {
                        break;
                    }
                }
                // Drop the live count - `stop_generation`'s wait condition.
                // The slot itself keeps the (now-finished) token: cancelling
                // a finished generation is a no-op, and the next request
                // replaces the slot wholesale.
                live.fetch_sub(1, Ordering::SeqCst);
            });
        if let Err(e) = spawned {
            self.live.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!("spawning the generation thread: {e}");
        }
        Ok(Box::pin(Events(rx)))
    }
}

/// The generation thread's events as sven's `ResponseStream`: the receiving
/// half of the channel the decode streams through. Dropping it is the
/// abandon signal - the sender sees a failed send and fires the cancel token.
struct Events(tokio::sync::mpsc::Receiver<anyhow::Result<ResponseEvent>>);

impl futures::Stream for Events {
    type Item = anyhow::Result<ResponseEvent>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.0.poll_recv(cx)
    }
}

/// A panic payload as text, however it was constructed.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "no message".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_of_the_standard_layout_resolves_to_its_checkpoint() {
        let dir = std::env::temp_dir().join(format!("loop-provider-base-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(resolve_base(&dir).is_err(), "nothing inside, no resolution");
        std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
        assert_eq!(resolve_base(&dir).unwrap(), dir.join("model.safetensors"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
