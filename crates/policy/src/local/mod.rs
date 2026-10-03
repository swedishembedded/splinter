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
//! served chat endpoint runs. The base is the [`crate::residency`]'s: one
//! resident copy per checkpoint, shared by every model on it, with the
//! model's LoRA adapter attached before each of its generations. This
//! module maps sven's request onto a [`brain::ChatRequest`] and the reply
//! back onto sven's event stream.

mod events;
mod reasoning;
mod request;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use brain::CancelToken;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent};

use crate::error::PolicyError;
use crate::residency::{Residency, Resident};
use events::events_from;
pub use reasoning::{opens_think_block, REASONING_REPLY_TOKENS};
use request::chat_request;
pub use request::Sampling;

/// Sampling defaults for agent work, applied per request. Low temperature:
/// an agent is executing a procedure, not writing prose; the small models
/// this provider serves drift into repetition well before they drift into
/// creativity at higher temperatures. The generation cap is bounded to what
/// one agentic step needs - a completion that has not concluded within a few
/// hundred tokens is looping, and a larger cap would let one never-ending
/// generation spend an entire attempt budget.
pub const AGENT_SAMPLING: Sampling = Sampling {
    max_new_tokens: 512,
    temperature: 0.2,
    top_k: 20,
};

/// Greedy decoding for measurement: agent sampling at temperature zero,
/// which brain decodes as argmax. A probe graded twice - in-process and on
/// a server - must get the same answer from the same weights, not two
/// draws.
pub const GREEDY_SAMPLING: Sampling = Sampling {
    temperature: 0.0,
    ..AGENT_SAMPLING
};

/// A chat model on a resident base, with its optional LoRA adapter,
/// ready to complete. One sequence decodes at a time on a base - it carries
/// its KV cache and attached adapter across a generation - so requests on
/// every model sharing it serialize behind its lock; that is the shape of a
/// single-user local agent, not a serving fleet.
pub struct LocalQwen {
    resident: Arc<Resident>,
    /// The adapter attached before each generation; `None` for the base.
    adapter: Option<PathBuf>,
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
    /// How each reply is sampled.
    sampling: Sampling,
    /// The least a reply may run before its cap, whatever the sampling says:
    /// a reasoning model spends its first tokens inside the think block its
    /// template opened.
    reply_floor: u32,
}

/// Where the weights come from and which adapter rides on top.
#[derive(Clone, Debug)]
pub struct LocalWeights {
    /// Checkpoint directory (or file) - the same layout brain's model store
    /// uses. A directory resolves to the checkpoint inside it.
    pub base: PathBuf,
    /// Optional LoRA adapter file, attached to the base.
    pub adapter: Option<PathBuf>,
    /// Inline context budget (tokens): the KV cache is built for exactly
    /// this many, so a prompt plus its generation must fit inside it.
    pub context_tokens: u32,
}

impl LocalQwen {
    /// The model `weights` names, on `residency`'s copy of its base: loaded
    /// if no other model holds it - the expensive step, done once per
    /// base, not per request - with the adapter attached, so an adapter
    /// that does not fit the base is refused here.
    pub fn load(
        residency: &Residency,
        weights: &LocalWeights,
        model_name: &str,
    ) -> Result<Self, PolicyError> {
        let base = resolve_base(&weights.base)?;
        let resident =
            residency.acquire(&base, weights.context_tokens, weights.adapter.as_deref())?;
        Ok(Self {
            resident,
            adapter: weights.adapter.clone(),
            model_name: model_name.to_string(),
            in_flight: Arc::new(Mutex::new(None)),
            live: Arc::new(AtomicUsize::new(0)),
            sampling: AGENT_SAMPLING,
            reply_floor: reasoning::reply_floor(&base),
        })
    }

    /// The same model sampling as `sampling` says: the base, the adapter
    /// and the in-flight generation are shared, so a request on either
    /// waits for (and supersedes) one running on the other, as two
    /// requests on one model do.
    #[must_use]
    pub fn resampled(&self, sampling: Sampling) -> Self {
        Self {
            resident: Arc::clone(&self.resident),
            adapter: self.adapter.clone(),
            model_name: self.model_name.clone(),
            in_flight: Arc::clone(&self.in_flight),
            live: Arc::clone(&self.live),
            sampling,
            reply_floor: self.reply_floor,
        }
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

/// What brain is asked to open for `checkpoint`: a `model.safetensors`
/// beside a `config.json` is a Hugging Face checkpoint, whose architecture is
/// read from that config, so it opens as its directory; any other file
/// (brain's own format, a GGUF) opens as itself.
pub fn load_source(checkpoint: &Path) -> PathBuf {
    match checkpoint.parent() {
        Some(dir)
            if checkpoint
                .file_name()
                .is_some_and(|n| n == "model.safetensors")
                && dir.join("config.json").is_file() =>
        {
            dir.to_path_buf()
        }
        _ => checkpoint.to_path_buf(),
    }
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
        let sampling = Sampling {
            max_new_tokens: self.sampling.max_new_tokens.max(self.reply_floor),
            ..self.sampling
        };
        let request = chat_request(&req, &sampling);
        let resident = Arc::clone(&self.resident);
        let adapter = self.adapter.clone();
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
                // The resident's lock is held for the whole generation, its
                // adapter switched to this model's first; the thread's own
                // handle keeps the base loaded until the generation ends.
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    resident.generate(adapter.as_deref(), |engine| {
                        engine.generate_stream(&request, &cancel, &mut |delta| {
                            let Some(event) = events::delta_event(delta) else {
                                return;
                            };
                            // A failed send means the consumer is gone - the
                            // turn was abandoned above this stream - so stop.
                            if tx.blocking_send(Ok(event)).is_err() {
                                cancel.cancel();
                            }
                        })
                    })
                }));
                let tail = match run {
                    Ok(Ok(response)) => events_from(response),
                    Ok(Err(e)) => vec![Err(anyhow::anyhow!("generation failed: {e}"))],
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

    #[test]
    fn a_hugging_face_checkpoint_is_opened_as_its_directory() {
        let root = std::env::temp_dir().join(format!("splinter-hf-source-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let hf = root.join("hf");
        let brain_format = root.join("brain");
        std::fs::create_dir_all(&hf).unwrap();
        std::fs::create_dir_all(&brain_format).unwrap();
        // The model's architecture lives in config.json; the bare file does not
        // carry it, so only the directory loads.
        std::fs::write(hf.join("model.safetensors"), b"x").unwrap();
        std::fs::write(hf.join("config.json"), b"{}").unwrap();
        // A brain-format file declares itself under its own name.
        std::fs::write(brain_format.join("model.brain.safetensors"), b"x").unwrap();
        std::fs::write(brain_format.join("config.json"), b"{}").unwrap();

        assert_eq!(load_source(&hf.join("model.safetensors")), hf);
        let brain_file = brain_format.join("model.brain.safetensors");
        assert_eq!(load_source(&brain_file), brain_file);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
