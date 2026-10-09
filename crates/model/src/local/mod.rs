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
pub use reasoning::{closed_think_pairs, opens_think_block, CLOSED_THINK, REASONING_REPLY_TOKENS};
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
    // Agent work wants the answer, not a reasoning preamble it cannot use as
    // tool input; a run that wants the reasoning asks for it.
    thinking: false,
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
    /// this many, so a prompt plus its generation must fit inside it. `None`
    /// is the largest context the checkpoint supports
    /// ([`maximum_context_tokens`]); a limit is how a model is made to fit
    /// a card.
    pub context_tokens: Option<u32>,
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
        let context_tokens = match weights.context_tokens {
            Some(tokens) => tokens,
            None => maximum_context_tokens(&base)?,
        };
        let resident = residency.acquire(&base, context_tokens, weights.adapter.as_deref())?;
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

    /// Another handle on this model, sampling as it does: the base, the
    /// adapter and the in-flight generation are shared.
    #[must_use]
    pub fn shared(&self) -> Self {
        self.resampled(self.sampling)
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
/// it; a file passes through. A sharded Hugging Face checkpoint has no single
/// file to resolve to, so its directory is the checkpoint. Mirrors brain's
/// own `resolve_base`.
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
    if is_sharded_hugging_face(specified) {
        return Ok(specified.to_path_buf());
    }
    // Anything else: refuse here, where the caller can name the directory,
    // rather than inside the checkpoint open.
    Err(PolicyError::NoCheckpoint {
        path: specified.to_path_buf(),
    })
}

/// Whether `dir` holds a Hugging Face checkpoint split across numbered
/// `model-*.safetensors` shards (or named by a shard index): its `config.json`
/// declares the architecture, so the directory is what brain opens.
fn is_sharded_hugging_face(dir: &Path) -> bool {
    if !dir.join("config.json").is_file() {
        return false;
    }
    if dir.join("model.safetensors.index.json").is_file() {
        return true;
    }
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
    })
}

/// The largest context `checkpoint` supports: the `max_position_embeddings`
/// of the `config.json` in its directory, or beside its file. A checkpoint with no such config (brain's
/// own format, a GGUF) states no maximum, so its context has to be named.
pub fn maximum_context_tokens(checkpoint: &Path) -> Result<u32, PolicyError> {
    let unknown = |reason: String| PolicyError::Load {
        path: checkpoint.to_path_buf(),
        reason: format!("{reason}; name a context with local:<checkpoint>@<tokens>"),
    };
    let source = load_source(checkpoint);
    let directory = if source.is_dir() {
        source.as_path()
    } else {
        source.parent().unwrap_or(Path::new("."))
    };
    let config = directory.join("config.json");
    let text = std::fs::read_to_string(&config).map_err(|e| {
        unknown(format!(
            "no context maximum: cannot read {}: {e}",
            config.display()
        ))
    })?;
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|c| c.get("max_position_embeddings")?.as_u64())
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)
        .ok_or_else(|| {
            unknown(format!(
                "{} has no max_position_embeddings",
                config.display()
            ))
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

/// The most tokens a reply may run when the request names no cap: the
/// sampling's, raised to `floor` when the model may reason, because a
/// reasoning model spends its first tokens inside the block it opened. A model
/// asked not to reason answers at once and needs no more than any other.
fn reply_cap(sampling: &Sampling, floor: u32) -> u32 {
    if sampling.thinking {
        sampling.max_new_tokens.max(floor)
    } else {
        sampling.max_new_tokens
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
            max_new_tokens: reply_cap(&self.sampling, self.reply_floor),
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
pub(crate) fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
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
    fn a_sharded_hugging_face_directory_resolves_to_itself() {
        let root = std::env::temp_dir().join(format!("policy-sharded-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sharded = root.join("sharded");
        let bare = root.join("bare");
        std::fs::create_dir_all(&sharded).unwrap();
        std::fs::create_dir_all(&bare).unwrap();
        // The 7B distill's layout: config, an index and numbered shards, no
        // single model.safetensors.
        std::fs::write(sharded.join("config.json"), b"{}").unwrap();
        std::fs::write(sharded.join("model.safetensors.index.json"), b"{}").unwrap();
        std::fs::write(sharded.join("model-00001-of-000002.safetensors"), b"x").unwrap();
        std::fs::write(sharded.join("model-00002-of-000002.safetensors"), b"x").unwrap();
        // Shards with no config name no architecture, so nothing to open.
        std::fs::write(bare.join("model-00001-of-000002.safetensors"), b"x").unwrap();

        assert_eq!(resolve_base(&sharded).unwrap(), sharded);
        assert_eq!(load_source(&resolve_base(&sharded).unwrap()), sharded);
        assert!(resolve_base(&bare).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_checkpoints_largest_context_is_read_from_its_config() {
        let root = std::env::temp_dir().join(format!("splinter-max-ctx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (hf, bare) = (root.join("hf"), root.join("bare"));
        std::fs::create_dir_all(&hf).unwrap();
        std::fs::create_dir_all(&bare).unwrap();
        std::fs::write(hf.join("model.safetensors"), b"x").unwrap();
        std::fs::write(
            hf.join("config.json"),
            br#"{"max_position_embeddings": 40960}"#,
        )
        .unwrap();
        std::fs::write(bare.join("model.safetensors"), b"x").unwrap();
        assert_eq!(
            maximum_context_tokens(&hf.join("model.safetensors")).unwrap(),
            40960
        );
        // A brain-format file in a store directory, its config beside it.
        std::fs::write(hf.join("model.brain.safetensors"), b"x").unwrap();
        assert_eq!(
            maximum_context_tokens(&hf.join("model.brain.safetensors")).unwrap(),
            40960
        );
        // No stated maximum: refused with the way to name one, never guessed.
        let refused = maximum_context_tokens(&bare.join("model.safetensors")).unwrap_err();
        assert!(refused.to_string().contains("@<tokens>"), "{refused}");
        std::fs::remove_dir_all(&root).unwrap();
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

    /// A model that may reason gets room to; one asked not to is held to the
    /// ordinary cap, and thinking is off unless a run asks for it.
    #[test]
    fn only_a_model_that_may_think_gets_the_reasoning_floor() {
        let on = Sampling {
            thinking: true,
            ..AGENT_SAMPLING
        };
        assert_eq!(reply_cap(&on, 4096), 4096);
        assert_eq!(
            reply_cap(&AGENT_SAMPLING, 4096),
            AGENT_SAMPLING.max_new_tokens
        );
        assert_eq!(reply_cap(&on, 0), AGENT_SAMPLING.max_new_tokens);
    }
}
