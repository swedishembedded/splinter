// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that run their model
// in-process - no separately versioned serving process to drift from the
// build. If your team needs expertise in local inference integration or
// provider abstractions, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The local model provider: brain's Qwen3 stack, linked in-process.
//!
//! sven's [`ModelProvider`] seam is what remote providers (OpenAI,
//! Anthropic, OpenRouter) hang off; this module hangs a LOCAL model off the
//! same seam, so the loop agent runs the same engine the wire providers do
//! (same tool loop, same event stream, same usage accounting) with no HTTP
//! hop and no dependency on a separately running, separately versioned
//! `brain serve`: Splinter is built against the brain crates it links,
//! and loads weights and adapters directly from disk at startup.
//!
//! The generation path is the one brain's own serving uses for a single
//! sequence: chat-template render (`qwen3::chat`), KV-cached decode
//! (`qwen3::sample`), tool-call scanning (`ChatScanner` via `SeqState`).
//! A trained LoRA adapter is folded into the base tensors before the model
//! is built (`qwen3::lora::fold_adapter_into`), the same fold
//! `qwen3::eval::score_chat` uses, so a served adapter is numerically the
//! model it was trained to be.

mod generate;
mod request;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use capability::CancelToken;
use checkpoint::weightio::WeightReader;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::lora;
use qwen3::model::Qwen;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent};

use generate::{events_from, generate_once, Generation};
use request::invocation_from;

/// Sampling defaults for agent work, applied per request. Low temperature:
/// an agent is executing a procedure, not writing prose; the small models
/// this provider serves drift into repetition well before they drift into
/// creativity at higher temperatures. The generation cap is bounded to what
/// one agentic step needs - a completion that has not concluded within a few
/// hundred tokens is looping, and at this hardware's measured decode rate a
/// larger cap would spend an entire attempt budget on one never-ending
/// generation.
const DEFAULT_MAX_NEW_TOKENS: usize = 512;
const DEFAULT_TEMPERATURE: f64 = 0.2;

/// Prefill chunk size, in prompt tokens. Prefill runs in chunks of this many
/// tokens so a cancellation lands within one chunk instead of after the
/// whole prompt: at this device's measured prefill rate a 5000-token agent
/// prompt is minutes of one uninterruptible device wait when prefilled in a
/// single call, which is what hung an abandoned timeout run until it
/// finished - and made the process exit under it crash. 512 chunks keep the
/// extra readbacks noise against the per-chunk compute while bounding the
/// cancellation latency to well under half a minute.
const PREFILL_CHUNK_TOKENS: usize = 512;

/// A loaded Qwen3 model (optionally with a folded LoRA adapter) plus its
/// tokenizer, ready to complete. One sequence decodes at a time - the model
/// itself carries the KV cache across a generation - so requests serialize
/// behind a lock; that is the shape of a single-user local agent, not a
/// serving fleet.
pub struct LocalQwen {
    model: Arc<Mutex<Qwen>>,
    head: Arc<Vec<f32>>,
    tok: Arc<QwenBpe>,
    eos: Arc<Vec<u32>>,
    model_name: String,
    max_new_tokens: usize,
    temperature: f64,
    /// Inline context budget: the KV cache is sized for exactly this many
    /// tokens, so a prompt plus its generation must fit inside it.
    context_tokens: u32,
    /// The in-flight generation's cancel token, if one is running. The
    /// runner arms it - through [`Self::stop_generation`] - when the attempt
    /// ends before the turn does, so a decode stops within one prefill chunk
    /// or one token instead of running to its cap.
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
    pub base: std::path::PathBuf,
    /// Optional LoRA adapter file, folded in at load.
    pub adapter: Option<std::path::PathBuf>,
    /// Inline context budget (tokens).
    pub context_tokens: u32,
}

impl LocalQwen {
    /// Loads weights, tokenizer and (optionally) an adapter, and builds the
    /// decode engine. This is the expensive step - do it once, at startup,
    /// not per request.
    pub fn load(weights: &LocalWeights, model_name: &str) -> anyhow::Result<Self> {
        let base = resolve_base(&weights.base)
            .with_context(|| format!("resolving weights at {}", weights.base.display()))?;
        let base = base.to_string_lossy().into_owned();
        let reader = WeightReader::open(&base).map_err(|e| anyhow::anyhow!("{base}: {e}"))?;
        // Tokenizer precedence, matching brain's own resident loader: an
        // explicit sibling tokenizer.json wins; a GGUF carries one embedded.
        let tokenizer_path = weights.base.join("tokenizer.json");
        let tok = if tokenizer_path.is_file() {
            let path = tokenizer_path.to_string_lossy().into_owned();
            QwenBpe::from_file(&path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?
        } else if let Some(gt) = reader.tokenizer() {
            QwenBpe::from_gguf(&gt)
                .map_err(|e| anyhow::anyhow!("loading tokenizer from GGUF metadata: {e}"))?
        } else {
            anyhow::bail!(
                "no tokenizer: expected {} beside the checkpoint",
                tokenizer_path.display()
            )
        };
        let eos = tok
            .encode("<|im_end|>")
            .first()
            .copied()
            .map(|t| vec![t])
            .unwrap_or_default();

        let ctx = weights.context_tokens.max(1);
        // Adapter serving is the `from_tensors_decode` path - the fold the
        // qwen3 crate documents for exactly this. Base-only stays on the
        // mmap streaming load, which never materializes the whole model on
        // the host.
        let model = if let Some(adapter) = &weights.adapter {
            let adapter = resolve_adapter_file(adapter)?;
            let adapter = adapter.to_string_lossy().into_owned();
            let mut tensors = checkpoint::load(&base).into_by_role("");
            lora::fold_adapter_into(&mut tensors, &adapter)
                .map_err(|e| anyhow::anyhow!("folding adapter {adapter}: {e}"))?;
            let mut cfg = qwen3::config::QwenConfig::from_json(&reader.config());
            cfg.lora = None;
            Qwen::from_tensors_decode(cfg, &tensors, ctx)
        } else {
            Qwen::from_reader_decode(&reader, ctx)
        };
        // The (tied) LM head is applied host-side by the sampler; read it
        // once here, off the built model, so both load paths - including the
        // adapter fold - go through one head derivation.
        let head = model.read_weight(model.cfg.head_weight());
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            head: Arc::new(head),
            tok: Arc::new(tok),
            eos: Arc::new(eos),
            model_name: model_name.to_string(),
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            context_tokens: ctx,
            in_flight: Arc::new(Mutex::new(None)),
            live: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Cancels the in-flight generation, if any, and waits up to `grace` for
    /// it to actually stop. The runner calls this on a raced attempt end -
    /// timeout, interrupt - so the process exits AFTER the device is quiet:
    /// exiting under a live prefill or decode crashed the process instead of
    /// ending it. Returns whether every generation has stopped; on `false`
    /// the caller should treat the exit as best-effort (the generation is
    /// cancelling and will stop, just not inside the grace window).
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

/// The adapter to fold: `--adapter` names either a LoRA safetensors file
/// directly or the loop's own promotion pointer (`train`'s `adapter.json`,
/// which names the currently promoted adapter). Accepting the pointer is
/// the whole point of writing it: a serving invocation keeps working as
/// adapters are re-trained, without being rewritten per promotion.
fn resolve_adapter_file(specified: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    if specified.extension().and_then(|e| e.to_str()) != Some("json") {
        return Ok(specified.to_path_buf());
    }
    let pointer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(specified).map_err(|e| {
            anyhow::anyhow!("reading adapter pointer {}: {e}", specified.display())
        })?)
        .map_err(|e| anyhow::anyhow!("parsing adapter pointer {}: {e}", specified.display()))?;
    let target = pointer
        .get("adapter")
        .and_then(|a| a.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "adapter pointer {} has no \"adapter\" path",
                specified.display()
            )
        })?;
    let target = std::path::PathBuf::from(target);
    anyhow::ensure!(
        target.is_file(),
        "adapter pointer {} names {} which does not exist",
        specified.display(),
        target.display()
    );
    Ok(target)
}

/// A directory pointing at a checkpoint resolves to the checkpoint inside
/// it; a file passes through. Mirrors brain's own `resolve_base`.
pub fn resolve_base(specified: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
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
    anyhow::bail!("no checkpoint file found under {}", specified.display())
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
        // A caller may ask for more output than the default (explore wants
        // one WHOLE JSON object per section). Give it all the room the KV
        // cache allows: clamped here, where the context size is known, so a
        // big ask degrades to "every token the engine can still hold"
        // instead of failing generate_once's prompt+max_new check.
        let mut bounded = req;
        if let Some(n) = bounded.max_output_tokens_override {
            bounded.max_output_tokens_override = Some(n.min(self.context_tokens));
        }
        let invocation = invocation_from(&bounded, self.max_new_tokens, self.temperature)?;
        // Generation owns the model's KV cache; hold the lock across the
        // whole decode, off the async runtime's threads. The Arcs make the
        // generation closure `'static` without copying the (hundreds-of-MB)
        // head per request.
        let (model, head, tok, eos) = (
            Arc::clone(&self.model),
            Arc::clone(&self.head),
            Arc::clone(&self.tok),
            Arc::clone(&self.eos),
        );
        let ctx = self.context_tokens;
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let cancel = CancelToken::armed();
        // Register this generation so a raced attempt end can stop it
        // through `stop_generation`, and account it in `live` for that
        // method's bounded wait.
        if let Some(slot) = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            slot.cancel();
        }
        *self.in_flight.lock().unwrap_or_else(|e| e.into_inner()) = Some(cancel.clone());
        self.live.fetch_add(1, Ordering::SeqCst);
        let live = Arc::clone(&self.live);
        // Generation runs on an OS thread the async runtime does not own and
        // never joins. A `spawn_blocking` task would make the runtime's
        // shutdown wait for a decode to run to its cap - an abandoned turn
        // (timeout, interrupt) hung the process for minutes inside
        // `Runtime::drop`. Here an abandoned stream is what arms the cancel:
        // its receiver is gone, the next send fails, and the decode stops at
        // the next step boundary. The runner additionally calls
        // `stop_generation` before exiting, so the device is quiet - not
        // merely abandoned - when the process ends.
        std::thread::Builder::new()
            .name("loop-generate".to_string())
            .spawn(move || {
                // A device error surfaces as a panic (brain's backend reports
                // wgpu errors that way); catch it here so it reaches the
                // consumer as a failed stream instead of a dead thread.
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let model = model
                        .lock()
                        .map_err(|e| anyhow::anyhow!("model lock poisoned: {e}"))?;
                    let gen = Generation {
                        head: &head,
                        tok: &tok,
                        eos: &eos,
                        context_tokens: ctx,
                        tx: &tx,
                        cancel: &cancel,
                    };
                    generate_once(&model, &invocation, &gen)
                }));
                let tail = match run {
                    Ok(Ok(outcome)) => events_from(outcome),
                    Ok(Err(e)) => vec![Err(anyhow::anyhow!("generation failed: {e:#}"))],
                    Err(panic) => vec![Err(anyhow::anyhow!(
                        "device panic during generation: {}",
                        panic_message(&panic)
                    ))],
                };
                for event in tail {
                    let _ = tx.blocking_send(event);
                }
                // Drop the live count - `stop_generation`'s wait condition.
                // The slot itself keeps the (now-finished) token: cancelling
                // a finished generation is a no-op, and the next request
                // replaces the slot wholesale.
                live.fetch_sub(1, Ordering::SeqCst);
            })
            .map_err(|e| anyhow::anyhow!("spawning the generation thread: {e}"))?;
        Ok(Box::pin(Events(rx)))
    }
}

/// The generation thread's events as sven's `ResponseStream`: the receiving
/// half of the channel the decode streams through. Dropping it is the
/// abandon signal - the sender sees a failed send and arms the cancel token.
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
#[must_use]
pub fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
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
    fn a_promotion_pointer_resolves_to_the_adapter_it_names() {
        let dir =
            std::env::temp_dir().join(format!("loop-provider-adapter-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let adapter = dir.join("adapter.safetensors");
        std::fs::write(&adapter, b"x").unwrap();

        // A pointer written by `train`'s promotion resolves to the file it
        // names, so a serving invocation can keep passing the stable path.
        let pointer = dir.join("adapter.json");
        std::fs::write(
            &pointer,
            serde_json::json!({"adapter": adapter.display().to_string()}).to_string(),
        )
        .unwrap();
        assert_eq!(resolve_adapter_file(&pointer).unwrap(), adapter);

        // A direct safetensors path passes through untouched.
        assert_eq!(resolve_adapter_file(&adapter).unwrap(), adapter);

        // A broken pointer fails loudly, naming what is missing.
        let dangling = dir.join("dangling.json");
        std::fs::write(
            &dangling,
            serde_json::json!({"adapter": dir.join("gone.safetensors").display().to_string()})
                .to_string(),
        )
        .unwrap();
        let err = resolve_adapter_file(&dangling).unwrap_err().to_string();
        assert!(
            err.contains("does not exist"),
            "refusal must name the missing target: {err}"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
