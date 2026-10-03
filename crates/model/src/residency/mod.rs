// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements device-memory ownership for local model
// serving and training, for its clients. If your team needs expertise in
// running several model variants on one accelerator, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The one owner of resident bases: at most one loaded copy of each base
//! checkpoint, shared by every model on it, each model's LoRA adapter
//! switched in before its generation.
//!
//! A model is a base, an optional adapter and a sampling. The policy, the
//! release gate's candidate arm and its champion arm differ only by
//! adapter, so they share one [`Resident`]: before each generation, under
//! the resident's lock, the model makes sure its adapter is the attached
//! one - brain's `attach_adapter` for another adapter, `detach_adapter` for
//! none, nothing when it already matches - and the resident records which
//! is attached, so the answer is the one its model's identity names.
//!
//! A base is loaded plain and its adapters are always attached, never
//! folded in at load: brain folds an adapter given at load into an fp32
//! base, and every switch away from a folded adapter re-reads the folded
//! linears from the checkpoint, while an attached adapter runs beside the
//! resident base and switching costs reading the adapter alone.
//!
//! The residency holds its bases weakly: a base is freed when the last
//! model using it drops. [`Residency::release_all`] frees every resident
//! base at once, for work that needs the device for itself (a fine-tune
//! loading its own copy, a separate serving process); a model whose base
//! was released loads it again on its next generation.
//!
//! Which bases load, and how, is a [`BaseLoader`]: brain's for real
//! weights ([`BrainLoader`]), a scripted one in specs.

#[cfg(any(test, feature = "scripted"))]
pub mod scripted;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use brain::chat::ChatDelta;
use brain::{CancelToken, ChatPipeline, ChatRequest, ChatResponse, TextGenerationPipeline};

use crate::error::PolicyError;

/// One loaded base, as the residency drives it: brain's chat pipeline, or
/// a spec's scripted stand-in. Errors are the engine's own, as text.
pub trait Engine: Send {
    /// Runs the adapter at `adapter` beside the base from the next
    /// generation on, replacing any attached one. A refused adapter leaves
    /// the engine serving what it served.
    fn attach_adapter(&mut self, adapter: &Path) -> Result<(), String>;

    /// Serves exactly the base again.
    fn detach_adapter(&mut self) -> Result<(), String>;

    /// Generates the next assistant turn of `request`, handing each piece
    /// to `on_delta` and stopping when `cancel` fires.
    fn generate_stream(
        &self,
        request: &ChatRequest,
        cancel: &CancelToken,
        on_delta: &mut dyn FnMut(ChatDelta),
    ) -> Result<ChatResponse, String>;
}

impl Engine for ChatPipeline {
    fn attach_adapter(&mut self, adapter: &Path) -> Result<(), String> {
        let path = adapter
            .to_str()
            .ok_or_else(|| format!("{} is not valid UTF-8", adapter.display()))?;
        ChatPipeline::attach_adapter(self, path).map_err(|e| e.to_string())
    }

    fn detach_adapter(&mut self) -> Result<(), String> {
        ChatPipeline::detach_adapter(self)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn generate_stream(
        &self,
        request: &ChatRequest,
        cancel: &CancelToken,
        on_delta: &mut dyn FnMut(ChatDelta),
    ) -> Result<ChatResponse, String> {
        ChatPipeline::generate_stream(self, request, cancel, on_delta).map_err(|e| e.to_string())
    }
}

/// Loads a base checkpoint, with no adapter, for a KV cache of
/// `context_tokens`.
pub trait BaseLoader: Send + Sync {
    /// Loads the checkpoint file `checkpoint` (already resolved, see
    /// [`crate::local::resolve_base`]).
    fn load(&self, checkpoint: &Path, context_tokens: u32) -> Result<Box<dyn Engine>, PolicyError>;
}

/// brain's loader: the checkpoint, and the tokenizer beside it when it
/// carries none of its own.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainLoader;

impl BaseLoader for BrainLoader {
    fn load(&self, checkpoint: &Path, context_tokens: u32) -> Result<Box<dyn Engine>, PolicyError> {
        let source = crate::local::load_source(checkpoint);
        let mut builder =
            TextGenerationPipeline::builder(utf8(&source)?).capacity(context_tokens.max(1));
        // A brain-format checkpoint carries no tokenizer; the one beside it
        // is the one it was trained with. A GGUF embeds its own.
        if let Some(tokenizer) = checkpoint
            .parent()
            .map(|dir| dir.join("tokenizer.json"))
            .filter(|path| path.is_file())
        {
            builder = builder.tokenizer(utf8(&tokenizer)?);
        }
        let pipeline = builder.load().map_err(|e| PolicyError::Load {
            path: checkpoint.to_path_buf(),
            reason: e.to_string(),
        })?;
        Ok(Box::new(ChatPipeline::from(pipeline)))
    }
}

/// `path` as the UTF-8 string brain's loaders take, or an error naming it.
fn utf8(path: &Path) -> Result<&str, PolicyError> {
    path.to_str().ok_or_else(|| PolicyError::NotUtf8 {
        path: path.to_path_buf(),
    })
}

/// What tells two residents apart: the checkpoint file, and the context
/// budget its KV cache is built for (a cache is sized at load, so two
/// budgets are two engines).
type BaseKey = (PathBuf, u32);

/// See the module documentation.
pub struct Residency {
    loader: Arc<dyn BaseLoader>,
    bases: Mutex<HashMap<BaseKey, Weak<Resident>>>,
}

impl Default for Residency {
    fn default() -> Self {
        Self::new(Arc::new(BrainLoader))
    }
}

impl std::fmt::Debug for Residency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Residency")
            .field("resident", &self.resident())
            .finish_non_exhaustive()
    }
}

impl Residency {
    /// A residency loading its bases with `loader`.
    #[must_use]
    pub fn new(loader: Arc<dyn BaseLoader>) -> Self {
        Self {
            loader,
            bases: Mutex::new(HashMap::new()),
        }
    }

    /// The resident base at `checkpoint` for a KV cache of
    /// `context_tokens`, loaded if it is not resident, with `adapter`
    /// attached: a refused adapter is refused here, where the model naming
    /// it is loaded, rather than on its first generation.
    pub fn acquire(
        &self,
        checkpoint: &Path,
        context_tokens: u32,
        adapter: Option<&Path>,
    ) -> Result<Arc<Resident>, PolicyError> {
        // One file named two ways (relative and absolute, through a link)
        // is one base.
        let checkpoint = std::fs::canonicalize(checkpoint).map_err(|source| PolicyError::Io {
            path: checkpoint.to_path_buf(),
            source,
        })?;
        let resident = {
            let mut bases = lock(&self.bases);
            bases.retain(|_, weak| weak.strong_count() > 0);
            let key = (checkpoint.clone(), context_tokens);
            match bases.get(&key).and_then(Weak::upgrade) {
                Some(resident) => resident,
                None => {
                    let resident = Arc::new(Resident {
                        checkpoint,
                        context_tokens,
                        loader: Arc::clone(&self.loader),
                        slot: Mutex::new(Slot::default()),
                    });
                    bases.insert(key, Arc::downgrade(&resident));
                    resident
                }
            }
        };
        // Loaded outside the map's lock: a load is slow, and only this
        // base's users need to wait for it.
        resident.prepare(&mut resident.lock(), adapter)?;
        Ok(resident)
    }

    /// Frees every resident base, waiting for a generation running on one
    /// to end first. The models on them live on; the next generation of
    /// each loads its base again.
    pub fn release_all(&self) {
        let residents: Vec<Arc<Resident>> = lock(&self.bases)
            .values()
            .filter_map(Weak::upgrade)
            .collect();
        for resident in residents {
            resident.release();
        }
    }

    /// How many bases are loaded right now.
    #[must_use]
    pub fn resident(&self) -> usize {
        lock(&self.bases)
            .values()
            .filter_map(Weak::upgrade)
            .filter(|resident| resident.lock().engine.is_some())
            .count()
    }
}

/// One base checkpoint, loaded while a model uses it and it has not been
/// released. Generations on it serialize behind its lock: the engine
/// carries its KV cache and its attached adapter across a generation.
pub struct Resident {
    checkpoint: PathBuf,
    context_tokens: u32,
    loader: Arc<dyn BaseLoader>,
    slot: Mutex<Slot>,
}

/// A resident's engine and the adapter attached to it.
#[derive(Default)]
struct Slot {
    /// `None` before the first load and after a release.
    engine: Option<Box<dyn Engine>>,
    /// The adapter attached to `engine`; `None` for the plain base.
    attached: Option<PathBuf>,
    /// An adapter switch started and neither finished nor was refused: it
    /// panicked (brain reports a device error that way), so what the
    /// engine serves is unknown and it is loaded afresh.
    switching: bool,
}

impl Resident {
    /// Runs `generate` on the engine with `adapter` attached, loading the
    /// base first if it was released. The lock is held throughout, so no
    /// other model switches the adapter under the generation.
    pub fn generate<T>(
        &self,
        adapter: Option<&Path>,
        generate: impl FnOnce(&dyn Engine) -> Result<T, String>,
    ) -> Result<T, PolicyError> {
        let mut slot = self.lock();
        let engine = self.prepare(&mut slot, adapter)?;
        generate(engine).map_err(|reason| PolicyError::Generate {
            path: self.checkpoint.clone(),
            adapter: adapter.map(Path::to_path_buf),
            reason,
        })
    }

    /// The adapter attached right now; `None` for the plain base.
    #[must_use]
    pub fn attached(&self) -> Option<PathBuf> {
        self.lock().attached.clone()
    }

    /// `slot` loaded, with `adapter` attached.
    fn prepare<'a>(
        &self,
        slot: &'a mut Slot,
        adapter: Option<&Path>,
    ) -> Result<&'a dyn Engine, PolicyError> {
        if slot.switching {
            slot.engine = None;
            slot.switching = false;
        }
        let engine = match &mut slot.engine {
            Some(engine) => engine,
            empty @ None => {
                slot.attached = None;
                empty.insert(self.loader.load(&self.checkpoint, self.context_tokens)?)
            }
        };
        if slot.attached.as_deref() != adapter {
            slot.switching = true;
            let switched = match adapter {
                Some(path) => engine.attach_adapter(path),
                None => engine.detach_adapter(),
            };
            // Finished or refused, the engine serves a known adapter: a
            // refused switch leaves it serving what it served.
            slot.switching = false;
            switched.map_err(|reason| PolicyError::Adapter {
                path: self.checkpoint.clone(),
                adapter: adapter.map(Path::to_path_buf),
                reason,
            })?;
            slot.attached = adapter.map(Path::to_path_buf);
        }
        Ok(&**engine)
    }

    /// Frees the engine, after any generation holding it has ended.
    fn release(&self) {
        let mut slot = self.lock();
        slot.engine = None;
        slot.attached = None;
    }

    fn lock(&self) -> MutexGuard<'_, Slot> {
        // A poisoned lock means a load, a switch or a generation panicked;
        // the slot still says what it knows (`switching` covers a switch
        // that did not finish).
        lock(&self.slot)
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
