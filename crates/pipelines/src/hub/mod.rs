// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distribution of curated training data,
// adapters and evaluation verdicts as immutable, verifiable versions, for its
// clients. If your team needs expertise in sharing learning artifacts between
// machines without losing their provenance, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Bundles of experience, datasets, adapters and verdicts, distributed as
//! immutable versions of a private Hugging Face dataset repository.
//!
//! [`publish`] writes what this state root holds as one tagged commit;
//! [`ingest`] reads a bundle back at a pinned revision into this root through
//! the stores' own APIs. [`bundle`] is the manifest both read and write;
//! [`withhold`] is the typed check of what is never published. The Hub is
//! reached through [`splinter_hub::Hub`], at the endpoint the caller hands
//! in.

pub mod bundle;
pub mod ingest;
pub mod publish;
pub mod withhold;

pub use ingest::{ingest, Ingested};
pub use publish::{publish, PublishRequest, Published};
pub use splinter_hub::{Hub, HubError, DEFAULT_ENDPOINT};

use std::io::Read as _;
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The branch every publish commits to.
pub const BRANCH: &str = "main";

/// The largest bundle manifest read.
const MANIFEST_LIMIT: u64 = 64 << 20;

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> OrchestratorError + '_ {
    move |source| OrchestratorError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// The SHA-256 (hex) and size of the file at `path`.
fn file_sha256(path: &Path) -> Result<(String, u64), OrchestratorError> {
    let mut file = std::fs::File::open(path).map_err(io(path))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut size = 0u64;
    loop {
        let n = file.read(&mut buffer).map_err(io(path))?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        size += n as u64;
    }
    Ok((
        hash.finalize().iter().map(|b| format!("{b:02x}")).collect(),
        size,
    ))
}

/// A working directory under the state root's `work/`, removed when
/// dropped, whatever happened.
struct Scratch(PathBuf);

impl Scratch {
    fn new(ctx: &Context, what: &str) -> Result<Self, OrchestratorError> {
        let path = ctx
            .root()
            .work()
            .join(splinter_store::new_id_with_prefix(what));
        std::fs::create_dir_all(&path).map_err(io(&path))?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
