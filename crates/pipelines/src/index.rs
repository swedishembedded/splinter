// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The retrieval index kept in the state: the vector of every passage of a
//! set of sources.
//!
//! Embedding a large body of writing takes far longer than answering from
//! it, so the vectors are made once. They are derived data, never a source
//! of truth: stored as an artifact, found by a pointer whose name is the
//! digest of the embedder's name and every passage's address and text, so
//! other sources, changed text or another embedder is another index, and an
//! index that cannot be read back is rebuilt rather than trusted.

use splinter_core::digest::Digest;
use splinter_knowledge::retrieve::{EmbedError, Embedder, Library, Passage};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::artifacts::ArtifactSpec;

const ROLE: &str = "retrieval-index";
const PRODUCER: &str = "splinter/retrieval";
/// The bytes before the vectors: how many, and how long.
const HEADER_BYTES: usize = 16;

/// The library of `passages`: read from the kept index when there is one for
/// them and `embedder`, else embedded now and kept.
pub fn library(
    ctx: &Context,
    passages: Vec<Passage>,
    embedder: &dyn Embedder,
) -> Result<Library, OrchestratorError> {
    let name = pointer_name(&passages, embedder);
    if let Some(vectors) = kept(ctx, &name) {
        if let Ok(library) = Library::from_vectors(passages.clone(), vectors) {
            return Ok(library);
        }
    }
    let library = Library::new(passages, embedder).map_err(retrieval)?;
    keep(ctx, &name, library.vectors());
    Ok(library)
}

fn retrieval(e: EmbedError) -> OrchestratorError {
    OrchestratorError::Refused(format!("retrieval: {e}"))
}

/// The pointer that names the index of `passages` made by `embedder`.
fn pointer_name(passages: &[Passage], embedder: &dyn Embedder) -> String {
    let mut bytes = Vec::new();
    let mut put = |text: &str| {
        bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
    };
    put(&embedder.name());
    for passage in passages {
        put(&passage.part);
        put(&passage.section.to_string());
        put(&passage.text);
    }
    format!("retrieval-index/{}", Digest::of(&bytes).hex())
}

/// The vectors the pointer `name` leads to; `None` where there is no index
/// or it cannot be read.
fn kept(ctx: &Context, name: &str) -> Option<Vec<Vec<f32>>> {
    let (_, artifact) = ctx.workspace().pointer(name).ok()??;
    let path = ctx.artifacts().path(&Digest::parse(&artifact).ok()?).ok()?;
    decode(&std::fs::read(path).ok()?)
}

/// Keeps `vectors` under the pointer `name`. A failure to keep costs only the
/// next question another embedding, so it is not the question's failure.
fn keep(ctx: &Context, name: &str, vectors: &[Vec<f32>]) {
    let stored = ctx
        .artifacts()
        .put_bytes(&encode(vectors), &ArtifactSpec::new(ROLE, PRODUCER));
    let Ok(artifact) = stored else { return };
    let current = ctx.workspace().pointer(name).ok().flatten();
    let _ = ctx.workspace().move_pointer(
        name,
        current.as_ref().map(|(_, value)| value.as_str()),
        artifact.digest.as_str(),
        &ctx.clock().utc_now(),
    );
}

fn encode(vectors: &[Vec<f32>]) -> Vec<u8> {
    let size = vectors.first().map_or(0, Vec::len);
    let mut bytes = Vec::with_capacity(HEADER_BYTES + vectors.len() * size * 4);
    bytes.extend_from_slice(&(vectors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(size as u64).to_le_bytes());
    for vector in vectors {
        for x in vector {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    bytes
}

fn decode(bytes: &[u8]) -> Option<Vec<Vec<f32>>> {
    let count = usize::try_from(u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?)).ok()?;
    let size = usize::try_from(u64::from_le_bytes(bytes.get(8..16)?.try_into().ok()?)).ok()?;
    let body = bytes.get(HEADER_BYTES..)?;
    if body.len() != count.checked_mul(size)?.checked_mul(4)? {
        return None;
    }
    let (words, _) = body.as_chunks::<4>();
    let mut floats = words.iter().map(|word| f32::from_le_bytes(*word));
    Some(
        (0..count)
            .map(|_| floats.by_ref().take(size).collect())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_survive_the_bytes_and_a_damaged_index_is_not_read() {
        let vectors = vec![vec![1.0, -2.5, 0.25], vec![0.0, 3.0, 4.5]];
        assert_eq!(decode(&encode(&vectors)), Some(vectors.clone()));
        let mut cut = encode(&vectors);
        cut.pop();
        assert_eq!(decode(&cut), None);
        assert_eq!(decode(&[1, 2, 3]), None);
        assert_eq!(decode(&encode(&[])), Some(Vec::new()));
    }
}
