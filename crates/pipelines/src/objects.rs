// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `objects propose`: a directory of documents, a model, and a pack of knowledge objects admitted by code.
//!
//! The documents are Markdown files with an optional header (`authority`, `version`, `date`). The model is asked one document at a time, in
//! date order, through a typed call whose postcondition is the admission check, so a refused proposal is corrected within the call's bounds
//! (`splinter_agent::object_proposer`). The result is the pack format of `splinter_knowledge::objects`; nothing is stored in the state root.

use std::path::Path;

use splinter_agent::object_proposer::SvenObjectProposer;
use splinter_core::model_ref::ModelRef;
use splinter_knowledge::objects::propose::{propose_pack, Proposed, Strategy, Vocabulary};
use splinter_knowledge::objects::Document;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What to propose from, and how.
pub struct ObjectsRequest<'a> {
    /// The pack's name.
    pub pack_id: &'a str,
    /// The directory of `*.md` documents.
    pub source: &'a Path,
    /// The model asked.
    pub policy: &'a ModelRef,
    /// The outcomes a rule may conclude; empty leaves them open.
    pub outcomes: Vec<String>,
    /// How many times a reply with refused proposals is sent back, per document.
    pub repairs: u32,
    /// How a document is put to the model.
    pub strategy: Strategy,
}

/// The documents of `directory`, in file name order.
///
/// # Errors
/// The directory or a file cannot be read, or holds no document.
pub fn read_documents(directory: &Path) -> Result<Vec<Document>, OrchestratorError> {
    let io =
        |e: std::io::Error| OrchestratorError::Refused(format!("{}: {e}", directory.display()));
    let mut paths: Vec<_> = std::fs::read_dir(directory)
        .map_err(io)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    paths.sort();
    let documents: Vec<Document> = paths
        .iter()
        .map(|p| {
            let raw = std::fs::read_to_string(p).map_err(io)?;
            Ok(Document::parse(
                p.file_stem().and_then(|s| s.to_str()).unwrap_or("document"),
                &raw,
            ))
        })
        .collect::<Result<_, OrchestratorError>>()?;
    if documents.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "{} holds no .md document",
            directory.display()
        )));
    }
    Ok(documents)
}

/// Proposes and admits the objects of the request's documents.
///
/// # Errors
/// The documents cannot be read, the model cannot be had, or a document gets no usable reply at all.
pub fn propose(ctx: &Context, request: &ObjectsRequest<'_>) -> Result<Proposed, OrchestratorError> {
    let documents = read_documents(request.source)?;
    let model = ctx.model(request.policy)?;
    let identity = model.identity.clone();
    let proposer = SvenObjectProposer::new(model).with_repairs(request.repairs);
    let vocabulary = Vocabulary {
        outcomes: request.outcomes.clone(),
    };
    ctx.block_on(propose_pack(
        request.pack_id,
        &documents,
        &proposer,
        &vocabulary,
        request.strategy,
    ))
    .map_err(|e| OrchestratorError::Model {
        model: identity,
        detail: e.to_string(),
    })
}
