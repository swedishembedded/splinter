// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements question answering over a learned model and
// its sources, for its clients. If your team needs expertise in grounded
// answers from local models, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `ask`: argument handling and output only.

use splinter_sdk::ask::{ask, ask_retrieving, RetrievingQuestion};
use splinter_sdk::rerank::ModelReranker;
use splinter_sdk::retrieval::{candidates_for, Rerank};
use splinter_sdk::{Context, Error};

use crate::cli::{AskArgs, RetrieveArgs};
use crate::output::emit;
use crate::session::Exit;

/// `ask`, closed-book, open-book or over retrieved passages.
pub fn ask_command(ctx: &Context, json: bool, args: &AskArgs) -> Result<Exit, Error> {
    let retrieve = &args.retrieval;
    if retrieve.sources.is_empty() {
        emit(
            json,
            &ask(ctx, &args.question, args.open_book.as_deref(), &args.policy)?,
        );
    } else {
        if args.open_book.is_some() {
            return Err(Error::Refused(
                "--open-book shows one whole source and --retrieve passages of \
                 several: name one of them"
                    .into(),
            ));
        }
        let embedder = ctx.embedder()?;
        let reader = reader_of(ctx, retrieve)?;
        let rerank = reader.as_ref().map(|reader| Rerank {
            reranker: reader,
            candidates: candidates_for(retrieve.passages),
        });
        emit(
            json,
            &ask_retrieving(
                ctx,
                &RetrievingQuestion {
                    question: &args.question,
                    sources: &retrieve.sources,
                    passages: retrieve.passages,
                    rerank,
                    policy: &args.policy,
                },
                &*embedder,
            )?,
        );
    }
    Ok(Exit::Ok)
}

/// The model that reads retrieval's candidates, when one was named.
pub fn reader_of<'a>(
    ctx: &'a Context,
    retrieve: &RetrieveArgs,
) -> Result<Option<ModelReranker<'a>>, Error> {
    retrieve
        .reranker
        .as_ref()
        .map(|reference| ModelReranker::new(ctx, reference))
        .transpose()
}
