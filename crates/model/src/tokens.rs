// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines whose data is sized in
// the tokens of the model it trains, for its clients. If your team needs
// expertise in fitting training records to a model's context, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Counting a text's tokens as the policy's tokenizer would: what sizes a
//! training record for the row a fine-tune trains it on.
//!
//! A chat fine-tune's row is sized by its longest record, so a record cut
//! by characters can double the row - and the memory and time of every
//! step - where its text tokenizes badly (tables, numbers, old spellings).
//! The count is taken through the chat template, as the record will be
//! rendered, less the template's own tokens around an empty turn.

use std::path::Path;

use brain::{ChatMessage, ChatRequest, ChatTokenizer};

use crate::PolicyError;

/// The tokens of a text, as one model's tokenizer counts them.
pub struct TokenCounter {
    tokenizer: ChatTokenizer,
    /// The tokens the template puts around an empty user turn.
    baseline: usize,
}

impl TokenCounter {
    /// The counter of the model whose tokenizer and chat template are in
    /// `model_dir`.
    pub fn for_model(model_dir: &Path) -> Result<Self, PolicyError> {
        let tokenizer =
            ChatTokenizer::from_model_dir(model_dir).map_err(|e| PolicyError::Load {
                path: model_dir.to_path_buf(),
                reason: e.to_string(),
            })?;
        let baseline = rendered(&tokenizer, "").ok_or_else(|| PolicyError::Load {
            path: model_dir.to_path_buf(),
            reason: "the chat template cannot render a user turn".into(),
        })?;
        Ok(Self {
            tokenizer,
            baseline,
        })
    }

    /// The tokens `text` takes in a turn, the template's own aside. A text
    /// the template cannot render counts as its characters, the most tokens
    /// it could take.
    #[must_use]
    pub fn count(&self, text: &str) -> usize {
        rendered(&self.tokenizer, text)
            .map_or_else(|| text.chars().count(), |n| n.saturating_sub(self.baseline))
    }
}

/// The tokens of a user turn of `text` rendered through the template;
/// `None` when the template cannot render it.
fn rendered(tokenizer: &ChatTokenizer, text: &str) -> Option<usize> {
    tokenizer
        .prompt_ids(&ChatRequest::new(vec![ChatMessage::user(text)]))
        .ok()
        .map(|ids| ids.len())
}
