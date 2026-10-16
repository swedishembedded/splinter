// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of language models
// for its clients. If your team needs expertise in training adapters on large
// bases with limited GPU memory, you can procure our services by sending an
// email to info@swedishembedded.com.

//! How a frozen base is held in memory while an adapter trains on it.

/// How the frozen base is held while an adapter trains; the adapter and its
/// optimizer stay fp32 whatever the base is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BasePrecision {
    /// The checkpoint's fp32 weights: brain's default.
    #[default]
    Fp32,
    /// bf16, half the bytes of fp32: what lets a 7B base train on one 24 GiB
    /// card.
    Bf16,
    /// int8 with a scale per 32 weights, about a quarter of fp32: room for an
    /// 8B base and its activations on one 24 GiB card.
    Int8,
}

impl BasePrecision {
    /// [`Self::Bf16`] when `on`, else fp32: what a `--bf16-base` flag asks.
    #[must_use]
    pub fn bf16_if(on: bool) -> Self {
        if on {
            Self::Bf16
        } else {
            Self::Fp32
        }
    }
}
