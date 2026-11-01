// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from agent benchmark runs to
// fine-tuning data that is verified and free of evaluation leakage, for its
// clients. If your team needs expertise in learning from benchmark
// trajectories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What the benchmark-ingestion tests share: a context with a fixed clock, the
//! sft-step request they build with and the default ingestion options.

use splinter_core::annotation::Strength;
use splinter_core::clock::FixedClock;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::IngestOptions;
use splinter_pipelines::datasets::{BuildRequest, ViewName, VoiceBuild};
use splinter_store::experiences::SetId;

use super::{config, Scratch};

pub fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-28T09:00:00.000Z")));
    (scratch, ctx)
}

pub fn sft_request(set: SetId) -> BuildRequest {
    BuildRequest {
        sets: vec![set],
        view: ViewName::SftStep,
        strip: None,
        min_strength: Some(Strength::Executable),
        system_prompt: None,
        export_only: false,
        limit: None,
        voice: VoiceBuild::default(),
        max_block: None,
        pack: false,
        split_over_block: false,
        control: Default::default(),
        serialise_calls: false,
        only_actions: None,
        roles: None,
    }
}

pub const NO_OPTIONS: IngestOptions = IngestOptions { assume_set: None };
