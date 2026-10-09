// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements voice agents that act through tools and
// answer aloud, for its clients. If your team needs expertise in agent
// runtimes behind a spoken interface, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Talk to a sven agent that answers as the persona.
//!
//! Each recording is recognised and sent to the agent as the user's turn; the
//! words the model writes are spoken sentence by sentence as they arrive. The
//! agent is sven's: its session, and its tools when a workspace is given.
//! Tool calls and their output are not spoken. The session opens with the
//! spoken disclosure that the voice is a portrayal.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use splinter_sdk::model::speech::{
    BrainRecognizer, BrainSynthesizer, CascadeListener, Synthesizer, DEFAULT_RECOGNIZER,
    DEFAULT_SYNTHESIZER,
};
use splinter_sdk::model::PolicyError;
use splinter_sdk::vocabulary::speech::{spoken_disclosure, Portrayal, SpeakerProfile};

use crate::batch::{spoken_turns, Batch, Names, Persona};

#[derive(Args)]
pub struct AgentArgs {
    /// The recognition model, in brain's model store.
    #[arg(long, default_value = DEFAULT_RECOGNIZER)]
    asr: String,
    /// The synthesis model, in brain's model store.
    #[arg(long, default_value = DEFAULT_SYNTHESIZER)]
    tts: String,
    /// The seed the persona's synthetic voice is rendered from.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// A directory of WAV recordings of the user, taken in file-name order.
    #[arg(long)]
    recordings: PathBuf,
    /// Where to write each spoken answer, and the opening disclosure.
    #[arg(long)]
    out_dir: PathBuf,
    /// The person the persona answers as.
    #[arg(long, default_value = "Samuel Adams")]
    persona: String,
    /// The text model's checkpoint directory.
    #[arg(long)]
    base: PathBuf,
    /// A persona adapter to attach to it.
    #[arg(long)]
    adapter: Option<PathBuf>,
    /// A directory the agent may read and change with sven's coding tools;
    /// without one it has no tools.
    #[arg(long)]
    workspace: Option<PathBuf>,
    /// Also keep the report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

/// Take every recording as a turn of one conversation with the agent.
pub fn run(args: &AgentArgs) -> Result<()> {
    let speaker = SpeakerProfile::new(args.seed, Portrayal::synthetic_theatrical());
    std::fs::create_dir_all(&args.out_dir)?;
    BrainSynthesizer::load(&args.tts)?
        .speak(&spoken_disclosure(&args.persona, &speaker), &speaker)?
        .save(args.out_dir.join("disclosure.wav"))
        .context("writing the spoken disclosure")?;

    let persona = Persona::load(&args.persona, &args.base, args.adapter.as_deref())?;
    let voice = persona.voice_agent(args.workspace.as_deref())?;
    let listener = CascadeListener::new(
        BrainRecognizer::load(&args.asr)?,
        |question: &str, on_text: &mut dyn FnMut(&str)| {
            voice
                .ask(question, on_text)
                .map_err(|e| PolicyError::Generate {
                    path: args.base.clone(),
                    adapter: args.adapter.clone(),
                    reason: e.to_string(),
                })
        },
    );
    spoken_turns(
        &Names {
            asr: &args.asr,
            tts: &args.tts,
            speaker,
        },
        &Batch {
            recordings: &args.recordings,
            questions: None,
            out_dir: &args.out_dir,
        },
        &args.persona,
        &listener,
        args.report.as_deref(),
    )
}
