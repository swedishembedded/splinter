// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements voice agents that learn how to speak from
// what their users tell them, for its clients. If your team needs expertise
// in turning spoken feedback into training data, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Teach the persona how to speak, by talking to it.
//!
//! The user says what they want ("Pronounce Jefferson as Jeff-er-son", or
//! "Talk as follows") and, for an imitation, speaks the example in the next
//! turn, the way a conversation goes. The directive is heard, read, and kept
//! as lessons; the example's recording is kept by digest.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Args;
use splinter_sdk::model::speech::{BrainRecognizer, Recognizer, DEFAULT_RECOGNIZER};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::speech::Portrayal;
use splinter_sdk::vocabulary::speech_lesson::{
    lessons_from, parse_directive, Lesson, Lexicon, Objective,
};

use crate::batch::{emit, read_clip};

#[derive(Args)]
pub struct TeachArgs {
    /// The recognition model, in brain's model store.
    #[arg(long, default_value = DEFAULT_RECOGNIZER)]
    asr: String,
    /// A recording of the user giving the directive.
    #[arg(long, conflicts_with = "said")]
    directive: Option<PathBuf>,
    /// The directive as typed, in place of a recording.
    #[arg(long)]
    said: Option<String>,
    /// A recording of the user speaking the example, for an imitation.
    #[arg(long)]
    example: Option<PathBuf>,
    /// Where the lessons are kept, one per line; appended to.
    #[arg(long)]
    lessons: PathBuf,
    /// Also keep the report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

/// Hear the directive, make its lessons and keep them.
pub fn teach(args: &TeachArgs) -> Result<()> {
    let heard = match (&args.directive, &args.said) {
        (Some(wav), _) => {
            BrainRecognizer::load(&args.asr)?
                .transcribe(&read_clip(wav)?)?
                .text
        }
        (None, Some(said)) => said.clone(),
        (None, None) => bail!("give --directive (a recording) or --said (typed)"),
    };
    let Some(directive) = parse_directive(&heard) else {
        bail!("not a directive about how to speak: {heard:?}");
    };
    let recording = args
        .example
        .as_ref()
        .map(|path| {
            std::fs::read(path)
                .map(|bytes| Digest::of(&bytes))
                .with_context(|| format!("reading {}", path.display()))
        })
        .transpose()?;
    let lessons = lessons_from(
        &heard,
        &directive,
        recording.as_ref(),
        &Portrayal::synthetic_theatrical(),
    )
    .map_err(|e| anyhow::anyhow!("{e}: record the example as the next turn and pass --example"))?;

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.lessons)
        .with_context(|| format!("opening {}", args.lessons.display()))?;
    for lesson in &lessons {
        writeln!(file, "{}", serde_json::to_string(lesson)?)?;
    }
    let kept = read_lessons(&args.lessons)?;
    emit(
        &serde_json::json!({
            "heard": heard,
            "made": lessons.iter().map(|l| l.objective).collect::<Vec<Objective>>(),
            "lessons_kept": kept.len(),
            "lexicon_words": Lexicon::from_lessons(&kept).len(),
        }),
        args.report.as_deref(),
    )
}

/// The lessons kept in `path`.
pub fn read_lessons(path: &std::path::Path) -> Result<Vec<Lesson>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("a lesson line is not a lesson"))
        .collect()
}
