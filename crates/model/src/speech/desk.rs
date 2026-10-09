// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements voice agents that learn how to speak from
// what their users tell them, for its clients. If your team needs expertise
// in turning spoken feedback into training data, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The desk in front of the model: what a user says about how to speak is
//! taken here, and everything else goes on to the model.
//!
//! A directive ("Pronounce X as Y") is not a question for the persona. The desk
//! reads it, offers what it would learn in the way it would say it, and keeps the
//! lesson only if the next turn plainly confirms it. A kept lesson is applied from
//! the next sentence spoken, can be taken back ("forget that"), and survives a
//! restart. Nothing a user says is kept without that confirmation.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use splinter_core::speech::Portrayal;
use splinter_core::speech_lesson::{
    lessons_from, parse_confirmation, parse_directive, parse_retraction, Directive, Lesson, Lexicon,
};

/// A lexicon several voices and the desk share, so a lesson kept on one turn is
/// heard on the next sentence.
pub type SharedLexicon = Arc<RwLock<Lexicon>>;

/// What the desk made of a turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Heard {
    /// Not about how to speak: the model answers it.
    Pass,
    /// About how to speak: say this, and do not ask the model.
    Say(String),
}

#[derive(Default)]
struct State {
    kept: Vec<Lesson>,
    offered: Option<Lesson>,
}

/// The lessons a conversation has taught and the offer awaiting an answer.
pub struct LessonDesk {
    portrayal: Portrayal,
    lexicon: SharedLexicon,
    state: Mutex<State>,
    file: Option<PathBuf>,
}

impl LessonDesk {
    /// A desk with no lessons, for the voice declared by `portrayal`.
    #[must_use]
    pub fn new(portrayal: Portrayal) -> Self {
        Self {
            portrayal,
            lexicon: Arc::default(),
            state: Mutex::default(),
            file: None,
        }
    }

    /// A desk that keeps its lessons in `file` (one JSON object per line),
    /// loading those already there.
    ///
    /// # Errors
    /// The file exists and cannot be read or is not lessons.
    pub fn open(file: &Path, portrayal: Portrayal) -> std::io::Result<Self> {
        let kept: Vec<Lesson> = match std::fs::read_to_string(file) {
            Ok(text) => text
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    serde_json::from_str(line)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                })
                .collect::<Result<_, _>>()?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };
        let desk = Self {
            portrayal,
            lexicon: Arc::new(RwLock::new(Lexicon::from_lessons(&kept))),
            state: Mutex::new(State {
                kept,
                offered: None,
            }),
            file: Some(file.to_path_buf()),
        };
        Ok(desk)
    }

    /// The lexicon the desk keeps, to hand to a voice.
    #[must_use]
    pub fn lexicon(&self) -> SharedLexicon {
        Arc::clone(&self.lexicon)
    }

    /// The lessons kept so far, oldest first.
    #[must_use]
    pub fn lessons(&self) -> Vec<Lesson> {
        self.lock().kept.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Take a turn. An offer awaiting an answer is settled by this turn: a plain
    /// yes keeps it, a plain no drops it, and anything else drops it and is
    /// then treated as a turn in its own right.
    pub fn hear(&self, heard: &str) -> Heard {
        let mut state = self.lock();
        if let Some(offer) = state.offered.take() {
            match parse_confirmation(heard) {
                Some(true) => {
                    state.kept.push(offer);
                    self.refresh(&state);
                    return Heard::Say("Kept.".to_string());
                }
                Some(false) => return Heard::Say("I will leave it as it was.".to_string()),
                None => {}
            }
        }
        if parse_retraction(heard) {
            return Heard::Say(if state.kept.pop().is_some() {
                self.refresh(&state);
                "I have taken that back.".to_string()
            } else {
                "There is nothing to take back.".to_string()
            });
        }
        match parse_directive(heard) {
            Some(Directive::Respell { word, spoken_as }) => {
                let directive = Directive::Respell {
                    word: word.clone(),
                    spoken_as: spoken_as.clone(),
                };
                match lessons_from(heard, &directive, None, &self.portrayal) {
                    Ok(mut lessons) if lessons.len() == 1 => {
                        state.offered = lessons.pop();
                        Heard::Say(format!(
                            "Here it is: {spoken_as}. Shall I keep saying {word} that way?"
                        ))
                    }
                    _ => Heard::Pass,
                }
            }
            Some(Directive::Imitate { .. }) => Heard::Say(
                "I cannot learn from an example yet. I can learn a pronunciation: say pronounce, \
                 the word, as, and how it is said."
                    .to_string(),
            ),
            None => Heard::Pass,
        }
    }

    /// Bring the shared lexicon and the file up to date with the kept lessons.
    /// The lexicon is rebuilt from the lessons, so taking one back restores
    /// exactly what the earlier ones taught.
    fn refresh(&self, state: &State) {
        *self.lexicon.write().unwrap_or_else(PoisonError::into_inner) =
            Lexicon::from_lessons(&state.kept);
        if let Some(file) = &self.file {
            let text: String = state
                .kept
                .iter()
                .filter_map(|lesson| serde_json::to_string(lesson).ok())
                .map(|line| line + "\n")
                .collect();
            // A lesson that cannot be written would be lost on restart; say so.
            let written = file.with_extension("tmp");
            if std::fs::write(&written, text).is_err() || std::fs::rename(&written, file).is_err() {
                warn_unwritten(file);
            }
        }
    }
}

/// The desk has no logger; a failed write is reported on standard error.
fn warn_unwritten(file: &Path) {
    eprintln!(
        "lessons: could not write {}; they will not survive a restart",
        file.display()
    );
}
