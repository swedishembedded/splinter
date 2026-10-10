// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specs of the lesson desk: how a user's spoken directives about how to speak
//! become lessons, and are confirmed, applied and taken back.

use splinter_core::speech::Portrayal;

use super::tests::{speaker, Log, Logging};
use super::*;

fn desk() -> LessonDesk {
    LessonDesk::new(Portrayal::synthetic_theatrical())
}

fn said_by(desk: &LessonDesk, heard: &str) -> String {
    match desk.hear(heard) {
        Heard::Say(reply) => reply,
        Heard::Retry(text) => format!("<retry: {text}>"),
        Heard::Pass => "<passed to the model>".to_string(),
    }
}

#[test]
fn a_confirmed_pronunciation_is_used_from_the_next_sentence_and_can_be_taken_back() {
    let desk = desk();
    let log = Log::default();
    let voice = Lexical::shared(Logging(&log), desk.lexicon());
    let speak = |text: &str| voice.speak(text, &speaker()).unwrap();

    speak("Jefferson wrote it.");
    let offer = said_by(&desk, "Pronounce Jefferson as Jeff-er-son.");
    assert!(
        offer.contains("Jeff-er-son"),
        "the offer lets the user hear the new way: {offer}"
    );
    speak("Jefferson wrote it again.");
    assert_eq!(said_by(&desk, "Yes."), "Kept.");
    assert_eq!(desk.lessons().len(), 1);
    speak("Jefferson wrote it.");
    assert_eq!(
        said_by(&desk, "Tell me about the tax."),
        "<passed to the model>"
    );

    assert_eq!(said_by(&desk, "Forget that."), "I have taken that back.");
    assert!(desk.lessons().is_empty());
    speak("Jefferson wrote it.");

    assert_eq!(
        log.into_inner().unwrap(),
        [
            "spoke Jefferson wrote it.",
            "spoke Jefferson wrote it again.",
            "spoke Jeff-er-son wrote it.",
            "spoke Jefferson wrote it.",
        ],
        "taught only after the user confirmed, and gone after it was taken back"
    );
}

#[test]
fn a_refused_or_unanswered_offer_teaches_nothing() {
    let desk = desk();
    said_by(&desk, "Pronounce Adams as Add-ums.");
    assert_eq!(said_by(&desk, "No."), "I will leave it as it was.");
    said_by(&desk, "Pronounce Adams as Add-ums.");
    assert_eq!(
        said_by(&desk, "What of the tax?"),
        "<passed to the model>",
        "an unrelated turn drops the offer"
    );
    assert_eq!(
        said_by(&desk, "Yes."),
        "<passed to the model>",
        "and a later yes confirms nothing"
    );
    assert!(desk.lessons().is_empty());
    assert!(!desk.has_kept());
    assert_eq!(
        said_by(&desk, "Forget that."),
        "There is nothing to take back."
    );
}

#[test]
fn the_desk_says_whether_it_holds_a_lesson_a_retraction_could_take_back() {
    let desk = desk();
    said_by(&desk, "Pronounce Adams as Add-ums.");
    said_by(&desk, "Yes.");
    assert!(desk.has_kept());
    said_by(&desk, "Forget that.");
    assert!(!desk.has_kept());
}

#[test]
fn lessons_survive_a_restart_and_a_lesson_never_changes_the_portrayal() {
    let dir = std::env::temp_dir().join(format!("splinter-desk-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("lessons.jsonl");
    let _ = std::fs::remove_file(&path);
    {
        let desk = LessonDesk::open(&path, Portrayal::synthetic_theatrical()).unwrap();
        said_by(&desk, "Pronounce Jefferson as Jeff-er-son.");
        said_by(&desk, "Yes.");
    }
    let reopened = LessonDesk::open(&path, Portrayal::synthetic_theatrical()).unwrap();
    assert_eq!(
        reopened.lexicon().read().unwrap().apply("Jefferson"),
        "Jeff-er-son"
    );
    assert!(reopened
        .lessons()
        .iter()
        .all(|l| l.portrayal == Portrayal::synthetic_theatrical()));
    said_by(&reopened, "Undo that.");
    let after = LessonDesk::open(&path, Portrayal::synthetic_theatrical()).unwrap();
    assert!(
        after.lessons().is_empty(),
        "taking a lesson back is kept too"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_imitation_is_declined_until_its_example_can_be_heard() {
    let desk = desk();
    let reply = said_by(&desk, "Talk as follows: Good morning.");
    assert!(
        reply.contains("pronounce"),
        "the user is told what can be taught: {reply}"
    );
    assert!(desk.lessons().is_empty());
}

fn take(text: &str, audio: &str) -> Take {
    Take {
        text: text.to_string(),
        audio: splinter_core::digest::Digest::of(audio.as_bytes()),
    }
}

#[test]
fn a_rejected_take_is_said_again_and_the_users_pick_becomes_a_preference() {
    let desk = desk();
    desk.spoke(take("We must resist.", "first"));
    assert_eq!(
        desk.hear("That sounded wrong."),
        Heard::Retry("We must resist.".into())
    );
    assert_eq!(
        desk.follow_up(),
        None,
        "nothing to ask until the new take is spoken"
    );
    desk.spoke(take("We must resist.", "second"));
    assert_eq!(desk.follow_up().as_deref(), Some("Was that better?"));

    assert_eq!(
        said_by(&desk, "I liked the tax part."),
        "<passed to the model>",
        "only a plain answer counts"
    );
    // The unclear answer dropped the comparison; ask again from the top.
    desk.spoke(take("We must resist.", "third"));
    desk.hear("Say that again.");
    desk.spoke(take("We must resist.", "fourth"));
    assert_eq!(said_by(&desk, "Better!"), "Noted.");

    let lessons = desk.lessons();
    assert_eq!(lessons.len(), 1);
    assert_eq!(
        lessons[0].objective,
        splinter_core::speech_lesson::Objective::Preference
    );
    match &lessons[0].exemplar {
        splinter_core::speech_lesson::Exemplar::Preferred {
            text,
            chosen,
            rejected,
        } => {
            assert_eq!(text, "We must resist.");
            assert_eq!(*chosen, splinter_core::digest::Digest::of(b"fourth"));
            assert_eq!(*rejected, splinter_core::digest::Digest::of(b"third"));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        desk.lexicon().read().unwrap().len(),
        0,
        "a preference teaches the lexicon nothing"
    );
}

#[test]
fn preferring_the_old_take_swaps_the_pair_and_approval_or_silence_teaches_nothing() {
    let desk = desk();
    assert_eq!(
        said_by(&desk, "That sounded wrong."),
        "I have not said anything yet."
    );
    desk.spoke(take("Aye.", "old"));
    assert_eq!(said_by(&desk, "That sounded good."), "Good.");
    desk.hear("That sounded wrong.");
    desk.spoke(take("Aye.", "new"));
    assert_eq!(said_by(&desk, "No."), "Noted.");
    match &desk.lessons()[0].exemplar {
        splinter_core::speech_lesson::Exemplar::Preferred {
            chosen, rejected, ..
        } => {
            assert_eq!(*chosen, splinter_core::digest::Digest::of(b"old"));
            assert_eq!(*rejected, splinter_core::digest::Digest::of(b"new"));
        }
        other => panic!("{other:?}"),
    }
}
