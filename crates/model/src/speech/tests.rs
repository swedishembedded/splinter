// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specs of the cascade: what a turn hears, answers and speaks, and what a
//! round trip says was lost.

use splinter_core::speech::{Portrayal, SpeakerProfile};

use super::scripted::{ScriptedRecognizer, ScriptedSynthesizer};
use super::*;

fn speaker() -> SpeakerProfile {
    SpeakerProfile::new(1, Portrayal::synthetic_theatrical())
}

fn said(text: &str) -> Clip {
    ScriptedSynthesizer.speak(text, &speaker()).unwrap()
}

#[test]
fn a_turn_answers_what_was_heard_and_speaks_the_answer() {
    let mut asked = Vec::new();
    let mut respond = |heard: &str| {
        asked.push(heard.to_string());
        Ok("Liberty is not given.".to_string())
    };
    let turn = take_turn(
        &ScriptedRecognizer::perfect(),
        &mut respond,
        &ScriptedSynthesizer,
        &speaker(),
        &said("What is liberty?"),
    )
    .unwrap();

    assert_eq!(asked, ["What is liberty?"]);
    assert_eq!(turn.heard, "What is liberty?");
    assert_eq!(turn.answer, "Liberty is not given.");
    assert_eq!(
        turn.reply,
        said("Liberty is not given."),
        "the answer is what was spoken"
    );
    assert_eq!(
        turn.timings.total(),
        turn.timings.recognise + turn.timings.respond + turn.timings.synthesise
    );
}

#[test]
fn silence_is_not_answered() {
    let mut called = false;
    let mut respond = |_: &str| {
        called = true;
        Ok("anything".to_string())
    };
    let err = take_turn(
        &ScriptedRecognizer::perfect(),
        &mut respond,
        &ScriptedSynthesizer,
        &speaker(),
        &said("   "),
    )
    .unwrap_err();

    assert!(matches!(err, PolicyError::Transcription { .. }), "{err}");
    assert!(!called, "the persona must not be asked to answer nothing");
}

#[test]
fn an_empty_answer_is_not_spoken() {
    let mut respond = |_: &str| Ok("  ".to_string());
    let err = take_turn(
        &ScriptedRecognizer::perfect(),
        &mut respond,
        &ScriptedSynthesizer,
        &speaker(),
        &said("Hello"),
    )
    .unwrap_err();

    assert!(matches!(err, PolicyError::Synthesis { .. }), "{err}");
}

#[test]
fn a_failing_stage_stops_the_turn_with_its_own_error() {
    let mut respond = |_: &str| {
        Err(PolicyError::Remote {
            spec: "policy".into(),
            reason: "down".into(),
        })
    };
    let err = take_turn(
        &ScriptedRecognizer::perfect(),
        &mut respond,
        &ScriptedSynthesizer,
        &speaker(),
        &said("Hello"),
    )
    .unwrap_err();

    assert!(err.to_string().contains("down"), "{err}");
}

#[test]
fn a_round_trip_with_perfect_hearing_loses_nothing() {
    let sentences = vec![
        "The quick brown fox.".to_string(),
        "Jumps over.".to_string(),
    ];
    let report = round_trip(
        &ScriptedSynthesizer,
        &ScriptedRecognizer::perfect(),
        &speaker(),
        &sentences,
    )
    .unwrap();

    assert_eq!(report.corpus_word_error_rate(), Some(0.0));
    assert_eq!(report.mostly_lost(), 0);
}

#[test]
fn the_corpus_rate_weighs_a_long_sentence_more_than_a_short_one() {
    // The listener loses the last word of each sentence: one of two words in
    // the short one (0.5), one of eight in the long one (0.125).
    let drop_last = ScriptedRecognizer::degrading(|text| {
        let mut words: Vec<&str> = text.split_whitespace().collect();
        words.pop();
        words.join(" ")
    });
    let sentences = vec![
        "Hello there".to_string(),
        "one two three four five six seven eight".to_string(),
    ];
    let report = round_trip(&ScriptedSynthesizer, &drop_last, &speaker(), &sentences).unwrap();

    assert_eq!(report.items[0].word_error_rate, 0.5);
    assert_eq!(report.items[1].word_error_rate, 0.125);
    assert_eq!(
        report.corpus_word_error_rate(),
        Some(0.2),
        "2 errors in 10 words"
    );
    assert_eq!(report.mostly_lost(), 0);
}

#[test]
fn a_round_trip_of_nothing_has_no_rate() {
    let report = round_trip(
        &ScriptedSynthesizer,
        &ScriptedRecognizer::perfect(),
        &speaker(),
        &[],
    )
    .unwrap();
    assert_eq!(report.corpus_word_error_rate(), None);
}

/// A synthesizer that mumbles for the seeds it is given and speaks the rest.
fn mumbling_on(seeds: &'static [u64]) -> impl Synthesizer {
    struct Mumbler(&'static [u64]);
    impl Synthesizer for Mumbler {
        fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
            let spoken = if self.0.contains(&speaker.seed()) {
                ""
            } else {
                text
            };
            ScriptedSynthesizer.speak(spoken, speaker)
        }
    }
    Mumbler(seeds)
}

#[test]
fn a_clip_is_kept_only_if_it_is_heard_back_and_the_next_seed_is_tried_when_it_is_not() {
    let speaker = SpeakerProfile::new(10, Portrayal::synthetic_theatrical());
    let kept = speak_verified(
        &mumbling_on(&[10, 11]),
        &ScriptedRecognizer::perfect(),
        &speaker,
        "Liberty is not given.",
        0.2,
        4,
    )
    .unwrap()
    .expect("the third seed speaks");

    assert_eq!(
        kept.speaker.seed(),
        12,
        "the seed that was kept is the one recorded"
    );
    assert_eq!(kept.attempts, 3);
    assert_eq!(kept.word_error_rate, 0.0);
    assert_eq!(kept.speaker.portrayal(), speaker.portrayal());
}

#[test]
fn a_sentence_no_seed_speaks_is_rejected_not_kept_badly() {
    let speaker = SpeakerProfile::new(10, Portrayal::synthetic_theatrical());
    let kept = speak_verified(
        &mumbling_on(&[10, 11, 12]),
        &ScriptedRecognizer::perfect(),
        &speaker,
        "Liberty is not given.",
        0.2,
        3,
    )
    .unwrap();

    assert!(kept.is_none());
}

#[test]
fn the_error_bound_decides_what_is_heard_well_enough() {
    let speaker = SpeakerProfile::new(1, Portrayal::synthetic_theatrical());
    let drops_a_word = ScriptedRecognizer::degrading(|t| {
        t.rsplit_once(' ')
            .map_or(t.clone(), |(head, _)| head.to_string())
    });
    let strict = speak_verified(
        &ScriptedSynthesizer,
        &drops_a_word,
        &speaker,
        "one two three four",
        0.2,
        1,
    )
    .unwrap();
    let lenient = speak_verified(
        &ScriptedSynthesizer,
        &drops_a_word,
        &speaker,
        "one two three four",
        0.25,
        1,
    )
    .unwrap();

    assert!(
        strict.is_none(),
        "one word in four is 0.25, over the bound of 0.2"
    );
    assert_eq!(lenient.unwrap().word_error_rate, 0.25);
}

/// Speaks each piece it is given as one sample per byte, so a spec can read
/// which pieces were spoken, and in what order, from the clip's length.
struct Counting(std::sync::Mutex<Vec<String>>);

impl Synthesizer for Counting {
    fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        self.0.lock().unwrap().push(text.to_string());
        ScriptedSynthesizer.speak(text, speaker)
    }
}

#[test]
fn a_long_answer_is_spoken_a_sentence_at_a_time_and_joined_in_order() {
    let inner = Counting(std::sync::Mutex::new(Vec::new()));
    let spoken = Sentences::new(inner, 30);
    let clip = spoken
        .speak(
            "First we resolved. Then we petitioned! Did they listen? No.",
            &speaker(),
        )
        .unwrap();

    let pieces = spoken_pieces(&spoken);
    assert_eq!(
        pieces,
        [
            "First we resolved.",
            "Then we petitioned!",
            "Did they listen?",
            "No."
        ]
    );
    let pauses = (16_000 * PAUSE_MILLIS as usize / 1000) * 3;
    let words: usize = pieces.iter().map(String::len).sum();
    assert_eq!(
        clip.samples().len(),
        words + pauses,
        "the pieces, with a pause between each"
    );
}

#[test]
fn a_sentence_longer_than_the_bound_is_broken_at_a_comma_or_the_bound() {
    let inner = Counting(std::sync::Mutex::new(Vec::new()));
    let spoken = Sentences::new(inner, 5);
    spoken
        .speak(
            "one two three, four five six seven eight nine ten eleven twelve.",
            &speaker(),
        )
        .unwrap();

    let pieces = spoken_pieces(&spoken);
    assert_eq!(
        pieces[0], "one two three,",
        "broken at the comma inside the bound"
    );
    assert!(
        pieces.iter().all(|p| p.split_whitespace().count() <= 5),
        "{pieces:?}"
    );
    assert_eq!(
        pieces.join(" "),
        "one two three, four five six seven eight nine ten eleven twelve."
    );
}

#[test]
fn a_short_text_is_spoken_whole_with_no_pause() {
    let inner = Counting(std::sync::Mutex::new(Vec::new()));
    let spoken = Sentences::new(inner, 30);
    let clip = spoken.speak("Hello there.", &speaker()).unwrap();

    assert_eq!(spoken_pieces(&spoken), ["Hello there."]);
    assert_eq!(clip.samples().len(), "Hello there.".len());
}

#[test]
fn text_with_no_words_is_not_spoken() {
    let spoken = Sentences::new(Counting(std::sync::Mutex::new(Vec::new())), 30);
    assert!(spoken.speak("  ", &speaker()).is_err());
}

fn spoken_pieces(spoken: &Sentences<Counting>) -> Vec<String> {
    spoken.inner().0.lock().unwrap().clone()
}

#[test]
fn a_stream_of_text_yields_a_sentence_once_its_end_is_known() {
    let mut stream = SentenceStream::new(30);
    assert_eq!(
        stream.push("Liberty is not given. We m"),
        ["Liberty is not given."]
    );
    assert!(
        stream.push("ust claim it.").is_empty(),
        "the end of the sentence is not known until a space follows"
    );
    assert_eq!(stream.push(" And then"), ["We must claim it."]);
    assert_eq!(stream.finish(), ["And then"]);
}

#[test]
fn a_stream_does_not_end_a_sentence_after_an_abbreviation() {
    let mut stream = SentenceStream::new(30);
    assert_eq!(stream.push("Mr. Adams spoke. "), ["Mr. Adams spoke."]);
    assert!(stream.finish().is_empty());
}

#[test]
fn a_stream_breaks_a_long_sentence_at_the_bound_like_the_whole_text_would_be() {
    let text = "one two three, four five six seven eight nine ten eleven twelve. ";
    let mut stream = SentenceStream::new(5);
    let mut pieces = stream.push(text);
    pieces.extend(stream.finish());
    assert_eq!(pieces.join(" "), text.trim());
    assert!(
        pieces.iter().all(|p| p.split_whitespace().count() <= 5),
        "{pieces:?}"
    );
}

type Log = std::sync::Mutex<Vec<String>>;

fn note(log: &Log, line: impl Into<String>) {
    log.lock().unwrap().push(line.into());
}

/// Wait until the log holds `line`, so a listener can prove that speaking
/// proceeds while it is still answering. A serial implementation never gets
/// there and fails the wait.
fn wait_for(log: &Log, line: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !log.lock().unwrap().iter().any(|l| l == line) {
        assert!(
            std::time::Instant::now() < deadline,
            "{line:?} was never logged: nothing was spoken while the listener wrote"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// A listener that answers in deltas, then waits to hear the first sentence
/// spoken before it finishes.
struct Replying<'a> {
    answer: &'a str,
    first_spoken: Option<&'a str>,
    log: &'a Log,
}

impl Listener for Replying<'_> {
    fn answer(&self, _clip: &Clip, on_text: &mut dyn FnMut(&str)) -> Result<String, PolicyError> {
        for delta in self.answer.split_inclusive(' ') {
            on_text(delta);
        }
        if let Some(first) = self.first_spoken {
            wait_for(self.log, first);
        }
        note(self.log, "listener done");
        Ok(self.answer.to_string())
    }
}

struct Logging<'a>(&'a Log);

impl Synthesizer for Logging<'_> {
    fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        note(self.0, format!("spoke {text}"));
        ScriptedSynthesizer.speak(text, speaker)
    }
}

#[test]
fn the_first_sentence_is_spoken_while_the_listener_is_still_answering() {
    let log = Log::default();
    let listener = Replying {
        answer: "We resolved. Then we petitioned. They did not listen.",
        first_spoken: Some("spoke We resolved."),
        log: &log,
    };
    let turn =
        take_spoken_turn(&listener, &Logging(&log), &speaker(), &said("question"), 30).unwrap();

    assert_eq!(
        turn.answer,
        "We resolved. Then we petitioned. They did not listen."
    );
    let log = log.into_inner().unwrap();
    assert_eq!(log[0], "spoke We resolved.", "{log:?}");
    let done = log.iter().position(|l| l == "listener done").unwrap();
    assert!(done > 0, "speaking began before the listener finished");
    assert_eq!(
        log.last().unwrap(),
        "spoke They did not listen.",
        "the last sentence waits for the end of the answer"
    );
    let spoken: Vec<&String> = log.iter().filter(|l| l.starts_with("spoke ")).collect();
    assert_eq!(
        spoken,
        [
            "spoke We resolved.",
            "spoke Then we petitioned.",
            "spoke They did not listen."
        ],
        "in the order written"
    );
    assert!(turn.timings.first_audio <= turn.timings.total);
    let words: usize = [
        "We resolved.",
        "Then we petitioned.",
        "They did not listen.",
    ]
    .iter()
    .map(|s| s.len())
    .sum();
    assert_eq!(
        turn.reply.samples().len(),
        words + (16_000 * PAUSE_MILLIS as usize / 1000) * 2
    );
}

#[test]
fn a_listener_that_says_nothing_is_not_spoken_for() {
    let log = Log::default();
    let listener = Replying {
        answer: "  ",
        first_spoken: None,
        log: &log,
    };
    let err =
        take_spoken_turn(&listener, &Logging(&log), &speaker(), &said("question"), 30).unwrap_err();
    assert!(matches!(err, PolicyError::Synthesis { .. }), "{err}");
}

#[test]
fn a_cascade_listener_answers_what_it_heard_and_is_spoken_as_it_writes() {
    let log = Log::default();
    let asked = std::sync::Mutex::new(String::new());
    let listener = CascadeListener::new(
        ScriptedRecognizer::perfect(),
        |question: &str, on_text: &mut dyn FnMut(&str)| {
            *asked.lock().unwrap() = question.to_string();
            for piece in ["It is ", "tyranny. ", "Resist."] {
                on_text(piece);
            }
            wait_for(&log, "spoke It is tyranny.");
            note(&log, "listener done");
            Ok("It is tyranny. Resist.".to_string())
        },
    );
    let turn = take_spoken_turn(
        &listener,
        &Logging(&log),
        &speaker(),
        &said("Is it just?"),
        30,
    )
    .unwrap();

    assert_eq!(
        *asked.lock().unwrap(),
        "Is it just?",
        "the model gets the transcript"
    );
    assert_eq!(turn.answer, "It is tyranny. Resist.");
    let log = log.into_inner().unwrap();
    assert_eq!(log[0], "spoke It is tyranny.", "{log:?}");
}

#[test]
fn a_stream_lets_the_first_words_go_at_a_clause_end_before_the_sentence_closes() {
    let mut stream = SentenceStream::new(30);
    let first = stream.push("Well, I think that we must resist, and ");
    assert_eq!(first, ["Well, I think that we must resist,"]);
    let mut rest = stream.push("never submit to tyranny. ");
    rest.extend(stream.finish());
    assert_eq!(rest, ["and never submit to tyranny."]);
}

#[test]
fn a_stream_without_a_clause_end_lets_the_first_ten_words_go() {
    let mut stream = SentenceStream::new(30);
    let first = stream.push("one two three four five six seven eight nine ten eleven twelve ");
    assert_eq!(first, ["one two three four five six seven eight nine ten"]);
    let mut rest = stream.push("thirteen. ");
    rest.extend(stream.finish());
    assert_eq!(rest, ["eleven twelve thirteen."]);
}

#[test]
fn only_the_first_piece_is_cut_early() {
    let mut stream = SentenceStream::new(30);
    stream.push("One two three four five. ");
    let next = stream.push("Six seven eight, nine ten eleven, twelve thirteen. ");
    assert_eq!(next, ["Six seven eight, nine ten eleven, twelve thirteen."]);
}

#[test]
fn a_taught_word_reaches_the_voice_as_taught_and_nothing_else_changes() {
    let mut lexicon = splinter_core::speech_lesson::Lexicon::new();
    lexicon.teach("Jefferson", "Jeff-er-son");
    let log = Log::default();
    let voice = Lexical::new(Logging(&log), lexicon);
    voice.speak("Jefferson wrote it.", &speaker()).unwrap();
    voice.speak("Adams agreed.", &speaker()).unwrap();
    assert_eq!(
        log.into_inner().unwrap(),
        ["spoke Jeff-er-son wrote it.", "spoke Adams agreed."]
    );
}
