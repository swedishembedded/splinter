// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for checking that recorded speech
// says what its transcript says before it becomes training data, for its
// clients. If your team needs expertise in curating voice datasets, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The recordings of an audio dataset heard by a recogniser, each judged
//! against its transcript before it may train anything.
//!
//! Every row of a Hugging Face audio folder's `metadata.jsonl` is sent, as
//! its file's own bytes, to a brain server's transcription route, which reads
//! the container (WAV at any rate and channel count, Ogg/Opus, WebM/Opus),
//! mixes it to mono and resamples it for the recogniser. What was heard is
//! compared with the row's transcript ([`splinter_model::speech::transcript`])
//! and the verdict is written into the row beside the evidence for it:
//!
//! | column | |
//! |---|---|
//! | `asr_model` | the recogniser |
//! | `asr_text` | what it heard |
//! | `asr_similarity`, `asr_length_ratio` | the comparison |
//! | `asr_audio_sha256` | the recording the hearing is of |
//! | `validation` | `pass`, `review` or `fail` |
//! | `validation_reason` | why, for anything but a pass |
//!
//! A row whose recording and recogniser are those of its last hearing is not
//! heard again, only judged again, so a transcript corrected in the file is
//! checked against the same hearing. A recording that cannot be read or
//! heard fails with the reason, and the other rows are still checked.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use splinter_data::audio_folder::{AudioFolder, Row};
use splinter_model::speech::transcript::{compare, MatchPolicy, TranscriptMatch, Verdict};
use splinter_model::speech::BrainRecognizer;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What to check, and by which bounds.
#[derive(Clone, Debug)]
pub struct SpeechCheck {
    /// The audio folder's `metadata.jsonl`.
    pub metadata: PathBuf,
    /// The column holding each recording's transcript.
    pub text_column: String,
    /// The recognition model, as the brain server lists it.
    pub recognizer: String,
    /// The bounds a match is judged by.
    pub policy: MatchPolicy,
    /// Hear every recording again, even one heard before by this recogniser.
    pub rehear: bool,
}

/// One recording's verdict.
#[derive(Clone, Debug, Serialize)]
pub struct HeardRow {
    /// The row's `id`, else its `file_name`.
    pub id: String,
    /// Pass, review or fail.
    pub verdict: Verdict,
    /// Why, for anything but a pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The similarity to the transcript; absent when nothing could be heard.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f64>,
    /// The hearing's length over the transcript's; absent when nothing could be heard.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length_ratio: Option<f64>,
    /// Whether this run heard it, or reused the hearing of an unchanged file.
    pub heard_now: bool,
}

/// The verdicts of a whole dataset.
#[derive(Clone, Debug, Serialize)]
pub struct SpeechVerification {
    /// The metadata file the verdicts were written into.
    pub metadata: String,
    /// The recognition model.
    pub recognizer: String,
    /// Each recording's verdict, in file order.
    pub rows: Vec<HeardRow>,
    /// Recordings that passed.
    pub passed: usize,
    /// Recordings for review.
    pub review: usize,
    /// Recordings that failed.
    pub failed: usize,
}

impl SpeechVerification {
    /// Whether every recording passed.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.review == 0 && self.failed == 0
    }
}

/// Hear and judge every recording of `check.metadata`, writing the verdicts
/// into it.
///
/// # Errors
/// The metadata cannot be read or written, or no brain server can be asked.
/// A single recording that cannot be heard is a failed row, not an error.
pub fn verify_speech(
    ctx: &Context,
    check: &SpeechCheck,
) -> Result<SpeechVerification, OrchestratorError> {
    let mut folder = AudioFolder::open(&check.metadata).map_err(OrchestratorError::Refused)?;
    let recognizer = BrainRecognizer::new(&ctx.brain()?, &check.recognizer);
    let heard = judge_rows(&mut folder, check, |bytes| {
        recognizer
            .transcribe_recording(bytes)
            .map(|t| t.text)
            .map_err(|e| e.to_string())
    })?;
    folder.save().map_err(OrchestratorError::Refused)?;
    Ok(heard)
}

/// [`verify_speech`] over any recogniser: `hear` turns a recording's bytes
/// into the words heard in it.
pub(crate) fn judge_rows(
    folder: &mut AudioFolder,
    check: &SpeechCheck,
    hear: impl Fn(&[u8]) -> Result<String, String>,
) -> Result<SpeechVerification, OrchestratorError> {
    let mut rows = Vec::new();
    for i in 0..folder.rows().len() {
        let path = folder.audio_path(&folder.rows()[i]);
        let row = &mut folder.rows_mut()[i];
        rows.push(judge_row(row, &path, check, &hear));
    }
    let count = |v: Verdict| rows.iter().filter(|r| r.verdict == v).count();
    Ok(SpeechVerification {
        metadata: check.metadata.display().to_string(),
        recognizer: check.recognizer.clone(),
        passed: count(Verdict::Pass),
        review: count(Verdict::Review),
        failed: count(Verdict::Fail),
        rows,
    })
}

fn judge_row(
    row: &mut Row,
    path: &Path,
    check: &SpeechCheck,
    hear: &impl Fn(&[u8]) -> Result<String, String>,
) -> HeardRow {
    let id = row
        .text("id")
        .or_else(|| row.text("file_name"))
        .unwrap_or_default()
        .to_string();
    let fail = |row: &mut Row, reason: String| {
        row.set("validation", Value::from("fail"));
        row.set("validation_reason", Value::from(reason.clone()));
        HeardRow {
            id: id.clone(),
            verdict: Verdict::Fail,
            reason: Some(reason),
            similarity: None,
            length_ratio: None,
            heard_now: false,
        }
    };
    let Some(transcript) = row.text(&check.text_column).map(str::to_string) else {
        return fail(
            row,
            format!("no transcript in column '{}'", check.text_column),
        );
    };
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) => return fail(row, format!("the recording {}: {e}", path.display())),
    };
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let unchanged = !check.rehear
        && row.text("asr_audio_sha256") == Some(digest.as_str())
        && row.text("asr_model") == Some(check.recognizer.as_str());
    let (heard_text, heard_now) = match (unchanged, row.text("asr_text")) {
        (true, Some(text)) => (text.to_string(), false),
        _ => match hear(&bytes) {
            Ok(text) => (text, true),
            Err(e) => return fail(row, format!("not heard: {e}")),
        },
    };
    let m: TranscriptMatch = compare(&transcript, &heard_text);
    let (verdict, reason) = check.policy.judge(&m);
    let round = |x: f64| (x * 1000.0).round() / 1000.0;
    row.set("asr_model", Value::from(check.recognizer.clone()));
    row.set("asr_text", Value::from(heard_text));
    row.set("asr_similarity", Value::from(round(m.similarity)));
    row.set("asr_length_ratio", Value::from(round(m.length_ratio)));
    row.set("asr_audio_sha256", Value::from(digest));
    row.set(
        "validation",
        serde_json::to_value(&verdict).unwrap_or(Value::Null),
    );
    row.set(
        "validation_reason",
        reason.clone().map_or(Value::Null, Value::from),
    );
    HeardRow {
        id,
        verdict,
        reason,
        similarity: Some(round(m.similarity)),
        length_ratio: Some(round(m.length_ratio)),
        heard_now,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const ROWS: &str = concat!(
        "{\"id\": \"good\", \"file_name\": \"good.wav\", \"text\": \"Sex laxar i en laxask.\"}\n",
        "{\"id\": \"twice\", \"file_name\": \"twice.wav\", \"text\": \"Sex laxar i en laxask.\"}\n",
        "{\"id\": \"other\", \"file_name\": \"other.wav\", \"text\": \"Sex laxar i en laxask.\"}\n",
        "{\"id\": \"gone\", \"file_name\": \"gone.wav\", \"text\": \"Sex laxar i en laxask.\"}\n",
    );

    /// A recogniser that hears what the file says: the test's "recordings"
    /// are their own transcripts.
    fn hear(calls: &Cell<usize>) -> impl Fn(&[u8]) -> Result<String, String> + '_ {
        move |bytes| {
            calls.set(calls.get() + 1);
            Ok(String::from_utf8_lossy(bytes).into_owned())
        }
    }

    fn dataset() -> (tempfile::TempDir, SpeechCheck) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("metadata.jsonl"), ROWS).unwrap();
        std::fs::write(dir.path().join("good.wav"), "sex laxar i en lax ask").unwrap();
        std::fs::write(
            dir.path().join("twice.wav"),
            "Sex laxar i en laxask. Sex laxar i en laxask.",
        )
        .unwrap();
        std::fs::write(dir.path().join("other.wav"), "Packa pappas kappsäck").unwrap();
        let check = SpeechCheck {
            metadata: dir.path().join("metadata.jsonl"),
            text_column: "text".into(),
            recognizer: "asr".into(),
            policy: MatchPolicy::default(),
            rehear: false,
        };
        (dir, check)
    }

    #[test]
    fn each_recording_is_judged_against_its_transcript_and_the_verdict_is_written_into_its_row() {
        let (_dir, check) = dataset();
        let calls = Cell::new(0);
        let mut folder = AudioFolder::open(&check.metadata).unwrap();
        let report = judge_rows(&mut folder, &check, hear(&calls)).unwrap();
        folder.save().unwrap();
        let verdicts: Vec<(&str, Verdict)> = report
            .rows
            .iter()
            .map(|r| (r.id.as_str(), r.verdict.clone()))
            .collect();
        assert_eq!(
            verdicts,
            [
                ("good", Verdict::Pass),
                ("twice", Verdict::Fail),
                ("other", Verdict::Fail),
                ("gone", Verdict::Fail)
            ]
        );
        assert!(report.rows[1]
            .reason
            .as_deref()
            .unwrap()
            .contains("repeated"));
        assert!(
            report.rows[3]
                .reason
                .as_deref()
                .unwrap()
                .contains("gone.wav"),
            "a missing file is named"
        );
        assert_eq!((report.passed, report.failed, calls.get()), (1, 3, 3));
        let written = AudioFolder::open(&check.metadata).unwrap();
        let good = &written.rows()[0];
        assert_eq!(good.text("validation"), Some("pass"));
        assert_eq!(good.text("asr_text"), Some("sex laxar i en lax ask"));
        assert_eq!(good.get("asr_similarity"), Some(&Value::from(1.0)));
        assert_eq!(
            good.text("text"),
            Some("Sex laxar i en laxask."),
            "the transcript is left as it was"
        );
    }

    #[test]
    fn an_unchanged_recording_is_not_heard_again_but_a_changed_one_or_transcript_is_judged_again() {
        let (dir, check) = dataset();
        let mut folder = AudioFolder::open(&check.metadata).unwrap();
        judge_rows(&mut folder, &check, hear(&Cell::new(0))).unwrap();
        folder.save().unwrap();

        // The "other" take is recorded again; the first row's transcript is corrected to what was said.
        std::fs::write(dir.path().join("other.wav"), "Sex laxar i en laxask.").unwrap();
        let mut folder = AudioFolder::open(&check.metadata).unwrap();
        folder.rows_mut()[0].set("text", Value::from("Packa pappas kappsäck."));
        let calls = Cell::new(0);
        let report = judge_rows(&mut folder, &check, hear(&calls)).unwrap();
        assert_eq!(calls.get(), 1, "only the changed recording is heard again");
        assert_eq!(report.rows[2].verdict, Verdict::Pass);
        assert!(report.rows[2].heard_now && !report.rows[0].heard_now);
        assert_eq!(
            report.rows[0].verdict,
            Verdict::Fail,
            "the same hearing against the new transcript"
        );

        let calls = Cell::new(0);
        judge_rows(
            &mut folder,
            &SpeechCheck {
                rehear: true,
                ..check.clone()
            },
            hear(&calls),
        )
        .unwrap();
        assert_eq!(calls.get(), 3, "rehear hears every readable recording");
    }

    #[test]
    fn a_recording_the_recogniser_refuses_fails_with_its_reason() {
        let (_dir, check) = dataset();
        let mut folder = AudioFolder::open(&check.metadata).unwrap();
        let report = judge_rows(&mut folder, &check, |_| {
            Err("unsupported audio: not a recognised container".into())
        })
        .unwrap();
        assert_eq!(report.failed, 4);
        assert!(report.rows[0]
            .reason
            .as_deref()
            .unwrap()
            .contains("unsupported audio"));
    }
}
