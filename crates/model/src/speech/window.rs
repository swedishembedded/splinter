// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A clip padded with silence to a fixed length, so every utterance fills the
//! same number of audio rows in a language model's input.

use super::Clip;
use crate::error::PolicyError;

/// `clip` followed by silence up to `seconds` seconds. A clip longer than the
/// window is refused, not cut: a model answering half a question is worse than
/// one that says the question did not fit.
pub fn padded_to(clip: &Clip, seconds: f32) -> Result<Clip, PolicyError> {
    let target = (f64::from(clip.sample_rate()) * f64::from(seconds)).round() as usize;
    if clip.samples().len() > target {
        return Err(PolicyError::Transcription {
            reason: format!(
                "a clip of {:.1} seconds does not fit a window of {seconds:.1} seconds",
                clip.seconds()
            ),
        });
    }
    let mut samples = clip.samples().to_vec();
    samples.resize(target, 0.0);
    Ok(Clip::new(samples, clip.sample_rate()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_clip_is_followed_by_silence_to_the_window() {
        let padded = padded_to(&Clip::new(vec![0.5; 8_000], 16_000), 1.0).unwrap();
        assert_eq!(padded.samples().len(), 16_000);
        assert!(padded.samples()[..8_000].iter().all(|s| *s == 0.5));
        assert!(padded.samples()[8_000..].iter().all(|s| *s == 0.0));
        assert_eq!(padded.sample_rate(), 16_000);
    }

    #[test]
    fn a_clip_exactly_the_window_is_unchanged_and_a_longer_one_is_refused() {
        let exact = Clip::new(vec![0.1; 16_000], 16_000);
        assert_eq!(padded_to(&exact, 1.0).unwrap(), exact);
        let err = padded_to(&Clip::new(vec![0.1; 16_001], 16_000), 1.0).unwrap_err();
        assert!(err.to_string().contains("does not fit"), "{err}");
    }

    #[test]
    fn the_window_is_in_seconds_at_the_clips_own_rate() {
        let padded = padded_to(&Clip::new(vec![0.0; 100], 24_000), 0.5).unwrap();
        assert_eq!(padded.samples().len(), 12_000);
    }
}
