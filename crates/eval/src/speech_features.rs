// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurements of synthetic speech that a
// listener's judgement can be learned from, for its clients. If your team needs
// expertise in turning audio into features a judge can be fitted on, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Cheap, interpretable features of one take, the input of the naturalness judge
//! of [`crate::speech_judge`].
//!
//! Every feature is computed from the samples alone, on 20 ms frames with a 10 ms
//! hop, and the level-dependent ones are relative to the loudest frames of the
//! same take so that playback gain does not move them. They describe pace
//! (duration, silence, longest pause), dynamics (spread of frame level), noise
//! (level of the quietest frames, energy in the highest band) and clipping. They
//! do not say what was spoken: a mumbled take with the right pace and level has
//! the features of a clear one, which is why the reward also carries the
//! recognisers' word error rate.

/// The names of the features, in the order [`features`] returns them.
pub const NAMES: [&str; 8] = [
    "seconds",
    "silence_fraction",
    "longest_pause_seconds",
    "level_spread_db",
    "noise_floor_db",
    "zero_crossing_rate",
    "high_band_share",
    "clipped_fraction",
];

const FRAME_SECONDS: f32 = 0.020;
const HOP_SECONDS: f32 = 0.010;
/// Frames this far below the loudest frames count as silence.
const SILENCE_BELOW_PEAK_DB: f32 = 30.0;
const FLOOR_DB: f32 = -120.0;

fn db(power: f32) -> f32 {
    (10.0 * power.max(1e-12).log10()).max(FLOOR_DB)
}

/// The features of a mono take, or `None` when it is shorter than one frame.
pub fn features(samples: &[f32], sample_rate: u32) -> Option<[f32; NAMES.len()]> {
    let frame = (FRAME_SECONDS * sample_rate as f32) as usize;
    let hop = (HOP_SECONDS * sample_rate as f32) as usize;
    if frame == 0 || hop == 0 || samples.len() < frame {
        return None;
    }
    let mut levels = Vec::new();
    let mut crossings = Vec::new();
    let mut highs = Vec::new();
    for window in samples.windows(frame).step_by(hop) {
        let power = window.iter().map(|x| x * x).sum::<f32>() / frame as f32;
        let change = window
            .windows(2)
            .map(|p| (p[1] - p[0]).powi(2))
            .sum::<f32>()
            / (frame - 1) as f32;
        let sign_changes = window
            .windows(2)
            .filter(|p| (p[0] >= 0.0) != (p[1] >= 0.0))
            .count();
        levels.push(db(power));
        crossings.push(sign_changes as f32 / (frame - 1) as f32);
        // The first difference attenuates low frequencies, so its energy over the
        // frame's energy is high when the spectrum leans toward the top.
        highs.push(change / power.max(1e-12));
    }
    let mut sorted = levels.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f32| sorted[((sorted.len() - 1) as f32 * q) as usize];
    let peak = at(0.95);
    let voiced: Vec<usize> = (0..levels.len())
        .filter(|&i| levels[i] > peak - SILENCE_BELOW_PEAK_DB)
        .collect();
    let silent = levels.len() - voiced.len();
    let (mut longest, mut run) = (0usize, 0usize);
    for &level in &levels {
        run = if level <= peak - SILENCE_BELOW_PEAK_DB {
            run + 1
        } else {
            0
        };
        longest = longest.max(run);
    }
    let mean = |values: &dyn Fn(usize) -> f32| {
        if voiced.is_empty() {
            0.0
        } else {
            voiced.iter().map(|&i| values(i)).sum::<f32>() / voiced.len() as f32
        }
    };
    let level_mean = mean(&|i| levels[i]);
    let spread = mean(&|i| (levels[i] - level_mean).powi(2)).sqrt();
    let clipped = samples.iter().filter(|x| x.abs() >= 0.999).count() as f32 / samples.len() as f32;
    Some([
        samples.len() as f32 / sample_rate as f32,
        silent as f32 / levels.len() as f32,
        longest as f32 * HOP_SECONDS,
        spread,
        at(0.10) - peak,
        mean(&|i| crossings[i]),
        mean(&|i| highs[i]),
        clipped,
    ])
}

/// `samples` with white noise added, `snr_db` below their mean power. The noise
/// is deterministic in `seed`, so a degraded copy of a take is reproducible.
#[must_use]
pub fn with_noise(samples: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    let power = samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32;
    let sigma = (power / 10f32.powf(snr_db / 10.0)).sqrt();
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    samples
        .iter()
        .map(|x| {
            // The sum of four uniforms has variance 1/3: scaled to unit variance it is close to Gaussian.
            let n = (uniform() + uniform() + uniform() + uniform() - 2.0) * 3f32.sqrt();
            x + sigma * n
        })
        .collect()
}

#[cfg(test)]
mod specs {
    use super::*;

    const RATE: u32 = 16_000;

    /// A voiced-sounding tone of the given length, gapped by silence.
    fn speech(seconds: &[(f32, bool)]) -> Vec<f32> {
        let mut out = Vec::new();
        for &(length, voiced) in seconds {
            for n in 0..(length * RATE as f32) as usize {
                let t = n as f32 / RATE as f32;
                out.push(if voiced {
                    0.3 * (2.0 * std::f32::consts::PI * 180.0 * t).sin()
                } else {
                    0.0
                });
            }
        }
        out
    }

    fn named(values: &[f32; NAMES.len()], name: &str) -> f32 {
        values[NAMES.iter().position(|n| *n == name).unwrap()]
    }

    #[test]
    fn a_take_shorter_than_one_frame_has_no_features() {
        assert!(features(&[0.1; 10], RATE).is_none());
    }

    #[test]
    fn a_longer_pause_shows_in_the_silence_features() {
        let short = features(&speech(&[(0.5, true), (0.1, false), (0.5, true)]), RATE).unwrap();
        let long = features(&speech(&[(0.5, true), (0.6, false), (0.5, true)]), RATE).unwrap();
        assert!(
            named(&long, "longest_pause_seconds") > named(&short, "longest_pause_seconds") + 0.4
        );
        assert!(named(&long, "silence_fraction") > named(&short, "silence_fraction"));
        assert!(named(&long, "seconds") > named(&short, "seconds"));
    }

    #[test]
    fn added_noise_raises_the_noise_floor_and_the_high_band() {
        let clean = speech(&[(0.4, true), (0.3, false), (0.4, true)]);
        let mut state = 12345u32;
        let noisy: Vec<f32> = clean
            .iter()
            .map(|x| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                x + 0.05 * ((state >> 8) as f32 / (1 << 24) as f32 - 0.5)
            })
            .collect();
        let (a, b) = (
            features(&clean, RATE).unwrap(),
            features(&noisy, RATE).unwrap(),
        );
        assert!(named(&b, "noise_floor_db") > named(&a, "noise_floor_db") + 10.0);
        assert!(named(&b, "high_band_share") > named(&a, "high_band_share"));
    }

    #[test]
    fn playback_gain_does_not_move_the_level_relative_features() {
        let base = speech(&[(0.4, true), (0.3, false), (0.4, true)]);
        let quiet: Vec<f32> = base.iter().map(|x| x * 0.25).collect();
        let (a, b) = (
            features(&base, RATE).unwrap(),
            features(&quiet, RATE).unwrap(),
        );
        for name in [
            "silence_fraction",
            "longest_pause_seconds",
            "level_spread_db",
            "zero_crossing_rate",
        ] {
            assert!((named(&a, name) - named(&b, name)).abs() < 1e-3, "{name}");
        }
    }

    #[test]
    fn added_noise_has_the_requested_signal_to_noise_ratio_and_is_reproducible() {
        let clean = speech(&[(1.0, true)]);
        let noisy = with_noise(&clean, 12.0, 7);
        let signal = clean.iter().map(|x| x * x).sum::<f32>();
        let noise = clean
            .iter()
            .zip(&noisy)
            .map(|(c, n)| (n - c).powi(2))
            .sum::<f32>();
        let measured = 10.0 * (signal / noise).log10();
        assert!((measured - 12.0).abs() < 0.5, "{measured}");
        assert_eq!(noisy, with_noise(&clean, 12.0, 7));
        assert_ne!(noisy, with_noise(&clean, 12.0, 8));
    }
}
