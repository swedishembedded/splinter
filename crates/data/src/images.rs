// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for models that read images,
// for its clients. If your team needs expertise in fine-tuning on
// conversations with pictures, you can procure our services by sending an
// email to info@swedishembedded.com.

//! What an image costs a training record.
//!
//! Brain's trainer reads a record's images with the vision tower: each image
//! becomes rows in the sequence the decoder trains on, one per 32 x 32 pixels
//! of the picture (an image above 4 megapixels is scaled down to that first),
//! so a record's length is its text plus these rows. A dataset built for a
//! block size has to count them, and must not count the picture's base64 as
//! text.

use crate::RecordBody;

/// The most images brain's trainer reads in one record; it refuses a record
/// with more.
pub const MAX_IMAGES_PER_RECORD: usize = 8;

/// The rows of the decoder's sequence one image never exceeds: a picture is
/// scaled to 4 megapixels, and each row stands for 32 x 32 pixels.
pub const MAX_ROWS_PER_IMAGE: usize = 4096;

/// The rows the image at `url` adds to a record: from its dimensions when the
/// URL is an inline PNG, else the most one image can take.
#[must_use]
pub fn image_rows(url: &str) -> usize {
    png_size(url).map_or(MAX_ROWS_PER_IMAGE, |(width, height)| {
        (width.div_ceil(32) * height.div_ceil(32)).min(MAX_ROWS_PER_IMAGE)
    })
}

/// The width and height of an inline PNG data URL, read from its header.
fn png_size(url: &str) -> Option<(usize, usize)> {
    let data = url.strip_prefix("data:image/png;base64,")?;
    // The signature (8 bytes) and the IHDR chunk's length, type, width and
    // height (16 bytes) are the first 24 bytes: 32 base64 characters.
    let bytes = decode_base64(data.get(..32)?)?;
    let word = |at: usize| -> Option<usize> {
        let four: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
        usize::try_from(u32::from_be_bytes(four)).ok()
    };
    (bytes.get(12..16)? == b"IHDR").then_some(())?;
    Some((word(16)?, word(20)?))
}

fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    for group in text.as_bytes().chunks(4) {
        let mut bits = 0u32;
        for &c in group {
            let value = match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => return None,
            };
            bits = bits << 6 | u32::from(value);
        }
        out.extend_from_slice(&bits.to_be_bytes()[1..]);
    }
    Some(out)
}

impl RecordBody {
    /// The record as the text a tokenizer measures and the rows its images
    /// add: its JSON with every image taken out, and the sum of
    /// [`image_rows`] over them.
    #[must_use]
    pub fn text_and_image_rows(&self) -> (String, usize) {
        let mut rows = 0;
        let mut without = self.clone();
        let messages = match &mut without {
            Self::Chat { messages, .. } | Self::Rewarded { messages, .. } => Some(messages),
            _ => None,
        };
        for message in messages.into_iter().flatten() {
            rows += message
                .images
                .drain(..)
                .map(|image| image_rows(&image.url))
                .sum::<usize>();
        }
        (serde_json::to_string(&without).unwrap_or_default(), rows)
    }
}
