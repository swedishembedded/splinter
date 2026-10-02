// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Framing of one compressed, checksummed unit of a segment.

use std::io::Read;

use crate::config::Compression;
use crate::error::{Error, Result};

pub(crate) const FRAME_HEADER: usize = 1 + 4 + 4 + 4;

/// The checksum covers the header as well as the stored bytes, so a damaged
/// length or codec is noticed like any other damage.
fn checksum(header: &[u8], stored: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(header);
    hasher.update(stored);
    hasher.finalize()
}

/// Wraps `raw` in a frame, compressing it as configured.
pub(crate) fn frame(raw: &[u8], compression: Compression) -> Result<Vec<u8>> {
    let (codec, stored) = match compression {
        Compression::None => (0u8, raw.to_vec()),
        Compression::Zstd(level) => (
            1u8,
            zstd::bulk::compress(raw, level)
                .map_err(|e| Error::corrupt("segment block", format!("cannot compress: {e}")))?,
        ),
    };
    let mut out = Vec::with_capacity(FRAME_HEADER + stored.len());
    out.push(codec);
    out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
    let crc = checksum(&out[..9], &stored);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&stored);
    Ok(out)
}

/// Checks a frame's checksum and returns the bytes inside it.
pub(crate) fn unframe(what: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() < FRAME_HEADER {
        return Err(Error::corrupt(what, "frame is shorter than its header"));
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or([0; 4]));
    let (raw_len, stored_len, crc) = (word(1) as usize, word(5) as usize, word(9));
    let stored = &bytes[FRAME_HEADER..];
    if stored.len() != stored_len {
        return Err(Error::corrupt(
            what,
            "frame length does not match its header",
        ));
    }
    if checksum(&bytes[..9], stored) != crc {
        return Err(Error::corrupt(what, "checksum mismatch"));
    }
    match bytes[0] {
        0 if raw_len == stored_len => Ok(stored.to_vec()),
        0 => Err(Error::corrupt(
            what,
            "an uncompressed frame has two lengths",
        )),
        1 => {
            // Read at most the stated length plus one byte, growing as data
            // arrives, so a lying header cannot ask for a huge allocation.
            let decoder = zstd::stream::read::Decoder::new(stored)
                .map_err(|e| Error::corrupt(what, format!("cannot decompress: {e}")))?;
            let mut raw = Vec::new();
            decoder
                .take(raw_len as u64 + 1)
                .read_to_end(&mut raw)
                .map_err(|e| Error::corrupt(what, format!("cannot decompress: {e}")))?;
            if raw.len() != raw_len {
                return Err(Error::corrupt(
                    what,
                    "decompressed length does not match its header",
                ));
            }
            Ok(raw)
        }
        other => Err(Error::corrupt(what, format!("unknown codec {other}"))),
    }
}
