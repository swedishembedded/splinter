// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Framing of one compressed, checksummed unit of a segment.

use crate::config::Compression;
use crate::error::{Error, Result};

pub(crate) const FRAME_HEADER: usize = 1 + 4 + 4 + 4;

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
    out.extend_from_slice(&crc32fast::hash(&stored).to_le_bytes());
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
    if crc32fast::hash(stored) != crc {
        return Err(Error::corrupt(what, "checksum mismatch"));
    }
    match bytes[0] {
        0 => Ok(stored.to_vec()),
        1 => zstd::bulk::decompress(stored, raw_len)
            .map_err(|e| Error::corrupt(what, format!("cannot decompress: {e}"))),
        other => Err(Error::corrupt(what, format!("unknown codec {other}"))),
    }
}
