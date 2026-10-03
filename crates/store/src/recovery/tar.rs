// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, recoverable state for learning
// agents, for its clients. If your team needs expertise in backup and
// disaster recovery of training data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The tar framing of an archive: plain ustar members with fixed owner, mode
//! and time, so the same members always make the same bytes, and a reader that
//! accepts nothing but what the writer makes.

use std::io::{self, Read, Write};

const BLOCK: usize = 512;
/// The longest member path ustar holds without its prefix field.
pub(super) const MAX_PATH: usize = 100;

fn octal(field: &mut [u8], value: u64) {
    let digits = field.len() - 1;
    let text = format!("{value:0digits$o}");
    field[..digits].copy_from_slice(text.as_bytes());
    field[digits] = 0;
}

fn checksum(header: &[u8; BLOCK]) -> u64 {
    header
        .iter()
        .enumerate()
        .map(|(i, b)| u64::from(if (148..156).contains(&i) { b' ' } else { *b }))
        .sum()
}

fn invalid(reason: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, reason.into())
}

/// Writes members and the closing blocks.
pub(super) struct Writer<W: Write> {
    out: W,
}

impl<W: Write> Writer<W> {
    pub(super) fn new(out: W) -> Self {
        Self { out }
    }

    /// Writes one member whose `size` bytes come from `data`.
    pub(super) fn member(&mut self, path: &str, size: u64, mut data: impl Read) -> io::Result<()> {
        if path.len() > MAX_PATH || !path.is_ascii() {
            return Err(invalid(format!("member path `{path}` does not fit ustar")));
        }
        let mut header = [0u8; BLOCK];
        header[..path.len()].copy_from_slice(path.as_bytes());
        octal(&mut header[100..108], 0o644);
        octal(&mut header[108..116], 0);
        octal(&mut header[116..124], 0);
        octal(&mut header[124..136], size);
        octal(&mut header[136..148], 0);
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let sum = checksum(&header);
        let text = format!("{sum:06o}\0 ");
        header[148..156].copy_from_slice(text.as_bytes());
        self.out.write_all(&header)?;
        let copied = io::copy(&mut data.by_ref().take(size), &mut self.out)?;
        if copied != size {
            return Err(invalid(format!("`{path}` ended before its {size} bytes")));
        }
        let padding = (BLOCK - (size as usize % BLOCK)) % BLOCK;
        self.out.write_all(&vec![0u8; padding])
    }

    /// Writes the end of the archive and hands back the sink.
    pub(super) fn finish(mut self) -> io::Result<W> {
        self.out.write_all(&[0u8; 2 * BLOCK])?;
        Ok(self.out)
    }
}

/// One member's header.
pub(super) struct Entry {
    pub(super) path: String,
    pub(super) size: u64,
}

/// Reads members in order. A member's data must be taken or skipped before
/// the next header is read.
pub(super) struct Reader<R: Read> {
    input: R,
    unread: u64,
    padding: usize,
}

impl<R: Read> Reader<R> {
    pub(super) fn new(input: R) -> Self {
        Self {
            input,
            unread: 0,
            padding: 0,
        }
    }

    /// The next member, or `None` at the end of the archive.
    pub(super) fn next(&mut self) -> io::Result<Option<Entry>> {
        self.skip_rest()?;
        let mut header = [0u8; BLOCK];
        self.input.read_exact(&mut header)?;
        if header.iter().all(|b| *b == 0) {
            return Ok(None);
        }
        if &header[257..263] != b"ustar\0" || header[156] != b'0' {
            return Err(invalid("not a member this tool wrote"));
        }
        let stored = std::str::from_utf8(&header[148..154])
            .ok()
            .and_then(|t| u64::from_str_radix(t, 8).ok())
            .ok_or_else(|| invalid("unreadable header checksum"))?;
        if stored != checksum(&header) {
            return Err(invalid("a member header is damaged"));
        }
        let end = header[..MAX_PATH]
            .iter()
            .position(|b| *b == 0)
            .unwrap_or(MAX_PATH);
        let path = String::from_utf8(header[..end].to_vec())
            .map_err(|_| invalid("a member path is not text"))?;
        let size = std::str::from_utf8(&header[124..135])
            .ok()
            .and_then(|t| u64::from_str_radix(t, 8).ok())
            .ok_or_else(|| invalid("unreadable member size"))?;
        self.unread = size;
        self.padding = (BLOCK - (size as usize % BLOCK)) % BLOCK;
        Ok(Some(Entry { path, size }))
    }

    /// The data of the current member, to read exactly.
    pub(super) fn data(&mut self) -> impl Read + '_ {
        Counted {
            inner: (&mut self.input).take(self.unread),
            unread: &mut self.unread,
        }
    }

    fn skip_rest(&mut self) -> io::Result<()> {
        let total = self.unread + self.padding as u64;
        let skipped = io::copy(&mut (&mut self.input).take(total), &mut io::sink())?;
        if skipped != total {
            return Err(invalid("the archive ends inside a member"));
        }
        self.unread = 0;
        self.padding = 0;
        Ok(())
    }
}

/// Counts what a reader hands out against what the member still holds.
struct Counted<'a, R: Read> {
    inner: io::Take<&'a mut R>,
    unread: &'a mut u64,
}

impl<R: Read> Read for Counted<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        *self.unread -= n as u64;
        Ok(n)
    }
}
