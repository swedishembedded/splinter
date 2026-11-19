// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// crash-safe append-only logs, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The pack journal's entries: how each is framed on disk, how a stretch of
//! journal is split into whole entries, and the index they build.

use std::collections::{BTreeMap, HashMap};

use super::MAX_ENTRY;
use crate::backend::{Key, Kind};

/// The journal's stable code for a kind. Never renumber: journals on disk
/// hold these.
fn kind_code(kind: Kind) -> u8 {
    match kind {
        Kind::Segment => 1,
        Kind::BlobPack => 2,
        Kind::Manifest => 3,
        Kind::Index => 4,
        Kind::Spool => 5,
        Kind::Cache => 6,
        Kind::Trash => 7,
        Kind::Signal => 8,
        Kind::Ref => 9,
        Kind::Pin => 10,
        Kind::Overlay => 11,
    }
}

fn kind_of_code(code: u8) -> Option<Kind> {
    Kind::ALL.into_iter().find(|kind| kind_code(*kind) == code)
}

/// Where an object's bytes lie, and what they hash to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Location {
    pub(super) pack: u32,
    pub(super) offset: u64,
    pub(super) len: u64,
    pub(super) modified_ns: u64,
    pub(super) digest: [u8; 32],
}

/// One journal entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Entry {
    /// The object's bytes are at `location`; replaces what was there.
    Put { key: Key, location: Location },
    /// The object is gone.
    Remove { key: Key },
    /// The object was written again now.
    Touch { key: Key, at_ns: u64 },
    /// The object now lives under another key.
    Move { from: Key, to: Key },
}

const PUT: u8 = 1;
const REMOVE: u8 = 2;
const TOUCH: u8 = 3;
const MOVE: u8 = 4;

fn put_key(out: &mut Vec<u8>, key: &Key) {
    out.push(kind_code(key.kind()));
    let name = key.name().as_bytes();
    // Key names are short validated ASCII; MAX_ENTRY bounds them anyway.
    out.extend_from_slice(&u16::try_from(name.len()).unwrap_or(u16::MAX).to_le_bytes());
    out.extend_from_slice(name);
}

impl Entry {
    /// The entry framed for the journal: its length, its body and the
    /// body's CRC-32.
    pub(super) fn frame(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(96);
        match self {
            Entry::Put { key, location } => {
                body.push(PUT);
                put_key(&mut body, key);
                body.extend_from_slice(&location.pack.to_le_bytes());
                body.extend_from_slice(&location.offset.to_le_bytes());
                body.extend_from_slice(&location.len.to_le_bytes());
                body.extend_from_slice(&location.modified_ns.to_le_bytes());
                body.extend_from_slice(&location.digest);
            }
            Entry::Remove { key } => {
                body.push(REMOVE);
                put_key(&mut body, key);
            }
            Entry::Touch { key, at_ns } => {
                body.push(TOUCH);
                put_key(&mut body, key);
                body.extend_from_slice(&at_ns.to_le_bytes());
            }
            Entry::Move { from, to } => {
                body.push(MOVE);
                put_key(&mut body, from);
                put_key(&mut body, to);
            }
        }
        let mut frame = Vec::with_capacity(body.len() + 8);
        frame.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_le_bytes());
        frame.extend_from_slice(&body);
        frame.extend_from_slice(&crc32fast::hash(&body).to_le_bytes());
        frame
    }

    pub(super) fn decode(body: &[u8]) -> Option<Self> {
        let mut cursor = Cursor { bytes: body, at: 0 };
        let op = cursor.u8()?;
        let entry = match op {
            PUT => {
                let key = cursor.key()?;
                let location = Location {
                    pack: u32::from_le_bytes(cursor.take(4)?.try_into().ok()?),
                    offset: cursor.u64()?,
                    len: cursor.u64()?,
                    modified_ns: cursor.u64()?,
                    digest: cursor.take(32)?.try_into().ok()?,
                };
                Entry::Put { key, location }
            }
            REMOVE => Entry::Remove { key: cursor.key()? },
            TOUCH => Entry::Touch {
                key: cursor.key()?,
                at_ns: cursor.u64()?,
            },
            MOVE => Entry::Move {
                from: cursor.key()?,
                to: cursor.key()?,
            },
            _ => return None,
        };
        (cursor.at == body.len()).then_some(entry)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let slice = self.bytes.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(slice)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn key(&mut self) -> Option<Key> {
        let kind = kind_of_code(self.u8()?)?;
        let len = u16::from_le_bytes(self.take(2)?.try_into().ok()?);
        let name = std::str::from_utf8(self.take(usize::from(len))?).ok()?;
        Key::new(kind, name).ok()
    }
}

/// Where parsing a stretch of journal stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stop {
    /// At the end of what was read.
    End,
    /// At an entry that is not all there yet: being written, or torn.
    Incomplete,
    /// At a whole-length entry that does not check out.
    Damaged {
        /// Whether it is the last thing in the journal, which a torn write
        /// can leave; damage with entries after it is not a torn write.
        last: bool,
    },
}

/// Splits `bytes` into entries, stopping at the first that is not whole.
/// Returns the entries, how many bytes they took and why it stopped.
pub(super) fn parse(bytes: &[u8]) -> (Vec<Entry>, usize, Stop) {
    let mut entries = Vec::new();
    let mut at = 0;
    loop {
        let rest = &bytes[at..];
        if rest.is_empty() {
            return (entries, at, Stop::End);
        }
        let Some(len) = rest.get(..4).map(|b| {
            usize::try_from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(usize::MAX)
        }) else {
            return (entries, at, Stop::Incomplete);
        };
        if len == 0 || len > MAX_ENTRY {
            // A crash can leave the tail zero-filled: torn, not damaged.
            let last = rest.iter().all(|b| *b == 0);
            return (entries, at, Stop::Damaged { last });
        }
        let frame = 4 + len + 4;
        if rest.len() < frame {
            return (entries, at, Stop::Incomplete);
        }
        let body = &rest[4..4 + len];
        let crc = u32::from_le_bytes([rest[4 + len], rest[5 + len], rest[6 + len], rest[7 + len]]);
        let decoded = (crc32fast::hash(body) == crc)
            .then(|| Entry::decode(body))
            .flatten();
        match decoded {
            Some(entry) => {
                entries.push(entry);
                at += frame;
            }
            None => {
                return (
                    entries,
                    at,
                    Stop::Damaged {
                        last: rest[frame..].iter().all(|b| *b == 0),
                    },
                )
            }
        }
    }
}

/// What the journal says, as of the bytes applied so far.
#[derive(Debug, Default)]
pub(super) struct Index {
    pub(super) objects: HashMap<Kind, BTreeMap<String, Location>>,
    /// Journal bytes applied, the header included.
    pub(super) applied: u64,
    /// Why the last read of the journal stopped.
    pub(super) stop: Option<Stop>,
    /// The end of the last object each pack holds.
    pub(super) pack_ends: BTreeMap<u32, u64>,
}

impl Index {
    pub(super) fn get(&self, key: &Key) -> Option<Location> {
        self.objects.get(&key.kind())?.get(key.name()).copied()
    }

    pub(super) fn apply(&mut self, entry: Entry) {
        match entry {
            Entry::Put { key, location } => {
                let end = location.offset + location.len;
                let pack_end = self.pack_ends.entry(location.pack).or_default();
                *pack_end = (*pack_end).max(end);
                self.objects
                    .entry(key.kind())
                    .or_default()
                    .insert(key.name().to_owned(), location);
            }
            Entry::Remove { key } => {
                if let Some(names) = self.objects.get_mut(&key.kind()) {
                    names.remove(key.name());
                }
            }
            Entry::Touch { key, at_ns } => {
                if let Some(location) = self
                    .objects
                    .get_mut(&key.kind())
                    .and_then(|names| names.get_mut(key.name()))
                {
                    location.modified_ns = at_ns;
                }
            }
            Entry::Move { from, to } => {
                let taken = self
                    .objects
                    .get_mut(&from.kind())
                    .and_then(|names| names.remove(from.name()));
                if let Some(location) = taken {
                    self.objects
                        .entry(to.kind())
                        .or_default()
                        .insert(to.name().to_owned(), location);
                }
            }
        }
    }

    /// The pack appends go to, and where in it they start.
    pub(super) fn append_point(&self) -> (u32, u64) {
        self.pack_ends
            .last_key_value()
            .map_or((0, 0), |(pack, end)| (*pack, *end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_round_trips_through_its_frame() {
        let key = Key::new(Kind::Segment, "ab.seg").unwrap();
        let to = Key::new(Kind::Trash, "segments.ab.seg").unwrap();
        let entries = vec![
            Entry::Put {
                key: key.clone(),
                location: Location {
                    pack: 3,
                    offset: 9,
                    len: 27,
                    modified_ns: 81,
                    digest: [7; 32],
                },
            },
            Entry::Touch {
                key: key.clone(),
                at_ns: 5,
            },
            Entry::Move {
                from: key.clone(),
                to,
            },
            Entry::Remove { key },
        ];
        let bytes: Vec<u8> = entries.iter().flat_map(Entry::frame).collect();
        assert_eq!(parse(&bytes), (entries, bytes.len(), Stop::End));
    }

    #[test]
    fn parsing_stops_at_a_torn_entry_and_tells_it_from_damage_inside() {
        let key = Key::new(Kind::Signal, "done/a").unwrap();
        let one = Entry::Remove { key }.frame();
        let mut torn = one.clone();
        torn.extend_from_slice(&one[..one.len() - 3]);
        assert_eq!(parse(&torn).2, Stop::Incomplete);
        assert_eq!(parse(&torn).1, one.len());

        let mut damaged = one.clone();
        damaged.extend_from_slice(&one);
        let last = damaged.len() - 1;
        damaged[last] ^= 0xff;
        assert_eq!(parse(&damaged).2, Stop::Damaged { last: true });

        let mut inside = one.clone();
        inside[5] ^= 0xff;
        inside.extend_from_slice(&one);
        assert_eq!(parse(&inside).2, Stop::Damaged { last: false });
    }
}
