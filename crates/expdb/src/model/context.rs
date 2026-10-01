// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A conversation or other append-only history as shared pages.
//!
//! Events are grouped into immutable pages stored in the blob store; the log
//! head lists the pages and the partly filled tail. Branches of a history
//! share every full page byte for byte, so a branch costs its tail and its
//! head, not a copy of everything before it.

use serde::{Deserialize, Serialize};

use crate::blob::{BlobRef, BlobStore};
use crate::error::{Error, Result};
use crate::id::ContentId;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Head {
    page_events: usize,
    pages: Vec<BlobRef>,
    tail: Option<BlobRef>,
    len: usize,
}

/// An append-only history of events, paged.
#[derive(Debug, Clone)]
pub struct ContextLog {
    page_events: usize,
    pages: Vec<BlobRef>,
    tail: Vec<Vec<u8>>,
}

fn encode_page(events: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for event in events {
        out.extend_from_slice(&(event.len() as u32).to_le_bytes());
        out.extend_from_slice(event);
    }
    out
}

fn decode_page(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut events = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let len_bytes = bytes
            .get(at..at + 4)
            .ok_or_else(|| Error::corrupt("context page", "truncated length"))?;
        let len = u32::from_le_bytes(len_bytes.try_into().unwrap_or([0; 4])) as usize;
        let event = bytes
            .get(at + 4..at + 4 + len)
            .ok_or_else(|| Error::corrupt("context page", "truncated event"))?;
        events.push(event.to_vec());
        at += 4 + len;
    }
    Ok(events)
}

fn load_head(store: &BlobStore, head: ContentId) -> Result<Head> {
    let bytes = store.get_by_id(&head)?;
    serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
        what: format!("context head {head}"),
        source,
    })
}

impl ContextLog {
    /// An empty log whose pages hold `page_events` events.
    pub fn new(page_events: usize) -> Self {
        Self {
            page_events: page_events.max(1),
            pages: Vec::new(),
            tail: Vec::new(),
        }
    }

    /// The number of events.
    pub fn len(&self) -> usize {
        self.pages.len() * self.page_events + self.tail.len()
    }

    /// Whether the log has no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends an event; a full page is stored at once.
    pub fn append(&mut self, store: &mut BlobStore, event: &[u8]) -> Result<()> {
        self.tail.push(event.to_vec());
        if self.tail.len() == self.page_events {
            let page = store.put(&encode_page(&self.tail))?;
            self.pages.push(page);
            self.tail.clear();
        }
        Ok(())
    }

    /// Stores the log's head and returns its id, which a state names as the
    /// part holding the history.
    pub fn commit(&self, store: &mut BlobStore) -> Result<ContentId> {
        let tail = if self.tail.is_empty() {
            None
        } else {
            Some(store.put(&encode_page(&self.tail))?)
        };
        let head = Head {
            page_events: self.page_events,
            pages: self.pages.clone(),
            tail,
            len: self.len(),
        };
        let bytes = serde_json::to_vec(&head).map_err(|source| Error::Encode {
            what: "context head",
            source,
        })?;
        Ok(store.put(&bytes)?.id)
    }

    /// Resumes a committed log, to continue or to branch it.
    pub fn load(store: &BlobStore, head: ContentId) -> Result<Self> {
        let head = load_head(store, head)?;
        let tail = match head.tail {
            Some(tail) => decode_page(&store.get(&tail)?)?,
            None => Vec::new(),
        };
        Ok(Self {
            page_events: head.page_events,
            pages: head.pages,
            tail,
        })
    }

    /// The first `upto` events of a committed log, reading only the pages
    /// that hold them.
    pub fn read(store: &BlobStore, head: ContentId, upto: usize) -> Result<Vec<Vec<u8>>> {
        let head = load_head(store, head)?;
        if upto > head.len {
            return Err(Error::invalid(
                "context length",
                format!("asked for {upto} events of {}", head.len),
            ));
        }
        let mut events = Vec::with_capacity(upto);
        for page in &head.pages {
            if events.len() >= upto {
                break;
            }
            events.extend(decode_page(&store.get(page)?)?);
        }
        if events.len() < upto {
            if let Some(tail) = head.tail {
                events.extend(decode_page(&store.get(&tail)?)?);
            }
        }
        events.truncate(upto);
        Ok(events)
    }
}
