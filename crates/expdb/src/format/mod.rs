// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The sealed segment: records and edges in checksummed, compressed column
//! blocks, with a footer of zone maps that lets a reader skip blocks.
//!
//! ```text
//! magic | record blocks | edge blocks | footer frame | tail
//! frame  = codec[1] raw_len[4] stored_len[4] crc[4] stored bytes
//! tail   = footer_offset[8] footer_len[4] tail_crc[4] magic
//! ```

mod columns;
mod frame;
mod segment;
mod zone;

pub use columns::Block;
pub use segment::{encode_segment, seal_segment, seal_segment_in, BlockInfo, Segment, SegmentInfo};
pub use zone::Zone;
