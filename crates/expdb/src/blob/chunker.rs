// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Content-defined chunking, so inserting bytes into a large object moves
//! chunk boundaries only near the edit.

use fastcdc::v2020::FastCDC;

use crate::config::Config;

/// The `(offset, length)` of each chunk of `data`. Empty data has none.
pub(crate) fn chunks(data: &[u8], config: &Config) -> Vec<(usize, usize)> {
    FastCDC::new(data, config.chunk_min, config.chunk_avg, config.chunk_max)
        .map(|chunk| (chunk.offset, chunk.length))
        .collect()
}
