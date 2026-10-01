// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A versioned Experience Graph database.
//!
//! Raw experience is written once and never changed. Evaluations, credit,
//! skills and training views are interpretations derived from it, so a better
//! algorithm can reinterpret old experience without any of it having been
//! thrown away.
//!
//! Everything lives in immutable, content-addressed files under one root the
//! caller passes in. A file is part of the database only when an immutable
//! manifest names it, so a crashed writer leaves an orphan nobody reads, and a
//! reader pinned to a manifest sees the same data for as long as it holds it.
//! There are no file locks on the write path: each writer owns a unique id.
#![warn(missing_docs)]
