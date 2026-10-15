// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Talk to a trained persona through tools you define.
//!
//! A tool is a name, a description and typed arguments, written in a file
//! ([`tools`]). A [`session::Session`] holds a conversation: each thing said
//! goes to the model with the tools offered, each call it makes is checked
//! against its tool and answered, and the model speaks again or calls again
//! until it has nothing more to call or the rounds allowed run out. Every
//! exchange is a [`trace`] of the words, the calls with their arguments and
//! the answers they got, and the tokens and time spent.
//!
//! A model that runs here samples, and each exchange is decoded from its own
//! seed ([`decoding`]), which its trace line records with the model's name.
//!
//! The same code serves a chat at the terminal and a batch of questions put
//! from a file, so what is traced in one is what is traced in the other.

pub mod decoding;
pub mod session;
pub mod tools;
pub mod trace;
