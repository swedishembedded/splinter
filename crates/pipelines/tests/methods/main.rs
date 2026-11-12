// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in falsifiable smoke tests of
// fine-tuning, preference, reinforcement-learning and retrieval pipelines then you can procure our services by
// sending an email to info@swedishembedded.com.

//! One falsifiable smoke per learning method, end to end through Splinter's own stages onto brain, on a toy model
//! that trains and samples on the CPU in seconds (`make test/methods` runs them and prints the table).
//!
//! Every smoke shares one world ([`world`]): a three-letter cipher the toy model has to learn, a verifier that
//! knows the cipher, and prompts split into those a method may learn from and those it is scored on. A method counts
//! only if its held-out accuracy moves, with the numbers before and after printed as a `METHOD|` line
//! ([`report`]); a method that cannot move a number is reported as a failure, never as a pass.

mod lab;
mod report;
mod world;

mod cpt;
mod distill;
mod dpo;
mod grpo;
mod process;
mod raft;
mod retrieval;
mod search;
mod selection;
mod selfplay;
mod sft;
mod skills;
