// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for `solve`: an episode compacts its context unless
//! `--no-compaction` asks for it whole.

use super::{command, Command};

#[test]
fn a_solve_compacts_unless_told_to_keep_the_episode_whole() {
    let Command::Solve(solve) = command(&["solve", "ab12", "--solver", "local:Qwen/Qwen3-1.7B"])
    else {
        panic!("solve");
    };
    assert!(
        !solve.no_compaction,
        "an episode compacts unless told not to"
    );
    let Command::Solve(whole) = command(&[
        "solve",
        "ab12",
        "--solver",
        "local:Qwen/Qwen3-1.7B",
        "--no-compaction",
    ]) else {
        panic!("solve");
    };
    assert!(whole.no_compaction);
}
