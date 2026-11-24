// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that the training records one system writes are the records
// another system's trainer reads, for its clients. If your team needs expertise in training-data contracts between
// agent runtimes and trainers then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The experiences the contract is held with: a passing tool-using trajectory, a failing attempt at the same task,
//! a conversation the person corrected, and a multi-step run. Each carries the action texts the expectations are
//! computed from.

#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::experience::Experience;
use splinter_data::Corpus;

use crate::common::{answered, call, experience, say, task, trajectory, user, verdict};

/// What one agent step does, as the experience says it: the text it answers with, or the call it makes.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Says(String),
    /// A reply the other speaker's next turn corrects: shown, never trained on.
    Corrected(String),
    Calls {
        name: String,
        arguments: String,
    },
}

impl Action {
    /// The text brain's template writes for the action: its content, then each call's name and arguments.
    pub fn rendered(&self) -> String {
        match self {
            Self::Says(text) | Self::Corrected(text) => text.clone(),
            Self::Calls { name, arguments } => format!("{name}{arguments}"),
        }
    }
}

/// An experience, its verdict and the agent actions of its trajectory in order.
pub struct Attempt {
    pub experience: Experience,
    pub outcome: Outcome,
    pub actions: Vec<Action>,
}

fn shell(command: &str) -> Action {
    Action::Calls {
        name: "shell".into(),
        arguments: json!({"command": command}).to_string(),
    }
}

/// The attempts: all are graded at the strength the views admit.
pub fn attempts() -> Vec<Attempt> {
    let listing = task("qa", "List the files of the project.", vec![]);
    let port = task("qa", "Which port does the dashboard use?", vec![]);
    let build = task("qa", "Build the project and report the result.", vec![]);

    let passing = experience(
        &listing,
        trajectory(vec![
            user(1, &listing.instruction),
            call(
                2,
                "c1",
                "shell",
                json!({"command": "ls"}),
                "part.py notes.md",
            ),
            say(3, "The project holds part.py and notes.md."),
        ]),
        Some("The project holds part.py and notes.md."),
        "2026-11-01T00:00:00.000Z",
    );
    let failing = answered(
        &listing,
        "The project is empty.",
        "2026-11-01T00:00:01.000Z",
    );
    let corrected = experience(
        &port,
        trajectory(vec![
            user(1, &port.instruction),
            say(2, "It uses port 8080."),
            user(3, "No, it uses port 9090."),
            say(4, "Port 9090, understood."),
        ]),
        Some("Port 9090, understood."),
        "2026-11-01T00:00:02.000Z",
    );
    let multi_step = experience(
        &build,
        trajectory(vec![
            user(1, &build.instruction),
            call(
                2,
                "c1",
                "shell",
                json!({"command": "make"}),
                "built 3 targets",
            ),
            call(
                3,
                "c2",
                "shell",
                json!({"command": "make test"}),
                "12 tests passed",
            ),
            say(4, "It builds and all tests pass."),
        ]),
        Some("It builds and all tests pass."),
        "2026-11-01T00:00:03.000Z",
    );

    vec![
        Attempt {
            experience: passing,
            outcome: Outcome::Pass,
            actions: vec![
                shell("ls"),
                Action::Says("The project holds part.py and notes.md.".into()),
            ],
        },
        Attempt {
            experience: failing,
            outcome: Outcome::Fail,
            actions: vec![Action::Says("The project is empty.".into())],
        },
        Attempt {
            experience: corrected,
            outcome: Outcome::Pass,
            actions: vec![
                Action::Corrected("It uses port 8080.".into()),
                Action::Says("Port 9090, understood.".into()),
            ],
        },
        Attempt {
            experience: multi_step,
            outcome: Outcome::Pass,
            actions: vec![
                shell("make"),
                shell("make test"),
                Action::Says("It builds and all tests pass.".into()),
            ],
        },
    ]
}

/// The attempts as a corpus, each graded by an executable verifier.
pub fn corpus_of(attempts: &[Attempt]) -> Corpus {
    let mut corpus = Corpus::new();
    for attempt in attempts {
        let note = verdict(&attempt.experience, attempt.outcome, Strength::Executable);
        corpus
            .insert(attempt.experience.clone(), vec![note])
            .unwrap();
    }
    corpus
}
