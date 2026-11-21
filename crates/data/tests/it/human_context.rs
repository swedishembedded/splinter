// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: an experience a person guided (a user step typed during the run,
//! `extra.sven.origin` `human`) is never a training candidate: the model
//! produced its later tokens with the person's message in context, which a
//! student never sees. The corpus takes it out and says how many.

use serde_json::json;
use splinter_data::{has_human_turn, Corpus};

use crate::common::{experience, say, task, trajectory, user};

fn typed(step: u64, text: &str) -> atif::TraceStep {
    let mut s = user(step, text);
    s.extra = Some(json!({"sven": {"origin": "human"}}));
    s
}

#[test]
fn a_guided_experience_is_withheld_and_an_unguided_one_kept() {
    let t = task("qa", "Write a.txt", vec![]);
    let guided = trajectory(vec![
        user(1, "Write a.txt"),
        say(2, "Wrote ho."),
        typed(3, "It must say hi."),
        say(4, "Fixed."),
    ]);
    // A second user turn alone (a dialogue, a workspace's feedback) is not a
    // person's guidance unless the run says so.
    let dialogue = trajectory(vec![
        user(1, "Write a.txt"),
        say(2, "Which text?"),
        user(3, "hi"),
        say(4, "Done."),
    ]);
    assert!(has_human_turn(&guided));
    assert!(!has_human_turn(&dialogue));

    let mut corpus = Corpus::new();
    corpus
        .insert(experience(&t, guided, None, "2026-11-21T00:00:00Z"), vec![])
        .unwrap();
    let kept = corpus
        .insert(
            experience(&t, dialogue, None, "2026-11-21T00:00:01Z"),
            vec![],
        )
        .unwrap();
    assert_eq!(corpus.withhold_human_context(), 1);
    assert_eq!(corpus.entries().len(), 1);
    assert!(corpus.get(&kept).is_some());
}
