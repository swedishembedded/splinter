// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: what was read out of the articles becomes training records, one
//! decision through each tool a model is taught with; and a model is examined
//! by putting each situation to it with its tool and reading the call it makes,
//! picking up where an interrupted examination stopped.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use synthetic_author::arms::{ask, run_items};
use synthetic_author::dataset::records_for;
use synthetic_author::exam::items;
use synthetic_author::mine::Verified;
use synthetic_author::persona::Persona;
use synthetic_author::scenario::Archetype;
use synthetic_author::tools::TRAINING;

fn decision(case: &str, chose: Archetype, offer: bool, reasons: &[&str]) -> Verified {
    Verified {
        case: case.into(),
        briefing:
            "A matter of the winter stores, which the community must settle before the roads close."
                .into(),
        offer_made: offer,
        chose,
        reasons: reasons.iter().map(|r| (*r).to_string()).collect(),
        dropped: 0,
    }
}

#[test]
fn a_decision_becomes_one_record_for_each_tool_a_model_is_taught_with() {
    let d = decision(
        "the Dunmere granary",
        Archetype::DeferToOutsider,
        true,
        &["We took the help."],
    );
    let records = records_for(7, "a0:the Dunmere granary", &d);
    assert_eq!(records.len(), TRAINING.len());
    let names: Vec<_> = records
        .iter()
        .map(|r| {
            r["messages"][2]["tool_calls"][0]["function"]["name"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    let expected: Vec<_> = TRAINING.iter().map(|s| s.name.to_string()).collect();
    assert_eq!(names, expected);
    for r in &records {
        assert_eq!(r["metadata"]["group"], "a0:the Dunmere granary");
        let user = r["messages"][1]["content"].as_str().unwrap();
        assert!(user.contains("accept the offer of outside help"));
        let args: Value = serde_json::from_str(
            r["messages"][2]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert!(args.to_string().contains("We took the help."));
    }
    // The same decision is put the same way every time.
    assert_eq!(records, records_for(7, "a0:the Dunmere granary", &d));
}

#[test]
fn the_option_of_outside_help_is_listed_only_when_an_offer_was_made() {
    let without = decision("the Osmere ferry", Archetype::HoldAndWait, false, &[]);
    for r in records_for(7, "g", &without) {
        let user = r["messages"][1]["content"].as_str().unwrap();
        assert!(!user.contains("outside help"), "{user}");
        assert!(user.contains("wait, and do nothing yet"));
    }
}

struct Calling {
    /// What each call to the model replies with, in order.
    replies: Mutex<Vec<Vec<ResponseEvent>>>,
    seen: Mutex<Vec<CompletionRequest>>,
}

#[async_trait::async_trait]
impl ModelProvider for Calling {
    fn name(&self) -> &str {
        "calling"
    }
    fn model_name(&self) -> &str {
        "calling-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seen.lock().unwrap().push(req);
        let events = {
            let mut replies = self.replies.lock().unwrap();
            if replies.is_empty() {
                vec![ResponseEvent::Done]
            } else {
                replies.remove(0)
            }
        };
        Ok(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
    }
}

fn exam_items() -> Vec<synthetic_author::exam::Item> {
    let p = Persona::generate(6);
    items(&p, 6, 3, &Default::default())
}

#[test]
fn a_tool_call_split_across_chunks_is_read_whole_and_a_reply_without_one_is_no_call() {
    let item = &exam_items()[0];
    let first = &item.schema;
    let half =
        json!({first.choice: first.label(item.options[0], 0), first.reason: "why"}).to_string();
    let (a, b) = half.split_at(10);
    let provider: Arc<dyn ModelProvider> = Arc::new(Calling {
        replies: Mutex::new(vec![
            vec![
                ResponseEvent::ToolCall {
                    index: 0,
                    id: "c".into(),
                    name: first.name.into(),
                    arguments: a.into(),
                },
                ResponseEvent::ToolCall {
                    index: 0,
                    id: "c".into(),
                    name: String::new(),
                    arguments: b.into(),
                },
                ResponseEvent::Done,
            ],
            vec![
                ResponseEvent::TextDelta("I would wait.".into()),
                ResponseEvent::Done,
            ],
        ]),
        seen: Mutex::new(Vec::new()),
    });
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (name, args) = rt
        .block_on(ask(&provider, "system", item, 64))
        .unwrap()
        .unwrap();
    assert_eq!(name, first.name);
    assert_eq!(args[first.choice], first.label(item.options[0], 0));
    assert!(rt
        .block_on(ask(&provider, "system", item, 64))
        .unwrap()
        .is_none());
}

/// A model that always takes the first option of the tool it is offered.
struct FirstOption;

#[async_trait::async_trait]
impl ModelProvider for FirstOption {
    fn name(&self) -> &str {
        "first"
    }
    fn model_name(&self) -> &str {
        "first-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let tool = &req.tools[0];
        let choice = tool.parameters["required"][0].as_str().unwrap().to_string();
        let label = tool.parameters["properties"][&choice]["enum"][0].clone();
        let args = json!({ choice: label, "x": "y" });
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::ToolCall {
                index: 0,
                id: "c".into(),
                name: tool.name.clone(),
                arguments: args.to_string(),
            }),
            Ok(ResponseEvent::Done),
        ])))
    }
}

#[test]
fn an_examination_grades_every_item_and_picks_up_where_it_stopped() {
    let all = exam_items();
    let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
    let dir = std::env::temp_dir().join(format!("synthetic-author-lab-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("arm.jsonl");
    let half = all.len() / 2;
    let first = run_items(&provider, "system", &all[..half], &out, 64).unwrap();
    assert_eq!(first.len(), half);
    let again = run_items(&provider, "system", &all, &out, 64).unwrap();
    assert_eq!(
        again.len(),
        all.len(),
        "all outcomes, the first half read back"
    );
    let lines = std::fs::read_to_string(&out).unwrap().lines().count();
    assert_eq!(lines, all.len(), "nothing was asked twice");
    let right = again.iter().filter(|o| o.correct).count();
    let expected = all.iter().filter(|i| i.expected == i.options[0]).count();
    assert_eq!(right, expected);
}

fn outcome(
    id: usize,
    correct: bool,
    topic_seen: bool,
    combo_seen: bool,
    tool_seen: bool,
) -> synthetic_author::arms::Outcome {
    synthetic_author::arms::Outcome {
        id: format!("i{id}"),
        correct,
        topic_seen,
        combo_seen,
        tool_seen,
        call: None,
    }
}

#[test]
fn a_report_gives_each_arm_s_accuracy_by_kind_and_a_paired_test_between_arms() {
    use synthetic_author::report::{accuracy, render};
    // 40 items: the tuned arm gets 36 right, the base 16, the base's right ones all among the tuned's.
    let kinds = |i: usize| (i.is_multiple_of(2), i % 4 < 2, i % 8 < 4);
    let tuned: Vec<_> = (0..40)
        .map(|i| {
            let (t, c, s) = kinds(i);
            outcome(i, i < 36, t, c, s)
        })
        .collect();
    let base: Vec<_> = (0..40)
        .map(|i| {
            let (t, c, s) = kinds(i);
            outcome(i, i < 16, t, c, s)
        })
        .collect();
    let a = accuracy(&tuned);
    assert_eq!((a.right, a.n), (36, 40));
    assert!(a.low < 0.9 && 0.9 < a.high, "{a:?}");
    let text = render(&[("base", base), ("tuned", tuned)], &[("tuned", "base")]);
    assert!(text.contains("tuned") && text.contains("0.900"), "{text}");
    assert!(text.contains("base") && text.contains("0.400"), "{text}");
    // 20 items the tuned arm got right and the base did not, none the other way.
    assert!(text.contains("tuned over base: 20 of 20"), "{text}");
    assert!(text.contains("p = "), "{text}");
}
