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
use synthetic_author::arms::{ask, run_items, run_items_with};
use synthetic_author::dataset::records_for;
use synthetic_author::exam::items;
use synthetic_author::mine::{MinedFacts, Verified};
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
        facts: MinedFacts::default(),
        chose,
        open: if offer {
            vec![
                Archetype::ActAlone,
                Archetype::DeferToOutsider,
                Archetype::HoldAndWait,
            ]
        } else {
            vec![
                Archetype::ActAlone,
                Archetype::ConsultThenAct,
                Archetype::HoldAndWait,
            ]
        },
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
fn a_record_lists_exactly_the_options_the_reader_found_open() {
    let without = decision(
        "the Osmere ferry",
        Archetype::HoldAndWait,
        false,
        &["We waited for the thaw."],
    );
    for r in records_for(7, "g", &without) {
        let user = r["messages"][1]["content"].as_str().unwrap();
        assert!(!user.contains("outside help"), "{user}");
        assert_eq!(user.matches("\n- ").count(), 3, "{user}");
        assert!(user.contains("wait, and do nothing yet"));
    }
    // Four ways read are four ways listed, and no way unread is added.
    let mut four = decision(
        "the Osmere ferry",
        Archetype::HoldAndWait,
        true,
        &["We waited for the thaw."],
    );
    four.open.push(Archetype::ConsultThenAct);
    for r in records_for(7, "g", &four) {
        let user = r["messages"][1]["content"].as_str().unwrap();
        assert_eq!(user.matches("\n- ").count(), 4, "{user}");
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
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("synthetic-author-lab");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("arm.jsonl");
    let half = all.len() / 2;
    let first = run_items(&provider, "system", &all[..half], &out, 64, "test-model").unwrap();
    assert_eq!(first.len(), half);
    let again = run_items(&provider, "system", &all, &out, 64, "test-model").unwrap();
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

#[test]
fn an_item_written_twice_by_overlapping_runs_is_counted_once() {
    let all = exam_items();
    let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("synthetic-author-dup");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("arm.jsonl");
    run_items(&provider, "system", &all, &out, 64, "test-model").unwrap();
    let once = std::fs::read_to_string(&out).unwrap();
    let doubled = format!("{once}{once}");
    std::fs::write(&out, doubled).unwrap();
    let read = run_items(&provider, "system", &all, &out, 64, "test-model").unwrap();
    assert_eq!(
        read.len(),
        all.len(),
        "each item once, however often written"
    );
}

#[test]
fn where_two_authors_would_decide_differently_a_model_is_counted_with_the_one_it_followed() {
    use synthetic_author::report::divergence;
    let persona = Persona::generate(6);
    let all = items(&persona, 6, 3, &Default::default());
    let provider: Arc<dyn ModelProvider> = Arc::new(FirstOption);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("synthetic-author-div");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let outcomes = run_items(
        &provider,
        "system",
        &all,
        &dir.join("arm.jsonl"),
        64,
        "test-model",
    )
    .unwrap();

    let itself = divergence(&all, &outcomes, &persona);
    assert_eq!(itself.differing, 0, "an author never differs from itself");

    let rival = persona.opposite();
    let d = divergence(&all, &outcomes, &rival);
    assert!(
        d.differing > 0,
        "the opposite author decides differently somewhere"
    );
    assert_eq!(d.own + d.rival + d.neither, d.differing);
    let own_right = all
        .iter()
        .zip(&outcomes)
        .filter(|(i, o)| {
            o.correct
                && rival.decide(&i.scenario).determined
                && rival.decide(&i.scenario).choice != i.expected
        })
        .count();
    assert_eq!(
        d.own, own_right,
        "the model's own author's answers are the graded-correct ones"
    );
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
        combo_trained: None,
        tool_seen,
        call: None,
        item: String::new(),
        model: String::new(),
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
    let text = render(
        &[("base", base), ("tuned", tuned)],
        &[("tuned", "base")],
        &[],
    );
    assert!(text.contains("tuned") && text.contains("0.900"), "{text}");
    assert!(text.contains("base") && text.contains("0.400"), "{text}");
    // 20 items the tuned arm got right and the base did not, none the other way.
    assert!(text.contains("tuned over base: 20 of 20"), "{text}");
    assert!(text.contains("p = "), "{text}");
}

fn full_facts() -> MinedFacts {
    MinedFacts {
        exceeds_authority: Some(true),
        urgent: Some(false),
        binds_future: Some(true),
        irreversible: Some(false),
        members_harmed: Some(true),
    }
}

#[test]
fn a_decision_whose_six_facts_were_all_read_also_makes_a_record_in_the_examination_s_own_words() {
    let mut d = decision(
        "the Dunmere granary",
        Archetype::DeferToOutsider,
        true,
        &["We took the help."],
    );
    d.facts = full_facts();
    let records = records_for(7, "g", &d);
    assert_eq!(
        records.len(),
        2 * TRAINING.len(),
        "one in the reader's briefing, one in the facts' words"
    );
    let users: Vec<String> = records
        .iter()
        .map(|r| r["messages"][1]["content"].as_str().unwrap().to_string())
        .collect();
    // The facts' words state each fact the reader found, and nothing it did not.
    let templated = users
        .iter()
        .find(|u| u.contains("gives no power"))
        .expect("a templated record");
    assert!(
        templated.contains("Whatever is settled will bind the community for a generation"),
        "{templated}"
    );
    assert!(
        templated.contains("There is no pressing hurry"),
        "{templated}"
    );
    assert!(
        templated.contains("Several families are already suffering"),
        "{templated}"
    );
}

#[test]
fn a_decision_with_a_fact_unread_makes_only_the_records_in_the_readers_briefing() {
    let mut d = decision(
        "the Osmere ferry",
        Archetype::HoldAndWait,
        false,
        &["We waited for the thaw."],
    );
    d.facts = MinedFacts {
        urgent: None,
        ..full_facts()
    };
    assert_eq!(records_for(7, "g", &d).len(), TRAINING.len());
}

#[test]
fn the_retrieval_arm_hands_the_model_the_paragraphs_that_share_the_most_with_the_situation() {
    use synthetic_author::cache::StoredArticle;
    use synthetic_author::retrieval::Notes;
    let item = exam_items().remove(0);
    let tells = format!(
        "In the matter of {} the steward weighed the cost and settled it by waiting for the neighbours to speak first.",
        item.case
    );
    let elsewhere = "The harvest festival was held on the lower field and the children sang until the evening came down over the hills.";
    let article = |index: usize, body: String| StoredArticle {
        index,
        topic: 0,
        title: format!("Essay {index}"),
        body,
        cases: vec![],
        attempts: 1,
    };
    let notes = Notes::new(&[
        article(0, format!("{elsewhere}\n\n{tells}")),
        article(
            1,
            format!("{elsewhere}\n\nAnother day, another field, another quarrel about the fence."),
        ),
    ]);
    let system = notes.system_for(&item, 1);
    assert!(
        system.contains(&tells),
        "the paragraph that tells the case is handed over"
    );
    assert!(
        !system.contains(elsewhere),
        "one paragraph asked for, one given"
    );
    assert!(system.starts_with(synthetic_author::prompts::SYSTEM));
}

#[test]
fn a_decision_read_from_an_essay_is_checked_against_what_the_beliefs_dictate() {
    use synthetic_author::lab::{audit, Mined, MinedResult};
    use synthetic_author::writer::plan;
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 2, 3);
    let episode = &planned[0].1[0];
    let right = persona.decide(&episode.scenario).choice;
    let wrong = episode
        .scenario
        .options
        .iter()
        .copied()
        .find(|o| *o != right)
        .unwrap();
    let mined = |case: &str, chose: Archetype| Mined {
        article: 0,
        case: case.to_string(),
        result: MinedResult::Kept(decision(case, chose, false, &[])),
    };
    let refused = Mined {
        article: 0,
        case: planned[0].1[1].case.name.clone(),
        result: MinedResult::Refused("no decision".into()),
    };
    let found = audit(
        &[mined(&episode.case.name, right), refused],
        &persona,
        seed,
        2,
    );
    assert_eq!(
        (found.kept, found.agree),
        (1, 1),
        "a refused case is not counted"
    );
    let found = audit(&[mined(&episode.case.name, wrong)], &persona, seed, 2);
    assert_eq!((found.kept, found.agree), (1, 0));
}

#[test]
fn the_audit_counts_the_decisions_whose_open_ways_are_the_ones_the_case_offered() {
    use synthetic_author::lab::{audit, Mined, MinedResult};
    use synthetic_author::writer::plan;
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 1, 3);
    let episode = &planned[0].1[0];
    let right = persona.decide(&episode.scenario).choice;
    let kept = |open: Vec<Archetype>| {
        let mut d = decision(&episode.case.name, right, false, &[]);
        d.open = open;
        Mined {
            article: 0,
            case: episode.case.name.clone(),
            result: MinedResult::Kept(d),
        }
    };
    let mut reordered = episode.scenario.options.clone();
    reordered.reverse();
    let mut padded = episode.scenario.options.clone();
    padded.extend(
        [
            Archetype::ActAlone,
            Archetype::ConsultThenAct,
            Archetype::DeferToOutsider,
            Archetype::HoldAndWait,
        ]
        .into_iter()
        .find(|o| !padded.contains(o)),
    );
    let found = audit(
        &[kept(reordered), kept(padded), kept(vec![right])],
        &persona,
        seed,
        1,
    );
    assert_eq!((found.kept, found.open_agree), (3, 1));
}

#[test]
fn a_case_name_that_two_articles_share_is_checked_against_the_article_that_told_it() {
    use synthetic_author::lab::{audit, Mined, MinedResult};
    use synthetic_author::writer::plan;
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 3, 3);
    let episode = &planned[0].1[0];
    let right = persona.decide(&episode.scenario).choice;
    let kept = |article: usize| Mined {
        article,
        case: episode.case.name.clone(),
        result: MinedResult::Kept(decision(&episode.case.name, right, false, &[])),
    };
    // The same name, read out of an article that does not tell that case.
    let found = audit(&[kept(0), kept(2)], &persona, seed, 3);
    assert_eq!(
        (found.kept, found.agree),
        (1, 1),
        "only the article that tells the case is checked against it"
    );
}

#[test]
fn an_exam_item_is_shown_as_the_model_is_asked_it_and_with_the_answer_the_beliefs_dictate() {
    use synthetic_author::lab::item_view;
    let persona = Persona::generate(6);
    let all = items(&persona, 6, 3, &Default::default());
    let item = &all[0];
    let view = item_view(item);
    assert_eq!(view["id"], item.id);
    assert_eq!(view["tool"], item.schema.name);
    assert_eq!(
        view["expected"],
        item.schema.label(
            item.expected,
            item.options
                .iter()
                .position(|o| *o == item.expected)
                .unwrap()
        )
    );
    let prompt = view["prompt"].as_str().unwrap();
    assert!(
        prompt.contains(&item.briefing),
        "the briefing is in the prompt"
    );
    assert!(
        prompt.contains(&format!("`{}`", item.schema.name)),
        "the tool is named"
    );
    assert_eq!(view["tool_seen"], item.tool_seen);
}

#[test]
fn what_the_writer_is_shown_carries_the_case_files_and_the_convictions_but_no_belief_statement() {
    use synthetic_author::writer::{plan, writer_input};
    let seed = splinter_sdk::model::seed::draw().value();
    let persona = Persona::generate(seed);
    let planned = plan(&persona, seed, 1, 3);
    let shown = writer_input(&persona, planned[0].0, &planned[0].1, 700, None);
    assert_eq!(shown["author"], persona.name);
    assert_eq!(shown["episodes"].as_array().unwrap().len(), 3);
    assert!(shown["episodes"][0]["case_file"]["case"].is_string());
    assert!(shown["episodes"][0]["you_did"].is_string());
    let text = shown.to_string();
    for belief in &persona.beliefs {
        assert!(
            !text.contains(belief.statement()),
            "the writer is given dispositions, not the statement a reader could look for"
        );
    }
}

#[test]
fn an_empty_cell_is_not_measured_and_an_arm_without_outcomes_is_named() {
    use synthetic_author::report::render;
    // Every item is of a seen topic: the cells of new topics hold nothing.
    let tuned: Vec<_> = (0..8)
        .map(|i| outcome(i, i < 6, true, i % 2 == 0, true))
        .collect();
    let text = render(
        &[("tuned", tuned)],
        &[("tuned", "base"), ("tuned", "named")],
        &["base", "named"],
    );
    let row = text.lines().find(|l| l.starts_with("| tuned")).unwrap();
    assert!(row.contains("not measured"), "{row}");
    assert!(!row.contains("0.000 (0/0)"), "{row}");
    assert!(text.contains("base: not measured"), "{text}");
    assert!(text.contains("tuned over base: not measured"), "{text}");
    assert!(text.contains("tuned over named: not measured"), "{text}");
}

#[test]
fn constant_policies_are_listed_beside_the_arms_and_compared_with_the_tuned_one() {
    use synthetic_author::exam::{draw, Spec, Trained, Version};
    use synthetic_author::report::{baselines, render, FIRST_LISTED};
    let persona = Persona::generate(6);
    let told = Default::default();
    let first = items(&persona, 6, 3, &told);
    let second = draw(
        &persona,
        6,
        &Spec {
            version: Version::V2,
            per_cell: 3,
            corpus_cases: &told,
            trained: &Trained::default(),
        },
    );
    for (exam, trained_columns) in [(&first, false), (&second, true)] {
        let mut arms = baselines(exam);
        let tuned = arms[0].1.clone();
        arms.insert(0, ("tuned", tuned));
        let text = render(&arms, &[("tuned", FIRST_LISTED)], &[]);
        assert!(text.contains(&format!("| {FIRST_LISTED} |")), "{text}");
        assert!(text.contains("| majority-option |"), "{text}");
        assert!(text.contains("tuned over first-listed: 0 of 0"), "{text}");
        assert_eq!(
            text.contains("combination trained"),
            trained_columns,
            "{text}"
        );
    }
}

/// Answers like [`FirstOption`] after a pause, counting how many requests are
/// in flight at once.
#[derive(Default)]
struct Busy {
    in_flight: std::sync::atomic::AtomicUsize,
    most: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl ModelProvider for Busy {
    fn name(&self) -> &str {
        "busy"
    }
    fn model_name(&self) -> &str {
        "busy-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        use std::sync::atomic::Ordering::SeqCst;
        let now = self.in_flight.fetch_add(1, SeqCst) + 1;
        self.most.fetch_max(now, SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        self.in_flight.fetch_sub(1, SeqCst);
        FirstOption.complete(req).await
    }
}

#[test]
fn a_batching_model_is_asked_as_many_items_at_once_as_it_has_slots_and_the_outcomes_keep_their_order(
) {
    use std::sync::atomic::Ordering::SeqCst;
    let all: Vec<_> = exam_items().into_iter().take(12).collect();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("synthetic-author-lab-slots");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for slots in [1, 4] {
        let busy = Arc::new(Busy::default());
        let provider: Arc<dyn ModelProvider> = busy.clone();
        let out = dir.join(format!("slots-{slots}.jsonl"));
        let outcomes = run_items_with(
            &provider,
            |_| "system".to_string(),
            &all,
            &out,
            64,
            "m",
            slots,
        )
        .unwrap();
        assert_eq!(busy.most.load(SeqCst), slots, "{slots} slots kept full");
        let ids: Vec<_> = outcomes.iter().map(|o| o.id.clone()).collect();
        let asked: Vec<_> = all.iter().map(|i| i.id.clone()).collect();
        assert_eq!(ids, asked, "outcomes in the order of the items");
        assert_eq!(
            std::fs::read_to_string(&out).unwrap().lines().count(),
            all.len(),
            "every answer written once"
        );
    }
}
