// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reconstruction benchmark's specs: the gate on a briefing, the replies, the scores and the comparison.

use super::*;
use crate::testkit::{doc, helper, words as distinct};

const LETTER: &str = "My dear Sir, I have received your Favor of the tenth, and I am of Opinion that the Towns ought not to wait upon the Ministry. Let each Town choose a Committee to write to the rest, that the Sense of the People may be known before the Assembly meets. I beg you will send me what Number of Towns have already acted.";

fn letter() -> Document {
    doc("l1", 1773, "James Warren", LETTER)
}

fn good() -> Drafted {
    Drafted {
        situation: format!("In the autumn of 1773 the Province is uneasy over the Governor's salary and the shipments of tea expected at the port. A correspondent in Plymouth reports that his neighbours are divided on whether to petition or to act, and asks how the towns should proceed. {}", distinct("ctx", 20)),
        request: "Advise your correspondent how the towns should proceed.".into(),
        key_points: vec![
            KeyPoint { point: "he advises not waiting on the ministry".into(), quote: "the Towns ought not to wait upon the Ministry".into() },
            KeyPoint { point: "he proposes committees in each town to write to the rest".into(), quote: "Let each Town choose a Committee to write to the rest".into() },
            KeyPoint { point: "he asks how many towns have already acted".into(), quote: "send me what Number of Towns have already acted".into() },
        ],
    }
}

#[test]
fn a_situation_told_in_the_helpers_words_with_a_grounded_rubric_passes() {
    assert_eq!(gate(&good(), &letter()), Ok(()));
}

#[test]
fn a_rubric_point_whose_quotation_is_not_in_the_letter_is_refused() {
    let mut d = good();
    d.key_points[1].quote = "the committees shall assemble at Boston in December".into();
    let why = gate(&d, &letter()).unwrap_err();
    assert!(why.contains("not in the letter"), "{why}");
}

#[test]
fn a_quotation_too_short_to_ground_a_point_is_refused() {
    let mut d = good();
    d.key_points[0].quote = "not wait".into();
    assert!(gate(&d, &letter()).unwrap_err().contains("short"));
}

#[test]
fn the_rubric_names_three_to_six_things() {
    let mut few = good();
    few.key_points.truncate(2);
    assert!(gate(&few, &letter()).unwrap_err().contains("key points"));
    let mut many = good();
    let extra = many.key_points[0].clone();
    many.key_points.extend(std::iter::repeat_n(extra, 5));
    assert!(gate(&many, &letter()).unwrap_err().contains("key points"));
}

#[test]
fn a_situation_that_repeats_the_letters_words_gives_the_answer_away_and_is_refused() {
    let mut d = good();
    d.situation.push_str(" Let each Town choose a Committee to write to the rest, that the Sense of the People may be known.");
    assert!(gate(&d, &letter())
        .unwrap_err()
        .contains("letter's own words"));
    let mut request = good();
    request.request = "Let each Town choose a Committee to write to the rest of them".into();
    assert!(gate(&request, &letter())
        .unwrap_err()
        .contains("letter's own words"));
}

#[test]
fn a_situation_too_thin_to_answer_is_refused() {
    let mut d = good();
    d.situation = "A man writes to him.".into();
    assert!(gate(&d, &letter()).unwrap_err().contains("short"));
}

#[test]
fn a_key_point_already_stated_in_the_situation_or_the_request_gives_the_answer_away() {
    let mut d = good();
    d.request =
        "Advise your correspondent whether the towns ought to wait upon the ministry or act."
            .into();
    let why = gate(&d, &letter()).unwrap_err();
    assert!(why.contains("gives away"), "{why}");
}

#[test]
fn what_the_situation_says_he_asks_or_argues_is_the_letters_content_and_is_refused() {
    for line in [
        "Adams asks the recipient to use his access to the ministers.",
        "Adams urges the towns to act together.",
        "Adams requests that the recipient recognise the loyalty of the colonists.",
    ] {
        let mut d = good();
        d.request = format!("{line} {}", d.request);
        assert!(
            gate(&d, &letter()).unwrap_err().contains("what Adams"),
            "{line}"
        );
    }
    let mut d = good();
    d.request = "A correspondent in Plymouth asks Adams how the towns should proceed.".into();
    assert_eq!(
        gate(&d, &letter()),
        Ok(()),
        "what is put to him is the stimulus, and allowed"
    );
}

#[test]
fn a_finished_briefing_that_leaks_is_found_and_one_that_does_not_is_not() {
    let clean = Briefing {
        id: "r".into(),
        doc_id: "l1".into(),
        situation: good().situation,
        request: good().request,
        key_points: good().key_points,
    };
    assert_eq!(leaks(&clean), None);
    let mut leaky = clean.clone();
    leaky.request = "Adams advises the towns to choose a committee to write to the rest.".into();
    assert!(leaks(&leaky).is_some());
}

#[test]
fn a_briefing_made_before_the_gate_that_leaks_is_briefed_again_and_a_clean_one_is_kept() {
    let letters = [letter()];
    let refs: Vec<&Document> = letters.iter().collect();
    let path = out();
    let mut leaky = Briefing {
        id: "recon-old".into(),
        doc_id: "l1".into(),
        situation: good().situation,
        request: "Adams advises the towns to choose a committee to write to the rest.".into(),
        key_points: good().key_points,
    };
    std::fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string(&BriefingResult {
                doc_id: "l1".into(),
                briefing: Some(leaky.clone()),
                error: None
            })
            .unwrap()
        ),
    )
    .unwrap();
    assert!(
        read_briefings(&path).unwrap().is_empty(),
        "a leaky briefing is not read as a briefing"
    );
    let good_json = serde_json::to_string(&good()).unwrap();
    let (h, _) = helper(&[&good_json]);
    assert_eq!(
        brief_all(&h, &refs, &path, None).unwrap(),
        1,
        "it is briefed again"
    );
    assert_eq!(read_briefings(&path).unwrap().len(), 1);
    leaky.request = good().request;
    assert_eq!(
        brief_all(&h, &refs, &path, None).unwrap(),
        0,
        "a clean briefing is not briefed again"
    );
}

#[test]
fn the_student_is_shown_the_situation_and_the_request_and_never_the_rubric() {
    let b = Briefing {
        id: "recon-1".into(),
        doc_id: "l1".into(),
        situation: good().situation,
        request: good().request,
        key_points: good().key_points,
    };
    let p = prompt(&b);
    assert!(p.contains("autumn of 1773") && p.contains("Advise your correspondent"));
    assert!(
        !p.contains("Committee to write to the rest") && !p.contains("ought not to wait"),
        "{p}"
    );
}

#[test]
fn a_briefing_is_drafted_from_the_letter_and_the_answer_is_not_in_what_the_student_sees() {
    let json = serde_json::to_string(&good()).unwrap();
    let (h, provider) = helper(&[&json]);
    let b = brief(&h, &letter()).unwrap();
    assert_eq!((b.doc_id.as_str(), b.key_points.len()), ("l1", 3));
    assert!(b.id.starts_with("recon-"));
    assert!(
        provider.sent().contains("Towns ought not to wait"),
        "the helper reads the letter"
    );
    assert!(!prompt(&b).contains("Towns ought not to wait"));
}

#[test]
fn a_draft_that_fails_the_gate_is_corrected_and_one_that_never_passes_is_an_error() {
    let mut leaky = good();
    leaky.situation.push_str(" Let each Town choose a Committee to write to the rest, that the Sense of the People may be known.");
    let (h, provider) = helper(&[
        &serde_json::to_string(&leaky).unwrap(),
        &serde_json::to_string(&good()).unwrap(),
    ]);
    assert_eq!(brief(&h, &letter()).unwrap().situation, good().situation);
    assert!(provider.requests() >= 2);
    let (stuck, _) = helper(&[&serde_json::to_string(&leaky).unwrap()]);
    assert!(brief(&stuck, &letter()).is_err());
}

// ---- replies, scores and the comparison ----

use splinter_sdk::measure::verifiers::quotation::TextIndex;

fn briefing(id: &str) -> Briefing {
    Briefing {
        id: id.into(),
        doc_id: "l1".into(),
        situation: good().situation,
        request: good().request,
        key_points: good().key_points,
    }
}

fn reply_of(text: &str) -> splinter_sdk::model::answer::Reply {
    splinter_sdk::model::answer::Reply {
        text: text.into(),
        tokens: 7,
        ..Default::default()
    }
}

fn out() -> std::path::PathBuf {
    tempfile::tempdir().unwrap().keep().join("out.jsonl")
}

#[test]
fn a_reply_is_kept_in_full_per_arm_and_a_run_resumes_where_it_stopped() {
    let briefings = [briefing("b1"), briefing("b2"), briefing("b3")];
    let path = out();
    let long = "x ".repeat(900);
    assert_eq!(
        answer_all(&briefings, &path, "base", Some(1), &mut |_| Ok(reply_of(
            &long
        )))
        .unwrap(),
        1
    );
    let first = read_replies(&path).unwrap();
    assert!(
        first[0].text.len() > 1000,
        "the reply is not cut for the record"
    );
    let mut asked = Vec::new();
    answer_all(&briefings, &path, "base", None, &mut |b| {
        asked.push(b.id.clone());
        Ok(reply_of("ok"))
    })
    .unwrap();
    assert_eq!(asked, ["b2", "b3"]);
    assert_eq!(
        answer_all(&briefings, &path, "tuned", None, &mut |_| Ok(reply_of(
            "ok"
        )))
        .unwrap(),
        3,
        "another arm answers the same briefings"
    );
}

#[test]
fn a_reply_the_model_gave_inside_its_reasoning_block_counts_and_one_that_ran_out_does_not() {
    let path = out();
    answer_all(&[briefing("b1")], &path, "base", None, &mut |_| {
        Ok(splinter_sdk::model::answer::Reply {
            thinking: "Dear Sir, I answer.".into(),
            ..Default::default()
        })
    })
    .unwrap();
    answer_all(&[briefing("b2")], &path, "base", None, &mut |_| {
        Ok(splinter_sdk::model::answer::Reply {
            thinking: "still weighing".into(),
            truncated: true,
            ..Default::default()
        })
    })
    .unwrap();
    let replies = read_replies(&path).unwrap();
    assert_eq!(replies[0].text, "Dear Sir, I answer.");
    assert_eq!(replies[1].text, "");
}

fn corpus() -> TextIndex {
    TextIndex::new([LETTER])
}

fn reply(id: &str, arm: &str, text: &str) -> Reply {
    Reply {
        briefing_id: id.into(),
        arm: arm.into(),
        text: text.into(),
        tokens: 5,
        truncated: false,
    }
}

fn judging(covered: &[bool]) -> Helper {
    let v = serde_json::json!({"covered": covered}).to_string();
    crate::testkit::responding(move |_| v.clone()).0
}

#[test]
fn a_reply_is_scored_by_the_judge_and_by_code() {
    let b = [briefing("b1")];
    let quote = "As I wrote, \"Let each Town choose a Committee to write to the rest\" always.";
    let invented = "As I wrote, \"liberty is the first gift of nature to every citizen of the commonwealth\" always, and I would use the internet.";
    let path = out();
    score_all(
        &judging(&[true, false, true]),
        &b,
        &[reply("b1", "base", invented)],
        &corpus(),
        &path,
        None,
    )
    .unwrap();
    let s = &read_scores(&path).unwrap()[0];
    assert_eq!(s.covered, [true, false, true]);
    assert_eq!((s.fabricated, s.quotations), (1, 1));
    assert_eq!(s.anachronism.as_deref(), Some("internet"));
    let real = out();
    score_all(
        &judging(&[true, true, true]),
        &b,
        &[reply("b1", "base", quote)],
        &corpus(),
        &real,
        None,
    )
    .unwrap();
    assert_eq!(
        read_scores(&real).unwrap()[0].fabricated,
        0,
        "a quotation that is in the letter is not fabricated"
    );
}

#[test]
fn an_empty_reply_scores_nothing_without_asking_the_judge() {
    let (h, provider) = helper(&["not used"]);
    let path = out();
    score_all(
        &h,
        &[briefing("b1")],
        &[reply("b1", "base", "  ")],
        &corpus(),
        &path,
        None,
    )
    .unwrap();
    assert_eq!(
        read_scores(&path).unwrap()[0].covered,
        [false, false, false]
    );
    assert_eq!(provider.requests(), 0);
}

#[test]
fn a_judge_that_fails_on_a_reply_is_recorded_retried_twice_and_then_given_up_on() {
    let (h, _) = helper(&["not json"]);
    let path = out();
    let replies = [reply("b1", "base", "Dear Sir, a reply.")];
    for _ in 0..3 {
        score_all(&h, &[briefing("b1")], &replies, &corpus(), &path, None).unwrap();
    }
    let scores = read_scores(&path).unwrap();
    assert_eq!(scores.len(), 2, "two attempts and no more");
    assert!(scores.iter().all(|s| s.error.is_some()));
}

#[test]
fn a_scored_reply_is_not_scored_again() {
    let path = out();
    let replies = [reply("b1", "base", "Dear Sir, a reply.")];
    let h = judging(&[true, true, false]);
    assert_eq!(
        score_all(&h, &[briefing("b1")], &replies, &corpus(), &path, None).unwrap(),
        1
    );
    assert_eq!(
        score_all(&h, &[briefing("b1")], &replies, &corpus(), &path, None).unwrap(),
        0
    );
}

fn scored(id: &str, covered: &[bool], fabricated: usize, anachronism: bool) -> Scored {
    Scored {
        briefing_id: id.into(),
        arm: "x".into(),
        covered: covered.to_vec(),
        fabricated,
        quotations: fabricated,
        anachronism: anachronism.then(|| "internet".to_string()),
        error: None,
    }
}

#[test]
fn two_arms_are_compared_on_the_briefings_both_were_scored_on() {
    let before = [
        scored("b1", &[true, false, false], 1, true),
        scored("b2", &[false, false, false], 0, false),
        scored("only-before", &[true, true, true], 0, false),
    ];
    let after = [
        scored("b1", &[true, true, true], 0, false),
        scored("b2", &[false, false, false], 0, false),
        Scored {
            error: Some("x".into()),
            covered: vec![],
            ..scored("b3", &[], 0, false)
        },
    ];
    let s = summarize(&before, &after);
    assert_eq!(s.n, 2);
    assert!((s.coverage_before - 1.0 / 6.0).abs() < 1e-9 && (s.coverage_after - 0.5).abs() < 1e-9);
    assert_eq!((s.gained, s.lost), (1, 0));
    assert!((s.fabricated_before - 0.5).abs() < 1e-9 && s.fabricated_after == 0.0);
    assert!((s.anachronism_before - 0.5).abs() < 1e-9 && s.anachronism_after == 0.0);
}

#[test]
fn a_judge_that_failed_its_controls_leaves_no_claim_about_coverage_but_the_code_measures_stand() {
    let s = summarize(
        &[scored("b1", &[true, false, false], 1, true)],
        &[scored("b1", &[true, true, true], 0, false)],
    );
    let failed = crate::judge::Calibration {
        real_letter: 0.4,
        other_letter: 0.6,
        briefings: 10,
    };
    let text = render(&s, &failed, "base", "tuned");
    assert!(
        text.contains("no claim is made about coverage")
            && !text.contains("Coverage of what the real letter does"),
        "{text}"
    );
    assert!(text.contains("fabricated quotation") && text.contains("after his death"));
    let passed = crate::judge::Calibration {
        real_letter: 0.9,
        other_letter: 0.1,
        briefings: 10,
    };
    let text = render(&s, &passed, "base", "tuned");
    assert!(
        text.contains("Coverage of what the real letter does") && !text.contains("no claim"),
        "{text}"
    );
}

#[test]
fn letters_are_briefed_once_a_failure_is_recorded_and_retried_a_bounded_number_of_times() {
    let letters = [letter()];
    let refs: Vec<&Document> = letters.iter().collect();
    let path = out();
    let (bad, _) = helper(&["not json"]);
    for _ in 0..3 {
        brief_all(&bad, &refs, &path, None).unwrap();
    }
    let lines = std::fs::read_to_string(&path).unwrap().lines().count();
    assert_eq!(lines, 2, "two attempts and no more");
    assert!(read_briefings(&path).unwrap().is_empty());
    let good_json = serde_json::to_string(&good()).unwrap();
    let (good_helper, _) = helper(&[&good_json]);
    let other = out();
    assert_eq!(brief_all(&good_helper, &refs, &other, None).unwrap(), 1);
    assert_eq!(
        brief_all(&good_helper, &refs, &other, None).unwrap(),
        0,
        "a briefed letter is not briefed again"
    );
    assert_eq!(read_briefings(&other).unwrap().len(), 1);
}
