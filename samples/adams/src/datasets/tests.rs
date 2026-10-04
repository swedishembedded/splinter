// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The dataset builder's specs: records, preference pairs and the benchmark.

use super::*;
use crate::document::Period;
use crate::principles::{Found, Status};
use crate::testkit::doc;

const WORDS_OF_HIS: &str = "Let the Committee write to every Town, that the Sense of the People may be known before the Assembly meets again.";

fn principle(n: usize) -> Principle {
    Principle {
        id: format!("P-{n}"),
        description: format!(
            "When the centre will not act, he had the towns state the claim together, variant {n}."
        ),
        trigger_conditions: vec!["a grievance unanswered".into()],
        expected_behavior: vec!["write to the towns".into()],
        qualifications: vec!["not when divided".into()],
        status: Status::SourceSupported,
        support: vec![Found {
            doc_id: "d1".into(),
            quote: "Let the Committee write to every Town".into(),
            period: Period::PreRevolution,
            recipient: None,
        }],
        dropped: Vec::new(),
    }
}

fn scenario(p: &Principle, case: Case) -> Scenario {
    Scenario {
        id: format!("scenario-{}", p.id),
        principle_id: p.id.clone(),
        case,
        domain: "software".into(),
        situation: "Five regional teams want different migration dates.".into(),
        observations: vec![
            "There are five regional teams.".into(),
            "The platform group announced one date.".into(),
        ],
        request: "Advise the head of engineering.".into(),
        conditions_present: vec!["a grievance unanswered".into()],
        conditions_missing: if case.applies() {
            Vec::new()
        } else {
            vec!["whether the teams agree".into()]
        },
    }
}

const FIT: &str = "Applicability: APPLIES\n\nI would have each team put its case in writing and gather them for the platform group.\n\nGrounding:\n- SOURCE_DIRECT: \"Let the Committee write to every Town\" [d1]\n- MODERN_OBSERVATION: There are five regional teams\n- PERSONA_TRANSFER: the committee method carried to the teams";

const NON_FIT: &str = "Applicability: NEEDS_INFORMATION\n\nI cannot say yet. Do the five teams agree among themselves?\n\nGrounding:\n- SOURCE_INFERRED: my papers show I gathered opinion before acting\n- MODERN_OBSERVATION: There are five regional teams";

fn docs() -> Vec<Document> {
    vec![doc("d1", 1770, "A B", WORDS_OF_HIS)]
}

fn result(p: &Principle, case: Case, answer: &str) -> transfer::Result {
    let s = scenario(p, case);
    transfer::Result {
        principle_id: p.id.clone(),
        variant: 0,
        case,
        answer: Some(respond::Answer {
            scenario_id: s.id.clone(),
            text: answer.into(),
        }),
        scenario: Some(s),
        error: None,
    }
}

#[test]
fn a_fit_is_broken_five_ways_and_the_check_refuses_every_one() {
    let d = docs();
    let refs: Vec<&Document> = d.iter().collect();
    let s = scenario(&principle(1), Case::Clear);
    assert_eq!(respond::check(FIT, &s, &refs), Ok(()));
    let made = breaks(FIT, &s, &refs);
    let kinds: Vec<Break> = made.iter().map(|(k, _)| *k).collect();
    for kind in [
        Break::FabricatedQuote,
        Break::DroppedBlock,
        Break::FlippedApplicability,
        Break::InventedFact,
        Break::Anachronism,
    ] {
        assert!(kinds.contains(&kind), "{kind:?} in {kinds:?}");
    }
    for (kind, text) in &made {
        assert_ne!(text, FIT, "{kind:?}");
        assert!(
            respond::check(text, &s, &refs).is_err(),
            "{kind:?} must fail the check: {text}"
        );
    }
}

#[test]
fn a_non_fit_is_broken_the_same_ways_and_its_flip_says_it_applies() {
    let d = docs();
    let refs: Vec<&Document> = d.iter().collect();
    let s = scenario(&principle(1), Case::MissingPrecondition);
    let made = breaks(NON_FIT, &s, &refs);
    let flipped = made
        .iter()
        .find(|(k, _)| *k == Break::FlippedApplicability)
        .unwrap();
    assert_eq!(Applicability::of(&flipped.1), Some(Applicability::Applies));
    assert!(made
        .iter()
        .all(|(_, t)| respond::check(t, &s, &refs).is_err()));
}

#[test]
fn with_no_passages_to_quote_a_quotation_is_the_fabrication() {
    let s = scenario(&principle(1), Case::MissingPrecondition);
    let made = breaks(NON_FIT, &s, &[]);
    let fabricated = made
        .iter()
        .find(|(k, _)| *k == Break::FabricatedQuote)
        .unwrap();
    assert!(fabricated.1.contains("SOURCE_DIRECT"));
    assert!(respond::check(&fabricated.1, &s, &[]).is_err());
}

#[test]
fn the_internalized_answer_says_what_his_papers_show_and_quotes_nothing() {
    let p = principle(1);
    let text = internalized(FIT, &p).unwrap();
    assert!(
        !text.contains("SOURCE_DIRECT")
            && !text.contains("[d1]")
            && !text.contains("\"Let the Committee"),
        "{text}"
    );
    assert!(text.contains("SOURCE_INFERRED") && text.contains("I would write to the towns"));
    assert!(
        !text.contains(&p.description),
        "the line is in the first person, not the principle's third-person statement: {text}"
    );
    let s = scenario(&p, Case::Clear);
    assert_eq!(respond::check(&text, &s, &[]), Ok(()), "{text}");
}

#[test]
fn an_answer_with_nothing_to_convert_is_its_own_internalized_form() {
    assert_eq!(
        internalized(NON_FIT, &principle(1)).as_deref(),
        Some(NON_FIT)
    );
}

#[test]
fn about_a_fifth_of_principles_are_the_benchmark_whatever_the_order() {
    let ids: Vec<String> = (0..500).map(|i| format!("P-{i}")).collect();
    let n = ids.iter().filter(|i| is_benchmark(i)).count();
    assert!((60..=140).contains(&n), "{n} of 500");
    assert_eq!(is_benchmark("P-7"), is_benchmark("P-7"));
}

fn split_principles() -> (Principle, Principle) {
    let all: Vec<Principle> = (0..100).map(principle).collect();
    let bench = all.iter().find(|p| is_benchmark(&p.id)).cloned().unwrap();
    let train = all.iter().find(|p| !is_benchmark(&p.id)).cloned().unwrap();
    (train, bench)
}

#[test]
fn a_training_principle_gives_two_supervised_records_and_preference_pairs_and_no_benchmark() {
    let (train, bench) = split_principles();
    let built = build(
        &[result(&train, Case::Clear, FIT)],
        &[train.clone(), bench],
        &docs(),
    );
    assert_eq!(built.sft.len(), 2, "retrieval and internalized");
    assert!(built.benchmark.is_empty());
    assert!(built.preference.len() >= 5, "{}", built.preference.len());
    let m = built.sft[0]["messages"].as_array().unwrap();
    assert_eq!(
        (
            m[0]["train"].as_bool(),
            m[1]["train"].as_bool(),
            m[2]["train"].as_bool()
        ),
        (Some(false), Some(false), Some(true))
    );
    assert_eq!(
        m[0]["content"].as_str(),
        Some(Framing::of_record(m[1]["content"].as_str().unwrap()).system(respond::SYSTEM))
    );
    let retrieval_prompt = m[1]["content"].as_str().unwrap();
    let internal_prompt = built.sft[1]["messages"][1]["content"].as_str().unwrap();
    assert!(retrieval_prompt.contains("[d1]") && !internal_prompt.contains("[d1]"));
}

#[test]
fn a_preference_pair_has_one_prompt_a_checked_answer_and_a_broken_one() {
    let (train, bench) = split_principles();
    let built = build(
        &[result(&train, Case::Clear, FIT)],
        &[train, bench],
        &docs(),
    );
    let pair = &built.preference[0];
    assert_eq!(pair["prompt"].as_array().unwrap().len(), 2);
    assert!(
        pair["prompt"][0].get("train").is_none(),
        "the prompt is never supervised"
    );
    assert_eq!(pair["chosen"]["role"].as_str(), Some("assistant"));
    assert_ne!(pair["chosen"]["content"], pair["rejected"]["content"]);
    assert!(pair["metadata"]["break"].is_string() && pair["metadata"]["scenario_id"].is_string());
}

#[test]
fn a_benchmark_principle_gives_questions_in_both_modes_and_no_training_data() {
    let (train, bench) = split_principles();
    let built = build(
        &[result(&bench, Case::Clear, FIT)],
        &[train, bench.clone()],
        &docs(),
    );
    assert!(built.sft.is_empty() && built.preference.is_empty());
    assert_eq!(built.benchmark.len(), 2);
    let modes: Vec<Mode> = built.benchmark.iter().map(|t| t.mode).collect();
    assert!(modes.contains(&Mode::Retrieval) && modes.contains(&Mode::Internalized));
    let retrieval = built
        .benchmark
        .iter()
        .find(|t| t.mode == Mode::Retrieval)
        .unwrap();
    assert_eq!(
        (retrieval.case, retrieval.evidence_docs.as_slice()),
        (Case::Clear, ["d1".to_string()].as_slice())
    );
    assert!(retrieval.prompt.contains("[d1]"));
    let internalized = built
        .benchmark
        .iter()
        .find(|t| t.mode == Mode::Internalized)
        .unwrap();
    assert!(internalized.evidence_docs.is_empty() && !internalized.prompt.contains("[d1]"));
}

#[test]
fn an_answer_that_fails_its_check_or_a_failed_result_is_left_out_with_the_reason() {
    let (train, bench) = split_principles();
    let bad = FIT.replace(
        "Let the Committee write to every Town",
        "Liberty is the first gift of nature to every citizen",
    );
    let failed = transfer::Result {
        principle_id: train.id.clone(),
        variant: 0,
        case: Case::Clear,
        scenario: None,
        answer: None,
        error: Some("x".into()),
    };
    let built = build(
        &[result(&train, Case::Clear, &bad), failed],
        &[train, bench],
        &docs(),
    );
    assert!(built.sft.is_empty() && built.preference.is_empty() && built.benchmark.is_empty());
    assert_eq!(built.excluded.len(), 2);
    assert!(
        built.excluded[0].1.contains("quotation"),
        "{:?}",
        built.excluded
    );
}

#[test]
fn a_benchmark_question_is_graded_by_the_rules_not_by_a_judge() {
    let (_, bench) = split_principles();
    let built = build(
        &[result(&bench, Case::Clear, FIT)],
        &[bench.clone(), principle(0)],
        &docs(),
    );
    let d = docs();
    let retrieval = built
        .benchmark
        .iter()
        .find(|t| t.mode == Mode::Retrieval)
        .unwrap();
    assert!(retrieval.verdict(FIT, &d).correct);
    assert!(
        !retrieval.verdict(NON_FIT, &d).correct,
        "wrong verdict for a fit"
    );
    assert!(
        !retrieval
            .verdict(
                &FIT.replace(
                    "Let the Committee write to every Town",
                    "Liberty is the first gift of nature to every citizen"
                ),
                &d
            )
            .correct
    );
    let internalized = built
        .benchmark
        .iter()
        .find(|t| t.mode == Mode::Internalized)
        .unwrap();
    assert!(
        !internalized.verdict(FIT, &d).correct,
        "with no passages shown, quoting him is fabrication"
    );
    assert!(
        internalized
            .verdict(&internalized_of(&bench, FIT), &d)
            .correct
    );
}

fn internalized_of(p: &Principle, text: &str) -> String {
    internalized(text, p).unwrap()
}

#[test]
fn the_files_round_trip_and_the_benchmark_is_readable_as_questions() {
    let (train, bench) = split_principles();
    let built = build(
        &[
            result(&train, Case::Clear, FIT),
            result(&bench, Case::Clear, FIT),
        ],
        &[train, bench],
        &docs(),
    );
    let dir = tempfile::tempdir().unwrap();
    write_all(&built, dir.path(), BENCHMARK_FILE, EXTRA_BENCHMARK_FILE).unwrap();
    assert_eq!(
        read_benchmark(&dir.path().join("benchmark.jsonl")).unwrap(),
        built.benchmark
    );
    let pairs = std::fs::read_to_string(dir.path().join("preference-transfer.jsonl")).unwrap();
    assert_eq!(pairs.lines().count(), built.preference.len());
    assert!(pairs
        .lines()
        .all(|l| serde_json::from_str::<Value>(l).unwrap()["chosen"]["role"] == "assistant"));
    assert!(std::fs::read_to_string(dir.path().join("excluded.jsonl"))
        .unwrap()
        .is_empty());
    assert!(!dir.path().join("sft-transfer.part").exists());
}

#[test]
fn a_benchmark_question_names_its_mode_as_its_kind_and_its_case_as_its_reference() {
    use splinter_sdk::model::exam::Question;
    let (_, bench) = split_principles();
    let built = build(
        &[result(&bench, Case::SurfaceAnalogy, NON_FIT)],
        &[bench],
        &docs(),
    );
    let kinds: Vec<&str> = built.benchmark.iter().map(|t| t.kind()).collect();
    assert!(kinds.contains(&"transfer-retrieval") && kinds.contains(&"transfer-internalized"));
    assert!(built
        .benchmark
        .iter()
        .all(|t| t.reference() == "surface_analogy" && t.split() == "exam"));
}

// ---- reconstruction records ----

use crate::reconstruct::{Briefing, KeyPoint};

fn briefing_for(doc_id: &str) -> Briefing {
    Briefing {
        id: format!("recon-{doc_id}"),
        doc_id: doc_id.into(),
        situation: "A correspondent asks how the towns should proceed.".into(),
        key_points: vec![KeyPoint {
            point: "p".into(),
            quote: "q".into(),
        }],
    }
}

#[test]
fn a_short_letter_is_its_own_target_and_a_long_one_is_cut_at_a_paragraph() {
    let short = "My dear Sir,\n\nI have your favor.\n\nYours, S. A.";
    assert_eq!(letter_target(short).as_deref(), Some(short));
    let para = |p: &str| {
        (0..120)
            .map(|i| format!("{p}{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let long = format!(
        "{}\n\n{}\n\n{}\n\n{}",
        para("a"),
        para("b"),
        para("c"),
        para("d")
    );
    let target = letter_target(&long).unwrap();
    assert_eq!(
        target.matches("\n\n").count(),
        1,
        "two paragraphs of 120 words fit, the third does not"
    );
    assert!(
        target.ends_with("b119"),
        "cut between paragraphs, not inside one"
    );
}

#[test]
fn a_first_paragraph_longer_than_the_limit_is_cut_at_the_last_sentence_that_fits() {
    let sentence = |n: usize| {
        (0..30)
            .map(|i| format!("s{n}w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
            + "."
    };
    let body = (0..20).map(sentence).collect::<Vec<_>>().join(" ");
    let target = letter_target(&body).unwrap();
    assert!(
        target.ends_with('.') && target.split_whitespace().count() <= TARGET_WORDS,
        "{}",
        target.split_whitespace().count()
    );
    assert!(
        letter_target("no sentence end here ".repeat(200).as_str()).is_none(),
        "nothing whole to keep"
    );
}

#[test]
fn a_reconstruction_record_trains_on_his_letter_and_never_on_the_situation() {
    let docs = vec![doc(
        "l1",
        1773,
        "James Warren",
        "My dear Sir,\n\nLet each Town choose a Committee to write to the rest.\n\nYours.",
    )];
    let allowed: std::collections::HashSet<String> = ["l1".to_string()].into();
    let records = reconstruction_sft(&[briefing_for("l1")], &docs, &allowed);
    assert_eq!(records.len(), 1);
    let m = records[0]["messages"].as_array().unwrap();
    assert_eq!(
        (m[1]["train"].as_bool(), m[2]["train"].as_bool()),
        (Some(false), Some(true))
    );
    assert_eq!(m[0]["content"].as_str(), Some(crate::reconstruct::SYSTEM));
    assert!(m[1]["content"]
        .as_str()
        .unwrap()
        .contains("how the towns should proceed"));
    assert!(m[2]["content"]
        .as_str()
        .unwrap()
        .contains("Let each Town choose a Committee"));
}

#[test]
fn a_letter_outside_the_allowed_set_is_never_a_target() {
    let docs = vec![doc(
        "held",
        1773,
        "James Warren",
        "My dear Sir,\n\nA held-out letter.\n\nYours.",
    )];
    let none: std::collections::HashSet<String> = std::collections::HashSet::new();
    assert!(reconstruction_sft(&[briefing_for("held")], &docs, &none).is_empty());
}

#[test]
fn a_preference_pair_is_framed_as_the_supervised_record_of_the_same_prompt() {
    let (train, bench) = split_principles();
    let built = build(
        &[result(&train, Case::Clear, FIT)],
        &[train, bench],
        &docs(),
    );
    for pair in &built.preference {
        let user = pair["prompt"][1]["content"].as_str().unwrap();
        let record = built
            .sft
            .iter()
            .find(|r| r["messages"][1]["content"].as_str() == Some(user))
            .expect("every pair's prompt is a supervised prompt");
        assert_eq!(
            pair["prompt"][0]["content"],
            record["messages"][0]["content"]
        );
    }
}

#[test]
fn a_few_shot_block_shows_the_first_n_training_examples_whole_and_nothing_else() {
    let (train, bench) = split_principles();
    let built = build(
        &[result(&train, Case::Clear, FIT)],
        &[train, bench],
        &docs(),
    );
    assert_eq!(few_shot_block(&built.sft, 0), "");
    let one = few_shot_block(&built.sft, 1);
    let first = &built.sft[0]["messages"];
    assert!(one.contains(first[1]["content"].as_str().unwrap()), "{one}");
    assert!(one.contains(first[2]["content"].as_str().unwrap()), "{one}");
    assert!(
        !one.contains("Example 2."),
        "the second record is not shown"
    );
}

#[test]
fn a_benchmark_that_has_been_written_is_frozen_and_a_changed_one_goes_under_a_new_name() {
    let (train, bench) = split_principles();
    let first = build(
        &[result(&bench, Case::Clear, FIT)],
        &[train.clone(), bench.clone()],
        &docs(),
    );
    let dir = tempfile::tempdir().unwrap();
    write_all(&first, dir.path(), BENCHMARK_FILE, EXTRA_BENCHMARK_FILE).unwrap();
    write_all(&first, dir.path(), BENCHMARK_FILE, EXTRA_BENCHMARK_FILE).unwrap();

    let mut changed = build(
        &[result(&bench, Case::SurfaceAnalogy, NON_FIT)],
        &[train, bench],
        &docs(),
    );
    changed.sft.clear();
    let before = std::fs::read_to_string(dir.path().join("benchmark.jsonl")).unwrap();
    let err = write_all(&changed, dir.path(), BENCHMARK_FILE, EXTRA_BENCHMARK_FILE)
        .unwrap_err()
        .to_string();
    assert!(err.contains("frozen") && err.contains("new name"), "{err}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("benchmark.jsonl")).unwrap(),
        before,
        "the frozen benchmark is untouched"
    );
    write_all(
        &changed,
        dir.path(),
        "benchmark-v2.jsonl",
        EXTRA_BENCHMARK_FILE,
    )
    .unwrap();
    assert!(dir.path().join("benchmark-v2.jsonl").exists());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("benchmark.jsonl")).unwrap(),
        before
    );
}

const CROWDED: &str = "Applicability: APPLIES\n\nI would have each team put its own case in writing.\n\nGrounding:\n- SOURCE_DIRECT: \"Let the Committee write to every Town\" [d1]\n- MODERN_OBSERVATION: There are five regional teams\n- MODERN_OBSERVATION: The platform group announced one date\n- MODERN_OBSERVATION: There are five regional teams again\n- PERSONA_TRANSFER: the committee method carried to the teams\n";

#[test]
fn a_target_keeps_two_restatements_of_the_facts_and_every_other_line() {
    let shaped = shape(CROWDED);
    assert_eq!(shaped.matches("MODERN_OBSERVATION").count(), 2);
    assert!(
        shaped.contains("There are five regional teams\n") && shaped.contains("announced one date")
    );
    assert!(
        !shaped.contains("again"),
        "the surplus restatement goes: {shaped}"
    );
    assert!(shaped.contains("SOURCE_DIRECT") && shaped.contains("PERSONA_TRANSFER"));
    assert_eq!(shape(&shaped), shaped, "shaping twice changes nothing");
    assert_eq!(shape(FIT), FIT, "an answer within the limit is untouched");
}

#[test]
fn an_invented_fact_replaces_a_restated_one_so_the_broken_answer_is_no_longer_than_the_good_one() {
    let p = principle(1);
    let s = scenario(&p, Case::Clear);
    let d = docs();
    let evidence: Vec<&Document> = d.iter().collect();
    let broken = breaks(FIT, &s, &evidence);
    for kind in [Break::InventedFact, Break::Anachronism] {
        let (_, text) = broken.iter().find(|(k, _)| *k == kind).unwrap();
        assert_eq!(
            text.lines().count(),
            FIT.lines().count(),
            "{kind:?} replaces a line: {text}"
        );
    }
}

#[test]
fn the_invented_material_is_drawn_from_a_pool_so_a_pair_does_not_teach_one_string() {
    let p = principle(1);
    let s = scenario(&p, Case::Clear);
    let d = docs();
    let evidence: Vec<&Document> = d.iter().collect();
    let mut facts = std::collections::HashSet::new();
    for n in 0..40 {
        let answer = FIT.replace("regional teams", &format!("regional teams of group {n}"));
        for (kind, text) in breaks(&answer, &s, &evidence) {
            if kind == Break::InventedFact {
                facts.insert(text.lines().nth(6).unwrap_or_default().to_string());
            }
        }
    }
    assert!(facts.len() >= 3, "{facts:?}");
}

/// A result for another scenario of `p`, set in `domain`.
fn later_result(p: &Principle, case: Case, domain: &str, variant: u8) -> transfer::Result {
    let mut r = result(p, case, FIT);
    r.variant = variant;
    let s = r.scenario.as_mut().unwrap();
    s.domain = domain.to_string();
    s.id = format!("{}-v{variant}", s.id);
    r.answer.as_mut().unwrap().scenario_id = s.id.clone();
    r
}

#[test]
fn more_scenarios_of_a_training_principle_train_unless_set_in_a_field_held_out_of_training() {
    let (train, bench) = split_principles();
    let results = [
        later_result(&train, Case::Clear, "a city council", 1),
        later_result(&train, Case::Clear, "a symphony orchestra", 2),
    ];
    let built = build(&results, &[train, bench], &docs());
    assert_eq!(
        built.sft.len(),
        2,
        "the council scenario trains, in both modes"
    );
    assert!(
        built.benchmark.is_empty(),
        "the frozen benchmark never grows"
    );
    assert_eq!(
        built.extra_benchmark.len(),
        2,
        "the orchestra scenario is asked in both modes"
    );
    assert!(built.extra_benchmark.iter().all(|t| t.slice == Slice::Ood));
}

#[test]
fn more_scenarios_of_a_benchmark_principle_never_train_and_extend_the_benchmark() {
    let (train, bench) = split_principles();
    let results = [later_result(&bench, Case::Clear, "a city council", 1)];
    let built = build(&results, &[train, bench], &docs());
    assert!(built.sft.is_empty() && built.preference.is_empty());
    assert_eq!(built.extra_benchmark.len(), 2);
    assert!(built.extra_benchmark.iter().all(|t| t.slice == Slice::New));
}

#[test]
fn the_frozen_benchmark_is_written_as_it_always_was_and_a_slice_names_itself_only_when_not_main() {
    use splinter_sdk::model::exam::Question;
    let (_, bench) = split_principles();
    let built = build(
        &[result(&bench, Case::Clear, FIT)],
        std::slice::from_ref(&bench),
        &docs(),
    );
    let line = serde_json::to_string(&built.benchmark[0]).unwrap();
    assert!(
        !line.contains("\"slice\"") && !line.contains("\"group\""),
        "{line}"
    );
    assert_eq!(built.benchmark[0].split(), "exam");
    let ood = TransferTask {
        slice: Slice::Ood,
        ..built.benchmark[0].clone()
    };
    assert!(serde_json::to_string(&ood)
        .unwrap()
        .contains("\"slice\":\"ood\""));
    assert_eq!(ood.split(), "ood");
}

#[test]
fn every_record_of_a_principle_names_the_same_group_so_a_holdout_never_divides_it() {
    let (train, bench) = split_principles();
    let results = [
        result(&train, Case::Clear, FIT),
        later_result(&train, Case::Clear, "a city council", 1),
    ];
    let built = build(&results, &[train.clone(), bench], &docs());
    assert!(built.sft.len() >= 3);
    for record in &built.sft {
        assert_eq!(
            record["metadata"]["group"].as_str(),
            Some(train.id.as_str()),
            "{record}"
        );
    }
    for pair in &built.preference {
        assert_eq!(pair["metadata"]["group"].as_str(), Some(train.id.as_str()));
    }
}

#[test]
fn a_letter_target_is_his_words_without_the_editors_footnotes() {
    let scanned = "My dear Sir,\n\nI have your favor.\n\n1 A copy is in S. A. Wells, Samuel Adams and the American Revolution, vol. i., pp. 363, 364.\n\nYours, S. A.";
    assert_eq!(
        letter_target(scanned).as_deref(),
        Some("My dear Sir,\n\nI have your favor.\n\nYours, S. A.")
    );
}

#[test]
fn every_training_scenario_is_also_a_question_to_sample_the_student_on_and_names_its_principle() {
    let (train, bench) = split_principles();
    let built = build(
        &[result(&train, Case::Clear, FIT)],
        &[train.clone(), bench],
        &docs(),
    );
    assert_eq!(built.train_questions.len(), 2, "both modes");
    assert!(built
        .train_questions
        .iter()
        .all(|t| t.slice == Slice::Train && t.group == train.id));
    let retrieval = built
        .train_questions
        .iter()
        .find(|t| t.mode == Mode::Retrieval)
        .unwrap();
    assert!(
        !retrieval.evidence_docs.is_empty(),
        "the passages it may quote"
    );
    assert!(built.benchmark.is_empty() && built.extra_benchmark.is_empty());
}
