// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training datasets for language models whose
// every example is checked by code rather than by the model that wrote it, for
// its clients. If your team needs expertise in building preference data that is
// correct by construction, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Training data and the frozen benchmark, from checked transfer results.
//!
//! A principle is on one side of the line: its scenarios are either all
//! training or all benchmark, so the benchmark asks about principles the model
//! was never shown applied. Training takes each accepted answer twice, with
//! the passages in the prompt (retrieval mode) and without (internalized mode,
//! where his own words are replaced by what the papers show). Preference pairs
//! are made by breaking an accepted answer in one named way and confirming that
//! the check refuses the result, so each pair is right by construction, not by
//! a model's say-so.

use serde_json::{json, Value};

use crate::curate::Document;
use crate::principles::Principle;
use crate::respond::{self, Applicability};
use crate::scenario::{Case, Scenario};
use crate::transfer;

/// How an accepted answer is broken to make its rejected twin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Break {
    /// A quotation that is in no document.
    FabricatedQuote,
    /// No grounding block.
    DroppedBlock,
    /// The opposite verdict on whether his method applies.
    FlippedApplicability,
    /// A fact the situation never gave.
    InventedFact,
    /// A word from after his death in a line labelled as his.
    Anachronism,
}

/// A quotation that is in no document of his.
const INVENTED_QUOTE: &str =
    "liberty is the first gift of nature to every citizen of the commonwealth";
/// A fact no situation gave.
const INVENTED_FACT: &str =
    "- MODERN_OBSERVATION: the organisation has 4000 employees across 17 offices";
/// A word from after his death, in a line labelled as his.
const ANACHRONISM: &str =
    "- SOURCE_INFERRED: he would have used the internet to write to every town";

/// The span of the first double-quoted passage in `line`, quotes excluded.
fn quote_span(line: &str) -> Option<(usize, usize)> {
    let start = line.find('"')? + 1;
    let end = start + line[start..].find('"')?;
    Some((start, end))
}

/// The header line of the grounding block.
fn block_start(lines: &[&str]) -> Option<usize> {
    lines.iter().position(|l| {
        l.trim()
            .trim_end_matches(':')
            .trim()
            .eq_ignore_ascii_case("grounding")
    })
}

fn flipped(applicability: Applicability) -> &'static str {
    match applicability {
        Applicability::Applies | Applicability::Partly => "NEEDS_INFORMATION",
        Applicability::DoesNotApply | Applicability::NeedsInformation => "APPLIES",
    }
}

/// Every way of breaking `answer` that the check then refuses, with the broken
/// text. A break that the check does not refuse is not offered: the pair would
/// teach nothing.
pub fn breaks(answer: &str, scenario: &Scenario, docs: &[&Document]) -> Vec<(Break, String)> {
    let lines: Vec<&str> = answer.lines().collect();
    let mut candidates: Vec<(Break, String)> = Vec::new();

    let mut replaced = false;
    let mut fabricated: Vec<String> = lines
        .iter()
        .map(|line| match quote_span(line) {
            Some((start, end))
                if !replaced && line.trim_start().starts_with("- SOURCE_DIRECT:") =>
            {
                replaced = true;
                format!("{}{}{}", &line[..start], INVENTED_QUOTE, &line[end..])
            }
            _ => (*line).to_string(),
        })
        .collect();
    if !replaced {
        fabricated.push(format!(
            "- SOURCE_DIRECT: \"{INVENTED_QUOTE}\" [cushing-1-1]"
        ));
    }
    candidates.push((Break::FabricatedQuote, fabricated.join("\n")));

    if let Some(at) = block_start(&lines) {
        candidates.push((
            Break::DroppedBlock,
            lines[..at].join("\n").trim_end().to_string(),
        ));
    }
    if let Some(stated) = Applicability::of(answer) {
        let first = lines.iter().position(|l| !l.trim().is_empty()).unwrap_or(0);
        let mut changed: Vec<String> = lines.iter().map(|l| (*l).to_string()).collect();
        changed[first] = format!("Applicability: {}", flipped(stated));
        candidates.push((Break::FlippedApplicability, changed.join("\n")));
    }
    candidates.push((Break::InventedFact, format!("{answer}\n{INVENTED_FACT}")));
    candidates.push((Break::Anachronism, format!("{answer}\n{ANACHRONISM}")));

    candidates
        .into_iter()
        .filter(|(_, text)| text != answer && respond::check(text, scenario, docs).is_err())
        .collect()
}

/// The answer as the student gives it with no passages to quote: each line
/// that quotes him becomes a line that says what his papers show.
pub fn internalized(answer: &str, principle: &Principle) -> Option<String> {
    let mut seen = false;
    let lines: Vec<String> = answer
        .lines()
        .filter_map(|line| {
            if !line.trim_start().starts_with("- SOURCE_DIRECT:") {
                return Some(line.to_string());
            }
            if std::mem::replace(&mut seen, true) {
                return None;
            }
            Some(format!(
                "- SOURCE_INFERRED: my papers show this habit: {}",
                principle.description
            ))
        })
        .collect();
    let text = lines.join("\n");
    (!text.contains("SOURCE_DIRECT")).then_some(text)
}

/// Whether a principle belongs to the benchmark, by hash of its id so the
/// side does not depend on order: one in five.
pub fn is_benchmark(principle_id: &str) -> bool {
    blake3::hash(principle_id.as_bytes()).as_bytes()[1].is_multiple_of(5)
}

/// How a benchmark question is put to the student.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// With passages from his papers in the prompt.
    Retrieval,
    /// With none: what the model has learned.
    Internalized,
}

/// One benchmark question: the prompt and what its answer is graded against.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TransferTask {
    pub id: String,
    pub scenario_id: String,
    pub mode: Mode,
    pub prompt: String,
    pub case: Case,
    pub observations: Vec<String>,
    /// The documents a retrieval-mode answer may quote.
    pub evidence_docs: Vec<String>,
}

impl TransferTask {
    /// Whether `answer` keeps every rule: right applicability for the case,
    /// grounding that holds against the evidence and the facts given.
    pub fn is_correct(&self, answer: &str, docs: &[Document]) -> bool {
        let scenario = Scenario {
            id: self.scenario_id.clone(),
            principle_id: String::new(),
            case: self.case,
            domain: String::new(),
            situation: String::new(),
            observations: self.observations.clone(),
            request: String::new(),
            conditions_present: Vec::new(),
            conditions_missing: Vec::new(),
        };
        let evidence: Vec<&Document> = docs
            .iter()
            .filter(|d| self.evidence_docs.contains(&d.id))
            .collect();
        respond::check(answer, &scenario, &evidence).is_ok()
    }
}

impl splinter_sdk::model::exam::Question for TransferTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> &str {
        match self.mode {
            Mode::Retrieval => "transfer-retrieval",
            Mode::Internalized => "transfer-internalized",
        }
    }
    fn split(&self) -> &str {
        "exam"
    }
    fn reference(&self) -> &str {
        match self.case {
            Case::Clear => "clear",
            Case::Weak => "weak",
            Case::MissingPrecondition => "missing_precondition",
            Case::SurfaceAnalogy => "surface_analogy",
        }
    }
    fn prompt(&self) -> &str {
        &self.prompt
    }
}

/// The most words of a letter a training target runs to: a long letter is cut
/// at a paragraph, never mid-sentence, so every target is his own words whole.
pub const TARGET_WORDS: usize = 350;

/// The letter's opening paragraphs, up to [`TARGET_WORDS`] words. A first
/// paragraph that is itself longer is cut at the last sentence that fits.
pub fn letter_target(body: &str) -> Option<String> {
    let mut kept: Vec<&str> = Vec::new();
    let mut used = 0usize;
    for paragraph in body.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        let n = paragraph.split_whitespace().count();
        if used + n > TARGET_WORDS {
            break;
        }
        kept.push(paragraph);
        used += n;
    }
    if kept.is_empty() {
        let first = body.split("\n\n").map(str::trim).find(|p| !p.is_empty())?;
        let cut: String = first
            .split_whitespace()
            .take(TARGET_WORDS)
            .collect::<Vec<_>>()
            .join(" ");
        let end = cut.rfind(['.', '?', '!'])?;
        return Some(cut[..=end].to_string());
    }
    Some(kept.join("\n\n"))
}

/// Supervised records of what he actually did: the situation a letter answered
/// as the prompt and the letter itself as the answer. Only letters in `allowed`
/// are used, so a held-out letter is never a target.
pub fn reconstruction_sft(
    briefings: &[crate::reconstruct::Briefing],
    docs: &[Document],
    allowed: &std::collections::HashSet<String>,
) -> Vec<Value> {
    briefings
        .iter()
        .filter(|b| allowed.contains(&b.doc_id))
        .filter_map(|b| {
            let letter = docs.iter().find(|d| d.id == b.doc_id)?;
            let target = letter_target(&letter.body)?;
            Some(json!({
                "messages": [
                    {"role": "system", "content": crate::reconstruct::SYSTEM, "train": false},
                    {"role": "user", "content": crate::reconstruct::prompt(b), "train": false},
                    {"role": "assistant", "content": target, "train": true},
                ],
                "tools": [],
            }))
        })
        .collect()
}

/// The datasets, and what was left out and why.
#[derive(Debug, Default)]
pub struct Built {
    pub sft: Vec<Value>,
    pub preference: Vec<Value>,
    pub benchmark: Vec<TransferTask>,
    pub excluded: Vec<(String, String)>,
}

fn sft_record(prompt: &str, answer: &str) -> Value {
    json!({
        "messages": [
            {"role": "system", "content": respond::SYSTEM, "train": false},
            {"role": "user", "content": prompt, "train": false},
            {"role": "assistant", "content": answer, "train": true},
        ],
        "tools": [],
    })
}

fn pair(
    prompt: &str,
    chosen: &str,
    rejected: &str,
    kind: Break,
    mode: Mode,
    scenario: &Scenario,
) -> Value {
    json!({
        "prompt": [
            {"role": "system", "content": respond::SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
        "tools": [],
        "metadata": {"break": kind, "mode": mode, "scenario_id": scenario.id, "principle_id": scenario.principle_id},
    })
}

/// Build the datasets from the transfer results. A result that failed, or
/// whose answer no longer passes its check, is left out with the reason.
pub fn build(results: &[transfer::Result], principles: &[Principle], docs: &[Document]) -> Built {
    let mut built = Built::default();
    for result in results {
        let id = result.principle_id.clone();
        if let Some(error) = &result.error {
            built.excluded.push((id, format!("failed: {error}")));
            continue;
        }
        let (Some(scenario), Some(answer), Some(principle)) = (
            &result.scenario,
            &result.answer,
            principles.iter().find(|p| p.id == id),
        ) else {
            built
                .excluded
                .push((id, "no scenario, answer or principle".to_string()));
            continue;
        };
        let cited: Vec<&str> = principle
            .support
            .iter()
            .map(|f| f.doc_id.as_str())
            .collect();
        let evidence: Vec<&Document> = docs
            .iter()
            .filter(|d| cited.contains(&d.id.as_str()))
            .collect();
        if let Err(why) = respond::check(&answer.text, scenario, &evidence) {
            built.excluded.push((id, why));
            continue;
        }
        let with_passages = respond::student_prompt(scenario, &principle.support);
        let without = respond::student_prompt(scenario, &[]);
        let own_words = internalized(&answer.text, principle)
            .filter(|t| respond::check(t, scenario, &[]).is_ok());

        if is_benchmark(&principle.id) {
            let task = |mode: Mode, prompt: &String, evidence_docs: Vec<String>| TransferTask {
                id: format!("transfer-{mode:?}-{}", scenario.id).to_lowercase(),
                scenario_id: scenario.id.clone(),
                mode,
                prompt: prompt.clone(),
                case: scenario.case,
                observations: scenario.observations.clone(),
                evidence_docs,
            };
            built.benchmark.push(task(
                Mode::Retrieval,
                &with_passages,
                cited.iter().map(|d| (*d).to_string()).collect(),
            ));
            built
                .benchmark
                .push(task(Mode::Internalized, &without, Vec::new()));
            continue;
        }
        built.sft.push(sft_record(&with_passages, &answer.text));
        for (kind, text) in breaks(&answer.text, scenario, &evidence) {
            built.preference.push(pair(
                &with_passages,
                &answer.text,
                &text,
                kind,
                Mode::Retrieval,
                scenario,
            ));
        }
        if let Some(own) = own_words {
            built.sft.push(sft_record(&without, &own));
            for (kind, text) in breaks(&own, scenario, &[]) {
                built.preference.push(pair(
                    &without,
                    &own,
                    &text,
                    kind,
                    Mode::Internalized,
                    scenario,
                ));
            }
        }
    }
    built
}

/// Write `sft-transfer.jsonl`, `preference-transfer.jsonl`, `benchmark.jsonl`
/// and `excluded.jsonl` under `dir`, each atomically.
///
/// # Errors
/// A file cannot be written.
pub fn write_all(built: &Built, dir: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let lines =
        |rows: Vec<Value>| -> String { rows.into_iter().map(|r| format!("{r}\n")).collect() };
    let files = [
        ("sft-transfer.jsonl", lines(built.sft.clone())),
        ("preference-transfer.jsonl", lines(built.preference.clone())),
        (
            "benchmark.jsonl",
            lines(built.benchmark.iter().map(|t| json!(t)).collect()),
        ),
        (
            "excluded.jsonl",
            lines(
                built
                    .excluded
                    .iter()
                    .map(|(id, why)| json!({"principle_id": id, "reason": why}))
                    .collect(),
            ),
        ),
    ];
    for (name, text) in files {
        let path = dir.join(name);
        let tmp = path.with_extension("part");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
    }
    Ok(())
}

/// The benchmark questions of a JSON-lines file.
///
/// # Errors
/// The file cannot be read or a line is not a question.
pub fn read_benchmark(path: &std::path::Path) -> anyhow::Result<Vec<TransferTask>> {
    std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Period;
    use crate::principles::{Found, Status};
    use crate::testkit::doc;

    const WORDS_OF_HIS: &str = "Let the Committee write to every Town, that the Sense of the People may be known before the Assembly meets again.";

    fn principle(n: usize) -> Principle {
        Principle {
            id: format!("P-{n}"),
            description: format!("When the centre will not act, he had the towns state the claim together, variant {n}."),
            trigger_conditions: vec!["a grievance unanswered".into()],
            expected_behavior: vec!["write to the towns".into()],
            qualifications: vec!["not when divided".into()],
            status: Status::SourceSupported,
            support: vec![Found { doc_id: "d1".into(), quote: "Let the Committee write to every Town".into(), period: Period::PreRevolution, recipient: None }],
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
        assert!(text.contains("SOURCE_INFERRED") && text.contains(&p.description));
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
        assert_eq!(m[0]["content"].as_str(), Some(respond::SYSTEM));
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
        assert!(
            pair["metadata"]["break"].is_string() && pair["metadata"]["scenario_id"].is_string()
        );
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
        assert!(retrieval.is_correct(FIT, &d));
        assert!(
            !retrieval.is_correct(NON_FIT, &d),
            "wrong verdict for a fit"
        );
        assert!(!retrieval.is_correct(
            &FIT.replace(
                "Let the Committee write to every Town",
                "Liberty is the first gift of nature to every citizen"
            ),
            &d
        ));
        let internalized = built
            .benchmark
            .iter()
            .find(|t| t.mode == Mode::Internalized)
            .unwrap();
        assert!(
            !internalized.is_correct(FIT, &d),
            "with no passages shown, quoting him is fabrication"
        );
        assert!(internalized.is_correct(&internalized_of(&bench, FIT), &d));
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
        write_all(&built, dir.path()).unwrap();
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
            request: "Advise him.".into(),
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
}
