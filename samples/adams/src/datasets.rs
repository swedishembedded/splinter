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
use crate::persona::Framing;
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

/// Quotations that are in no document of his. A pool, so that what a pair
/// teaches is not one memorised string.
const INVENTED_QUOTES: [&str; 6] = [
    "liberty is the first gift of nature to every citizen of the commonwealth",
    "no assembly was ever so wise as the plain judgment of the people met together",
    "let every man set his hand to the petition before the winter is out",
    "the cause of the colonies is the cause of every honest man in the world",
    "i have never known a ministry that was not afraid of a united people",
    "whatever is decided in secret will be undone in the open before the year ends",
];
/// Facts no situation gave.
const INVENTED_FACTS: [&str; 6] = [
    "- MODERN_OBSERVATION: the organisation has 4000 employees across 17 offices",
    "- MODERN_OBSERVATION: the board has already voted to cut the budget in half",
    "- MODERN_OBSERVATION: the previous director resigned over the same dispute last spring",
    "- MODERN_OBSERVATION: a regulator has opened an investigation into the matter",
    "- MODERN_OBSERVATION: most of the staff have said they will walk out on Friday",
    "- MODERN_OBSERVATION: the largest customer has threatened to leave within a month",
];
/// Lines labelled as his that use a word from after his death.
const ANACHRONISMS: [&str; 6] = [
    "- SOURCE_INFERRED: he would have used the internet to write to every town",
    "- SOURCE_INFERRED: he would have kept the correspondence in a database",
    "- SOURCE_INFERRED: he would have sent the same email to every committee",
    "- SOURCE_INFERRED: he would have posted the resolution on a website",
    "- SOURCE_INFERRED: he would have convened the towns online",
    "- SOURCE_INFERRED: he would have wanted the software to keep a record",
];

/// One of `pool`, chosen by a hash of `key`, so the same answer is always
/// broken the same way and different answers are broken differently.
fn pick<'a>(pool: &[&'a str], key: &str) -> &'a str {
    pool[usize::from(blake3::hash(key.as_bytes()).as_bytes()[0]) % pool.len()]
}

/// `answer` with the first line starting with any of `labels` replaced by
/// `line`, so the broken answer is as long as the good one; with no such line,
/// `line` is added at the end.
fn replacing_line(answer: &str, labels: &[&str], line: &str) -> String {
    let mut done = false;
    let mut lines: Vec<String> = answer
        .lines()
        .map(|l| {
            let hit = !done && labels.iter().any(|label| l.trim_start().starts_with(label));
            if hit {
                done = true;
                line.to_string()
            } else {
                l.to_string()
            }
        })
        .collect();
    if !done {
        lines.push(line.to_string());
    }
    lines.join("\n")
}

/// The most lines restating the situation's facts that a training answer
/// keeps: the situation is in the prompt, and a block that copies it back is
/// loss spent on copying.
const MAX_RESTATED_FACTS: usize = 2;

/// An answer as it is taught: at most [`MAX_RESTATED_FACTS`] lines of
/// MODERN_OBSERVATION, every other line as it was. Idempotent, and it only
/// removes lines the rules do not need, so a checked answer stays checked.
pub fn shape(answer: &str) -> String {
    let mut kept = 0usize;
    answer
        .lines()
        .filter(|line| {
            if !line.trim_start().starts_with("- MODERN_OBSERVATION:") {
                return true;
            }
            kept += 1;
            kept <= MAX_RESTATED_FACTS
        })
        .collect::<Vec<_>>()
        .join("\n")
}

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
                format!(
                    "{}{}{}",
                    &line[..start],
                    pick(&INVENTED_QUOTES, answer),
                    &line[end..]
                )
            }
            _ => (*line).to_string(),
        })
        .collect();
    if !replaced {
        fabricated.push(format!(
            "- SOURCE_DIRECT: \"{}\" [cushing-1-1]",
            pick(&INVENTED_QUOTES, answer)
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
    candidates.push((
        Break::InventedFact,
        replacing_line(
            answer,
            &["- MODERN_OBSERVATION:"],
            pick(&INVENTED_FACTS, answer),
        ),
    ));
    candidates.push((
        Break::Anachronism,
        replacing_line(
            answer,
            &["- PERSONA_TRANSFER:", "- SOURCE_INFERRED:"],
            pick(&ANACHRONISMS, answer),
        ),
    ));

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
            // In the first person, from what he is recorded as doing, not the
            // principle's third-person statement of it.
            principle
                .expected_behavior
                .first()
                .map(|done| format!("- SOURCE_INFERRED: my papers show that I would {done}"))
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

/// Which part of the benchmark a question belongs to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Slice {
    /// The first scenario of a benchmark principle: the frozen benchmark.
    #[default]
    Main,
    /// Another scenario of a benchmark principle.
    New,
    /// A scenario set in a field held out of training.
    Ood,
}

impl Slice {
    fn is_main(&self) -> bool {
        *self == Slice::Main
    }
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
    /// Which part of the benchmark; the frozen one writes nothing here, so its
    /// file is what it always was.
    #[serde(default, skip_serializing_if = "Slice::is_main")]
    pub slice: Slice,
}

impl TransferTask {
    /// How `answer` was graded against the evidence and the facts it was given:
    /// right when it keeps every rule, with each rule's own result alongside.
    pub fn verdict(&self, answer: &str, docs: &[Document]) -> splinter_sdk::model::exam::Verdict {
        let scenario = Scenario {
            id: self.scenario_id.clone(),
            principle_id: String::new(),
            case: self.case,
            domain: String::new(),
            // The prompt holds the situation, the facts, the request and the
            // passages: everything a number in the answer may have come from.
            situation: self.prompt.clone(),
            observations: self.observations.clone(),
            request: String::new(),
            conditions_present: Vec::new(),
            conditions_missing: Vec::new(),
        };
        let evidence: Vec<&Document> = docs
            .iter()
            .filter(|d| self.evidence_docs.contains(&d.id))
            .collect();
        let rules = respond::assess(answer, &scenario, &evidence);
        splinter_sdk::model::exam::Verdict {
            correct: rules.iter().all(|(_, broken)| broken.is_none()),
            checks: rules
                .into_iter()
                .map(|(rule, broken)| (rule.to_string(), broken.is_none()))
                .collect(),
        }
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
        match self.slice {
            Slice::Main => "exam",
            Slice::New => "new",
            Slice::Ood => "ood",
        }
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
                    {"role": "system", "content": Framing::of_record(&crate::reconstruct::prompt(b)).system(crate::reconstruct::SYSTEM), "train": false},
                    {"role": "user", "content": crate::reconstruct::prompt(b), "train": false},
                    {"role": "assistant", "content": target, "train": true},
                ],
                "tools": [],
            }))
        })
        .collect()
}

/// Examples to show a model in its system message instead of training it: the
/// first `n` of `records` (chat records of training principles), each as the
/// question and the whole answer. Empty for none.
pub fn few_shot_block(records: &[Value], n: usize) -> String {
    let mut out = String::new();
    for (i, record) in records.iter().take(n).enumerate() {
        let message = |role: usize| record["messages"][role]["content"].as_str().unwrap_or("");
        out.push_str(&format!(
            "Example {}.\nQuestion:\n{}\n\nAnswer:\n{}\n\n",
            i + 1,
            message(1),
            message(2)
        ));
    }
    out
}

/// The benchmark's file name unless a new one is asked for: a written
/// benchmark is frozen under its name, so a changed one needs another.
pub const BENCHMARK_FILE: &str = "benchmark.jsonl";
/// The file the extra questions go to unless another is asked for.
pub const EXTRA_BENCHMARK_FILE: &str = "benchmark-extra.jsonl";
/// The ledger that pins each benchmark file to its content.
const FROZEN_LEDGER: &str = "FROZEN.json";

/// The datasets, and what was left out and why.
#[derive(Debug, Default)]
pub struct Built {
    pub sft: Vec<Value>,
    pub preference: Vec<Value>,
    /// The benchmark of the first scenario of each benchmark principle: the
    /// frozen one.
    pub benchmark: Vec<TransferTask>,
    /// More scenarios to ask: later ones of the benchmark principles, and any
    /// set in a field held out of training. A separate file, never mixed into
    /// the frozen benchmark.
    pub extra_benchmark: Vec<TransferTask>,
    pub excluded: Vec<(String, String)>,
}

fn sft_record(prompt: &str, answer: &str) -> Value {
    json!({
        "messages": [
            {"role": "system", "content": Framing::of_record(prompt).system(respond::SYSTEM), "train": false},
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
            {"role": "system", "content": Framing::of_record(prompt).system(respond::SYSTEM)},
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
        let taught = shape(&answer.text);
        let with_passages = respond::student_prompt(scenario, &principle.support);
        let without = respond::student_prompt(scenario, &[]);
        let own_words =
            internalized(&taught, principle).filter(|t| respond::check(t, scenario, &[]).is_ok());

        let held_out_field = crate::domains::DOMAINS
            .iter()
            .any(|d| d.held_out && d.name == scenario.domain);
        let slice = match (result.variant, is_benchmark(&principle.id), held_out_field) {
            (0, true, _) => Some(Slice::Main),
            (_, true, _) => Some(Slice::New),
            (_, false, true) => Some(Slice::Ood),
            (_, false, false) => None,
        };
        if let Some(slice) = slice {
            let task = |mode: Mode, prompt: &String, evidence_docs: Vec<String>| TransferTask {
                id: format!("transfer-{mode:?}-{}", scenario.id).to_lowercase(),
                scenario_id: scenario.id.clone(),
                mode,
                prompt: prompt.clone(),
                case: scenario.case,
                observations: scenario.observations.clone(),
                evidence_docs,
                slice,
            };
            let questions = if slice == Slice::Main {
                &mut built.benchmark
            } else {
                &mut built.extra_benchmark
            };
            questions.push(task(
                Mode::Retrieval,
                &with_passages,
                cited.iter().map(|d| (*d).to_string()).collect(),
            ));
            questions.push(task(Mode::Internalized, &without, Vec::new()));
            continue;
        }
        built.sft.push(sft_record(&with_passages, &taught));
        for (kind, text) in breaks(&taught, scenario, &evidence) {
            built.preference.push(pair(
                &with_passages,
                &taught,
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

/// Write `sft-transfer.jsonl`, `preference-transfer.jsonl`, the benchmark
/// (`benchmark_file`) and `excluded.jsonl` under `dir`, each atomically.
///
/// # Errors
/// A file cannot be written, or `benchmark_file` is frozen at other content.
pub fn write_all(
    built: &Built,
    dir: &std::path::Path,
    benchmark_file: &str,
    extra_file: &str,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let lines =
        |rows: Vec<Value>| -> String { rows.into_iter().map(|r| format!("{r}\n")).collect() };
    let files = [
        ("sft-transfer.jsonl", lines(built.sft.clone())),
        ("preference-transfer.jsonl", lines(built.preference.clone())),
        (
            benchmark_file,
            lines(built.benchmark.iter().map(|t| json!(t)).collect()),
        ),
        (
            extra_file,
            lines(built.extra_benchmark.iter().map(|t| json!(t)).collect()),
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
    // The benchmark is checked before anything is written, so a refused one
    // leaves every file as it was.
    let ledger = splinter_sdk::data::frozen::Ledger::at(dir.join(FROZEN_LEDGER));
    let benchmark_text = &files[2].1;
    ledger.check(benchmark_file, benchmark_text.as_bytes())?;
    // The extra questions are still growing as scenarios are made: only a
    // file someone has pinned is protected.
    ledger.check(extra_file, files[3].1.as_bytes())?;
    for (name, text) in &files {
        let path = dir.join(name);
        let tmp = path.with_extension("part");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
    }
    ledger.pin(benchmark_file, benchmark_text.as_bytes())?;
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
mod tests;
