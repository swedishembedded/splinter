// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents that grow
// their own training data. If your team needs expertise in data extraction
// from technical documents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Fact exploration: a markdown fact sheet becomes chat-training records.
//!
//! `explore` splits a document at its headings, asks the configured model
//! to enumerate EVERY factual claim of each section as a question/answer
//! pair, and appends one training record per fact to a JSONL file in the
//! SAME schema `learn` writes to the experience pool - so a trained-on
//! questions file is trainer input without translation.
//!
//! Two rules keep the dataset honest:
//!
//! - Strict parse: a reply that is not exactly one
//!   `{"facts": [{"question", "answer"}, ...]}` object is counted as a
//!   failure for its section and skipped, not salvaged - a half-parsed
//!   reply would silently weight whatever the model felt like emitting.
//! - Dedup by normalized question text: the same fact appearing in two
//!   sections (or re-run over a grown document) trains once.

use splinter_policy::{complete_text, ModelSelection};
use splinter_store::{write_atomic, StateRoot};

use anyhow::Context;
use splinter_store::runs::{Limits, RunManifest};
use splinter_store::trace::Trace;

/// Everything one exploration needs, as configured by the caller.
#[derive(Clone, Debug)]
pub struct ExploreOptions {
    /// The markdown fact sheet to read.
    pub file: std::path::PathBuf,
    /// The JSONL training file to write atomically at the end.
    pub out: std::path::PathBuf,
    /// Section size cap in lines; a section past it starts a new chunk at
    /// the next heading or paragraph boundary. `None` means uncapped.
    pub chunk_lines: Option<usize>,
    /// The model that extracts the facts.
    pub model: ModelSelection,
    /// Device identifiers OUTSIDE the document's scope. Every other
    /// accepted fact also trains a negative variant with one of these
    /// substituted, answered by [`NOT_COVERED`], so the adapter learns
    /// where its knowledge ends instead of answering other chips with
    /// this chip's numbers.
    pub scope_negatives: Vec<String>,
}

/// What one exploration produced, for the CLI's summary line.
#[derive(Debug, PartialEq, Eq)]
pub struct ExploreSummary {
    pub run_id: String,
    pub sections: usize,
    pub facts: usize,
    pub parse_failures: usize,
    /// Questions the anchor gate refused: the generator produced them
    /// without the device identifier the title carries.
    pub unanchored: usize,
    /// Facts the traceability gate refused: the answer states a number
    /// the section does not carry.
    pub untraceable: usize,
}

/// One strict parse of a facts reply.
#[derive(Debug, PartialEq, Eq)]
pub struct Facts {
    pub pairs: Vec<(String, String)>,
}

/// Splits `text` into sections at markdown headings (`##` / `###`).
///
/// Three invariants:
/// - A table row is never split from its section: a heading line or a
///   blank line opens a new chunk, and a markdown table contains neither,
///   so a table always stays whole.
/// - `chunk_lines` caps a section's size: when a section exceeds the cap,
///   the next heading starts a fresh chunk (content in flight is kept
///   with the section that holds it - splitting mid-table is what the
///   cap must never cause).
/// - A section that stays over the cap with no heading in sight (the
///   common fact-sheet shape: one `#`, then bullets) also chunks at the
///   next blank line. Without this, a heading-less document was ONE
///   section however large - too much for a small generator to enumerate,
///   which is how coverage collapses. The split point is a paragraph
///   boundary, and a heading is never parked alone: if the content in
///   flight is only a heading, it stays for the paragraph that follows.
/// - Every section carries the document's title line (the first `#`
///   heading). Chunking cuts later sections off from the document's
///   subject, and the generator is told to use the identifiers its
///   section shows - a chunk that no longer names the device cannot
///   produce an anchored question about it.
pub(crate) fn split_sections(text: &str, chunk_lines: Option<usize>) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut overflow = false;
    let title = document_title(text);
    for line in text.lines() {
        let is_heading = line.starts_with("## ") || line.starts_with("### ");
        let mut close = is_heading && (!current.trim().is_empty() || overflow);
        if !close {
            if let Some(cap) = chunk_lines {
                let is_blank = line.trim().is_empty();
                let last_non_empty_is_heading = current
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .is_some_and(|l| l.starts_with('#'));
                close = is_blank
                    && !current.trim().is_empty()
                    && !last_non_empty_is_heading
                    && current.lines().count() >= cap;
            }
        }
        if close {
            sections.push(std::mem::take(&mut current));
            overflow = false;
        }
        current.push_str(line);
        current.push('\n');
        if let Some(cap) = chunk_lines {
            if current.lines().count() >= cap {
                // Over the cap: the next heading or paragraph boundary
                // MUST open a new chunk.
                overflow = true;
            }
        }
    }
    if !current.trim().is_empty() {
        sections.push(current);
    }
    // Re-unite every section with the document's subject, and drop a
    // section that holds nothing but it - a title with no content gives
    // the generator no facts to enumerate.
    if let Some(title) = title {
        for section in &mut sections {
            if !section.contains(title) {
                let content = section.trim_start();
                *section = format!("{title}\n\n{content}");
            }
        }
        sections.retain(|section| section.lines().any(|l| !l.trim().is_empty() && l != title));
    }
    sections
}

/// Strips optional markdown code fences around a reply, so a model that
/// answered perfectly inside ```json fences still parses. Fences are the
/// one tolerated decoration; prose around the object is not.
fn strip_fences(reply: &str) -> &str {
    let trimmed = reply.trim();
    let without = trimmed
        .strip_prefix("```")
        .and_then(|r| {
            r.trim_start_matches(|c: char| c.is_ascii_alphanumeric())
                .strip_prefix('\n')
        })
        .unwrap_or(trimmed);
    without
        .strip_suffix("```")
        .map(|r| r.trim())
        .unwrap_or(without)
}

/// Strict parse: the reply must be EXACTLY one JSON object of the shape
/// `{"facts": [{"question": string, "answer": string}, ...]}` after
/// trimming whitespace and stripping code fences. Anything else - prose,
/// an array, a missing key, a non-string field - is a failure.
pub(crate) fn parse_facts_reply(reply: &str) -> anyhow::Result<Facts> {
    let value: serde_json::Value = serde_json::from_str(strip_fences(reply))
        .context("reply is not exactly one JSON object")?;
    let object = value
        .as_object()
        .with_context(|| "reply is not a JSON object".to_string())?;
    let facts = object
        .get("facts")
        .with_context(|| "reply object has no \"facts\" key".to_string())?;
    let list = facts
        .as_array()
        .with_context(|| "\"facts\" is not an array".to_string())?;
    let mut pairs = Vec::new();
    for item in list {
        let entry = item
            .as_object()
            .with_context(|| "a fact entry is not an object".to_string())?;
        let question = entry
            .get("question")
            .and_then(|v| v.as_str())
            .with_context(|| "a fact entry has no string \"question\"".to_string())?;
        let answer = entry
            .get("answer")
            .and_then(|v| v.as_str())
            .with_context(|| "a fact entry has no string \"answer\"".to_string())?;
        pairs.push((question.to_string(), answer.to_string()));
    }
    Ok(Facts { pairs })
}

/// Strict parse for `ask`: the reply must be exactly one
/// `{"answer": string}` object (fences tolerated, prose is not).
pub(crate) fn parse_answer_reply(reply: &str) -> anyhow::Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(strip_fences(reply)).with_context(|| {
            // A parse failure is a scored event; the raw reply is the evidence
            // a repair decision needs, so it rides in the error chain.
            format!("reply is not exactly one JSON object: {reply:?}")
        })?;
    let object = value
        .as_object()
        .with_context(|| "reply is not a JSON object".to_string())?;
    let answer = object
        .get("answer")
        .and_then(|v| v.as_str())
        .with_context(|| "reply object has no string \"answer\"".to_string())?;
    Ok(answer.to_string())
}

/// The fixed reply a question about an out-of-scope device trains toward:
/// the scope boundary. Without such examples an adapter answers
/// out-of-family questions with in-family numbers - the measured probe
/// returned "168 MHz" for an STM32F103 whose true maximum is 72 MHz - and
/// a confident wrong number on the wrong chip is worse than an honest
/// refusal.
pub(crate) const NOT_COVERED: &str = "That device is not covered by this fact sheet.";

/// The negative variant of one anchored question: the in-scope device
/// identifier is replaced with an out-of-scope one, keeping the question
/// otherwise verbatim. `None` when the question does not name the
/// identifier (the anchor gate makes that rare, not impossible).
fn negative_question(
    question: &str,
    identifier: &str,
    negative: &str,
    identifiers: &[String],
) -> Option<String> {
    let start = question.find(identifier)?;
    let mut end = start + identifier.len();
    // A question names the fact sheet's devices as a slash-separated run
    // ("STM32F405 / STM32F407"). Substituting one member leaves a covered
    // device in the question, which trains a compound-question refusal
    // instead of the scope boundary the probe asks for; the whole run goes.
    loop {
        let tail = &question[end..];
        let ws = tail.len() - tail.trim_start().len();
        let Some(after) = tail[ws..].strip_prefix('/') else {
            break;
        };
        let ws2 = after.len() - after.trim_start().len();
        let candidate = &question[end + ws + 1 + ws2..];
        let Some((_, id_len)) = identifiers.iter().find_map(|id| {
            candidate
                .get(..id.len())
                .filter(|head| head.eq_ignore_ascii_case(id))
                .map(|_| (id, id.len()))
        }) else {
            break;
        };
        end += ws + 1 + ws2 + id_len;
    }
    Some(format!(
        "{}{negative}{}",
        &question[..start],
        &question[end..]
    ))
}

/// The number tokens of `text`, each with the unit word that binds to it
/// (one optional space, then a run of letters / ° / /, plural-insensitive).
/// Thousand separators are part of the token and normalized on compare.
fn number_tokens(text: &str) -> Vec<(String, Option<String>)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i;
        while j < chars.len() {
            let separator_before_digit = (chars[j] == ',' || chars[j] == '.')
                && j + 1 < chars.len()
                && chars[j + 1].is_ascii_digit();
            if chars[j].is_ascii_digit() || separator_before_digit {
                j += 1;
            } else {
                break;
            }
        }
        let number = chars[start..j].iter().collect::<String>().replace(',', "");
        let mut k = j;
        if k < chars.len() && chars[k] == ' ' {
            k += 1;
        }
        let unit_start = k;
        while k < chars.len() && (chars[k].is_alphabetic() || chars[k] == '°' || chars[k] == '/') {
            k += 1;
        }
        let unit = if k > unit_start {
            let word: String = chars[unit_start..k].iter().collect();
            // Plural-insensitive, but keep genuine short units ("ms") whole.
            let trimmed = word.trim_end_matches('s');
            Some(if word.len() > 2 && trimmed.len() < word.len() {
                trimmed.to_lowercase()
            } else {
                word.to_lowercase()
            })
        } else {
            None
        };
        out.push((number, unit));
        i = j;
    }
    out
}

/// Does `line` carry `number` as a standalone token? A digit on either
/// side means the match is inside a larger number ("14" is not in "114");
/// a '.' after it means the match is the head of a decimal ("4" is not in
/// "4.223").
fn line_has_number(line: &str, number: &str) -> bool {
    for (at, _) in line.match_indices(number) {
        let before = line[..at].chars().next_back();
        let after = line[at + number.len()..].chars().next();
        let standalone = before.is_none_or(|c| !c.is_ascii_digit())
            && after.is_none_or(|c| !c.is_ascii_digit() && c != '.');
        if standalone {
            return true;
        }
    }
    false
}

/// The traceability gate: every number the answer states must be
/// traceable to the section. The generator's measured failure mode is
/// inventing plausible numbers ("160 MHz" for a 168 MHz rating, four DMA
/// streams for eight, 100,000,000 erase cycles for 10,000), a trained
/// wrong number is worse than no fact, and the recall score cannot catch
/// it because recall compares against the same wrong reference. A number
/// matches when one section line carries it as a standalone token and -
/// when the answer binds it to a unit word - that unit too.
fn answer_numbers_traceable(answer: &str, section: &str) -> bool {
    number_tokens(answer).into_iter().all(|(number, unit)| {
        section.lines().any(|line| {
            line_has_number(line, &number)
                && unit
                    .as_deref()
                    .is_none_or(|u| line.to_lowercase().contains(u))
        })
    })
}

/// Normalized question text for dedup: lowercase, whitespace collapsed.
fn normalize(question: &str) -> String {
    question
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The device identifiers a document title names: alphanumeric tokens of
/// four or more characters that mix letters and digits ("STM32F407",
/// "DS8626"). A pure-word title names no device, and the anchor gate
/// disables itself for such a document rather than rejecting everything.
fn title_identifiers(title: &str) -> Vec<String> {
    title
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| {
            t.len() >= 4
                && t.chars().any(|c| c.is_ascii_digit())
                && t.chars().any(|c| c.is_ascii_alphabetic())
        })
        .map(str::to_string)
        .collect()
}

/// The anchor gate: does the question name a device the document's title
/// names? The check is case-insensitive containment on the title's
/// identifiers. With no identifiers to anchor on, every question passes -
/// the gate refuses only what it can name.
fn question_is_anchored(question: &str, identifiers: &[String]) -> bool {
    let lower = question.to_lowercase();
    identifiers.is_empty()
        || identifiers
            .iter()
            .any(|id| lower.contains(&id.to_lowercase()))
}

/// One training record, in the SAME schema `learn` appends to the pool:
/// the question as context (not supervised), the answer as the supervised
/// turn.
pub(crate) fn training_record(run_id: &str, question: &str, answer: &str) -> serde_json::Value {
    // The assistant side teaches the reply shape `ask` parses, not just the
    // fact: fine-tuning on bare answers trains the wrapper away, and a
    // model that answers "84 MHz" without the {"answer": ...} object then
    // fails every strict parse of its own correct reply.
    let reply = serde_json::json!({ "answer": answer }).to_string();
    serde_json::json!({
        "messages": [
            { "role": "user", "content": question, "train": false },
            { "role": "assistant", "content": reply, "train": true },
        ],
        "metadata": { "run_id": run_id, "verified_by": [] },
    })
}

/// The exact prompt one section sees. It demands the whole shape, and it
/// says what "covers the section" means: every factual claim, not a sample.
/// It also demands subject-anchored questions: a fine-tuned model learns
/// whatever association the question text carries, so "What is the maximum
/// CPU clock frequency?" taught without the chip's name answers any chip
/// question with this chip's number - and hallucinates when the family IS
/// named, because the named form was never seen. The question is the
/// retrieval key; it must carry the subject.
fn facts_prompt(section: &str) -> String {
    format!(
        "Below is one section of a hardware fact sheet. Enumerate EVERY factual claim it \
         makes - every number, unit, limit, relationship and conditional - as question/answer \
         pairs. Do not sample, do not summarize: each distinct fact gets its own pair, and a \
         question must be answerable from this section alone.\n\n\
         Every question must NAME THE SPECIFIC DEVICE OR COMPONENT it is about, so the \
         question is self-contained and answerable with no other context. Use the exact \
         identifiers from the section: not \"What is the maximum frequency?\" but \"What is \
         the maximum CPU clock frequency of <device>?\"; not \"How many streams does DMA \
         have?\" but \"How many streams does DMA1 have on <device>?\" - where <device> is \
         the device the title names, spelled exactly as the title spells it. A question \
         that would fit a different device unchanged is wrong. The section opens with the \
         document's title, which names the device this fact sheet describes: every question \
         must name that device exactly as the title spells it.\n\n\
         Reply with EXACTLY one JSON object and nothing else - no prose, no code fences:\n\
         {{\"facts\": [{{\"question\": string, \"answer\": string}}, ...]}}\n\n\
         SECTION:\n{section}"
    )
}

/// Runs the whole exploration, tracing to its own run dir like `run` does:
/// manifest, one event per section, and an outcome with the counts.
/// The document's title line - the first level-1 heading - which names
/// the subject every chunk must carry and every question must anchor on.
fn document_title(text: &str) -> Option<&str> {
    text.lines().find(|l| l.starts_with("# "))
}

pub(crate) fn run(root: &StateRoot, options: ExploreOptions) -> anyhow::Result<ExploreSummary> {
    let text = std::fs::read_to_string(&options.file)
        .with_context(|| format!("reading {}", options.file.display()))?;
    let sections = split_sections(&text, options.chunk_lines);
    anyhow::ensure!(
        !sections.is_empty(),
        "{}: no sections found",
        options.file.display()
    );
    let identifiers = document_title(&text)
        .map(title_identifiers)
        .unwrap_or_default();

    let run_id = splinter_store::new_id_with_prefix("explore");
    let dir = root.run_dir(&run_id);
    let trace = Trace::open(&dir, &run_id, 1)?;

    let mut manifest = RunManifest {
        schema: 1,
        run_id: run_id.clone(),
        workspace: options.file.display().to_string(),
        task: format!("explore facts from {}", options.file.display()),
        status: "pending".into(),
        attempts: 1,
        started_ts: splinter_store::clock::utc_now(),
        updated_ts: splinter_store::clock::utc_now(),
        model: options.model.identity(),
        base_url: match &options.model {
            ModelSelection::Remote(remote) => remote.base_url.clone(),
            ModelSelection::Local(_) => None,
        },
        local_adapter: options.model.local().and_then(|w| w.adapter.clone()),
        limits: Limits::default(),
    };
    write_atomic(
        &dir.join("run.json"),
        &serde_json::to_string_pretty(&manifest)?,
    )?;

    // One provider for the whole run: the load is the expensive step, and
    // every section wants the same model anyway.
    let provider = options.model.provider()?;
    let rt = tokio::runtime::Runtime::new()?;

    let mut records: Vec<serde_json::Value> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut parse_failures = 0usize;
    let mut unanchored = 0usize;
    let mut untraceable = 0usize;
    let mut negatives_emitted = 0usize;
    for (n, section) in sections.iter().enumerate() {
        let prompt = facts_prompt(section);
        let result: anyhow::Result<String> =
            rt.block_on(async { complete_text(provider.as_ref(), &prompt).await });
        let mut payload = match result {
            Ok(reply) => match parse_facts_reply(&reply) {
                Ok(facts) => {
                    let mut added = 0usize;
                    let mut rejected = 0usize;
                    let mut untraceable_section = 0usize;
                    for (question, answer) in facts.pairs {
                        let key = normalize(&question);
                        if !seen.insert(key) {
                            continue; // already present: skipped, not rewritten
                        }
                        // The gate refuses what the instruction asked for
                        // and the generator half-delivered: a question the
                        // title cannot anchor would train this chip's
                        // answers onto other chips' questions.
                        if !question_is_anchored(&question, &identifiers) {
                            rejected += 1;
                            continue;
                        }
                        // The traceability gate refuses invented numbers:
                        // a trained wrong number is worse than no fact,
                        // and recall against the same wrong reference can
                        // never catch it.
                        if !answer_numbers_traceable(&answer, section) {
                            untraceable_section += 1;
                            continue;
                        }
                        records.push(training_record(&run_id, &question, &answer));
                        added += 1;
                        // One negative per second accepted fact: enough to
                        // teach the boundary without letting the shared
                        // abstention target outweigh the facts themselves.
                        if added.is_multiple_of(2) && !options.scope_negatives.is_empty() {
                            let id = identifiers.iter().find(|id| question.contains(id.as_str()));
                            if let (Some(id), Some(negative)) = (
                                id,
                                options
                                    .scope_negatives
                                    .get(negatives_emitted % options.scope_negatives.len()),
                            ) {
                                negatives_emitted += 1;
                                if let Some(negative) =
                                    negative_question(&question, id, negative, &identifiers)
                                {
                                    records.push(training_record(&run_id, &negative, NOT_COVERED));
                                }
                            }
                        }
                    }
                    unanchored += rejected;
                    untraceable += untraceable_section;
                    serde_json::json!({
                        "section": n, "facts": added,
                        "unanchored": rejected, "untraceable": untraceable_section
                    })
                }
                Err(e) => {
                    parse_failures += 1;
                    serde_json::json!({ "section": n, "facts": 0, "parse_failure": format!("{e:#}") })
                }
            },
            Err(e) => {
                parse_failures += 1;
                serde_json::json!({ "section": n, "facts": 0, "completion_error": format!("{e:#}") })
            }
        };
        trace.event("section_explored", &mut payload)?;
    }

    // The dataset is written atomically at the end: a crash mid-exploration
    // leaves no half-file the trainer could read as complete.
    let mut jsonl = String::new();
    for record in &records {
        jsonl.push_str(&serde_json::to_string(record)?);
        jsonl.push('\n');
    }
    write_atomic(&options.out, &jsonl)?;

    let summary = ExploreSummary {
        run_id: run_id.clone(),
        sections: sections.len(),
        facts: records.len(),
        parse_failures,
        unanchored,
        untraceable,
    };
    let mut outcome = serde_json::json!({
        "schema": 1,
        "run_id": run_id,
        "status": if parse_failures > 0 { "completed_with_parse_failures" } else { "completed" },
        "file": options.file.display().to_string(),
        "out": options.out.display().to_string(),
        "sections": summary.sections,
        "facts": summary.facts,
        "parse_failures": summary.parse_failures,
        "unanchored": summary.unanchored,
        "untraceable": summary.untraceable,
    });
    trace.event("explore_outcome", &mut outcome)?;
    write_atomic(
        &dir.join("outcome.json"),
        &serde_json::to_string_pretty(&outcome)?,
    )?;
    manifest.status = "completed".into();
    manifest.updated_ts = splinter_store::clock::utc_now();
    write_atomic(
        &dir.join("run.json"),
        &serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(summary)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Headings split; a table stays with its heading's section.
    #[test]
    fn sections_split_at_headings_and_tables_stay_whole() {
        let text = "# Title\n\nintro\n\n## Timers\n\nTIM1 is 16-bit.\n\n\
            | Block | Limit |\n|---|---|\n| SPI1 | 42 Mbit/s |\n\n## Clocks\n\nHSI is 16 MHz.\n";
        let sections = split_sections(text, None);
        assert_eq!(sections.len(), 3, "{sections:#?}");
        assert!(sections[1].contains("TIM1 is 16-bit."));
        // The whole table rides with the section that opened it.
        assert!(sections[1].contains("| SPI1 | 42 Mbit/s |"));
        assert!(sections[2].contains("HSI is 16 MHz."));
    }

    /// A section over the cap starts a new chunk at the next heading - and
    /// content in flight stays with the section that holds it, so the cap
    /// never tears a table apart.
    #[test]
    fn chunk_lines_caps_sections_at_the_next_heading() {
        let text = "## A\n\none\n\n## B\n\ntwo\n\n## C\n\nthree\n";
        let sections = split_sections(text, Some(4));
        assert_eq!(sections.len(), 3, "{sections:#?}");
        assert!(sections[0].starts_with("## A"));
        assert!(sections[1].starts_with("## B"));
        assert!(sections[2].starts_with("## C"));
    }

    /// A heading-less document must still chunk: many fact sheets use a
    /// single `#` plus bullets, and under the old heading-only rule the
    /// whole document was one section - too large for a small generator,
    /// which then enumerates a fraction of the claims. Blank lines are the
    /// paragraph boundaries every markdown document has.
    #[test]
    fn an_overflowing_headingless_document_chunks_at_paragraph_boundaries() {
        let text = "# Device\n\nalpha fact\n\nbeta fact\n\ngamma fact\n\ndelta fact\n";
        let sections = split_sections(text, Some(2));
        assert_eq!(sections.len(), 4, "{sections:#?}");
        assert!(sections[0].contains("# Device"));
        assert!(sections[0].contains("alpha fact"), "{}", sections[0]);
        assert!(sections[1].contains("beta fact"), "{}", sections[1]);
        assert!(sections[2].contains("gamma fact"), "{}", sections[2]);
        assert!(sections[3].contains("delta fact"), "{}", sections[3]);
    }

    /// The boundary rule must never park a heading alone: a heading with
    /// nothing under it gives the generator no facts to enumerate.
    #[test]
    fn a_chunk_boundary_never_orphans_a_heading() {
        let text = "# Device\n\n## Clocks\n\nHSI is 16 MHz.\n\nLSI is 32 kHz.\n";
        let sections = split_sections(text, Some(2));
        assert_eq!(sections.len(), 2, "{sections:#?}");
        assert!(sections[0].contains("## Clocks"));
        // The heading rides with its first paragraph, never alone.
        assert!(sections[0].contains("HSI is 16 MHz."), "{}", sections[0]);
        assert!(sections[1].contains("LSI is 32 kHz."), "{}", sections[1]);
    }

    /// Chunks past the document's own section are cut off from the
    /// document's subject: the generator is told to use the exact
    /// identifiers from the section it sees, so a chunk that no longer
    /// contains the chip name produces unanchored questions ("What is the
    /// maximum frequency of the SPI/I²S?") - the anchoring instruction
    /// cannot be followed from a section that never names the device.
    /// Every chunk therefore carries the document's title line.
    #[test]
    fn every_chunk_carries_the_document_subject() {
        let text = "# STM32F407 fact sheet\n\n## Clocks\n\nHSI is 16 MHz.\n\n## Timers\n\nTIM1 is 16-bit.\n";
        let sections = split_sections(text, None);
        assert_eq!(sections.len(), 2, "{sections:#?}");
        for s in &sections {
            assert!(
                s.contains("STM32F407 fact sheet"),
                "chunk lost the document subject: {s}"
            );
        }
    }

    /// A well-formed reply parses into its question/answer pairs.
    #[test]
    fn a_well_formed_facts_reply_parses() {
        let reply = r#"{"facts": [
            {"question": "What is the max SPI1 clock?", "answer": "42 Mbit/s"},
            {"question": "How many SPI controllers?", "answer": "3"}
        ]}"#;
        let facts = parse_facts_reply(reply).unwrap();
        assert_eq!(facts.pairs.len(), 2);
        assert_eq!(facts.pairs[0].0, "What is the max SPI1 clock?");
        assert_eq!(facts.pairs[0].1, "42 Mbit/s");
    }

    /// Malformed JSON and prose-wrapped JSON are refusals, not salvages.
    #[test]
    fn malformed_and_prose_wrapped_replies_are_refused() {
        assert!(parse_facts_reply("not json at all").is_err());
        assert!(parse_facts_reply("{\"facts\": [{\"question\": 1}]}").is_err());
        assert!(parse_facts_reply("{\"facts\": []").is_err());
        // An array, not an object.
        assert!(parse_facts_reply("[{\"facts\": []}]").is_err());
        let wrapped =
            "Here are the facts:\n{\"facts\": [{\"question\": \"q\", \"answer\": \"a\"}]}";
        assert!(
            parse_facts_reply(wrapped).is_err(),
            "prose around the object is a parse failure"
        );
    }

    /// Fences ARE tolerated - they are decoration, not prose.
    #[test]
    fn fenced_replies_parse() {
        let reply = "```json\n{\"facts\": [{\"question\": \"q\", \"answer\": \"a\"}]}\n```";
        let facts = parse_facts_reply(reply).unwrap();
        assert_eq!(facts.pairs.len(), 1);
    }

    /// The generator instruction must demand subject-anchored questions.
    /// A question that does not name its subject trains a string-matcher:
    /// the adapter answers "What is the maximum CPU clock frequency?" and
    /// breaks the moment "of the STM32F407" is appended - measured on the
    /// first STM32 training run (168 MHz learned, "48 MHz" hallucinated
    /// when the chip family was named). The prompt is the contract.
    #[test]
    fn the_generator_instruction_demands_subject_anchored_questions() {
        let prompt = facts_prompt("## Clocks\n\nHSI is 16 MHz.\n");
        let lower = prompt.to_lowercase();
        for phrase in [
            "name the specific device",
            "not \"what is the maximum frequency?\" but",
            "fit a different device unchanged",
            // The title line rides in every chunk precisely so the
            // instruction can point the generator at it.
            "the section opens with the document's title",
        ] {
            assert!(
                lower.contains(phrase),
                "the instruction must say {phrase:?}; prompt:\n{prompt}"
            );
        }
    }

    /// An ask reply parses to its answer string; prose is a refusal.
    #[test]
    fn ask_replies_parse_strictly() {
        assert_eq!(
            parse_answer_reply("{\"answer\": \"42 Mbit/s\"}").unwrap(),
            "42 Mbit/s"
        );
        assert_eq!(
            parse_answer_reply("```\n{\"answer\": \"168 MHz\"}\n```").unwrap(),
            "168 MHz"
        );
        assert!(parse_answer_reply("The answer is 168 MHz.").is_err());
        assert!(parse_answer_reply("{\"result\": \"168 MHz\"}").is_err());
    }

    /// Dedup is by normalized question text: case and whitespace collapse.
    #[test]
    fn question_dedup_normalizes_text() {
        assert_eq!(
            normalize("  What  is the MAX? "),
            normalize("what is the max?")
        );
        let mut seen = std::collections::HashSet::new();
        assert!(seen.insert(normalize("What is the max?")));
        assert!(!seen.insert(normalize("what is  the max?")));
    }

    /// The anchor gate is the pipeline-side enforcement of the generator
    /// instruction. An unanchored question is the retrieval key that leaks
    /// this chip's answers onto other chips' questions (the measured defect:
    /// a trained adapter answering a CPU-clock question about the wrong
    /// family with this family's SDIO number), so the pipeline drops it
    /// rather than training it.
    #[test]
    fn the_anchor_gate_refuses_questions_the_title_does_not_anchor() {
        let ids = title_identifiers("# STM32F405 / STM32F407 \u{2014} Technical Fact Sheet");
        assert_eq!(ids, vec!["STM32F405", "STM32F407"], "{ids:?}");
        assert!(question_is_anchored(
            "What is the maximum CPU clock frequency of the STM32F407?",
            &ids
        ));
        assert!(!question_is_anchored(
            "What is the maximum frequency of the USART/UART?",
            &ids
        ));
        // A title without an identifier names no device: the gate disables
        // itself instead of rejecting every question.
        let generic = title_identifiers("A Technical Fact Sheet");
        assert!(generic.is_empty());
        assert!(question_is_anchored("Any question at all", &generic));
    }

    /// Scope negatives teach the boundary an all-positive dataset cannot
    /// express: the same question with an out-of-scope device substituted
    /// must train the abstention, because the measured alternative - a
    /// facts-only adapter - answered "168 MHz" for an STM32F103 whose true
    /// maximum is 72 MHz. A confident wrong number on the wrong chip is
    /// worse than an honest refusal.
    #[test]
    fn a_scope_negative_swaps_the_device_and_trains_the_boundary() {
        let ids: Vec<String> = ["STM32F405", "STM32F407"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let question = "What is the maximum CPU clock frequency of the STM32F407?";
        let negative = negative_question(question, "STM32F407", "STM32F103", &ids).unwrap();
        assert_eq!(
            negative,
            "What is the maximum CPU clock frequency of the STM32F103?"
        );
        // No identifier in the question: no negative variant exists.
        assert_eq!(
            negative_question("What is 2+2?", "STM32F407", "STM32F103", &ids),
            None
        );
    }

    /// A question usually names the fact sheet's devices as a
    /// slash-separated run ("STM32F405 / STM32F407"). Substituting one
    /// member leaves the covered device in the question, which trains a
    /// compound-question refusal instead of the scope boundary - the probe
    /// asks about one out-of-scope device alone. The whole run must go.
    #[test]
    fn a_scope_negative_replaces_the_whole_device_run() {
        let ids: Vec<String> = ["STM32F405", "STM32F407"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let question = "What is the maximum CPU clock frequency of the STM32F405 / STM32F407?";
        let negative = negative_question(question, "STM32F405", "STM32F103", &ids).unwrap();
        assert_eq!(
            negative,
            "What is the maximum CPU clock frequency of the STM32F103?"
        );
        // A non-identifier token after the slash stops the run.
        let question = "How many streams does DMA1 on the STM32F405 / its sibling have?";
        let negative = negative_question(question, "STM32F405", "STM32F103", &ids).unwrap();
        assert_eq!(
            negative,
            "How many streams does DMA1 on the STM32F103 / its sibling have?"
        );
    }

    /// The traceability gate: every number an answer states must be
    /// traceable to the section. The generator's measured failure mode is
    /// inventing plausible numbers - "160 MHz" for a 168 MHz rating, four
    /// DMA streams for eight, 100,000,000 erase cycles for 10,000 - and
    /// the recall score cannot catch it, because recall compares against
    /// the same wrong reference. Unit words bind to the number they
    /// follow (plural-insensitive, case-insensitive) and must co-occur on
    /// one section line; thousand separators are normalized.
    #[test]
    fn an_answer_number_untraceable_to_the_section_is_refused() {
        let section = "- **CPU:** up to **168 MHz**. - **DMA:** DMA1 and DMA2, 8 streams each.\n- Endurance: 10,000 erase cycles. 3 × 12-bit ADCs.\n";
        assert!(answer_numbers_traceable(
            "The maximum CPU clock frequency is 168 MHz.",
            section
        ));
        assert!(answer_numbers_traceable("8 streams", section));
        assert!(answer_numbers_traceable("12-bit ADCs", section));
        assert!(!answer_numbers_traceable(
            "The maximum CPU clock frequency is 160 MHz.",
            section
        ));
        assert!(!answer_numbers_traceable("4 streams", section));
        assert!(!answer_numbers_traceable(
            "The maximum flash endurance is 100,000,000 operations.",
            section
        ));
        // A number inside a larger number is not a match ("14" is not in
        // "114"); a bare number matches a decimal that starts with it only
        // at a token boundary ("4" is not in "4.223").
        let sizes = "WLCSP90, approximately 4.223×3.969 mm / 72; LQFP144, 20×20 mm / 114.";
        assert!(!answer_numbers_traceable("114 GPIO pins", sizes));
        assert!(answer_numbers_traceable("4.223 mm", sizes));
        assert!(!answer_numbers_traceable("14 mm", sizes));
    }

    /// An explore-produced record is exactly what the trainer's dataset reader
    /// parses,
    /// and the assistant side teaches the shape `ask` parses: the answer
    /// wrapped as one {"answer": ...} object. Training on bare answers
    /// makes a fine-tuned model drop the wrapper and every strict parse
    /// then fails on the model's own (correct) reply.
    #[test]
    fn an_explore_record_is_valid_pool_input() {
        let record = training_record("explore-test", "What is the max?", "42 Mbit/s");
        assert_eq!(
            record["messages"][0],
            serde_json::json!({"role": "user", "content": "What is the max?", "train": false})
        );
        assert_eq!(
            record["messages"][1],
            serde_json::json!({"role": "assistant", "content": r#"{"answer":"42 Mbit/s"}"#, "train": true})
        );
        assert_eq!(record["metadata"]["run_id"], "explore-test");
        assert_eq!(record["metadata"]["verified_by"], serde_json::json!([]));
    }
}
